import os
import io
import time
import pickle
import hashlib
import pymupdf
from PIL import Image
import numpy as np
import streamlit as st
import faiss
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from sentence_transformers import SentenceTransformer
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY is missing from your .env file.")
    st.stop()
client = genai.Client(api_key=api_key)
GEMINI_MODEL = "gemini-3.6-flash"
st.set_page_config(
    page_title="ResearchBuddy",
    page_icon="📚",
    layout="wide"
)
st.title("📚 ResearchBuddy")
st.caption("Lightweight Multimodal Academic PDF Assistant")
# ============================================================
# 3. CACHE DIRECTORY
# ============================================================

CACHE_DIR = ".rb_cache"
CACHE_VERSION = "v2_chunks"

os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedder():
    """
    Local embedding model.
    Runs locally and does not use Gemini API quota.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


# ============================================================
# 5. PDF → IMAGES
# ============================================================

def pdf_to_images(pdf_bytes):
    """
    Convert every PDF page into an image.

    These images are later used as visual context for Gemini.
    """

    doc = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    images = []

    for page_num in range(len(doc)):

        page = doc[page_num]

        pix = page.get_pixmap(
            dpi=120,
            alpha=False
        )

        img = Image.open(
            io.BytesIO(
                pix.tobytes("png")
            )
        )

        images.append(img)

    return images, doc


# ============================================================
# 6. EXTRACT TEXT PAGE BY PAGE
# ============================================================

def extract_text_per_page(doc):
    """
    Extract text separately for every PDF page.
    """

    return [
        page.get_text("text")
        for page in doc
    ]


# ============================================================
# 7. TEXT CHUNKING
# ============================================================

def create_chunks(
    texts,
    chunk_size=1000,
    overlap=200
):
    """
    Split each page into smaller overlapping chunks.

    Returns:
        chunks       -> actual text chunks
        chunk_pages  -> original PDF page for each chunk
    """

    chunks = []
    chunk_pages = []

    for page_idx, page_text in enumerate(texts):

        page_text = page_text.strip()

        # Skip completely empty pages
        if not page_text:
            continue

        start = 0
        text_length = len(page_text)

        while start < text_length:

            end = min(
                start + chunk_size,
                text_length
            )

            chunk = page_text[start:end].strip()

            if chunk:
                chunks.append(chunk)
                chunk_pages.append(page_idx)

            # Stop when we reach the end
            if end >= text_length:
                break

            start += chunk_size - overlap

    return chunks, chunk_pages


# ============================================================
# 8. CREATE EMBEDDINGS
# ============================================================

def get_text_embeddings(texts):
    """
    Convert text chunks into normalized embeddings.

    Normalization allows us to use FAISS Inner Product
    as cosine similarity.
    """

    cleaned = []

    for text in texts:

        text = text.strip()

        if text:
            cleaned.append(text)
        else:
            cleaned.append(
                "Empty document context."
            )

    embeddings = embedder.encode(
        cleaned,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    return np.asarray(
        embeddings,
        dtype=np.float32
    )


# ============================================================
# 9. BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):
    """
    Build FAISS Inner Product index.

    Because embeddings are normalized,
    inner product = cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# 10. CACHE KEY
# ============================================================

def cache_key_for(pdf_bytes):

    file_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()

    return f"{CACHE_VERSION}_{file_hash}"


# ============================================================
# 11. GEMINI RETRY FUNCTION
# ============================================================

def generate_with_retry(
    contents,
    model=GEMINI_MODEL,
    retries=3
):
    """
    Call Gemini with retry handling for quota errors.
    """

    for attempt in range(retries):

        try:

            response = client.models.generate_content(
                model=model,
                contents=contents
            )

            return response

        except ClientError as e:

            error_text = str(e)

            if (
                "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
            ):

                wait = 2 ** attempt

                st.warning(
                    f"Gemini quota limit reached. "
                    f"Retrying in {wait} seconds..."
                )

                time.sleep(wait)

            else:

                raise

    raise RuntimeError(
        "Gemini quota exhausted after retries. "
        "Please try again later."
    )


# ============================================================
# 12. SESSION STATE
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = None

if "pages_text" not in st.session_state:
    st.session_state.pages_text = []

if "pages_images" not in st.session_state:
    st.session_state.pages_images = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "chunk_pages" not in st.session_state:
    st.session_state.chunk_pages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

if "retrieved_pages" not in st.session_state:
    st.session_state.retrieved_pages = []

if "best_page" not in st.session_state:
    st.session_state.best_page = None


# ============================================================
# 13. SIDEBAR - DOCUMENT INGESTION
# ============================================================

with st.sidebar:

    st.header("📄 Document Ingestion")

    uploaded_file = st.file_uploader(
        "Upload a Research Paper (PDF)",
        type=["pdf"]
    )

    process_button = st.button(
        "🚀 Process Paper",
        use_container_width=True
    )

    if uploaded_file and process_button:

        with st.spinner(
            "Processing PDF locally..."
        ):

            # ------------------------------------------------
            # Read PDF
            # ------------------------------------------------

            pdf_bytes = uploaded_file.read()

            key = cache_key_for(
                pdf_bytes
            )

            cache_path = os.path.join(
                CACHE_DIR,
                f"{key}.pkl"
            )

            # ------------------------------------------------
            # Check cache
            # ------------------------------------------------

            if os.path.exists(cache_path):

                with open(
                    cache_path,
                    "rb"
                ) as f:

                    cached = pickle.load(f)

                images = cached["images"]
                texts = cached["texts"]
                chunks = cached["chunks"]
                chunk_pages = cached["chunk_pages"]
                embeddings = cached["embeddings"]

                st.info(
                    "Loaded document from local cache."
                )

            else:

                # ------------------------------------------------
                # Convert PDF to images
                # ------------------------------------------------

                images, doc = pdf_to_images(
                    pdf_bytes
                )

                # ------------------------------------------------
                # Extract text
                # ------------------------------------------------

                texts = extract_text_per_page(
                    doc
                )

                # ------------------------------------------------
                # Create chunks
                # ------------------------------------------------

                chunks, chunk_pages = create_chunks(
                    texts,
                    chunk_size=1000,
                    overlap=200
                )

                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                embeddings = get_text_embeddings(
                    chunks
                )

                # ------------------------------------------------
                # Save cache
                # ------------------------------------------------

                with open(
                    cache_path,
                    "wb"
                ) as f:

                    pickle.dump(
                        {
                            "images": images,
                            "texts": texts,
                            "chunks": chunks,
                            "chunk_pages": chunk_pages,
                            "embeddings": embeddings
                        },
                        f
                    )

            # ------------------------------------------------
            # Build FAISS
            # ------------------------------------------------

            index = build_faiss_index(
                embeddings
            )

            # ------------------------------------------------
            # Save everything to session
            # ------------------------------------------------

            st.session_state.pages_images = images
            st.session_state.pages_text = texts

            st.session_state.chunks = chunks
            st.session_state.chunk_pages = chunk_pages

            st.session_state.index = index

            st.session_state.chat_history = []

            st.session_state.retrieved_chunks = []
            st.session_state.retrieved_pages = []
            st.session_state.best_page = None

            st.success(
                f"✅ Processed {len(images)} pages "
                f"into {len(chunks)} searchable chunks."
            )


# ============================================================
# 14. MAIN APPLICATION
# ============================================================

if st.session_state.index is not None:

    col1, col2 = st.columns(
        [1, 1]
    )

    # ========================================================
    # LEFT COLUMN - CHAT
    # ========================================================

    with col1:

        st.subheader(
            "💬 Interactive Query"
        )

        # ----------------------------------------------------
        # Display previous conversation
        # ----------------------------------------------------

        for msg in st.session_state.chat_history:

            with st.chat_message(
                msg["role"]
            ):

                st.write(
                    msg["content"]
                )

        # ----------------------------------------------------
        # User question
        # ----------------------------------------------------

        user_query = st.chat_input(
            "Ask something about the research paper..."
        )

        if user_query:

            # ------------------------------------------------
            # Store user message
            # ------------------------------------------------

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": user_query
                }
            )

            with st.chat_message("user"):

                st.write(
                    user_query
                )

            # ------------------------------------------------
            # Assistant response
            # ------------------------------------------------

            with st.chat_message("assistant"):

                with st.spinner(
                    "Searching relevant chunks and analyzing..."
                ):

                    try:

                        # ========================================
                        # STEP 1: EMBED USER QUERY
                        # ========================================

                        query_embedding = embedder.encode(
                            [user_query],
                            normalize_embeddings=True
                        )

                        query_vector = np.asarray(
                            query_embedding,
                            dtype=np.float32
                        )

                        # ========================================
                        # STEP 2: SEARCH FAISS
                        # ========================================

                        total_chunks = len(
                            st.session_state.chunks
                        )

                        # Retrieve top 4 chunks
                        k = min(
                            4,
                            total_chunks
                        )

                        scores, indices = (
                            st.session_state.index.search(
                                query_vector,
                                k
                            )
                        )

                        top_chunk_indices = indices[0]

                        top_scores = scores[0]

                        # ========================================
                        # STEP 3: REMOVE INVALID INDICES
                        # ========================================

                        valid_results = []

                        for chunk_idx, score in zip(
                            top_chunk_indices,
                            top_scores
                        ):

                            if chunk_idx >= 0:

                                valid_results.append(
                                    (
                                        int(chunk_idx),
                                        float(score)
                                    )
                                )

                        # ========================================
                        # STEP 4: BUILD CONTEXT
                        # ========================================

                        context_parts = []

                        retrieved_page_numbers = []

                        for (
                            chunk_idx,
                            score
                        ) in valid_results:

                            page_idx = (
                                st.session_state
                                .chunk_pages[chunk_idx]
                            )

                            chunk_text = (
                                st.session_state
                                .chunks[chunk_idx]
                            )

                            page_number = (
                                page_idx + 1
                            )

                            retrieved_page_numbers.append(
                                page_number
                            )

                            context_parts.append(
                                f"""
--- Page {page_number} ---
Relevance score: {score:.3f}

{chunk_text}
"""
                            )

                        context_text = "\n\n".join(
                            context_parts
                        )

                        # ========================================
                        # STEP 5: FIND BEST PAGE
                        # ========================================

                        if valid_results:

                            best_chunk_idx = (
                                valid_results[0][0]
                            )

                            best_page_idx = (
                                st.session_state
                                .chunk_pages[
                                    best_chunk_idx
                                ]
                            )

                            st.session_state.best_page = (
                                best_page_idx
                            )

                        else:

                            st.session_state.best_page = None

                        # ========================================
                        # STEP 6: SAVE RETRIEVED INFO
                        # ========================================

                        st.session_state.retrieved_chunks = (
                            valid_results
                        )

                        st.session_state.retrieved_pages = (
                            sorted(
                                set(
                                    retrieved_page_numbers
                                )
                            )
                        )

                        # ========================================
                        # STEP 7: CREATE GEMINI PROMPT
                        # ========================================

                        prompt = f"""
You are ResearchBuddy, an AI research assistant
for academic papers.

Your job is to answer the user's question using
ONLY the retrieved information from the uploaded
research paper.

IMPORTANT RULES:

1. Use only the provided document context.
2. Do not invent information.
3. If the answer is not supported by the context,
   clearly say that the information was not found.
4. Give a clear and concise answer.
5. When using information from a page, cite it like:
   [Page 4]
6. If multiple pages support the answer, cite all
   relevant pages.
7. Do not cite pages that do not support your answer.
8. Distinguish between findings reported by the
   researchers and your own interpretation.
9. Do not use outside knowledge.

USER QUESTION:

{user_query}

RETRIEVED DOCUMENT CONTEXT:

{context_text}

Now answer the user's question.
"""

                        # ========================================
                        # STEP 8: SEND BEST PAGE IMAGE
                        # ========================================

                        contents = [prompt]

                        if (
                            st.session_state.best_page
                            is not None
                        ):

                            best_image = (
                                st.session_state
                                .pages_images[
                                    st.session_state
                                    .best_page
                                ]
                            )

                            contents.append(
                                best_image
                            )

                        # ========================================
                        # STEP 9: GEMINI
                        # ========================================

                        response = generate_with_retry(
                            contents
                        )

                        answer = response.text

                        if not answer:

                            answer = (
                                "I could not generate "
                                "an answer from the "
                                "retrieved document context."
                            )

                    except Exception as e:

                        answer = (
                            f"⚠️ Error generating answer:\n\n"
                            f"{e}"
                        )

                    # ============================================
                    # DISPLAY ANSWER
                    # ============================================

                    st.write(answer)

                    # ============================================
                    # SAVE ASSISTANT MESSAGE
                    # ============================================

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )


    # ========================================================
    # RIGHT COLUMN - VISUAL CONTEXT
    # ========================================================

    with col2:

        st.subheader(
            "🔎 Retrieved Visual Context"
        )

        if (
            st.session_state.best_page
            is not None
        ):

            best_page = (
                st.session_state.best_page
            )

            st.image(
                st.session_state.pages_images[
                    best_page
                ],
                caption=(
                    f"Most Relevant Page "
                    f"{best_page + 1}"
                ),
                use_container_width=True
            )

            # ------------------------------------------------
            # Retrieved chunks information
            # ------------------------------------------------

            if st.session_state.retrieved_chunks:

                st.markdown(
                    "### Retrieved Chunks"
                )

                for (
                    chunk_idx,
                    score
                ) in st.session_state.retrieved_chunks:

                    page_idx = (
                        st.session_state
                        .chunk_pages[
                            chunk_idx
                        ]
                    )

                    st.markdown(
                        f"""
**Page {page_idx + 1}**  
Similarity: `{score:.3f}`
"""
                    )

        else:

            st.info(
                "Ask a question to view "
                "the most relevant page."
            )


# ============================================================
# 15. NO DOCUMENT MESSAGE
# ============================================================

else:

    st.info(
        "📄 Upload a research paper in the sidebar "
        "and click **Process Paper** to begin."
    )