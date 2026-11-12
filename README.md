# 🔬 ResearchBuddy AI – Intelligent Research Paper Assistant

An AI-powered research assistant that helps users understand, explore, and interact with research papers through intelligent document processing and Retrieval-Augmented Generation (RAG).

ResearchBuddy AI allows users to upload research papers in PDF format and ask questions about their content. It retrieves relevant information and generates meaningful answers, making research easier, faster, and more accessible.

## 🚀 Features

- **PDF Upload:** Upload research papers in PDF format.
- **Text Extraction:** Extract text and content from research documents.
- **Visual Understanding:** Analyze document pages, charts, diagrams, and other visual elements using Gemini 2.0 Flash.
- **Semantic Search:** Find relevant information based on the meaning of user queries.
- **RAG-Based Question Answering:** Generate context-aware answers using retrieved document information.
- **Vector Search:** Store and retrieve document embeddings efficiently using ChromaDB.
- **Interactive Querying:** Ask questions and explore research papers conversationally.

## 🧠 System Architecture

ResearchBuddy AI follows a Retrieval-Augmented Generation (RAG) architecture.

1. **Document Ingestion:** Users upload research papers in PDF format.
2. **Document Processing:** Extract text and process PDF pages using PyMuPDF.
3. **Visual Analysis:** Use Gemini 2.0 Flash to understand charts, diagrams, and visual content.
4. **Text Chunking:** Divide extracted content into smaller, meaningful chunks.
5. **Embedding Generation:** Generate semantic embeddings using Sentence Transformers (`all-MiniLM-L6-v2`).
6. **Vector Storage:** Store embeddings and document chunks in ChromaDB.
7. **Retrieval:** Retrieve relevant document chunks based on the user's query.
8. **Answer Generation:** Generate context-aware responses using the retrieved information and an LLM.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend development |
| Gemini 2.0 Flash | Visual understanding of research papers |
| Sentence Transformers | Text embedding generation |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| PyMuPDF | PDF processing and text extraction |
| RAG | Context-aware question answering |
| LLM | Response generation |

## 📂 Project Structure

```text
ResearchBuddy-AI/
│
├── app.py
├── requirements.txt
├── .env
├── README.md
│
├── uploads/
│   └── research_papers/
│
├── processing/
│   ├── pdf_extraction.py
│   ├── text_chunking.py
│   └── visual_analysis.py
│
├── embeddings/
│   └── embedding_generator.py
│
├── vectorstore/
│   └── chroma_db/
│
└── retrieval/
    └── query_engine.py
```

*Note: This is a suggested structure. Adjust the folder names and files to match your actual project.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ResearchBuddy-AI.git
cd ResearchBuddy-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Mac/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file and add your API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Never upload your API keys or `.env` file to GitHub.

### 5. Run the Application

```bash
python app.py
```

## 💡 Example Use Cases

Users can upload a research paper and ask questions such as:

- What is the main objective of this research paper?
- Explain the methodology used in this paper.
- What algorithms are discussed?
- Summarize the key findings.
- Explain the charts and diagrams.
- What are the limitations of this research?

## 🎯 Key Learning Outcomes

- Understanding Retrieval-Augmented Generation (RAG) architecture.
- Working with PDF document processing.
- Learning semantic search and vector embeddings.
- Integrating Sentence Transformers with ChromaDB.
- Exploring multimodal AI and visual document understanding.
- Building practical AI-powered applications.

## 🔮 Future Enhancements

- Support multiple research papers simultaneously.
- Add citation-based answers with page references.
- Improve table and mathematical formula extraction.
- Add research paper summarization.
- Implement user authentication and chat history.
- Deploy the application for public access.

## 👨‍💻 Author

**Dharshan**

B.Tech – Artificial Intelligence and Machine Learning

Interested in Artificial Intelligence, Machine Learning, Generative AI, and Full-Stack Development.

---

⭐ If you find this project interesting, feel free to star the repository!
