<h1> Bajaj-RAG: Retrieval-Augmented Generation for Insurance Document QA </h1>

This project implements a Retrieval-Augmented Generation (RAG) pipeline tailored for answering queries related to Bajaj insurance documents using LLMs and vector databases.

 <h1> 🧠 Features  </h1>

📄 Upload and index insurance-related PDFs
🔍 Use FAISS + Sentence Transformers for document retrieval
🤖 Answer user queries using LLMs with retrieved document chunks
🛠️ Modular architecture with FastAPI backend
📦 HuggingFace + LangChain integration

 <h1>🚀 Quick Start  </h1>

Clone the repository
git clone https://github.com/Kabir2007/hackrx-rag-api.git
cd hackrx-rag-api.git

Create and activate a virtual environment

Linux/macOS

bash Copy Edit python -m venv venv source venv/bin/activate Windows

bash Copy Edit python -m venv venv venv\Scripts\activate Install dependencies

bash Copy Edit pip install -r requirements.txt

 <h1> 🔧 Tech Stack  </h1>

LLMs: OpenAI GPT, Google Gemini, HuggingFace Transformers
Embeddings: sentence-transformers/all-MiniLM-L6-v2
Vector Store: FAISS
Sparse Retriever: BM25
Frameworks: LangChain, FastAPI, PyPDF2
