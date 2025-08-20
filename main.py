import os
import PyPDF2
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import logging
from typing import List

# Load environment variables from .env file manually
def load_env_file():
    """Load environment variables from .env file without python-dotenv dependency."""
    try:
        if os.path.exists('.env'):
            # Try different encodings to handle BOM and various file formats
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open('.env', 'r', encoding=encoding) as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    break  # If successful, break out of the encoding loop
                except UnicodeDecodeError:
                    continue  # Try next encoding
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

load_env_file()

# Fallback: Set API key directly if not loaded from .env
if not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBzZSJ2b60YLpcSF_Job0D__rMwbcCZS8g"

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG Model API",
    description="Multi-layer RAG system for document Q&A",
    version="1.0.0"
)

# Add CORS middleware to allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for API I/O ---
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    refined_question: str
    status: str

class StatusResponse(BaseModel):
    status: str
    message: str

# --- RAG System Core Logic ---
class RAGSystem:
    def __init__(self):
        self.embedding = None
        self.layer1_retriever = None
        self.llm = None
        self.layer2_db = None
        self.refine_query_chain = None
        self.final_rag_chain = None
        self.initialized = False

    def initialize_system(self, gemini_api_key: str, layer1_db_path: str = "faiss_layer1_db"):
        """Initializes the core components: LLM, embeddings, and Layer 1 retriever."""
        try:
            logger.info("🚀 Initializing RAG system...")
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

            logger.info("   - Loading embedding model...")
            self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            if os.path.exists(layer1_db_path):
                logger.info(f"   - Loading Layer 1 DB from {layer1_db_path}...")
                try:
                    db = FAISS.load_local(layer1_db_path, self.embedding, allow_dangerous_deserialization=True)
                    
                    docs = list(db.docstore._dict.values())
                    faiss_retriever = db.as_retriever(search_kwargs={"k": 5})
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 5
                    
                    self.layer1_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])
                    logger.info("   - Layer 1 hybrid retriever created.")
                except UnicodeDecodeError as e:
                    logger.error(f"   - Unicode error loading FAISS database: {e}")
                    raise FileNotFoundError(f"Error loading Layer 1 database: Unicode encoding issue")
                except Exception as e:
                    logger.error(f"   - Error loading FAISS database: {e}")
                    raise FileNotFoundError(f"Error loading Layer 1 database: {e}")
            else:
                raise FileNotFoundError(f"Layer 1 database not found at {layer1_db_path}")

            logger.info("   - Initializing Google Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)
            
            self.initialized = True
            logger.info("✅ RAG system initialized successfully!")
        except Exception as e:
            logger.error(f"❌ Error initializing RAG system: {e}")
            raise

    def create_layer2_from_pdf(self, pdf_path: str):
        """Creates the Layer 2 vector store from a given PDF file path."""
        try:
            logger.info(f"   - Processing PDF: {pdf_path}")
            reader = PyPDF2.PdfReader(pdf_path)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())

            if not text.strip():
                raise ValueError("No text could be extracted from the PDF.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            docs = [Document(page_content=chunk) for chunk in text_splitter.split_text(text)]
            
            self.layer2_db = FAISS.from_documents(docs, self.embedding)
            logger.info("   - Layer 2 vector store created successfully.")
        except Exception as e:
            logger.error(f"❌ Error creating Layer 2 from PDF: {e}")
            raise

    def setup_chains(self):
        """Sets up the LangChain runnables for the complete RAG pipeline."""
        if not self.layer2_db:
            raise ValueError("Cannot setup chains without an initialized Layer 2 database.")

        # --- Helper Function ---
        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        # --- Chain 1: Refine the user's question ---
        refine_prompt = PromptTemplate.from_template(
            """Refine the original question based on the initial context from a general database. 
            This creates a better query for searching the specific policy document.
            Original Question: {question}
            Initial Context:
            ---
            {context}
            ---
            Refined Question:"""
        )
        self.refine_query_chain = (
            {"context": self.layer1_retriever | format_docs, "question": RunnablePassthrough()}
            | refine_prompt
            | self.llm
            | StrOutputParser()
        )

        # --- Chain 2: Generate the final answer ---
        layer2_retriever = self.layer2_db.as_retriever(search_kwargs={"k": 7})
        final_prompt = PromptTemplate.from_template(
            """You are an expert assistant. Answer the user's question based ONLY on the final context from the specific document.
            If the information isn't in the context, say so. Do not make up answers.
            Final Context:
            ---
            {context}
            ---
            Question: {question}
            Answer:"""
        )
        self.final_rag_chain = (
            {
                "context": self.refine_query_chain | layer2_retriever | format_docs,
                "question": self.refine_query_chain,
            }
            | final_prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("   - Query chains setup complete.")

    def query(self, question: str):
        """Processes a query through the full RAG pipeline."""
        if not self.final_rag_chain or not self.refine_query_chain:
            raise ValueError("RAG chains are not set up. Please upload a PDF first.")

        logger.info(f"   - Invoking RAG chain for question: '{question}'")
        # Invoke the chains separately to get both the refined question and the final answer
        refined_question = self.refine_query_chain.invoke(question)
        final_answer = self.final_rag_chain.invoke(question)
        
        return {
            "answer": final_answer,
            "refined_question": refined_question,
            "status": "success"
        }

# --- Global RAG System Instance ---
rag_system = RAGSystem()

# --- FastAPI Events and Endpoints ---

@app.on_event("startup")
async def startup_event():
    """Initializes the RAG system when the server starts."""
    try:
        # Use the provided API key as the default, but allow override via environment variable
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key:
            logger.error("❌ GOOGLE_API_KEY not found as an env variable or in the code.")
            return
        rag_system.initialize_system(gemini_api_key)
    except Exception as e:
        logger.error(f"❌ Critical error during startup: {e}")

@app.get("/", response_model=StatusResponse, tags=["Status"])
async def root():
    """Root endpoint to confirm the API is running."""
    return StatusResponse(status="ok", message="RAG Model API is running")

@app.get("/health", response_model=StatusResponse, tags=["Status"])
async def health_check():
    """Health check to confirm the RAG system is initialized."""
    if rag_system.initialized:
        return StatusResponse(status="healthy", message="RAG system is initialized and ready")
    return StatusResponse(status="unhealthy", message="RAG system is not initialized")

@app.post("/upload-pdf", response_model=StatusResponse, tags=["Document"])
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF to create the Layer 2 context for querying."""
    if not rag_system.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Cannot process file.")
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    # Securely save the uploaded file to a temporary location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", mode='wb') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        logger.info(f"📄 PDF '{file.filename}' uploaded temporarily to '{tmp_path}'.")
        # Process the temporary file
        rag_system.create_layer2_from_pdf(tmp_path)
        # Set up the chains now that Layer 2 is ready
        rag_system.setup_chains()

    except Exception as e:
        logger.error(f"❌ Error during PDF processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up the temporary file
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.info(f"   - Cleaned up temporary file: {tmp_path}")

    return StatusResponse(status="success", message=f"PDF '{file.filename}' loaded and processed.")

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(request: QueryRequest):
    """Process a question using the full RAG pipeline against the uploaded PDF."""
    if not rag_system.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized.")
    if not rag_system.layer2_db or not rag_system.final_rag_chain:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded. Please use the /upload-pdf endpoint first.")

    try:
        result = rag_system.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Main Execution Block ---
if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Assumes you save this script as 'main.py'
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
