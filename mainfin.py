import os
import tempfile
import logging
import asyncio
import aiohttp
import aiofiles
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import hashlib
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl, Field
import PyPDF2

from dotenv import load_dotenv
load_dotenv()

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_QUESTIONS = 20
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 150
    RETRIEVER_K = 7
    CACHE_TTL = 3600  # 1 hour
    DOWNLOAD_TIMEOUT = 60
    
config = Config()

# Enhanced Request/Response Models
class HackRxRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL to the PDF document")
    questions: List[str] = Field(..., min_items=1, max_items=config.MAX_QUESTIONS, description="List of questions")

class HackRxResponse(BaseModel):
    answers: List[str]
    processing_time: float
    document_hash: str

class HealthResponse(BaseModel):
    status: str
    rag_initialized: bool
    version: str
    uptime: float

# Custom Exceptions
class PDFProcessingError(Exception):
    pass

class DocumentDownloadError(Exception):
    pass

# Enhanced RAG System with Caching and Better Resource Management
class EnhancedRAGSystem:
    def __init__(self):
        self.embedding = None
        self.layer1_retriever = None
        self.llm = None
        self.initialized = False
        self.text_splitter = None
        self.document_cache: Dict[str, tuple] = {}  # hash -> (vectorstore, timestamp)
        self.start_time = time.time()
        
    def initialize_system(self, gemini_api_key: str, layer1_db_path: str = "faiss_layer1_db"):
        """Initialize the RAG system with better error handling."""
        try:
            logger.info("🚀 Initializing Enhanced RAG system...")
            os.environ["GOOGLE_API_KEY"] = gemini_api_key

            # Initialize components
            logger.info("📊 Loading embedding model...")
            self.embedding = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Initialize text splitter once
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.CHUNK_SIZE,
                chunk_overlap=config.CHUNK_OVERLAP
            )

            # Load Layer 1 DB if exists
            if os.path.exists(layer1_db_path):
                logger.info(f"📚 Loading Layer 1 DB from {layer1_db_path}...")
                try:
                    db = FAISS.load_local(
                        layer1_db_path, 
                        self.embedding, 
                        allow_dangerous_deserialization=True
                    )
                    docs = list(db.docstore._dict.values())
                    faiss_retriever = db.as_retriever(
                        search_type="mmr", 
                        search_kwargs={"k": 5}
                    )
                    bm25_retriever = BM25Retriever.from_documents(docs)
                    bm25_retriever.k = 5
                    self.layer1_retriever = EnsembleRetriever(
                        retrievers=[bm25_retriever, faiss_retriever],
                        weights=[0.5, 0.5]
                    )
                    logger.info("✅ Layer 1 hybrid retriever created.")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load Layer 1 DB: {e}")
                    self.layer1_retriever = None
            else:
                logger.info("ℹ️ Layer 1 DB not found. Running without query refinement.")
                self.layer1_retriever = None

            # Initialize LLM
            logger.info("🤖 Initializing Google Gemini LLM...")
            self.llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash", 
                temperature=0.2
            )

            self.initialized = True
            logger.info("✅ Enhanced RAG system initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
            raise RuntimeError(f"Failed to initialize RAG system: {e}")

    def _generate_document_hash(self, url: str) -> str:
        """Generate a hash for document caching."""
        return hashlib.md5(url.encode()).hexdigest()

    def _cleanup_cache(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.document_cache.items()
            if current_time - timestamp > config.CACHE_TTL
        ]
        for key in expired_keys:
            del self.document_cache[key]
            logger.debug(f"🧹 Removed expired cache entry: {key}")

    async def download_pdf_async(self, url: str) -> str:
        """Asynchronously download PDF with better error handling."""
        temp_file = None
        try:
            logger.info(f"📥 Downloading PDF from: {url}")
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=config.DOWNLOAD_TIMEOUT)
            ) as session:
                async with session.get(str(url)) as response:
                    # Validate response
                    if response.status != 200:
                        raise DocumentDownloadError(f"HTTP {response.status}: {response.reason}")
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if 'pdf' not in content_type and not str(url).lower().endswith('.pdf'):
                        logger.warning(f"⚠️ Unexpected content type: {content_type}")
                    
                    # Check file size
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > config.MAX_FILE_SIZE:
                        raise DocumentDownloadError(f"File too large: {content_length} bytes")
                    
                    # Create temporary file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                    temp_file.close()
                    
                    # Download with size check
                    downloaded_size = 0
                    async with aiofiles.open(temp_file.name, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            downloaded_size += len(chunk)
                            if downloaded_size > config.MAX_FILE_SIZE:
                                raise DocumentDownloadError("File too large during download")
                            await f.write(chunk)
                    
                    logger.info(f"📄 PDF downloaded: {temp_file.name} ({downloaded_size} bytes)")
                    return temp_file.name
                    
        except asyncio.TimeoutError:
            raise DocumentDownloadError("Download timeout")
        except aiohttp.ClientError as e:
            raise DocumentDownloadError(f"Network error: {e}")
        except Exception as e:
            # Cleanup on error
            if temp_file and os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise DocumentDownloadError(f"Download failed: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text with better error handling and validation."""
        try:
            logger.info(f"📖 Extracting text from: {Path(pdf_path).name}")
            
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    raise PDFProcessingError("PDF is encrypted and cannot be processed")
                
                # Extract text from all pages
                text_parts = []
                for i, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to extract text from page {i+1}: {e}")
                
                text = "\n".join(text_parts)
                
                if not text.strip():
                    raise PDFProcessingError("No text could be extracted from PDF")
                
                logger.info(f"✅ Extracted {len(text)} characters from {len(reader.pages)} pages")
                return text
                
        except PyPDF2.errors.PdfReadError as e:
            raise PDFProcessingError(f"Invalid PDF file: {e}")
        except Exception as e:
            raise PDFProcessingError(f"Text extraction failed: {e}")

    def create_vector_store(self, text: str) -> FAISS:
        """Create vector store from text with optimization."""
        try:
            # Create document chunks
            chunks = self.text_splitter.split_text(text)
            if not chunks:
                raise PDFProcessingError("No text chunks created")
            
            docs = [Document(page_content=chunk) for chunk in chunks]
            logger.info(f"📊 Created {len(docs)} document chunks")
            
            # Create FAISS vector store
            vectorstore = FAISS.from_documents(docs, self.embedding)
            return vectorstore
            
        except Exception as e:
            raise PDFProcessingError(f"Vector store creation failed: {e}")

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        """Format documents for prompt."""
        return "\n\n".join(doc.page_content for doc in docs)

    async def process_single_question(self, question: str, vectorstore: FAISS) -> str:
        """Process a single question with the document."""
        try:
            # Create retriever
            retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_K})
            
            # Refine question if Layer 1 is available
            refined_question = question
            if self.layer1_retriever:
                logger.debug("🔍 Refining question using Layer 1...")
                refine_prompt = PromptTemplate.from_template(
                    """Refine the original question to be more specific and clear using the initial context.

Original Question: {question}
Initial Context: {context}

Refined Question:"""
                )
                refine_chain = (
                    {"context": self.layer1_retriever | self.format_docs, "question": RunnablePassthrough()}
                    | refine_prompt
                    | self.llm
                    | StrOutputParser()
                )
                refined_question = refine_chain.invoke(question)
                logger.debug(f"📝 Refined: {refined_question}")
            
            # Final answer generation
            final_prompt = PromptTemplate.from_template(
                """You are an expert assistant. Answer the user's question based ONLY on the provided context.
If the answer is not clearly found in the context, say "I cannot find this information in the provided document."
Be precise and concise in your response.

Context:
{context}

Question: {question}

Answer:"""
            )
            
            final_chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | final_prompt
                | self.llm
                | StrOutputParser()
            )
            
            answer = final_chain.invoke(refined_question)
            return answer.strip()
            
        except Exception as e:
            logger.error(f"❌ Question processing failed: {e}")
            return f"Error processing question: {str(e)}"

    async def answer_questions(self, pdf_url: str, questions: List[str]) -> tuple[List[str], str]:
        """Process PDF and answer questions with caching and async operations."""
        if not self.initialized:
            raise RuntimeError("RAG system is not initialized")
        
        # Generate document hash for caching
        doc_hash = self._generate_document_hash(pdf_url)
        
        # Clean up expired cache entries
        self._cleanup_cache()
        
        # Check cache first
        vectorstore = None
        if doc_hash in self.document_cache:
            vectorstore, _ = self.document_cache[doc_hash]
            logger.info(f"📋 Using cached vector store for document: {doc_hash[:8]}")
        
        pdf_path = None
        try:
            if vectorstore is None:
                # Download and process document
                pdf_path = await self.download_pdf_async(pdf_url)
                text = self.extract_text_from_pdf(pdf_path)
                vectorstore = self.create_vector_store(text)
                
                # Cache the vector store
                self.document_cache[doc_hash] = (vectorstore, time.time())
                logger.info(f"💾 Cached vector store for document: {doc_hash[:8]}")
            
            # Process all questions concurrently
            logger.info(f"🤔 Processing {len(questions)} questions...")
            tasks = [
                self.process_single_question(question, vectorstore)
                for question in questions
            ]
            
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"❌ Question {i+1} failed: {answer}")
                    processed_answers.append(f"Error: {str(answer)}")
                else:
                    processed_answers.append(answer)
            
            return processed_answers, doc_hash
            
        finally:
            # Clean up temporary file
            if pdf_path and os.path.exists(pdf_path):
                try:
                    os.unlink(pdf_path)
                    logger.debug(f"🧹 Cleaned up: {pdf_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cleanup {pdf_path}: {e}")

# Initialize RAG system
rag_system = EnhancedRAGSystem()

# Security
security = HTTPBearer()
EXPECTED_API_KEY = os.getenv("API_KEY")

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Bearer token with better error messages."""
    if not EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: API key not set"
        )
    
    if credentials.credentials != EXPECTED_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        logger.error("❌ GOOGLE_API_KEY environment variable not set")
        raise RuntimeError("GOOGLE_API_KEY not configured")
    
    try:
        rag_system.initialize_system(gemini_api_key)
        logger.info("🚀 Application startup complete")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Application shutdown")

# FastAPI app with lifespan management
app = FastAPI(
    title="Enhanced HackRx RAG API",
    description="Efficiently process PDF documents from URLs and answer multiple questions with caching.",
    version="4.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Main endpoint
@app.post("/hackrx/run", response_model=HackRxResponse, tags=["RAG"])
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """
    Process a PDF document from URL and answer multiple questions efficiently.
    Features: Async processing, caching, concurrent question processing.
    """
    if not rag_system.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG system not initialized"
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"🚀 Processing {len(request.questions)} questions for: {request.documents}")
        
        answers, doc_hash = await rag_system.answer_questions(
            str(request.documents), 
            request.questions
        )
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Completed in {processing_time:.2f}s")
        
        return HackRxResponse(
            answers=answers,
            processing_time=round(processing_time, 2),
            document_hash=doc_hash[:8]  # Short hash for reference
        )
        
    except DocumentDownloadError as e:
        logger.error(f"📥 Download error: {e}")
        raise HTTPException(status_code=400, detail=f"Document download failed: {e}")
    except PDFProcessingError as e:
        logger.error(f"📄 PDF error: {e}")
        raise HTTPException(status_code=422, detail=f"PDF processing failed: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Enhanced health check
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with system status."""
    uptime = time.time() - rag_system.start_time
    
    return HealthResponse(
        status="healthy" if rag_system.initialized else "initializing",
        rag_initialized=rag_system.initialized,
        version="4.0.0",
        uptime=round(uptime, 2)
    )

# Cache management endpoint
@app.post("/admin/clear-cache", tags=["Admin"])
async def clear_cache(token: str = Depends(verify_token)):
    """Clear document cache (admin only)."""
    cache_size = len(rag_system.document_cache)
    rag_system.document_cache.clear()
    logger.info(f"🧹 Cleared {cache_size} cache entries")
    
    return {"message": f"Cleared {cache_size} cache entries", "status": "success"}

# Cache status endpoint
@app.get("/admin/cache-status", tags=["Admin"])
async def cache_status(token: str = Depends(verify_token)):
    """Get cache status information."""
    current_time = time.time()
    cache_info = []
    
    for doc_hash, (_, timestamp) in rag_system.document_cache.items():
        age = current_time - timestamp
        cache_info.append({
            "document_hash": doc_hash[:8],
            "age_seconds": round(age, 2),
            "expires_in": round(config.CACHE_TTL - age, 2)
        })
    
    return {
        "total_cached_documents": len(rag_system.document_cache),
        "cache_ttl_seconds": config.CACHE_TTL,
        "cached_documents": cache_info
    }