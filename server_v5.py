from flask import Flask, request, Response, session
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic.v1 import BaseModel, Field
from typing import List
from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START
import requests
import pandas as pd
import traceback
import re
import json
from datetime import datetime
import uuid
import threading
import time

# Load environment variables first
load_dotenv()

# Initialize Flask app FIRST - this is critical for Render
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')

# Global initialization state
INITIALIZATION_STATUS = {
    "status": "initializing",
    "message": "Starting up medical AI assistant...",
    "error": None
}

# Global variables for components
model = None
llm = None
retriever = None
vectorstore = None
prior_questions = None
prior_question_lookup = {}
retrieval_grader = None
question_rewriter = None
conversational_rag_chain = None
crags = None

# In-memory conversation storage
conversation_memory = {}

def normalize(text):
    """Clean text for matching while preserving essential words"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# WSL-specific file paths
import platform
if "microsoft" in platform.uname().release.lower():
    SCRAPED_DIR = "/data/scraped"
else:
    SCRAPED_DIR = "data/scraped"

class GraphState(TypedDict):
    """Represents the state of our graph with conversation context."""
    question: str
    generation: str
    web_search: str
    documents: List[str]
    conversation_history: List[dict]
    user_intent: str
    is_greeting: bool
    conversation_id: str
    exact_match: bool
    is_medical: bool

def get_or_create_conversation(conversation_id=None):
    """Get existing conversation or create new one"""
    if not conversation_id:
        conversation_id = str(uuid.uuid4())
    
    if conversation_id not in conversation_memory:
        conversation_memory[conversation_id] = {
            "history": [],
            "user_context": {},
            "created_at": datetime.now().isoformat()
        }
    
    return conversation_id, conversation_memory[conversation_id]

def is_medical_topic(question):
    """Check if question is related to medical topics, prioritizing skin cancer"""
    medical_keywords = [
        "skin cancer", "melanoma", "basal cell", "squamous cell", "bcc", "scc",
        "actinic keratosis", "sunburn", "sun exposure", "uv", "mole", "nevus",
        "skin lesion", "skin biopsy", "skin tumour", "skin tumor", "skin growth",
        "skin check", "skin screening", "skin protection", "sunscreen", "sunblock",
        "skin rash", "skin spot", "skin mark", "skin abnormality", "skin doctor",
        "dermatologist", "skin specialist", "skin condition", "skin disease",
        "cancer", "tumor", "disease", "symptom", "treatment", "diagnosis",
        "health", "medical", "doctor", "physician", "therapy", "surgery",
        "medication", "prognosis", "biopsy", "dermatology"
    ]
    q = question.lower()
    return any(k in q for k in medical_keywords)

def detect_user_intent(message):
    """Detect intent and validate medical relevance"""
    message_lower = message.lower().strip()
    
    # Greetings
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy']
    if any(greeting in message_lower for greeting in greetings):
        return "greeting", True, is_medical_topic(message_lower)
    
    # Questions
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 'do you', 'is', 'are']
    if any(word in message_lower for word in question_words) or message.endswith('?'):
        return "question", False, is_medical_topic(message_lower)
    
    # Casual conversation
    casual_phrases = ['thank you', 'thanks', 'okay', 'ok', 'i see', 'understood', 'bye', 'goodbye']
    if any(phrase in message_lower for phrase in casual_phrases):
        return "casual", True, is_medical_topic(message_lower)
    
    # Default to question if unclear
    return "question", False, is_medical_topic(message_lower)

def create_fallback_system():
    """Create a minimal fallback system for when full initialization fails"""
    global model
    
    try:
        # Try to create at least a basic model
        model = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=400,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )
        
        print("Fallback system created with basic model")
        return True
    except Exception as e:
        print(f"Failed to create fallback system: {e}")
        return False

def initialize_components_background():
    """Initialize all components in background thread"""
    global INITIALIZATION_STATUS, model, llm, retriever, vectorstore
    global prior_questions, prior_question_lookup, retrieval_grader
    global question_rewriter, conversational_rag_chain, crags
    
    try:
        print("Starting background initialization...")
        INITIALIZATION_STATUS["message"] = "Loading AI models..."
        
        # Initialize models first
        model = ChatTogether(
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            temperature=0.7,
            max_tokens=400,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )

        llm = ChatTogether(
            model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
            temperature=0,
            max_retries=1,
            together_api_key=os.getenv("TOGETHER_API_KEY")
        )
        
        INITIALIZATION_STATUS["message"] = "Setting up document retrieval..."
        
        # Setup basic document system
        try:
            doc_splits = setup_retriever_txt()
            if not doc_splits:
                print("No documents from files, creating fallback")
                doc_splits = [Document(page_content="Medical information fallback", metadata={"source": "fallback"})]
        except Exception as e:
            print(f"Error with text retriever: {e}")
            doc_splits = [Document(page_content="Medical information fallback", metadata={"source": "fallback"})]

        # Setup embeddings and vectorstore
        model_name = "BAAI/bge-base-en-v1.5"
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True},
            model_kwargs={'device': 'cpu'}
        )

        vectorstore = FAISS.from_documents(documents=doc_splits, embedding=embedding_model)
        retriever = vectorstore.as_retriever()
        
        INITIALIZATION_STATUS["message"] = "Loading knowledge base..."
        
        # Setup prior questions with error handling
        try:
            prior_questions, prior_question_lookup = setup_prior_questions()
        except Exception as e:
            print(f"Error loading prior questions: {e}")
            prior_question_lookup = {}

        INITIALIZATION_STATUS["message"] = "Finalizing components..."
        
        # Setup remaining components
        retrieval_grader = setup_grader()
        question_rewriter = setup_rewriter()
        
        # Setup conversational chain
        conversational_prompt = setup_conversational_prompt()
        conversational_rag_chain = conversational_prompt | model | StrOutputParser()
        
        # Build workflow
        crags = build_workflow()
        
        INITIALIZATION_STATUS["status"] = "ready"
        INITIALIZATION_STATUS["message"] = "Medical AI Assistant is ready!"
        print("Background initialization completed successfully!")
        
    except Exception as e:
        error_msg = f"Initialization failed: {str(e)}"
        print(error_msg)
        INITIALIZATION_STATUS["status"] = "error"
        INITIALIZATION_STATUS["error"] = error_msg
        
        # Try to create fallback system
        if create_fallback_system():
            INITIALIZATION_STATUS["status"] = "fallback"
            INITIALIZATION_STATUS["message"] = "Running in limited mode"

def setup_retriever_txt():
    """Sets up a retriever from locally scraped text files, filtering for medical content."""
    try:
        loader = DirectoryLoader(
            'data/scraped_test/',
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'autodetect_encoding': True, 'encoding': 'utf-8'}
        )
        docs = loader.load()

        processed_docs = []
        for i, doc in enumerate(docs):
            content = doc.page_content.split('\n', 1)
            if len(content) > 1:
                url = content[0].replace('URL: ', '').strip()
                text = content[1].strip()
                if is_medical_topic(text):
                    doc.page_content = text
                    doc.metadata['source'] = url
                    processed_docs.append(doc)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", " ", ""]
        )

        doc_splits = text_splitter.split_documents(processed_docs)
        return doc_splits
    except Exception as e:
        print(f"Error setting up text retriever: {e}")
        return []

def setup_prior_questions():
    """Load prior questions from CSV, filtering for medical content"""
    try:
        filepath = 'data/text_docs/prior_questions.csv'
        if not os.path.exists(filepath):
            print(f"Prior questions file not found at {filepath}")
            return None, {}
            
        df = pd.read_csv(filepath)
        df = df.dropna()
        df = df[df["questions"].apply(is_medical_topic)]

        prior_question_lookup = {}
        for _, row in df.iterrows():
            original_q = str(row["questions"]).strip()
            norm_q = normalize(original_q)
            answer = str(row["answers"]).strip()
            prior_question_lookup[norm_q] = answer

        df["qa_pair"] = "Q: " + df["questions"] + " A: " + df["answers"]
        documents = [
            Document(page_content=row["qa_pair"], metadata={"source": "qa_pairs.csv"})
            for _, row in df.iterrows()
        ]

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_prior = FAISS.from_documents(documents=documents, embedding=embedding_model)
        
        return vectorstore_prior, prior_question_lookup
        
    except Exception as e:
        print(f"Error loading prior questions: {e}")
        return None, {}

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the medical question, 'yes' or 'no'")

def setup_grader():
    structured_llm_grader = llm.with_structured_output(GradeDocuments, method="json_mode", include_raw=False)
    system = """You are a relevance grader for medical topics, particularly skin cancer. 
    Return JSON with "binary_score" as "yes" if the document is relevant to the medical question, or "no" if it is not."""
    grade_prompt = ChatPromptTemplate([
        ("system", system),
        ("human", 'Retrieved document:\n{document}\n\nUser medical question: {question}'),
    ])
    return grade_prompt | structured_llm_grader

def setup_rewriter():
    """Converts the medical question to a simpler version for web search"""
    system = """You are a question re-writer for medical topics, particularly skin cancer.
    Convert the input question to a simpler version that is easier for web search.
    Focus on key medical terms and intent. The response should be one sentence long."""
    re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Here is the initial medical question: \n\n {question} \n Formulate a simplified medical question."),
    ])
    return re_write_prompt | llm | StrOutputParser()

def setup_conversational_prompt():
    """Setup conversational prompt for medical topics"""
    system_message = """You are a helpful, friendly AI assistant specializing in medical topics, particularly skin cancer.

    Guidelines:
    - Be natural and conversational.
    - Only respond to questions related to medical topics, especially skin cancer.
    - If greeting, respond warmly and ask how you can help with medical questions.
    - For medical questions, provide detailed answers using the context documents.
    - If no relevant documents, acknowledge it and provide a general medical response if possible.
    - If the question is not medical-related, politely redirect to medical topics.
    
    Chat History: {chat_history}
    Retrieved Medical Context: {context}
    Current Medical Question: {question}
    
    Response:"""
    
    return ChatPromptTemplate.from_template(system_message)

def build_workflow():
    """Build the conversational workflow"""
    # This is a simplified version of your workflow
    # You can expand this once the basic deployment works
    workflow = StateGraph(GraphState)
    
    # Add basic nodes for now
    workflow.add_node("generate", simple_generate)
    workflow.add_edge(START, "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def simple_generate(state):
    """Simplified generation for initial deployment"""
    question = state["question"]
    
    if not is_medical_topic(question):
        return {
            "generation": "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question.",
            "question": question
        }
    
    # Use the model if available, otherwise fallback
    if model:
        try:
            prompt = f"You are a medical AI assistant. Please answer this medical question: {question}"
            response = model.invoke(prompt)
            return {"generation": response, "question": question}
        except Exception as e:
            print(f"Error with model: {e}")
    
    return {
        "generation": "I'm currently initializing. Please try again in a moment.",
        "question": question
    }

# CRITICAL: Define routes IMMEDIATELY after app creation
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint - MUST respond quickly"""
    return {
        "status": "healthy", 
        "message": "Medical AI Assistant server is running",
        "initialization": INITIALIZATION_STATUS["status"]
    }

@app.route("/status", methods=["GET"])
def get_status():
    """Get initialization status"""
    return INITIALIZATION_STATUS

@app.route("/chat", methods=["POST"])
def chat():
    """Main chat endpoint"""
    if INITIALIZATION_STATUS["status"] == "initializing":
        return Response(
            "System is still starting up. Please try again in a moment.".encode('utf-8'),
            status=503,
            content_type='text/plain; charset=utf-8'
        )
    
    if INITIALIZATION_STATUS["status"] == "error":
        return Response(
            f"System initialization failed: {INITIALIZATION_STATUS['error']}".encode('utf-8'),
            status=503,
            content_type='text/plain; charset=utf-8'
        )
    
    try:
        data = request.json
        prompt = data.get("prompt") or data.get("user_prompt", "")
        
        if isinstance(prompt, dict):
            prompt = prompt.get("prompt") or prompt.get("user_prompt", "")
        prompt = str(prompt)
        
        intent, is_casual, is_medical = detect_user_intent(prompt)
        
        if not is_medical and not is_casual:
            return Response(
                "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question.".encode('utf-8'),
                content_type='text/plain; charset=utf-8'
            )
        
        # Simple response for now
        if INITIALIZATION_STATUS["status"] == "fallback":
            if model:
                try:
                    response = model.invoke(f"You are a medical AI assistant. Answer this question: {prompt}")
                    return Response(response.encode('utf-8'), content_type='text/plain; charset=utf-8')
                except:
                    pass
            
            return Response(
                "I'm currently running in limited mode. Please try again later.".encode('utf-8'),
                content_type='text/plain; charset=utf-8'
            )
        
        # Full system response
        if crags:
            inputs = {"question": prompt, "conversation_history": [], "is_medical": is_medical}
            result = crags.invoke(inputs)
            generation = result.get("generation", "I couldn't generate a response.")
            return Response(generation.encode('utf-8'), content_type='text/plain; charset=utf-8')
        
        return Response(
            "System not ready yet. Please try again.".encode('utf-8'),
            content_type='text/plain; charset=utf-8'
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        traceback.print_exc()
        return Response(
            "An error occurred processing your request.".encode('utf-8'),
            status=500,
            content_type='text/plain; charset=utf-8'
        )

@app.route("/new_conversation", methods=["POST"])
def new_conversation():
    """Start a new conversation"""
    conv_id = str(uuid.uuid4())
    conversation_memory[conv_id] = {
        "history": [],
        "user_context": {},
        "created_at": datetime.now().isoformat()
    }
    return {"conversation_id": conv_id, "message": "New medical conversation started!"}

# Start background initialization
def start_background_init():
    """Start the background initialization thread"""
    init_thread = threading.Thread(target=initialize_components_background, daemon=True)
    init_thread.start()
    print("Background initialization thread started")

# CRITICAL: Start initialization immediately when module loads
if __name__ == "__main__":
    print("Starting Medical AI Assistant server...")
    start_background_init()
    
    # Get port from environment (Render sets this automatically)
    port = int(os.getenv("PORT", 10000))
    host = "0.0.0.0"  # MUST be 0.0.0.0 for Render
    
    print(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
else:
    # When running with gunicorn, start initialization immediately
    print("Starting background initialization for production...")
    start_background_init()