from flask import Flask, request, Response, session
from flask_cors import CORS  # Add this import at the top
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
    is_medical: bool  # Track if query is medical-related

# In-memory conversation storage
conversation_memory = {}

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

def setup_retriever_txt():
    """Sets up a retriever from locally scraped text files, filtering for medical content."""
    loader = DirectoryLoader(
        'data/scraped_test/',
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'autodetect_encoding': True, 'encoding': 'utf-8'}
    )
    docs = loader.load()

    processed_docs = []
    print("Starting to process documents...")
    for i, doc in enumerate(docs):
        content = doc.page_content.split('\n', 1)
        if len(content) > 1:
            url = content[0].replace('URL: ', '').strip()
            text = content[1].strip()
            # Filter for medical content
            if is_medical_topic(text):
                doc.page_content = text
                doc.metadata['source'] = url
                processed_docs.append(doc)
                print(f"Processed medical document {i + 1}/{len(docs)}: URL = {url}")
            else:
                print(f"Skipped non-medical document {i + 1}/{len(docs)}: URL = {url}")

    print(f"Finished processing {len(processed_docs)} medical documents. Starting to split documents...")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=40, separators=["\n\n", "\n", " ", ""]
    )

    doc_splits = text_splitter.split_documents(processed_docs)
    return doc_splits

def setup_retriever_urls():
    """Load retriever with text from webpages, filtering for medical content"""
    file_url = open('data/urls/webaseloader.txt')
    with file_url as file_url:
        url_txt = file_url.read()
    
    urls = [url for url in url_txt.split("\n") if url.strip()]
    docs = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            loaded_docs = loader.load()
            for doc in loaded_docs:
                if is_medical_topic(doc.page_content):
                    docs.append(doc)
        except Exception as e:
            print(f"Error loading URL {url}: {e}")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=40
    )
    print("Splitting medical documents...")
    return text_splitter.split_documents(docs)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the medical question, 'yes' or 'no'"
    )

def setup_grader():
    structured_llm_grader = llm.with_structured_output(
        GradeDocuments,
        method="json_mode",
        include_raw=False
    )
    
    system = """You are a relevance grader for medical topics, particularly skin cancer. 
    Return JSON with "binary_score" as "yes" if the document is relevant to the medical question, or "no" if it is not."""
    grade_prompt = ChatPromptTemplate(
        [
            ("system", system),
            ("human", 'Retrieved document:\n{document}\n\nUser medical question: {question}'),
        ]
    )
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader

def setup_rewriter():
    """Converts the medical question to a simpler version for web search"""
    system = """You are a question re-writer for medical topics, particularly skin cancer.
    Convert the input question to a simpler version that is easier for web search.
    Focus on key medical terms and intent.
    The response should be one sentence long."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial medical question: \n\n {question} \n Formulate a simplified medical question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter

def setup_conversational_prompt():
    """Setup conversational prompt for medical topics"""
    system_message = """You are a helpful, friendly AI assistant specializing in medical topics, particularly skin cancer.

    Guidelines:
    - Be natural and conversational.
    - Only respond to questions related to medical topics, especially skin cancer.
    - Reference previous messages only if truly needed, but DO NOT say things like 
      "as I mentioned before", "you already asked", or similar.
    - If greeting, respond warmly and ask how you can help with medical questions.
    - If casual conversation, respond appropriately but steer toward medical topics.
    - For medical questions, provide detailed answers using the context documents.
    - If no relevant documents, acknowledge it and provide a general medical response if possible.
    - If the question is not medical-related, politely redirect to medical topics.
    - Maintain conversation flow and context without repeating prior question notices.
    - Be personable and engaging.
    
    Chat History:
    {chat_history}
    
    Retrieved Medical Context:
    {context}
    
    Current Medical Question: {question}
    
    Response:"""
    
    return ChatPromptTemplate.from_template(system_message)

def setup_prior_questions():
    """Load prior questions from CSV, filtering for medical content"""
    try:
        filepath = 'data/text_docs/prior_questions.csv'
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows from prior questions file.")
        df = df.dropna()

        # Filter for medical questions
        df = df[df["questions"].apply(is_medical_topic)]
        print(f"Filtered to {len(df)} medical-related questions.")

        prior_question_lookup = {}
        for _, row in df.iterrows():
            original_q = str(row["questions"]).strip()
            norm_q = normalize(original_q)
            answer = str(row["answers"]).strip()
            prior_question_lookup[norm_q] = answer
            print(f"Added to lookup: '{norm_q}' -> '{answer[:50]}...'")

        df["qa_pair"] = "Q: " + df["questions"] + " A: " + df["answers"]

        documents = [
            Document(
                page_content=row["qa_pair"],
                metadata={"source": "qa_pairs.csv"}
            )
            for _, row in df.iterrows()
        ]
        print(f"Processed {len(documents)} medical documents with combined Q&A.")

        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore_prior = FAISS.from_documents(documents=documents, 
                                            embedding=embedding_model)

        print("Medical prior questions vectorstore created successfully.")
        print(f"Prior medical questions setup complete. {len(prior_question_lookup)} questions in lookup.")
        
        return vectorstore_prior, prior_question_lookup
        
    except Exception as e:
        print(f"Error loading prior questions: {e}")
        traceback.print_exc()
        return None, {}

def handle_conversation_start(state):
    """Handle conversation initialization and medical relevance check"""
    print("---CONVERSATION HANDLER---")
    question = state["question"]
    conversation_history = state.get("conversation_history", [])
    
    intent, is_casual, is_medical = detect_user_intent(question)
    
    if not is_medical and not is_casual:
        return {
            **state,
            "user_intent": intent,
            "is_greeting": is_casual,
            "is_medical": False,
            "generation": "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question."
        }
    
    return {
        **state,
        "user_intent": intent,
        "is_greeting": is_casual,
        "is_medical": is_medical,
        "exact_match": False
    }

def priorq_retriever(state):
    """PRIORITY 1: Retrieve from prior medical questions database"""
    print("---PRIORITY 1: RETRIEVE FROM PRIOR MEDICAL QUESTIONS---")
    question = state["question"]
    is_greeting = state.get("is_greeting", False)
    is_medical = state.get("is_medical", True)

    if not is_medical:
        return {
            "documents": [],
            "question": question,
            "is_greeting": is_greeting,
            "exact_match": False,
            "generation": "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question."
        }

    if is_greeting:
        return {
            "documents": [], 
            "question": question, 
            "is_greeting": is_greeting,
            "exact_match": False
        }

    norm_question = normalize(question)
    print(f"Checking prior medical questions for: '{norm_question}'")
    
    if norm_question in prior_question_lookup:
        print("EXACT MATCH FOUND in prior medical questions!")
        answer = prior_question_lookup[norm_question]
        doc = Document(
            page_content=f"A: {answer}",
            metadata={"source": "qa_pairs.csv", "exact_match": True}
        )
        return {
            "documents": [doc], 
            "question": question, 
            "is_greeting": is_greeting,
            "exact_match": True,
            "generation": answer
        }
    else:
        print(f"No exact match found in prior medical questions.")
        return {
            "documents": [], 
            "question": question, 
            "is_greeting": is_greeting,
            "exact_match": False
        }

def retrieve_local_docs(state):
    """PRIORITY 2: Retrieve medical documents from local sources"""
    print("---PRIORITY 2: RETRIEVE FROM LOCAL MEDICAL DOCUMENTS---")
    question = state["question"]
    is_greeting = state.get("is_greeting", False)
    is_medical = state.get("is_medical", True)
    
    if not is_medical:
        return {
            "documents": [],
            "question": question,
            "web_search": "No",
            "is_greeting": is_greeting,
            "generation": "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question."
        }
    
    if is_greeting:
        return {
            "documents": [],
            "question": question,
            "web_search": "No",
            "is_greeting": is_greeting
        }
    
    documents = retriever.invoke(question)
    print(f"Retrieved {len(documents)} local medical documents")
    
    return {
        "documents": documents,
        "question": question,
        "web_search": "No",
        "is_greeting": is_greeting
    }

def grade_documents_local(state):
    """Grade local medical documents for relevance"""
    print("---GRADE LOCAL MEDICAL DOCUMENT RELEVANCE---")
    question = state["question"]
    documents = state["documents"]
    is_greeting = state.get("is_greeting", False)
    is_medical = state.get("is_medical", True)
    
    if not is_medical:
        return {
            "documents": [],
            "question": question,
            "web_search": "No",
            "generation": "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question."
        }
    
    if is_greeting or not documents:
        return {"documents": documents, "question": question, "web_search": "No"}
    
    filtered_docs = []
    
    for doc in documents[:5]:
        try:
            score = retrieval_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            print(f"Local medical doc grader output: {score}")
            if hasattr(score, "binary_score") and score.binary_score == "yes":
                filtered_docs.append(doc)
                print("---GRADE: LOCAL MEDICAL DOCUMENT RELEVANT---")
            else:
                print("---GRADE: LOCAL MEDICAL DOCUMENT NOT RELEVANT---")
        except Exception as e:
            print(f"Error grading medical document: {e}")
            continue
    
    if filtered_docs:
        print(f"Found {len(filtered_docs)} relevant local medical documents")
        web_search = "No"
    else:
        print("No relevant local medical documents found, will try web search")
        web_search = "Yes"
    
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def transform_query(state):
    """Transform medical query for web search"""
    print("---TRANSFORM MEDICAL QUERY FOR WEB SEARCH---")
    user_input = state["question"]
    rewritten = question_rewriter.invoke({"question": user_input})
    print(f"Original: {user_input}")
    print(f"Rewritten: {rewritten}")
    return {"question": rewritten, "documents": state["documents"], "web_search": "Yes"}

def google_search(query, api_key, search_id, num_results=4):
    """Perform a Google Custom Search for medical topics"""
    query = f"{query} site:*.edu | site:*.gov | site:*.org medical skin cancer"  # Restrict to reputable medical sources
    url = (
        f"https://www.googleapis.com/customsearch/v1"
        f"?key={api_key}&cx={search_id}&q={query}&num={num_results}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        results = response.json()
        urls = []
        seen = set()
        for item in results.get('items', []):
            link = item.get('link')
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            if link and link not in seen and is_medical_topic(snippet):
                urls.append({"url": link, "title": title, "snippet": snippet})
                seen.add(link)
        return urls
    except Exception as e:
        print(f"Google Search API error: {e}")
        return []

def web_search(state):
    """PRIORITY 3: Web search for medical topics"""
    print("---PRIORITY 3: MEDICAL WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    is_medical = state.get("is_medical", True)

    if not is_medical:
        fallback = Document(
            page_content="I'm sorry, I can only answer questions about medical topics, particularly skin cancer.",
            metadata={"source": "fallback"}
        )
        return {"documents": [fallback], "question": question}

    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine = os.getenv("GOOGLE_SEARCH_ID")

    if not api_key or not search_engine:
        print("Google API credentials not found")
        fallback = Document(
            page_content="Web search is not available at the moment. Please try rephrasing your medical question.",
            metadata={"source": "fallback"}
        )
        return {"documents": [fallback], "question": question}

    results = google_search(question, api_key, search_engine)
    urls = [r["url"] for r in results]

    if not urls:
        fallback = Document(
            page_content="No relevant medical information found online.",
            metadata={"source": "fallback"}
        )
        return {"documents": [fallback], "question": question}

    new_docs = []
    for url in urls[:3]:
        try:
            if add_url_to_file(url, "data/urls/websearched_urls.txt"):
                loader = WebBaseLoader(url)
                loaded = loader.load()
                for doc in loaded:
                    if is_medical_topic(doc.page_content):
                        doc.metadata["source"] = url
                        new_docs.append(doc)
        except Exception as e:
            print(f"Error loading medical URL {url}: {e}")

    if not new_docs:
        fallback = Document(
            page_content="No relevant medical information found online.",
            metadata={"source": "fallback"}
        )
        documents = [fallback]
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=40
        )
        splits = text_splitter.split_documents(new_docs)
        valid_splits = [
            split for split in splits
            if hasattr(split, "page_content") and split.page_content and len(split.page_content.strip()) > 10
        ]

        if valid_splits:
            try:
                vectorstore.add_documents(valid_splits)
                print(f"Added {len(valid_splits)} new medical documents to vectorstore")
            except Exception as e:
                print(f"Error adding medical documents to vectorstore: {e}")
            documents = valid_splits[:3]
        else:
            fallback = Document(
                page_content="No relevant medical information found online.",
                metadata={"source": "fallback"}
            )
            documents = [fallback]

    return {"documents": documents, "question": question}

def generate_conversational_response(state):
    """Generate conversational response for medical topics"""
    print("---GENERATE MEDICAL CONVERSATIONAL RESPONSE---")
    question = state["question"]
    documents = state.get("documents", [])
    conversation_history = state.get("conversation_history", [])
    is_greeting = state.get("is_greeting", False)
    user_intent = state.get("user_intent", "question")
    exact_match = state.get("exact_match", False)
    is_medical = state.get("is_medical", True)
    
    if not is_medical:
        return {
            "documents": [],
            "question": question,
            "generation": "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question.",
            "urls": [],
            "conversation_history": conversation_history,
            "is_greeting": is_greeting,
            "exact_match": exact_match
        }
    
    if "generation" in state and exact_match:
        print("Using exact match answer from prior medical questions")
        generation = state["generation"]
        urls = []
    else:
        chat_history_text = ""
        if conversation_history:
            for entry in conversation_history[-3:]:
                chat_history_text += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"
        
        # Prioritize medical question over greeting if both are detected
        if is_greeting and user_intent == "greeting" and not is_medical_topic(question):
            if not conversation_history:
                generation = "Hello! I'm here to help with medical questions, especially about skin cancer. How can I assist you today?"
            else:
                generation = "What else can I help you with regarding medical topics?"
        elif user_intent == "casual":
            generation = conversational_rag_chain.invoke({
                "context": "",
                "question": question,
                "chat_history": chat_history_text
            })
        else:
            context_text = "\n\n".join([doc.page_content for doc in documents]) if documents else "No specific medical context available."
            generation = conversational_rag_chain.invoke({
                "context": context_text,
                "question": question,
                "chat_history": chat_history_text
            })
    
        urls = []
        for doc in documents:
            url = doc.metadata.get("source")
            if url and url.startswith("http"):
                urls.append(url)

    generation = clean_repeat_phrases(generation)

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "urls": urls,
        "conversation_history": conversation_history,
        "is_greeting": is_greeting,
        "exact_match": exact_match
    }

def add_url_to_file(url, file_path):
    """Add URL to file if not already present"""
    try:
        with open(file_path, 'r') as file:
            existing_urls = set(file.read().splitlines())
    except FileNotFoundError:
        existing_urls = set()
    
    if url not in existing_urls:
        existing_urls.add(url)
        with open(file_path, 'w') as file:
            for url in existing_urls:
                file.write(url + '\n')
        return True
    return False

def decide_to_generate_prior(state):
    """Determine next step after checking prior medical questions"""
    print("---DECIDE AFTER PRIOR MEDICAL QUESTIONS---")
    exact_match = state.get("exact_match", False)
    is_greeting = state.get("is_greeting", False)
    is_medical = state.get("is_medical", True)
    
    if not is_medical:
        return "generate"
    
    if is_greeting:
        return "generate"
    
    if exact_match:
        print("---DECISION: EXACT MATCH FOUND IN PRIOR MEDICAL QUESTIONS, GENERATE---")
        return "generate"
    else:
        print("---DECISION: NO EXACT MATCH, RETRIEVE FROM LOCAL MEDICAL DOCS---")
        return "retrieve_local"

def decide_to_generate_local(state):
    """Decide whether to generate or search web after local medical docs"""
    print("---DECIDE AFTER LOCAL MEDICAL DOCUMENTS---")
    web_search = state["web_search"]
    is_greeting = state.get("is_greeting", False)
    documents = state.get("documents", [])
    is_medical = state.get("is_medical", True)
    
    if not is_medical:
        return "generate"
    
    if is_greeting:
        return "generate"
    
    if web_search == "Yes" and not documents:
        print("---DECISION: NO LOCAL MEDICAL DOCUMENTS RELEVANT, TRANSFORM QUERY FOR WEB SEARCH---")
        return "transform_query"
    else:
        print("---DECISION: GENERATE FROM LOCAL MEDICAL DOCUMENTS---")
        return "generate"

# Initialize models
load_dotenv()

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

# Setup retrievers and components
print("Setting up medical retrievers...")
doc_splits = setup_retriever_txt()

model_name = "BAAI/bge-base-en-v1.5"
encode_kwargs = {'normalize_embeddings': True}

embedding_model_base_retriever = HuggingFaceEmbeddings(
    model_name=model_name,
    encode_kwargs=encode_kwargs,
    model_kwargs={'device': 'cpu'}
)

vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embedding_model_base_retriever
)
retriever = vectorstore.as_retriever()

# Add URL documents to the same vectorstore
try:
    url_docs = setup_retriever_urls()
    vectorstore.add_documents(url_docs)
    print(f"Added {len(url_docs)} medical URL documents to vectorstore")
except Exception as e:
    print(f"Error loading medical URL documents: {e}")

# Setup prior medical questions
print("Loading prior medical questions...")
prior_questions, prior_question_lookup = setup_prior_questions()

# DEBUG: Print some info about prior questions
print("=== DEBUGGING PRIOR MEDICAL QUESTIONS ===")
print(f"Prior medical question lookup has {len(prior_question_lookup)} entries")
if prior_question_lookup:
    print("First 5 normalized medical questions:")
    for i, key in enumerate(list(prior_question_lookup.keys())[:5]):
        print(f"  {i+1}. '{key}'")
else:
    print("WARNING: No prior medical questions loaded!")

print("Initializing medical components...")
retrieval_grader = setup_grader()
question_rewriter = setup_rewriter()

# Setup conversational RAG chain
conversational_prompt = setup_conversational_prompt()
conversational_rag_chain = (
    conversational_prompt
    | model
    | StrOutputParser()
)

# Build the conversational workflow
workflow = StateGraph(GraphState)

# Define nodes
workflow.add_node("conversation_handler", handle_conversation_start)
workflow.add_node("priorq_retriever", priorq_retriever)
workflow.add_node("retrieve_local", retrieve_local_docs)
workflow.add_node("grade_local_docs", grade_documents_local)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)
workflow.add_node("generate", generate_conversational_response)

# Build Graph
workflow.add_edge(START, "conversation_handler")
workflow.add_edge("conversation_handler", "priorq_retriever")

workflow.add_conditional_edges(
    "priorq_retriever",
    decide_to_generate_prior,
    {
        "generate": "generate",
        "retrieve_local": "retrieve_local",
    },
)

workflow.add_edge("retrieve_local", "grade_local_docs")

workflow.add_conditional_edges(
    "grade_local_docs",
    decide_to_generate_local,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)

workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.add_edge("generate", END)

# Compile
print("Compiling medical conversational workflow...")
crags = workflow.compile()

# Flask App
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})  # Ensure CORS for /chat endpoint
app.secret_key = 'your-secret-key-here'

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    prompt = data.get("prompt") or data.get("user_prompt")
    conversation_id = data.get("conversation_id")

    if isinstance(prompt, dict):
        prompt = prompt.get("prompt") or prompt.get("user_prompt")
    if not isinstance(prompt, str):
        prompt = str(prompt)

    intent, is_casual, is_medical = detect_user_intent(prompt)

    if not is_medical and not is_casual:
        return Response(
            "I'm sorry, I can only assist with medical topics, particularly skin cancer. Please ask a medical-related question.".encode('utf-8'),
            content_type='text/plain; charset=utf-8'
        )

    conv_id, conversation = get_or_create_conversation(conversation_id)

    def generate():
        inputs = {
            "question": prompt,
            "conversation_history": conversation["history"],
            "conversation_id": conv_id,
            "is_medical": is_medical
        }
        try:
            for output in crags.stream(inputs):
                if 'generate' in output:
                    generation_value = output['generate']['generation']
                    conversation["history"].append({
                        "user": prompt,
                        "assistant": generation_value,
                        "timestamp": datetime.now().isoformat()
                    })
                    yield generation_value.encode('utf-8')
        except Exception as e:
            print("Error during streaming:", e)
            traceback.print_exc()
            error_response = f"I apologize, but I encountered an error: {e}"
            yield error_response.encode('utf-8')

    return Response(generate(), content_type='text/plain; charset=utf-8')

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

@app.route("/conversation_history/<conversation_id>", methods=["GET"])
def get_conversation_history(conversation_id):
    """Get conversation history"""
    if conversation_id in conversation_memory:
        return {"conversation_id": conversation_id, "history": conversation_memory[conversation_id]["history"]}
    else:
        return {"error": "Conversation not found"}, 404

@app.route("/test_prior", methods=["POST"])
def test_prior():
    """Test endpoint to check prior medical questions matching"""
    data = request.json
    question = data.get("question", "")
    
    norm_question = normalize(question)
    
    result = {
        "original_question": question,
        "normalized_question": norm_question,
        "found_in_lookup": norm_question in prior_question_lookup,
        "is_medical": is_medical_topic(question),
        "total_prior_questions": len(prior_question_lookup),
        "sample_keys": list(prior_question_lookup.keys())[:5]
    }
    
    if norm_question in prior_question_lookup:
        result["answer"] = prior_question_lookup[norm_question]
    
    return result

def clean_repeat_phrases(text):
    """Remove phrases indicating repeated answers or prior questions."""
    patterns = [
        r"\bsince (we|i) (already )?(discussed|talked about|covered) (this|that|it)[^.,;:!?]*[.,;:!?]?",
        r"\bi('ve| have)? (already )?(answered|addressed|responded to) (this|that|it)[^.,;:!?]*[.,;:!?]?",
        r"\b(as|like) (i|we) (said|mentioned|discussed|covered) (before|earlier)[^.,;:!?]*[.,;:!?]?",
        r"\bto answer your question again[^.,;:!?]*[.,;:!?]?",
        r"\byou('ve)? (already )?asked (this|that)? question[^.,;:!?]*[.,;:!?]?",
        r"\bi('ve)? got this question (again|right here|already)[^.,;:!?]*[.,;:!?]?",
        r"\bjust to (summarize|recap)[^.,;:!?]*[.,;:!?]?",
        r"\bhere('s| is) a (quick )?summary[^.,;:!?]*[.,;:!?]?",
    ]
    for pat in patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[\s,;:.-]+', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def test_prior_questions_matching():
    """Test function to debug prior medical questions matching"""
    print("=== TESTING PRIOR MEDICAL QUESTIONS MATCHING ===")
    
    test_questions = [
        "What is skin cancer?",
        "How is melanoma treated?",
        "What are the symptoms of basal cell carcinoma?"
    ]
    
    for q in test_questions:
        norm_q = normalize(q)
        print(f"Original: '{q}'")
        print(f"Normalized: '{norm_q}'")
        if norm_q in prior_question_lookup:
            print(f"MATCH FOUND: {prior_question_lookup[norm_q][:100]}...")
        else:
            print("NO MATCH")
        print("-" * 50)
    
    print(f"Total medical questions in lookup: {len(prior_question_lookup)}")
    print("First 5 keys in lookup:")
    for i, key in enumerate(list(prior_question_lookup.keys())[:5]):
        print(f"  {i+1}. '{key}'")

# Commented out for Gunicorn deployment
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 10000))
#     host = "0.0.0.0"
#     app.run(host=host, port=port, debug=os.getenv("FLASK_DEBUG", "False") == "True")