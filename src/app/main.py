"""
Streamlit application for multi-PDF Retrieval-Augmented Generation (RAG)
using Ollama + LangChain.

This refactor focuses on:
 - Accepting and persisting multiple uploaded PDFs
 - Creating one Chroma collection per PDF (persisted on disk)
 - Allowing selection of one or more collections to run retrieval over
 - Keeping RAG prompt + few-shot behavior in-process (easy to tune)

Run using the provided run.py which launches this file via Streamlit.
"""
from pathlib import Path
import streamlit as st
import logging
import os
import json
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import hashlib
from typing import List, Any, Optional, Dict
from io import BytesIO
import requests
import base64
from PIL import Image
from dotenv import load_dotenv, find_dotenv
import chromadb
import openai
from datetime import datetime
from langdetect import detect

# LangChain / vector imports
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough

from components.pdf_indexing import handle_uploaded_files, list_collections, delete_collection, delete_all_collections, reindex_collection, get_collection_info, load_collections_metadata, bootstrap_collections_from_uploads, get_embedding_function, load_chroma_collection
from components.llm_clients import call_generate_api, chat_with_model
from components.utils import img_to_base64, safe_extract_model_names

# Suppress torch warning spam early to avoid messages during imports
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# Basic config
st.set_page_config(
    page_title="AI Hackathon - Unified AI App",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Persistence locations
BASE_DIR = Path(".")
PDF_UPLOAD_DIR = BASE_DIR / "data" / "pdfs" / "uploads"
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_DIRECTORY = str(BASE_DIR / "data" / "vectors")  # Chroma persistent directory
METADATA_PATH = BASE_DIR / "data" / "pdf_collections.json"

# RAG + prompt configuration
SYSTEM_INSTRUCTIONS = (
    "You are an intelligent, context-aware technical assistant specialized in service and maintenance of the LiuGong CLG835H Wheel Loader (and similar models). "
    "You have access to technical documents including the CLG835H Service Manual, Operation & Maintenance Manual, and Symptom–Cause–Remedy diagnostic sheets.\n\n"

    "Core Behaviors:\n"
    "- **Context Persistence**: When a user mentions a specific model (e.g., CLG835H), remember it as the active context for the entire conversation. Use this context for all subsequent questions unless explicitly changed. If no model context exists and the question requires it, ask: 'Please tell me the model for which you are looking for information or help.'\n"
    "- **Adaptive Reasoning**: Analyze each question to identify missing parameters (e.g., model, operating hours, specific conditions). If key information is missing, ask specifically for it rather than giving vague responses. Think step-by-step before answering.\n"
    "- **Answer Style Based on Intent**:\n"
      "  - For diagnostic/causes questions: Provide only likely causes in a clear, structured technical format (bullet points).\n"
      "  - For remedy/fix questions: Provide only corrective actions or step-by-step procedures.\n"
      "  - For combined questions: Separate into 'Likely Causes:' and 'Recommended Remedies:' sections.\n"
    "- **Domain Intelligence**: Base answers ONLY on the provided Context below. Do not invent, assume, or add any information not explicitly present in the Context. Cite sources accurately from the Context.\n"
    "- **Conversational Intelligence**: Link follow-up questions logically to previous context. End responses with helpful follow-up prompts when appropriate, like 'Would you like me to provide the recommended bleeding procedure next?'\n\n"

    "Response Guidelines:\n"
    "- Be concise yet informative, using bullet points for lists.\n"
    "- Include safety warnings or cautions for maintenance actions involving hazards.\n"
    "- Maintain conversation continuity and remember what was discussed.\n"
    "- Append citations like [source: CLG835H4F_ServiceManual.pdf] (Section: Hydraulic System).\n"
    "- Use exact specifications (fluids, torque, intervals) as in manuals.\n"
    "- Be professional, precise, and engaging like a senior service engineer.\n"
    "- If the requested information is not present in the Context, respond exactly: 'The current manuals don't mention that. Please try searching with alternate terms.'\n"
)
FEW_SHOT_EXAMPLES = (
    "Example 1:\nQ: What is the engine model used in CLG835H Wheel Loader?\n"
    "A: - Perkins 1204F-E44AT engine\n"
    "- Rated net power: 97.9 kW\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: General Information)\n\n"

    "Example 2:\nQ: What hydraulic oil should be used for the CLG835H in cold weather?\n"
    "A: - Use HV-46 low-temperature anti-wear hydraulic oil for -25°C to 40°C\n"
    "- Capacity: approximately 120 L (32 US gal)\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Oils Use Specifications)\n\n"

    "Example 3:\nQ: What is the standard torque for M16 bolts on this loader?\n"
    "A: - 305 ± 25 N·m (225 ± 18 lbf·ft)\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Standard Torque Table)\n\n"

    "Example 4:\nQ: How often should engine oil be changed?\n"
    "A: - Every 500 service hours after initial 100-hour service\n"
    "- Use SAE 15W-40 API CJ-4 diesel engine oil\n"
    "- Capacity: 12 L (3 US gal)\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Maintenance Schedule)\n\n"

    "Example 5:\nQ: The loader is overheating. What could be the causes?\n"
    "A: Likely causes:\n"
    "- Low coolant level or coolant leakage\n"
    "- Blocked radiator fins\n"
    "- Fan drive malfunction\n"
    "- Overloaded engine or clogged air filter\n"
    "[source: SymptomCauseRemedy_Sheet.xlsx]\n\n"

    "Example 6:\nQ: How do I fix the overheating issue?\n"
    "A: Recommended remedies:\n"
    "- Check and refill coolant; inspect for leaks\n"
    "- Clean radiator and check fan operation\n"
    "- Reduce load and ensure air filter is clean\n"
    "Would you like me to provide the detailed cooling system inspection procedure?\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Cooling System)\n\n"

    "Example 7:\nQ: What precautions must be taken before servicing the hydraulic system?\n"
    "A: - Lower all attachments to the ground\n"
    "- Shut off the engine and relieve hydraulic pressure\n"
    "- Engage articulation lock\n"
    "- Disconnect battery before opening any line\n"
    "[source: LiuGong_OperatorAndMaintenance.pdf] (Section: Safety Information)\n\n"

    "Example 8:\nQ: What is the rated operating mass of CLG835H Wheel Loader?\n"
    "A: - 11,200 kg (24,696 lb)\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Technical Specification)\n\n"

    "Example 9:\nQ: How can I identify the product serial number (PIN)?\n"
    "A: - Located on the left side of the machine front frame, under the boom pin\n"
    "- The last six characters represent the machine’s product serial number\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Manual Identification)\n\n"

    "Example 10:\nQ: What should I do if the transmission oil appears milky?\n"
    "A: - Indicates water contamination\n"
    "- Drain the oil, replace the filter, and refill with SAE 15W-40 Mobil Delvac\n"
    "- Inspect breather and seals for moisture ingress\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Oils Use Specifications)\n\n"

    "Example 11:\nQ: The brakes are spongy. What are the causes and how to fix it?\n"
    "A: Likely Causes:\n"
    "- Air contamination in the brake hydraulic circuit\n"
    "- Insufficient oil level in the master cylinder reservoir\n"
    "- Brake system leakage at a caliper or line connection\n"
    "- Excessive wear of the brake pads (wet axle)\n"
    "Recommended Remedies:\n"
    "- Bleed the brake system to remove air\n"
    "- Check and refill the master cylinder reservoir with appropriate brake fluid\n"
    "- Inspect for leaks in the brake lines and calipers, repair as necessary\n"
    "- Examine brake pads for wear and replace if excessively worn\n"
    "[source: CLG835H4F_ServiceManual.pdf] (Section: Brake System Diagnostics)\n\n"

    "Example 12:\nQ: How many service hours has the machine been running?\n"
    "A: Could you please confirm the operating hours or service interval you're referring to?\n\n"

    "Example 13:\nQ: What oil should I use?\n"
    "A: Please tell me the model for which you are looking for information or help.\n\n"

    "Example 14:\nQ: If the provided Context does not include the requested information, how should the assistant respond?\n"
    "A: The current manuals don't mention that. Please try searching with alternate terms.\n\n"
)
RAG_TOP_K = 100  # per collection (increased for better retrieval)
EXCERPT_MAX_CHARS = 2000

# New: disclaimer shown at the start of any response that is NOT sourced from the local document DB
DISCLAIMER_TEXT = "**⚠️ Disclaimer: The following information is not sourced from the system database, service manuals, or uploaded datasets. Its accuracy, authenticity, and completeness have not been verified against official or manufacturer documentation. Please validate this information from authorized or reliable sources before applying it..**\n\n"

# NOTE: create_vector_db_from_path, load_chroma_collection and handle_uploaded_files
# are implemented in `components.pdf_indexing`. Local duplicate implementations were
# removed to avoid re-indexing the same files multiple times and to ensure a single
# source of truth for collection metadata and persistence.

from langchain.schema import Document
import re


def plan_retrieval_strategy(question: str, conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Agentic RAG: Plan the retrieval strategy based on the question type and context.
    Returns a plan dict with steps to execute.
    """
    # Check if the question is about LiuGong/CLG835H models; otherwise, treat as general
    question_lower = question.lower()
    is_liugong_related = "liugong" in question_lower or "clg835h" in question_lower or "clg" in question_lower
    if not is_liugong_related:
        return {
            "intent": "general",
            "primary_document": None,
            "secondary_documents": [],
            "section_filter": None,
            "use_double_rag": False,
            "search_keywords": [question],
            "rationale": "Question not related to LiuGong/CLG835H models"
        }

    plan_prompt = f"""
You are an expert retrieval planner for technical documentation. Analyze the question and determine the optimal retrieval strategy.

Question: {question}

Available document types:
- SymptomCauseRemedy_Sheet.xlsx: Contains symptom-cause-remedy mappings for diagnostics
- CLG835H4F_ServiceManual.pdf: Service manual with detailed procedures and specifications
- LiuGong_OperatorAndMaintenance.pdf: Operation and maintenance manual

Based on the question, plan a retrieval strategy with these steps:
1. Identify the primary intent (diagnostic/causes, remedy/fix, specification, maintenance, etc.)
2. Determine which documents to check first, second, etc.
3. Specify any section filtering (e.g., "Brake System", "Hydraulic System")
4. Indicate if double RAG is needed (broad section search then narrow)
5. Suggest search terms or keywords to use

Return your plan in this exact JSON format:
{{
    "intent": "diagnostic|remedy|specification|maintenance|general",
    "primary_document": "SymptomCauseRemedy_Sheet.xlsx|CLG835H4F_ServiceManual.pdf|LiuGong_OperatorAndMaintenance.pdf",
    "secondary_documents": ["list of other docs"],
    "section_filter": "specific section name or null",
    "use_double_rag": true|false,
    "search_keywords": ["keyword1", "keyword2"],
    "rationale": "brief explanation"
}}

Plan:"""

    try:
        plan_response = chat_with_model(
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            user_prompt=plan_prompt,
            system_prompt="You are a retrieval planning expert. Return only valid JSON.",
            temperature=0.0
        )
        
        # Parse the JSON response
        plan = json.loads(plan_response.strip())
        logger.info(f"Planned retrieval strategy: {plan}")
        return plan
    except Exception as e:
        logger.exception("Failed to plan retrieval strategy, using default")
        # Default fallback plan
        return {
            "intent": "general",
            "primary_document": "CLG835H4F_ServiceManual.pdf",
            "secondary_documents": ["SymptomCauseRemedy_Sheet.xlsx", "LiuGong_OperatorAndMaintenance.pdf"],
            "section_filter": None,
            "use_double_rag": False,
            "search_keywords": [question],
            "rationale": "Fallback plan due to planning error"
        }


def perform_double_rag(question: str, selected_collections: List[str], section: str, keywords: List[str], rag_top_k: int = RAG_TOP_K) -> List[Any]:
    """
    Double RAG: First find the broad section, then narrow down to specific content.
    """
    logger.info(f"Performing double RAG for section: {section}, keywords: {keywords}")
    
    # Step 1: Search for the broad section
    section_query = f"{section} section"
    section_docs = []
    
    for coll in selected_collections:
        try:
            chroma = load_chroma_collection(coll)
            docs = chroma.similarity_search(section_query, k=rag_top_k * 2)
            section_docs.extend(docs)
        except Exception:
            logger.exception(f"Failed to search section in collection: {coll}")
    
    # Filter docs that contain the section name
    filtered_section_docs = []
    section_lower = section.lower()
    for doc in section_docs:
        content = getattr(doc, 'page_content', '').lower()
        if section_lower in content:
            filtered_section_docs.append(doc)
    
    if not filtered_section_docs:
        logger.warning(f"No documents found for section: {section}, falling back to keyword search")
        # Fallback to direct keyword search
        return perform_keyword_search(question, selected_collections, keywords, rag_top_k)
    
    # Step 2: Within the section docs, search for specific keywords
    refined_docs = []
    for doc in filtered_section_docs[:rag_top_k]:  # Limit to top section docs
        content = getattr(doc, 'page_content', '').lower()
        if any(kw.lower() in content for kw in keywords):
            refined_docs.append(doc)
    
    if not refined_docs:
        # If no matches with keywords, return the section docs
        refined_docs = filtered_section_docs[:rag_top_k]
    
    logger.info(f"Double RAG found {len(refined_docs)} refined documents")
    return refined_docs


def perform_keyword_search(question: str, selected_collections: List[str], keywords: List[str], rag_top_k: int = RAG_TOP_K) -> List[Any]:
    """
    Perform semantic search using the provided keywords.
    """
    search_query = " ".join(keywords) if keywords else question
    all_docs = []
    
    for coll in selected_collections:
        try:
            chroma = load_chroma_collection(coll)
            docs = chroma.similarity_search(search_query, k=rag_top_k)
            all_docs.extend(docs)
        except Exception:
            logger.exception(f"Failed to search in collection: {coll}")
    
    # Sort by relevance (assuming Chroma returns in order)
    return all_docs[:rag_top_k]


def check_response_completeness(response: str, question: str) -> Dict[str, Any]:
    """
    Check if the response is complete or needs iterative refinement.
    Returns dict with completeness assessment and suggested follow-up queries.
    """
    completeness_prompt = f"""
Analyze if this response adequately answers the question. Consider:
- Does it provide specific, actionable information?
- Are there missing details that could be found with additional search?
- Is the response too vague or generic?
- Does it cite sources appropriately?

Question: {question}
Response: {response}

Return JSON in this format:
{{
    "is_complete": true|false,
    "missing_info": ["what's missing"],
    "suggested_queries": ["follow-up search query 1", "query 2"],
    "confidence": 0.0-1.0
}}

Analysis:"""

    try:
        analysis = chat_with_model(
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            user_prompt=completeness_prompt,
            system_prompt="You are an expert at analyzing response completeness. Return only valid JSON.",
            temperature=0.0
        )
        
        import json
        result = json.loads(analysis.strip())
        logger.info(f"Response completeness: {result}")
        return result
    except Exception as e:
        logger.exception("Failed to check completeness")
        return {
            "is_complete": True,
            "missing_info": [],
            "suggested_queries": [],
            "confidence": 0.8
        }


def iterative_rag_refinement(question: str, initial_response: str, selected_collections: List[str], 
                           suggested_queries: List[str], rag_top_k: int = RAG_TOP_K) -> str:
    """
    Perform iterative RAG by running additional searches for missing information.
    """
    if not suggested_queries:
        return initial_response
    
    logger.info(f"Performing iterative refinement with queries: {suggested_queries}")
    
    # Check if initial response has disclaimer
    has_disclaimer = initial_response.startswith(DISCLAIMER_TEXT)
    if has_disclaimer:
        clean_initial = initial_response[len(DISCLAIMER_TEXT):].lstrip()
    else:
        clean_initial = initial_response
    
    additional_contexts = []
    for query in suggested_queries[:2]:  # Limit to 2 additional queries
        try:
            # Perform search for the suggested query
            additional_docs = perform_keyword_search(query, selected_collections, [query], rag_top_k // 2)
            if additional_docs:
                context = build_context_from_docs(additional_docs, max_chars_per_doc=EXCERPT_MAX_CHARS // 2)
                additional_contexts.append(f"Additional context for '{query}':\n{context}")
        except Exception:
            logger.exception(f"Failed iterative search for: {query}")
    
    if not additional_contexts:
        return initial_response
    
    # Combine initial response with additional context
    additional_info_text = "\n\n".join(additional_contexts)
    refinement_prompt = f"""
You have an initial response to a question, but additional relevant information was found.
Integrate the new information to provide a more complete answer.

Question: {question}

Initial Response: {clean_initial}

Additional Information:
{additional_info_text}

Provide an improved, integrated response that combines the best of both. Keep the same style and format as the initial_response.
If the additional information doesn't add value, keep the original response.

Improved Response:"""

    try:
        refined_response = chat_with_model(
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            user_prompt=refinement_prompt,
            system_prompt=None,
            temperature=0.0
        )
        logger.info("Iterative refinement completed")
        # Add disclaimer back if it was present
        if has_disclaimer:
            return DISCLAIMER_TEXT + refined_response
        else:
            return refined_response
    except Exception:
        logger.exception("Iterative refinement failed")
        return initial_response


def extract_part_tokens(question: str) -> List[str]:
    """Extract candidate part-number tokens from a question (must contain a digit).

    Returns tokens in normalized form.
    """
    # match alphanum sequences optionally containing - or / or _ (e.g. 587859/a or AB-123)
    tokens = re.findall(r"[A-Za-z0-9][_A-Za-z0-9\-/]{1,}", question)
    # filter tokens that contain at least one digit and are not too short
    tokens = [t.strip().upper() for t in tokens if any(c.isdigit() for c in t) and len(t) >= 3]
    return tokens


def exact_search_in_collection(collection_name: str, token: str) -> List[Document]:
    """Perform a fast substring search across the stored documents/metadatas of a chroma collection.

    Returns a list of langchain Document objects for matching chunks.
    """
    matches: List[Document] = []
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        coll = client.get_collection(name=collection_name)
        # request documents and metadatas
        try:
            res = coll.get(include=["documents", "metadatas", "ids"])  # type: ignore
        except Exception:
            # fallback to simpler get
            res = coll.get()
        docs = res.get("documents", []) if isinstance(res, dict) else []
        metadatas = res.get("metadatas", []) if isinstance(res, dict) else []
        for d, m in zip(docs, metadatas):
            try:
                text = (d or "").upper()
                # check in document text or in any metadata values
                in_text = token in text
                in_meta = False
                if isinstance(m, dict):
                    for v in m.values():
                        try:
                            if token in str(v).upper():
                                in_meta = True
                                break
                        except Exception:
                            continue
                if in_text or in_meta:
                    matches.append(Document(page_content=str(d), metadata=m or {"source": collection_name}))
            except Exception:
                continue
    except Exception:
        logger.exception(f"Exact search failed for collection: {collection_name}")
    return matches

def normalize_part_token(token: str) -> List[str]:
    """Return normalized variants of a part token to treat different separators as equivalent.

    Examples:
    - '587859/A' -> ['587859/A','587859-A','587859A','587859 / A']
    - 'AB-123' -> ['AB-123','AB/123','AB123']
    """
    t = token.strip().upper()
    variants = {t}
    # replace common separators
    variants.add(t.replace('/', '-'))
    variants.add(t.replace('-', '/'))
    variants.add(t.replace('/', ''))
    variants.add(t.replace('-', ''))
    # add space-separated version
    variants.add(t.replace('/', ' / '))
    variants.add(t.replace('-', ' - '))
    # also add lowercase forms just in case stored text is mixed
    variants.update({v.lower() for v in list(variants)})
    return list(variants)

def is_summary_request(question: str, conversation_history: List[Dict[str, str]] = None) -> bool:
    """Heuristic: detect if the question is asking for a summary, abstract, or overview.
    Also considers conversation context for better understanding.
    Be more selective to avoid triggering on specific questions.
    """
    question_lower = question.lower()

    # Direct summary keywords
    summary_keywords = ["summary", "abstract", "overview", "synopsis", "recap", "summarize", "summarise"]

    # More specific summary phrases
    specific_summary_phrases = ["what is this", "tell me about this", "explain this document", "what's in this document", "give me an overview of this", "brief me on this document", "what does this document cover"]

    # Check if it contains summary keywords
    has_summary_keyword = any(kw in question_lower for kw in summary_keywords)
    
    # Check if it contains specific summary phrases
    has_specific_phrase = any(phrase in question_lower for phrase in specific_summary_phrases)
    
    # For questions like "tell me about X", only trigger if X is generic (like "this", "the document")
    if "tell me about" in question_lower:
        # Extract what comes after "tell me about"
        after_tell_me = question_lower.split("tell me about", 1)[1].strip()
        # If it's specific (contains nouns, not just "this" or "the document"), don't treat as summary
        if len(after_tell_me.split()) > 1 or not any(word in after_tell_me for word in ["this", "the document", "the manual", "the pdf"]):
            return False

    if has_summary_keyword or has_specific_phrase:
        return True

    # Check for document-related questions that imply summary intent
    doc_indicators = ["pdf", "document", "manual", "book", "file", "content", "this document"]
    analysis_indicators = ["what is", "tell me", "explain", "describe", "about this"]

    if any(doc in question_lower for doc in doc_indicators) and any(analysis in question_lower for analysis in analysis_indicators):
        return True

    # Check conversation context for summary-related intent
    if conversation_history:
        recent_questions = [msg['content'] for msg in conversation_history[-4:] if msg['role'] == 'user']
        for prev_q in recent_questions:
            prev_lower = prev_q.lower()
            if any(kw in prev_lower for kw in summary_keywords):
                # If previous question was about summary and current is related, continue summary mode
                if any(word in question_lower for word in ["pdf", "document", "manual", "book", "content", "what", "how", "explain", "future", "commercial", "gdp", "growth"]):
                    return True

    return False

# Create a Chroma collection from a PDF file path
def create_vector_db_from_path(path: Path, collection_name: Optional[str] = None) -> str:
    """
    Create a Chroma vector collection from a PDF file on disk.

    Returns the collection_name used.
    """
    logger.info(f"Creating vector DB for file: {path}")
    loader = UnstructuredPDFLoader(str(path))
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # Use a deterministic collection name if none provided
    if collection_name is None:
        collection_name = f"pdf_{abs(hash(str(path)))}"
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY,
        collection_name=collection_name,
    )
    logger.info(f"Created collection: {collection_name}")
    return collection_name

# Load an existing Chroma collection (returns a Chroma instance)
def load_chroma_collection(collection_name: str) -> Chroma:
    try:
        # Use the same PersistentClient backend as the indexer to avoid mismatched stores
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        return Chroma(client=client, collection_name=collection_name, embedding_function=get_embedding_function())
    except Exception:
        # Fallback: keep the previous persist_directory-based Chroma construction
        embeddings = get_embedding_function()
        return Chroma(persist_directory=PERSIST_DIRECTORY, collection_name=collection_name, embedding_function=embeddings)

# Build context text with small excerpts and source citations
def build_context_from_docs(docs: List[Any], max_chars_per_doc: int = EXCERPT_MAX_CHARS) -> str:
    parts = []
    for i, d in enumerate(docs[: RAG_TOP_K * 5]):  # allow a slightly larger pool then trim in template
        src = d.metadata.get("source") or d.metadata.get("filename") or f"doc_{i}"
        excerpt = getattr(d, "page_content", str(d))[:max_chars_per_doc]
        if len(excerpt) >= max_chars_per_doc:
            excerpt = excerpt.rsplit("\n", 1)[0] + "…"
        parts.append(f"[source: {src}]\n{excerpt}\n")
    return "\n\n".join(parts).strip() or " "


def build_context_from_selected_docs(docs: List[Any], max_chars_per_doc: int = EXCERPT_MAX_CHARS) -> str:
    """Build context text from an explicit ordered list of docs (no additional slicing).

    This ensures the same docs used to create the context are used for source attribution.
    """
    parts = []
    for i, d in enumerate(docs):
        src = None
        try:
            if hasattr(d, 'metadata') and isinstance(d.metadata, dict):
                src = d.metadata.get('source') or d.metadata.get('filename')
        except Exception:
            src = None
        if not src:
            src = f"doc_{i}"
        excerpt = (getattr(d, 'page_content', str(d)) or '')[:max_chars_per_doc]
        if len(excerpt) >= max_chars_per_doc:
            excerpt = excerpt.rsplit("\n", 1)[0] + "…"
        parts.append(f"[source: {src}]\n{excerpt}\n")
    return "\n\n".join(parts).strip() or " "

# Process a user question by retrieving across selected collections
def process_question_across_collections(
    question: str,
    selected_collections: List[str],
    selected_model: str,
    conversation_history: List[Dict[str, str]] = None,
    rag_top_k: int = RAG_TOP_K,
    max_chars_per_doc: int = EXCERPT_MAX_CHARS,
    allow_partial: bool = False,
    debug: bool = False,
) -> str:
    """
    Retrieves documents from each selected collection, builds context, and queries the LLM.
    Includes conversation history for continuity when provided.
    """
    # Handle basic conversational queries
    conversational_responses = {
        "hi": "Hi there! I'm doing well, thank you. How may I help you with the manuals today?",
        "hello": "Hello! I'm here to assist with service technician queries. How can I help?",
        "how are you": "I'm doing great, thanks! Ready to help with any questions about the manuals. What can I assist you with?",
        "what's up": "Not much, just here to help with technical queries. How may I assist you?",
        "hey": "Hey! I'm good. How can I help you today?",
        "thanks": "Thanks and have a great day!",
        "thank you": "Thanks and have a great day!",
        "no thanks": "Thanks and have a great day!",
        "i don't need any info": "Thanks and have a great day!",
        "i don't need any information": "Thanks and have a great day!",
        "no more": "Thanks and have a great day!",
        "that's all": "Thanks and have a great day!",
    }
    question_lower = question.lower().strip()
    if question_lower in conversational_responses:
        response = conversational_responses[question_lower]
        return response

    # If no collections selected, do general chat
    if not selected_collections:
        try:
            response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=question, system_prompt=None, temperature=0.7)
            # Mark clearly that this answer is not from the system DB
            return DISCLAIMER_TEXT + response
        except Exception:
            logger.exception("General chat failed")
            return DISCLAIMER_TEXT + "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

    logger.info(f"Processing question across collections: {selected_collections} using model {selected_model}")
    logger.info(f"Question: {question}")
    all_docs = []

    # If the user explicitly asked for a summary/abstract/overview, run a summary flow
    try:
        if is_summary_request(question, conversation_history):
            logger.info("Detected summary/abstract request — running summary flow")
            summary_pool = []
            for coll in selected_collections:
                try:
                    chroma = load_chroma_collection(coll)
                    # try to retrieve a larger pool of relevant chunks
                    try:
                        pool = chroma.similarity_search(question, k=max(10, rag_top_k * 3))
                    except Exception:
                        pool = chroma.similarity_search(question, k=max(5, rag_top_k))
                    summary_pool.extend(pool)
                except Exception:
                    logger.exception(f"Failed retrieving docs for summary from collection: {coll}")
            # deduplicate by content while preserving order
            seen_keys = set()
            deduped = []
            for d in summary_pool:
                try:
                    key = getattr(d, 'page_content', '')[:500]
                except Exception:
                    key = str(d)[:500]
                if key and key not in seen_keys:
                    seen_keys.add(key)
                    deduped.append(d)
            docs_for_context = deduped[: max(10, rag_top_k * max(1, len(selected_collections)))]
            context_text = build_context_from_selected_docs(docs_for_context, max_chars_per_doc=max_chars_per_doc)
            if not context_text.strip():
                return DISCLAIMER_TEXT + "The current manuals don’t mention that. Please try searching with alternate terms.\n\nWhat other help do you need or what other information do you need?"
            summary_prompt = (
                "You are a technical summarizer for service technicians. Use ONLY the information present in the Context below to produce a concise abstract or summary tailored for a technician.\n"
                "- Provide a short abstract (3-6 sentences) or a brief bullet summary if the content is list-like.\n"
                "- Keep it factual and include one-line sources if available.\n\n"
                "Context:\n{context}\n\nRequest: Produce a concise abstract of the Context.\n\nAbstract:"
            )
            full_prompt = summary_prompt.format(context=context_text)
            try:
                resp = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=full_prompt, system_prompt=None, temperature=0.0)
                logger.info("Summary generated successfully")
                # append sources if model didn't include any
                if "[source:" not in resp and docs_for_context:
                    seen = []
                    for d in docs_for_context:
                        src = None
                        if hasattr(d, 'metadata') and isinstance(d.metadata, dict):
                            src = d.metadata.get('source') or d.metadata.get('filename')
                        if src and src not in seen:
                            seen.append(src)
                        if len(seen) >= 5:
                            break
                    if seen:
                        resp = resp + f"\n\n[sources: {', '.join(seen)}]"
                return resp
            except Exception:
                logger.exception("Summary generation failed")
                return "Sorry — something went wrong while generating the summary.\n\nWhat other help do you need or what other information do you need?"
    except Exception:
        logger.exception("Summary flow failed, falling back to normal retrieval")

    # Attempt exact part-number extraction + exact substring search across collections before semantic retrieval
    try:
        tokens = extract_part_tokens(question)
        exact_docs: List[Any] = []
        if tokens:
            logger.info(f"Extracted part tokens for exact search: {tokens}")
            for coll in selected_collections:
                try:
                    for t in tokens:
                        # build normalized variants (treat separators as equivalent)
                        variants = normalize_part_token(t)
                        for v in variants:
                            matches = exact_search_in_collection(coll, v)
                            if matches:
                                exact_docs.extend(matches)
                except Exception:
                    logger.exception(f"Exact-search failed for collection {coll}")
        # deduped exact_docs while preserving order
        seen_keys = set()
        deduped_exact = []
        for d in exact_docs:
            key = (getattr(d, 'page_content', '')[:200], tuple(sorted((d.metadata or {}).items())))
            if key not in seen_keys:
                seen_keys.add(key)
                deduped_exact.append(d)
        if deduped_exact:
            logger.info(f"Found {len(deduped_exact)} exact-match chunks — these will be prioritized in context")
    except Exception:
        logger.exception("Exact-match extraction/search failed")
        deduped_exact = []

    retrieved = []  # tuples of (doc, score, collection_name)
    for coll in selected_collections:
        try:
            chroma = load_chroma_collection(coll)
            count = chroma._collection.count()
            logger.info(f"Collection {coll} has {count} documents")
            # prefer a method that returns scores if available
            try:
                docs_and_scores = chroma.similarity_search_with_score(question, k=rag_top_k)
            except Exception:
                # some Chroma wrappers expose different method names or don't return scores
                try:
                    docs_and_scores = chroma.similarity_search_with_relevance_scores(question, k=rag_top_k)
                except Exception:
                    # fallback to score-less search
                    docs = chroma.similarity_search(question, k=rag_top_k)
                    docs_and_scores = [(d, None) for d in docs]
            logger.info(f"Search in {coll} returned {len(docs_and_scores)} results")
            # normalize to (doc, score)
            for item in docs_and_scores:
                if isinstance(item, tuple) and len(item) == 2:
                    doc, score = item
                else:
                    doc, score = item, None
                retrieved.append((doc, score, coll))
                # Debug: log source of retrieved documents
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', 'unknown')
                    logger.debug(f"Retrieved from {coll}: source={source}, score={score}")
        except Exception:
            logger.exception(f"Failed to query collection: {coll}")

    # If we have numeric scores, sort by score descending (higher score = more similar). Otherwise keep retrieval order
    try:
        scores_present = [r[1] for r in retrieved if r[1] is not None]
        if scores_present and all(isinstance(s, (int, float)) for s in scores_present):
            # Sort by file type priority first (PDF > DOCX > TXT), then by score
            def sort_key(item):
                doc, score, coll = item
                # Determine file type priority (lower number = higher priority)
                if hasattr(doc, 'metadata') and doc.metadata:
                    source = doc.metadata.get('source', '').lower()
                    if source.endswith('.pdf'):
                        priority = 0  # Highest priority
                    elif source.endswith('.docx'):
                        priority = 1
                    elif source.endswith('.txt'):
                        priority = 2  # Lowest priority
                    else:
                        priority = 3
                else:
                    priority = 3
                
                # Use score if available, otherwise use 0
                score_val = score if score is not None else 0
                return (priority, -score_val)  # Sort by priority asc, then score desc
            
            retrieved.sort(key=sort_key)
        # build docs list from sorted retrieved list
        all_docs = [r[0] for r in retrieved]
        logger.info(f"Retrieved {len(all_docs)} documents from {len(selected_collections)} collections")
    except Exception:
        # fallback: flatten any retrieved docs in original order
        all_docs = [r[0] for r in retrieved]
        logger.info(f"Retrieved {len(all_docs)} documents (fallback sorting) from {len(selected_collections)} collections")

    # Decide how many docs to include in the context. The UI exposes rag_top_k per collection,
    # so include up to rag_top_k * number_of_selected_collections documents total (best overall after sorting).
    try:
        num_docs = int(rag_top_k) * max(1, len(selected_collections))
    except Exception:
        num_docs = int(rag_top_k)

    # Advanced RAG: Plan retrieval strategy
    try:
        retrieval_plan = plan_retrieval_strategy(question, conversation_history)
        intent = retrieval_plan.get("intent", "general")
        primary_doc = retrieval_plan.get("primary_document")
        secondary_docs = retrieval_plan.get("secondary_documents", [])
        section_filter = retrieval_plan.get("section_filter")
        use_double_rag = retrieval_plan.get("use_double_rag", False)
        search_keywords = retrieval_plan.get("search_keywords", [question])
        
        logger.info(f"Using advanced RAG plan: intent={intent}, double_rag={use_double_rag}, section={section_filter}")
    except Exception:
        logger.exception("Advanced RAG planning failed, using standard approach")
        retrieval_plan = None
        use_double_rag = False
        search_keywords = [question]

    # Check if intent is general, if so, do general chat instead of RAG
    if retrieval_plan and retrieval_plan.get("intent") == "general":
        try:
            conversation_context = ""
            if conversation_history and len(conversation_history) > 1:
                recent_history = conversation_history[-6:-1]  # Exclude current question
                if recent_history:
                    conversation_context = "\n\nPrevious conversation:\n"
                    for msg in recent_history:
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        content = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
                        conversation_context += f"{role}: {content}\n"

            general_prompt = f"{conversation_context}Please answer this question helpfully: {question}"
            response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=general_prompt, system_prompt=None, temperature=0.7)
            # Prepend disclaimer for content not sourced from the local document DB
            return DISCLAIMER_TEXT + response
        except Exception:
            logger.exception("General chat failed")
            return DISCLAIMER_TEXT + "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

    # Proceed with RAG for non-general intents
    # Perform retrieval based on plan
        # Double RAG: section-based narrowing
        try:
            docs_for_context = perform_double_rag(question, selected_collections, section_filter, search_keywords, rag_top_k)
        except Exception:
            logger.exception("Double RAG failed, falling back to standard search")
            docs_for_context = perform_keyword_search(question, selected_collections, search_keywords, rag_top_k)
    else:
        # Standard or agentic keyword-based search
        docs_for_context = perform_keyword_search(question, selected_collections, search_keywords, rag_top_k)

    # Prioritize documents based on plan
    if retrieval_plan and primary_doc:
        # Sort docs to prioritize primary document type
        primary_docs = []
        secondary_docs_list = []
        
        for doc in docs_for_context:
            source = getattr(doc, 'metadata', {}).get('source', '').lower()
            if primary_doc.lower() in source:
                primary_docs.append(doc)
            elif any(sec.lower() in source for sec in secondary_docs):
                secondary_docs_list.append(doc)
        
        # Reorder: primary docs first, then secondary, then others
        other_docs = [d for d in docs_for_context if d not in primary_docs and d not in secondary_docs_list]
        docs_for_context = primary_docs + secondary_docs_list + other_docs
        logger.info(f"Prioritized docs: {len(primary_docs)} primary, {len(secondary_docs_list)} secondary, {len(other_docs)} other")

    # If exact-match chunks were found earlier, prioritize them
    try:
        combined_docs = docs_for_context
        if 'deduped_exact' in locals() and deduped_exact:
            # build set of keys for exact docs
            exact_keys = set()
            for d in deduped_exact:
                try:
                    key = (getattr(d, 'page_content', '')[:200], tuple(sorted((d.metadata or {}).items())))
                except Exception:
                    key = (str(getattr(d, 'page_content', ''))[:200], None)
                exact_keys.add(key)
            # filter out any docs already present in exact set
            filtered_all = []
            for d in docs_for_context:
                try:
                    key = (getattr(d, 'page_content', '')[:200], tuple(sorted((d.metadata or {}).items())))
                except Exception:
                    key = (str(getattr(d, 'page_content', ''))[:200], None)
                if key not in exact_keys:
                    filtered_all.append(d)
            combined_docs = deduped_exact + filtered_all
            logger.info(f"Prioritized {len(deduped_exact)} exact-match chunks at the front of context")
    except Exception:
        combined_docs = docs_for_context

    docs_for_context = combined_docs[: num_docs]
    
    has_retrieved_docs = len(docs_for_context) > 0
    
    context_text = build_context_from_selected_docs(docs_for_context, max_chars_per_doc=max_chars_per_doc)

    # Debug: log the context text
    logger.info(f"Built context text (length: {len(context_text)}): {context_text[:500]}...")
    
    # Debug: log sources of documents in context
    context_sources = []
    for doc in docs_for_context:
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get('source', 'unknown')
            context_sources.append(source)
    logger.info(f"Context sources: {context_sources}")

    # If there is no context (no docs retrieved), do general chat instead of strict RAG
    if not context_text.strip():
        try:
            # For general questions or when no document context available, use general chat with conversation context
            conversation_context = ""
            if conversation_history and len(conversation_history) > 1:
                recent_history = conversation_history[-6:-1]  # Exclude current question
                if recent_history:
                    conversation_context = "\n\nPrevious conversation:\n"
                    for msg in recent_history:
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        content = msg['content'][:300] + "..." if len(msg['content']) > 300 else msg['content']
                        conversation_context += f"{role}: {content}\n"

            general_prompt = f"{conversation_context}Please answer this question helpfully: {question}"
            response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=general_prompt, system_prompt=None, temperature=0.7)
            # Prepend disclaimer for content not sourced from the local document DB
            return DISCLAIMER_TEXT + response
        except Exception:
            logger.exception("General chat fallback failed")
            return DISCLAIMER_TEXT + "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

    # Build conversation history context if available
    conversation_context = ""
    if conversation_history and len(conversation_history) > 1:  # More than just the current question
        # Get the last few exchanges (up to 6 messages to keep context manageable)
        recent_history = conversation_history[-6:-1]  # Exclude the current question
        if recent_history:
            conversation_context = "\n\nPrevious Conversation:\n"
            for msg in recent_history:
                role = "User" if msg['role'] == 'user' else "Assistant"
                # Truncate long messages to keep prompt size manageable
                content = msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content']
                conversation_context += f"{role}: {content}\n"
            conversation_context += "\n"

    # Adjust strictness of instructions when partial answers are allowed
    if allow_partial:
        local_system = SYSTEM_INSTRUCTIONS.replace(
            "reply exactly: 'I don't know — I need more information.'",
            "if the Context lacks complete information, give a concise best-effort answer and append a short clarification like 'I may be missing information.' If you cannot answer at all, say exactly 'I don't know — I need more information.'",
        )
    else:
        local_system = SYSTEM_INSTRUCTIONS

    combined_template = (
        local_system
        + "\n\nExamples:\n"
        + FEW_SHOT_EXAMPLES
        + "\n\nContext:\n{context}"
        + conversation_context
        + "\n\nQuestion: {question}\n\nAnswer:"
    )
    # Format the prompt
    full_prompt = combined_template.format(context=context_text, question=question)
    
    # Use Azure OpenAI instead of Ollama
    try:
        response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=full_prompt, system_prompt=None, temperature=0.0)
        logger.info(f"LLM response: {response}")

        # If the model didn't include a source, append the actual retrieved sources (unique, in-order)
        appended_sources = False
        if ("[source:" not in response and "[sources:" not in response) and docs_for_context:
            seen = []
            for d in docs_for_context:
                src = None
                if hasattr(d, 'metadata') and isinstance(d.metadata, dict):
                    src = d.metadata.get('source') or d.metadata.get('filename')
                if not src:
                    # try to fallback to a best-effort string representation
                    try:
                        src = str(d.metadata) if hasattr(d, 'metadata') else None
                    except Exception:
                        src = None
                if src and src not in seen:
                    seen.append(src)
                if len(seen) >= 5:
                    break
            if seen:
                # append up to 5 unique sources in a single line
                sources_line = ", ".join(seen)
                response = response + f"\n\n[sources: {sources_line}]"
                appended_sources = True

        # Iterative RAG: Check response completeness and refine if needed
        try:
            completeness_check = check_response_completeness(response, question)
            if not completeness_check.get("is_complete", True) and completeness_check.get("suggested_queries"):
                logger.info("Response incomplete, performing iterative refinement")
                response = iterative_rag_refinement(
                    question=question,
                    initial_response=response,
                    selected_collections=selected_collections,
                    suggested_queries=completeness_check["suggested_queries"],
                    rag_top_k=rag_top_k
                )
        except Exception:
            logger.exception("Iterative RAG refinement failed, using original response")
    except Exception:
        logger.exception("LLM query failed")
        response = "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

    # Determine if the final response should be considered sourced from the DB
    sourced_from_db = ("[source:" in response) or ("[sources:" in response) or appended_sources

    # Add disclaimer only when the response is NOT sourced from the local document DB
    if not sourced_from_db and not response.startswith(DISCLAIMER_TEXT):
        response = DISCLAIMER_TEXT + response

    return response


def generate_conversation_summary(conversation_history, max_length=500):
    """Generate a summary of the conversation history using Azure OpenAI."""
    if not conversation_history:
        return "No conversation history available."

    # Format conversation for summarization, focusing on the dialogue flow
    conversation_text = ""
    for msg in conversation_history[-20:]:  # Last 20 messages for context
        role = "User" if msg['role'] == 'user' else "Assistant"
        # Truncate long messages but keep key questions/answers
        content = msg['content']
        if len(content) > 200:
            # Try to keep the question/answer essence
            if role == "User":
                content = content[:200] + "..."
            else:
                # For assistant, try to extract key points
                lines = content.split('\n')
                key_lines = [line for line in lines if not line.startswith('[') and not line.startswith('What other')]
                content = '\n'.join(key_lines[:3])[:200] + "..." if key_lines else content[:200] + "..."
        conversation_text += f"{role}: {content}\n"

    summary_prompt = f"""Please provide a concise summary of this conversation dialogue, focusing on the topics discussed, questions asked, and key responses exchanged between the user and assistant. Do not summarize document content or technical details from manuals - focus only on the conversation flow and main discussion points. Keep it under {max_length} characters.

Conversation:
{conversation_text}

Summary:"""

    try:
        summary = chat_with_model(
            deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            user_prompt=summary_prompt,
            system_prompt="You are a helpful assistant that creates clear, concise conversation summaries focusing on dialogue and discussion points.",
            temperature=0.3
        )
        return summary[:max_length] + "..." if len(summary) > max_length else summary
    except Exception as e:
        logger.exception("Failed to generate conversation summary")
        return f"Unable to generate summary: {str(e)}"


def add_to_unified_history(mode, role, content):
    """Add a message to the unified conversation history."""
    if 'unified_conversation' not in st.session_state:
        st.session_state['unified_conversation'] = []

    # Add timestamp and mode information
    timestamp = datetime.now().strftime("%H:%M:%S")
    unified_msg = {
        'timestamp': timestamp,
        'mode': mode,
        'role': role,
        'content': content
    }

    st.session_state['unified_conversation'].append(unified_msg)

    # Keep only last 100 messages to prevent memory issues
    if len(st.session_state['unified_conversation']) > 100:
        st.session_state['unified_conversation'] = st.session_state['unified_conversation'][-100:]


def detect_language(text: str) -> str:
    """Detect the language of the input text."""
    try:
        return detect(text)
    except Exception:
        return 'en'  # Default to English

def translate_text(text: str, from_lang: str, to_lang: str) -> str:
    """Translate text using Azure OpenAI."""
    if from_lang == to_lang:
        return text
    try:
        prompt = f"Translate the following text from {from_lang} to {to_lang}. Only return the translated text, no explanations: {text}"
        translated = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=prompt, system_prompt=None, temperature=0.0)
        return translated.strip()
    except Exception:
        logger.exception("Translation failed")
        return text  # Fallback to original text


def unified_page():
    st.title("AI Hackathon")

    # Set up session logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Session started. Log file: {log_filename}")

    # Mode selection at the top
    mode = st.radio(
        "Select interaction mode:",
        ["💬 Chat", "📄 PDF & Chat", "🌋 Image Analysis"],
        key="unified_mode",
        horizontal=True
    )

    models_info = ollama.list()
    # robust extraction of model names
    available_models = safe_extract_model_names(models_info)

    if not available_models:
        st.warning("No models found in Ollama. Ensure the Ollama daemon is running and models are pulled.")
        return

    # Ensure collections metadata is persisted across reruns
    if 'collections_meta' not in st.session_state:
        st.session_state['collections_meta'] = list_collections()

    # Use sidebar for configuration
    with st.sidebar:
        st.subheader("Configuration & Uploads")
        selected_model = st.selectbox("Pick a model", available_models, key="unified_selected_model")

        st.markdown("---")
        st.subheader("Upload an image (optional)")
        uploaded_image = st.file_uploader("Image for multimodal questions", type=["png", "jpg", "jpeg"], key="unified_image_upload")

        st.markdown("---")
        st.subheader("Upload Documents (Auto-indexed)")
        uploaded_pdfs = st.file_uploader("Upload PDF, Word (.doc, .docx), or Text (.txt) files", type=["pdf", "doc", "docx", "txt"], accept_multiple_files=True, key="unified_pdf_upload")

        # Track processed files to avoid re-indexing
        if 'processed_files' not in st.session_state:
            st.session_state['processed_files'] = set()

        # Auto-index uploaded PDFs (only if not already processed)
        if uploaded_pdfs:
            # Get checksums of uploaded files to check if they've been processed
            current_file_checksums = set()
            for uploaded in uploaded_pdfs:
                file_bytes = uploaded.getvalue()
                checksum = hashlib.sha256(file_bytes).hexdigest()
                current_file_checksums.add(checksum)
            
            # Find new files that haven't been processed yet
            new_files = []
            for i, uploaded in enumerate(uploaded_pdfs):
                file_bytes = uploaded.getvalue()
                checksum = hashlib.sha256(file_bytes).hexdigest()
                if checksum not in st.session_state['processed_files']:
                    new_files.append(uploaded)
            
            if new_files:
                # Provide a visual progress bar while indexing
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text(f"Auto-indexing {len(new_files)} new file(s)...")

                def _progress_cb(completed: int, total: int):
                    try:
                        progress_bar.progress(completed / total)
                        status_text.text(f"Indexing {completed}/{total} files...")
                    except Exception:
                        pass

                try:
                    # Index the new files only
                    new_collections = handle_uploaded_files(new_files, PDF_UPLOAD_DIR, progress_callback=_progress_cb)
                    # Update the session state with the new collections
                    st.session_state['collections_meta'].update(new_collections)
                    # Mark these files as processed
                    for uploaded in new_files:
                        file_bytes = uploaded.getvalue()
                        checksum = hashlib.sha256(file_bytes).hexdigest()
                        st.session_state['processed_files'].add(checksum)
                    st.success(f"✅ Successfully indexed {len(new_collections)} collection(s): {', '.join(new_collections.keys())}")
                except Exception as e:
                    st.error(f"❌ Error indexing documents: {e}")
                finally:
                    progress_bar.empty()
                    status_text.empty()
            else:
                st.info("ℹ️ All uploaded files have already been indexed.")

        st.markdown("---")
        st.subheader("Database Management")

        # Show current collections
        collections_meta = st.session_state.get('collections_meta', {})
        if collections_meta:
            st.write(f"**Available Document Collections:** {len(collections_meta)}")
            for name in collections_meta.keys():
                # Get detailed info for each collection
                info = get_collection_info(name)
                filename = info.get('filename', 'Unknown')
                st.write(f"• {name}: {filename}")

            # Refresh collections button
            if st.button("🔄 Refresh Collections", key="refresh_collections"):
                with st.spinner("Refreshing collections..."):
                    try:
                        st.session_state['collections_meta'] = list_collections()
                        st.success("✅ Collections refreshed!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error refreshing collections: {e}")

            # Clean database button
            if st.button("🗑️ Clean Database", key="clean_database"):
                with st.spinner("Cleaning database..."):
                    try:
                        delete_all_collections()
                        # Clear local data
                        if os.path.exists(PDF_UPLOAD_DIR):
                            shutil.rmtree(PDF_UPLOAD_DIR)
                            PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                        st.session_state['collections_meta'] = {}
                        st.success("✅ Database cleaned! All collections and uploaded files removed.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error cleaning database: {e}")
        else:
            st.info("No document collections available. Upload documents above to get started.")

        # Auto-select all available collections for document queries
        selected_collections = list(collections_meta.keys()) if collections_meta else []
        if selected_collections:
            st.info(f"Auto-selected {len(selected_collections)} collection(s) for document queries.")
        else:
            st.info("No collections available for selection.")

        st.markdown("---")
        st.subheader("Conversation Summary")

        # Show conversation statistics
        unified_history = st.session_state.get('unified_conversation', [])
        if unified_history:
            total_messages = len(unified_history)
            chat_messages = len([m for m in unified_history if m['mode'] == '💬 Chat'])
            pdf_messages = len([m for m in unified_history if m['mode'] == '📄 PDF & Chat'])
            image_messages = len([m for m in unified_history if m['mode'] == '🌋 Image Analysis'])

            st.write(f"**Total Messages:** {total_messages}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chat", chat_messages)
            with col2:
                st.metric("PDF & Chat", pdf_messages)
            with col3:
                st.metric("Image Analysis", image_messages)

            # Generate summary button
            if st.button("Generate Conversation Summary", key="generate_summary"):
                with st.spinner("Generating summary..."):
                    # Convert unified history to simple format for summarization
                    simple_history = [
                        {'role': msg['role'], 'content': msg['content']}
                        for msg in unified_history
                    ]
                    summary = generate_conversation_summary(simple_history)
                    st.session_state['conversation_summary'] = summary

            # Display summary if available
            if 'conversation_summary' in st.session_state:
                st.markdown("**Conversation Summary:**")
                st.info(st.session_state['conversation_summary'])

                # Export summary button
                if st.button("Export Summary", key="export_summary"):
                    # Create a downloadable text file with the summary
                    summary_content = f"AI Hackathon Conversation Summary\n{'='*40}\n\n{st.session_state['conversation_summary']}\n\n{'='*40}\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Messages: {total_messages}\nChat: {chat_messages}, PDF & Chat: {pdf_messages}, Image Analysis: {image_messages}"
                    st.download_button(
                        label="Download Summary as Text",
                        data=summary_content,
                        file_name="conversation_summary.txt",
                        mime="text/plain",
                        key="download_summary"
                    )

                # Clear summary button
                if st.button("Clear Summary", key="clear_summary"):
                    del st.session_state['conversation_summary']
                    st.rerun()
        else:
            st.info("No conversation history yet. Start chatting to see summary features!")

        st.markdown("---")
        st.subheader("Language Settings")
        selected_language = st.selectbox("Response Language", ["English", "German", "French"], key="selected_language")
        lang_codes = {"English": "en", "German": "de", "French": "fr"}
        selected_lang_code = lang_codes[selected_language]

    # Main area for interaction
    st.subheader("Interaction")

    if mode == "💬 Chat":
        st.markdown("**Chat Mode:** Ask general questions (Azure OpenAI)")
        # Initialize conversation history if not present
        if 'chat_conversation' not in st.session_state:
            st.session_state['chat_conversation'] = []

        # Display conversation history
        for msg in st.session_state['chat_conversation']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        # Chat input
        user_input = st.chat_input("Enter your question:")
        if user_input:
            # Detect input language and translate to English for processing
            input_lang = detect_language(user_input)
            translated_input = translate_text(user_input, input_lang, 'en')
            
            # Add user message to conversation (original language)
            st.session_state['chat_conversation'].append({"role": "user", "content": user_input})
            add_to_unified_history("💬 Chat", "user", user_input)
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Thinking..."):
                try:
                    response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=translated_input, system_prompt=None, temperature=0.7)
                    # Translate response back to selected language
                    final_response = translate_text(response, 'en', selected_lang_code)
                    # Add assistant message to conversation (translated)
                    st.session_state['chat_conversation'].append({"role": "assistant", "content": final_response})
                    add_to_unified_history("💬 Chat", "assistant", final_response)
                    with st.chat_message("assistant"):
                        st.write(final_response)
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.session_state['chat_conversation'].append({"role": "assistant", "content": error_msg})
                    add_to_unified_history("💬 Chat", "assistant", error_msg)
                    with st.chat_message("assistant"):
                        st.write(error_msg)

    elif mode == "📄 PDF & Chat":
        st.markdown("**Document & Chat Mode:** Ask questions about uploaded documents (PDF, Word, Text) or general questions (Azure OpenAI - auto-searches all collections)")
        # chat interface
        # Initialize conversation history if not present
        if 'pdf_chat_conversation' not in st.session_state:
            st.session_state['pdf_chat_conversation'] = []

        # Display conversation history
        for msg in st.session_state['pdf_chat_conversation']:
            with st.chat_message(msg['role']):
                st.write(msg['content'])

        # Chat input
        user_input = st.chat_input("Ask about PDFs or general questions:")
        if user_input:
            # Detect input language and translate to English for processing
            input_lang = detect_language(user_input)
            translated_input = translate_text(user_input, input_lang, 'en')
            
            # Add user message to conversation (original language)
            st.session_state['pdf_chat_conversation'].append({"role": "user", "content": user_input})
            add_to_unified_history("📄 PDF & Chat", "user", user_input)
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Thinking..."):
                try:
                    response = process_question_across_collections(
                        question=translated_input,  # Use translated question
                        selected_collections=selected_collections,
                        selected_model=selected_model,
                        conversation_history=st.session_state['pdf_chat_conversation'],
                        rag_top_k=RAG_TOP_K,
                        max_chars_per_doc=EXCERPT_MAX_CHARS
                    )
                    # Translate response back to selected language
                    final_response = translate_text(response, 'en', selected_lang_code)
                    # Add assistant message to conversation (translated)
                    st.session_state['pdf_chat_conversation'].append({"role": "assistant", "content": final_response})
                    add_to_unified_history("📄 PDF & Chat", "assistant", final_response)
                    with st.chat_message("assistant"):
                        st.write(final_response)
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.session_state['pdf_chat_conversation'].append({"role": "assistant", "content": error_msg})
                    add_to_unified_history("📄 PDF & Chat", "assistant", error_msg)
                    with st.chat_message("assistant"):
                        st.write(error_msg)

    elif mode == "🌋 Image Analysis":
        st.markdown("**Image Analysis Mode:** Ask questions about the uploaded image (Ollama Vision)")
        if not uploaded_image:
            st.warning("Please upload an image.")
        else:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Initialize conversation history if not present
            if 'image_conversation' not in st.session_state:
                st.session_state['image_conversation'] = []

            # Display conversation history
            for msg in st.session_state['image_conversation']:
                with st.chat_message(msg['role']):
                    st.write(msg['content'])

            # Chat input
            user_input = st.chat_input("Enter your question about the image:")
            if user_input:
                # Detect input language and translate to English for processing
                input_lang = detect_language(user_input)
                translated_input = translate_text(user_input, input_lang, 'en')
                
                # Add user message to conversation (original language)
                st.session_state['image_conversation'].append({"role": "user", "content": user_input})
                add_to_unified_history("🌋 Image Analysis", "user", user_input)
                with st.chat_message("user"):
                    st.write(user_input)

                with st.spinner("Analyzing image..."):
                    try:
                        # Convert image to base64
                        img_b64 = img_to_base64(image)
                        # Use Ollama API directly for vision with translated prompt
                        response = call_generate_api(model=selected_model, prompt=translated_input, images_b64=[img_b64])
                        # Translate response back to selected language
                        final_response = translate_text(response, 'en', selected_lang_code)
                        # Add assistant message to conversation (translated)
                        st.session_state['image_conversation'].append({"role": "assistant", "content": final_response})
                        add_to_unified_history("🌋 Image Analysis", "assistant", final_response)
                        with st.chat_message("assistant"):
                            st.write(final_response)
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.session_state['image_conversation'].append({"role": "assistant", "content": error_msg})
                        add_to_unified_history("🌋 Image Analysis", "assistant", error_msg)
                        with st.chat_message("assistant"):
                            st.write(error_msg)


def main():
    unified_page()


if __name__ == "__main__":
    # Load .env from repository root so environment variables are available
    load_dotenv(find_dotenv())

    main()