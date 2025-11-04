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
    "You are a retrieval-augmented assistant for service technicians in heavy machinery manufacturing and automobile sectors. Follow these rules strictly:\n"
    "- Use ONLY the information present in the provided Context section below to answer the Question. Do NOT use external knowledge.\n"
    "- If the Context does not contain sufficient information to answer, reply exactly: 'The current manuals don’t mention that. Please try searching with alternate terms.'\n"
    "- Be concise: prefer up to 10 short bullet points when possible, especially for lists of parts or steps.\n"
    "- For parts catalogs, accurately match part numbers to descriptions, specifications, or components.\n"
    "- If the question asks for a description of a specific part number, extract and summarize the relevant details from the context.\n"
    "- If the question asks for a part number based on a description, find the matching part number in the context.\n"
    "- If you used retrieved content, append a one-line citation like [source: filename] (use the source provided in Context).\n"
    "- Do NOT fabricate details, dates, or steps not present in Context.\n"
    "- Provide direct, technical answers suitable for technicians, including any relevant specs or notes.\n"
)

FEW_SHOT_EXAMPLES = (
    "Example 1:\nQ: What is the engine displacement of the Bullet EFI?\nA: - 499 cc single-cylinder, 4-stroke air-cooled engine\n[source: BulletEFI_OwnersManual.pdf] (Section: Technical Specifications)\n\n"
    "Example 2:\nQ: How should the motorcycle be refueled safely?\nA: - Turn off the engine before refueling\n- Avoid open flames or sparks\n- Fill only to the bottom of the filler neck insert\n[source: BulletEFI_OwnersManual.pdf] (Section: Safe Operating Rules)\n\n"
    "Example 3:\nQ: What is the recommended tyre pressure for solo riding?\nA: - Front: 18 PSI\n- Rear: 28 PSI\n[source: BulletEFI_OwnersManual.pdf] (Section: Technical Specifications)\n\n"
    "Example 4:\nQ: What oil should be used for the Classic EFI engine?\nA: - Use 15W50 API SL JASO MA semi-synthetic oil\n- Capacity: 2.3 to 2.5 litres including filter\n[source: ClassicEFI_OwnersManual.pdf] (Section: Recommended Oils)\n\n"
    "Example 5:\nQ: What are the key safety steps before starting the Classic EFI?\nA: - Check brakes, clutch, gear shifter, tyre pressures, fuel and oil levels\n- Ensure proper riding gear\n[source: ClassicEFI_OwnersManual.pdf] (Section: Safe Operating Rules)\n\n"
    "Example 6:\nQ: How can I prevent theft of the motorcycle?\nA: - Always lock the steering head\n- Remove the ignition key after parking\n[source: ClassicEFI_OwnersManual.pdf] (Section: Rules of the Road)\n\n"
    "Example 7:\nQ: What is the maximum power output of the Continental GT?\nA: - 21.4 kW @ 5100 RPM\n[source: ContiGT_OwnersManual.pdf] (Section: Technical Specifications)\n\n"
    "Example 8:\nQ: How do I use the fuel level indicator on the Continental GT?\nA: - Fuel bars decrease toward 'E' as fuel lowers\n- When last bar blinks (<3L), refuel immediately\n[source: ContiGT_OwnersManual.pdf] (Section: Operation of Controls)\n\n"
    "Example 9:\nQ: What are the recommended tyre pressures for solo riding (Continental GT)?\nA: - Front: 20 PSI\n- Rear: 30 PSI\n[source: ContiGT_OwnersManual.pdf] (Section: Technical Specifications)\n\n"
    "Example 10:\nQ: If the provided Context does not include the requested information, how should the assistant respond?\nA: The current manuals don’t mention that. Please try searching with alternate terms.\n\n"
)
RAG_TOP_K = 10  # per collection (increased for better retrieval)
EXCERPT_MAX_CHARS = 2000

# NOTE: create_vector_db_from_path, load_chroma_collection and handle_uploaded_files
# are implemented in `components.pdf_indexing`. Local duplicate implementations were
# removed to avoid re-indexing the same files multiple times and to ensure a single
# source of truth for collection metadata and persistence.

from langchain.schema import Document
import re


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

def is_summary_request(question: str) -> bool:
    """Heuristic: detect if the question is asking for a summary, abstract, or overview.

    This is used to trigger a different retrieval + LLM prompt strategy.
    """
    # simple keyword-based heuristics for now
    keywords = ["summary", "abstract", "overview", "synopsis", "recap"]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)

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
    rag_top_k: int = RAG_TOP_K,
    max_chars_per_doc: int = EXCERPT_MAX_CHARS,
    allow_partial: bool = False,
    debug: bool = False,
) -> str:
    """
    Retrieves documents from each selected collection, builds context, and queries the LLM.
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
            return response + "\n\nWhat other help do you need or what other information do you need?"
        except Exception:
            logger.exception("General chat failed")
            return "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

    logger.info(f"Processing question across collections: {selected_collections} using model {selected_model}")
    all_docs = []

    # If the user explicitly asked for a summary/abstract/overview, run a summary flow
    try:
        if is_summary_request(question):
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
                return "The current manuals don’t mention that. Please try searching with alternate terms.\n\nWhat other help do you need or what other information do you need?"
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
                return resp + "\n\nWhat other help do you need or what other information do you need?"
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
        # deduplicate exact_docs while preserving order
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
        except Exception:
            logger.exception(f"Failed to query collection: {coll}")

    # If we have numeric scores, sort by score descending (higher score = more similar). Otherwise keep retrieval order
    try:
        scores_present = [r[1] for r in retrieved if r[1] is not None]
        if scores_present and all(isinstance(s, (int, float)) for s in scores_present):
            # sort by score descending
            retrieved.sort(key=lambda x: x[1], reverse=True)
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

    # If exact-match chunks were found earlier, prioritize them and avoid duplicates
    try:
        combined_docs = all_docs
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
            for d in all_docs:
                try:
                    key = (getattr(d, 'page_content', '')[:200], tuple(sorted((d.metadata or {}).items())))
                except Exception:
                    key = (str(getattr(d, 'page_content', ''))[:200], None)
                if key not in exact_keys:
                    filtered_all.append(d)
            combined_docs = deduped_exact + filtered_all
            logger.info(f"Prioritized {len(deduped_exact)} exact-match chunks at the front of context")
    except Exception:
        combined_docs = all_docs

    docs_for_context = combined_docs[: num_docs]
    context_text = build_context_from_selected_docs(docs_for_context, max_chars_per_doc=max_chars_per_doc)

    # Debug: log the context text
    logger.info(f"Built context text (length: {len(context_text)}): {context_text[:500]}...")

    # If there is no context (no docs retrieved), do general chat instead of strict RAG
    if not context_text.strip():
        try:
            # For general questions or when no PDF context available, use general chat
            general_prompt = f"Please answer this question helpfully: {question}"
            response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=general_prompt, system_prompt=None, temperature=0.7)
            return response + "\n\nWhat other help do you need or what other information do you need?"
        except Exception:
            logger.exception("General chat fallback failed")
            return "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

    # If there is minimal context (less than 100 chars), also do general chat to allow smooth conversation
    if len(context_text.strip()) < 100:
        try:
            # For general questions or when insufficient PDF context, use general chat
            general_prompt = f"Please answer this question helpfully: {question}"
            response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=general_prompt, system_prompt=None, temperature=0.7)
            return response + "\n\nWhat other help do you need or what other information do you need?"
        except Exception:
            logger.exception("General chat fallback failed")
            return "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"

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
        + "\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    # Format the prompt
    full_prompt = combined_template.format(context=context_text, question=question)
    
    # Use Azure OpenAI instead of Ollama
    try:
        response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=full_prompt, system_prompt=None, temperature=0.0)
        logger.info(f"LLM response: {response}")
        
        # If the response indicates no information in manuals, fallback to general chat
        if "manuals don't mention" in response.lower():
            general_prompt = f"Please answer this question helpfully: {question}"
            try:
                response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=general_prompt, system_prompt=None, temperature=0.7)
                # For general chat, don't append sources
                return response + "\n\nWhat other help do you need or what other information do you need?"
            except Exception:
                logger.exception("General chat fallback failed")
                return "Sorry — something went wrong.\n\nWhat other help do you need or what other information do you need?"
        
        # If the model didn't include a source, append the actual retrieved sources (unique, in-order)
        if "[source:" not in response and docs_for_context:
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
        return response + "\n\nWhat other help do you need or what other information do you need?"
    except Exception:
        logger.exception("LLM invocation failed")
        return "Sorry — something went wrong while generating the answer.\n\nWhat other help do you need or what other information do you need?"


def unified_page():
    st.title("AI Hackathon")

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
        st.subheader("Upload PDFs (optional)")
        uploaded_pdfs = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True, key="unified_pdf_upload")

        # Only index when the user explicitly requests it to avoid re-indexing on every Streamlit rerun
        if uploaded_pdfs:
            st.info(f"{len(uploaded_pdfs)} file(s) ready to index. Click 'Index uploaded PDFs' to create collections.")
            index_clicked = st.button("Index uploaded PDFs", key="index_uploaded_pdfs")
        else:
            index_clicked = False

        if uploaded_pdfs and index_clicked:
            # Provide a visual progress bar while indexing using the backend progress_callback
            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress_cb(completed: int, total: int):
                try:
                    progress_bar.progress(completed / total)
                    status_text.text(f"Indexing {completed}/{total} files...")
                except Exception:
                    pass

            try:
                # Index the uploaded PDFs
                new_collections = handle_uploaded_files(uploaded_pdfs, PDF_UPLOAD_DIR, progress_callback=_progress_cb)
                # Update the session state with the new collections
                st.session_state['collections_meta'].update(new_collections)
                st.success(f"Successfully indexed {len(new_collections)} collection(s): {', '.join(new_collections.keys())}")
                # Clear the uploaded files after indexing to prevent re-indexing
                st.session_state['unified_pdf_upload'] = []
                st.rerun()  # Refresh the page to clear the file uploader
            except Exception as e:
                st.error(f"Error indexing PDFs: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

        st.markdown("---")
        st.subheader("Select Collections")
        collections_meta = st.session_state.get('collections_meta', {})
        if collections_meta:
            selected_collections = st.multiselect(
                "Choose one or more PDF collections to search",
                options=list(collections_meta.keys()),
                default=list(collections_meta.keys())[:1],  # Default to first collection if available
                key="unified_selected_collections"
            )
        else:
            selected_collections = []
            st.info("No PDF collections available. Upload and index PDFs to get started.")

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
            # Add user message to conversation
            st.session_state['chat_conversation'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Thinking..."):
                try:
                    response = chat_with_model(deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"), user_prompt=user_input, system_prompt=None, temperature=0.7)
                    # Add assistant message to conversation
                    st.session_state['chat_conversation'].append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.session_state['chat_conversation'].append({"role": "assistant", "content": error_msg})
                    with st.chat_message("assistant"):
                        st.write(error_msg)

    elif mode == "📄 PDF & Chat":
        st.markdown("**PDF & Chat Mode:** Ask questions about the selected PDF collections or general questions (Azure OpenAI)")
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
            # Add user message to conversation
            st.session_state['pdf_chat_conversation'].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.spinner("Thinking..."):
                try:
                    response = process_question_across_collections(
                        question=user_input,
                        selected_collections=selected_collections,
                        selected_model=selected_model,
                        rag_top_k=RAG_TOP_K,
                        max_chars_per_doc=EXCERPT_MAX_CHARS
                    )
                    # Add assistant message to conversation
                    st.session_state['pdf_chat_conversation'].append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.write(response)
                except Exception as e:
                    error_msg = f"Error: {e}"
                    st.session_state['pdf_chat_conversation'].append({"role": "assistant", "content": error_msg})
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
                # Add user message to conversation
                st.session_state['image_conversation'].append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.write(user_input)

                with st.spinner("Analyzing image..."):
                    try:
                        # Convert image to base64
                        img_b64 = img_to_base64(image)
                        # Use Ollama API directly for vision
                        response = call_generate_api(model=selected_model, prompt=user_input, images_b64=[img_b64])
                        # Add assistant message to conversation
                        st.session_state['image_conversation'].append({"role": "assistant", "content": response})
                        with st.chat_message("assistant"):
                            st.write(response)
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.session_state['image_conversation'].append({"role": "assistant", "content": error_msg})
                        with st.chat_message("assistant"):
                            st.write(error_msg)


def main():
    unified_page()


if __name__ == "__main__":
    # Load .env from repository root so environment variables are available
    load_dotenv(find_dotenv())

    main()