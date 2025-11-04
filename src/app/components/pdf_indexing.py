from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import os
from hashlib import sha256
from datetime import datetime
import time
import concurrent.futures
from PyPDF2 import PdfReader
from langchain.schema import Document

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Use local sentence-transformers for fast batch embeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import chromadb
from chromadb.config import Settings

from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = str(Path(".") / "data" / "vectors")
METADATA_PATH = Path(".") / "data" / "pdf_collections.json"


def get_embedding_function():
    """Return the embedding function for Chroma."""
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def load_chroma_collection(collection_name: str) -> Chroma:
    """Load a Chroma collection by name."""
    from chromadb import PersistentClient
    client = PersistentClient(path=PERSIST_DIRECTORY)
    return Chroma(client=client, collection_name=collection_name, embedding_function=get_embedding_function())


def load_collections_metadata() -> Dict[str, Dict[str, str]]:
    """Load metadata and normalize to new format: {collection_name: {"filename": ..., "checksum": ...}}"""
    if METADATA_PATH.exists():
        try:
            raw = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            # Normalize older format where values were filenames (str)
            normalized: Dict[str, Dict[str, str]] = {}
            for k, v in raw.items():
                if isinstance(v, str):
                    normalized[k] = {"filename": v, "checksum": ""}
                elif isinstance(v, dict):
                    # already new format
                    normalized[k] = {"filename": v.get("filename", ""), "checksum": v.get("checksum", "")}
                else:
                    normalized[k] = {"filename": str(v), "checksum": ""}
            return normalized
        except Exception:
            logger.exception("Failed loading metadata, returning empty mapping")
            return {}
    return {}


def save_collections_metadata(meta: Dict[str, Dict[str, str]]) -> None:
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_checksum_bytes(data: bytes) -> str:
    h = sha256()
    h.update(data)
    return h.hexdigest()


# instantiate a local sentence-transformers model (shared across calls)
# model choice can be tuned; all-MiniLM-L6-v2 is a good balance of speed/quality
_SB_MODEL = None
def _get_sentence_transformer():
    global _SB_MODEL
    if _SB_MODEL is None:
        try:
            _SB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            # fallback to a small default name if download fails
            _SB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SB_MODEL


def create_vector_db_from_path(path: Path, collection_name: Optional[str] = None) -> str:
    """Create a Chroma vector collection from a PDF and attach richer metadata to each chunk.

    Returns the collection_name used.
    """
    # Use PyPDF2 for better text extraction, similar to _index_file_worker
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
    except Exception:
        pages = [""]

    # Build langchain Document objects per page
    docs = []
    for i, pg in enumerate(pages):
        docs.append(Document(page_content=pg, metadata={"source": path.name, "page": i + 1}))

    # Use smaller chunks for better retrieval precision
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.split_documents(docs)

    # Attach helpful metadata to each chunk: source filename and checksum (and page if available)
    try:
        with open(path, 'rb') as f:
            file_bytes = f.read()
            checksum = _compute_checksum_bytes(file_bytes)
    except Exception:
        checksum = ""

    for c in chunks:
        # ensure metadata dict exists
        if not hasattr(c, 'metadata') or c.metadata is None:
            c.metadata = {}
        c.metadata['source'] = path.name
        c.metadata['checksum'] = checksum

    if collection_name is None:
        collection_name = f"pdf_{abs(hash(str(path)))}"

    # Batch embed using sentence-transformers and add to a single chromadb PersistentClient
    try:
        texts = [getattr(c, 'page_content', '') or '' for c in chunks]
        metadatas = [c.metadata if hasattr(c, 'metadata') and isinstance(c.metadata, dict) else {} for c in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]

        model = _get_sentence_transformer()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        if isinstance(embeddings, np.ndarray):
            embedded = embeddings.tolist()
        else:
            embedded = [list(e) for e in embeddings]

        # Use PersistentClient everywhere to avoid backend mismatches
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.create_collection(name=collection_name)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embedded)

        # Ensure LangChain Chroma wrapper is available for compatibility
        try:
            lc_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            try:
                # LangChain wrapper will operate against the same persistent store
                chroma_lc = Chroma(client=lc_client, collection_name=collection_name, embedding_function=get_embedding_function())
                chroma_lc.persist()
            except Exception:
                pass
        except Exception:
            pass

        return collection_name
    except Exception:
        logger.exception("Batch embedding with sentence-transformers failed; falling back to langchain Chroma.from_documents")
        try:
            # fallback: use LangChain Chroma with sentence-transformers via a simple wrapper
            Chroma.from_documents(documents=chunks, embedding=_get_sentence_transformer().encode, persist_directory=PERSIST_DIRECTORY, collection_name=collection_name)
            return collection_name
        except Exception:
            logger.exception("Fallback Chroma.from_documents also failed")
            raise


def _index_file_worker(path: Path, collection_name: Optional[str] = None) -> Dict[str, Any]:
    """Index a single PDF file: extract text with PyPDF2, split, embed and persist to Chroma.

    Returns a dict with keys: collection_name, filename, checksum, profile
    """
    profile = {}
    t0 = time.time()
    # extract pages
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
    except Exception:
        # fallback to empty
        pages = [""]
    profile['extract_time'] = time.time() - t0

    # build langchain Document objects per page
    docs = []
    for i, pg in enumerate(pages):
        docs.append(Document(page_content=pg, metadata={"source": path.name, "page": i + 1}))

    t1 = time.time()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = splitter.split_documents(docs)
    profile['split_count'] = len(chunks)
    profile['split_time'] = time.time() - t1

    try:
        with open(path, 'rb') as f:
            checksum = _compute_checksum_bytes(f.read())
    except Exception:
        checksum = ''

    # attach metadata
    for c in chunks:
        if not hasattr(c, 'metadata') or c.metadata is None:
            c.metadata = {}
        c.metadata['source'] = path.name
        c.metadata['checksum'] = checksum

    t2 = time.time()

    # create or overwrite collection
    if collection_name is None:
        collection_name = f"pdf_{checksum[:16]}" if checksum else f"pdf_{abs(hash(str(path)))}"

    # Batch embed all chunks at once using sentence-transformers
    try:
        texts = [getattr(c, 'page_content', '') or '' for c in chunks]
        metadatas = [c.metadata if hasattr(c, 'metadata') and isinstance(c.metadata, dict) else {} for c in chunks]
        ids = [f"{collection_name}_{i}" for i in range(len(texts))]

        model = _get_sentence_transformer()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=64)
        if isinstance(embeddings, np.ndarray):
            embedded = embeddings.tolist()
        else:
            embedded = [list(e) for e in embeddings]

        # Use PersistentClient to write the collection once
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
        collection = client.create_collection(name=collection_name)
        collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embedded)

        # ensure LangChain Chroma wrapper is usable for compatibility
        try:
            lc_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            chroma_lc = Chroma(client=lc_client, collection_name=collection_name, embedding_function=get_embedding_function())
            chroma_lc.persist()
        except Exception:
            pass

    except Exception:
        logger.exception("Batch embedding or chromadb add failed, falling back to Chroma.from_documents")
        try:
            client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
            Chroma.from_documents(documents=chunks, embedding=get_embedding_function(), client=client, collection_name=collection_name)
        except Exception:
            logger.exception("Fallback Chroma.from_documents also failed")

    profile['embed_persist_time'] = time.time() - t2
    profile['total_time'] = time.time() - t0

    return {"collection_name": collection_name, "filename": str(path.name), "checksum": checksum, "profile": profile}


def index_pdfs_concurrent(paths: List[Path], max_workers: int = 4, progress_callback=None) -> Dict[str, Dict[str, str]]:
    """Index multiple PDF files concurrently using threads.

    progress_callback(completed, total) will be called if provided.
    Returns mapping collection_name -> {filename, checksum, last_indexed}
    """
    results = {}
    total = len(paths)
    completed = 0
    # Use threads to avoid pickling issues and allow IO-bound embedding calls to run concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, total or 1)) as ex:
        future_to_path = {ex.submit(_index_file_worker, p): p for p in paths}
        for fut in concurrent.futures.as_completed(future_to_path):
            p = future_to_path[fut]
            try:
                out = fut.result()
                coll = out['collection_name']
                results[coll] = {"filename": out['filename'], "checksum": out['checksum'], "last_indexed": datetime.utcnow().isoformat()}
                # write profile info to disk for inspection
                try:
                    profile_path = Path(PERSIST_DIRECTORY).parent / 'index_profiles.json'
                    profile_path.parent.mkdir(parents=True, exist_ok=True)
                    existing = {}
                    if profile_path.exists():
                        try:
                            existing = json.loads(profile_path.read_text(encoding='utf-8'))
                        except Exception:
                            existing = {}
                    existing[coll] = out['profile']
                    profile_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
                except Exception:
                    logger.exception('Failed saving profile info')
            except Exception:
                logger.exception(f'Indexing failed for {p}')
            completed += 1
            if progress_callback:
                try:
                    progress_callback(completed, total)
                except Exception:
                    pass
    return results


# Update handle_uploaded_files to use concurrent indexing when multiple files are provided
def handle_uploaded_files(uploaded_files: List[Any], upload_dir: Path, max_workers: int = 4, progress_callback=None) -> Dict[str, str]:
    """
    Save files to disk and create a Chroma collection per file. Uses concurrent indexing for multiple files.

    Returns mapping collection_name -> original_filename
    """
    meta = load_collections_metadata()
    saved_paths = []
    for uploaded in uploaded_files:
        filename = uploaded.name
        target_path = upload_dir / filename
        # Avoid overwriting: if filename exists, create unique suffix
        if target_path.exists():
            base, ext = os.path.splitext(filename)
            counter = 1
            while (upload_dir / f"{base}_{counter}{ext}").exists():
                counter += 1
            target_path = upload_dir / f"{base}_{counter}{ext}"
        with open(target_path, "wb") as f:
            f.write(uploaded.getvalue())
        saved_paths.append(target_path)

    # If only one file, index directly to preserve previous behavior
    if len(saved_paths) == 1:
        out = _index_file_worker(saved_paths[0])
        coll = out['collection_name']
        meta[coll] = {"filename": out['filename'], "checksum": out['checksum'], "last_indexed": datetime.utcnow().isoformat()}
        save_collections_metadata(meta)
        return {k: v.get('filename', '') if isinstance(v, dict) else v for k, v in meta.items()}

    # For multiple files, index concurrently
    indexed = index_pdfs_concurrent(saved_paths, max_workers=max_workers, progress_callback=progress_callback)
    for coll, info in indexed.items():
        meta[coll] = info
    save_collections_metadata(meta)
    return {k: v.get('filename', '') if isinstance(v, dict) else v for k, v in meta.items()}


def reindex_collection(collection_name: str, upload_dir: Path) -> bool:
    """Force re-index an existing collection using the stored filename in metadata.

    Returns True on success, False otherwise.
    """
    meta = load_collections_metadata()
    info = meta.get(collection_name)
    if not info:
        return False
    filename = info.get('filename') if isinstance(info, dict) else str(info)
    target_path = upload_dir / filename
    if not target_path.exists():
        return False
    try:
        # Recreate collection with the same name to overwrite existing collection data
        collection_name_actual = create_vector_db_from_path(target_path, collection_name)
        # update checksum and timestamp
        try:
            with open(target_path, 'rb') as f:
                checksum = _compute_checksum_bytes(f.read())
        except Exception:
            checksum = ''
        meta[collection_name_actual] = {"filename": str(target_path.name), "checksum": checksum, "last_indexed": datetime.utcnow().isoformat()}
        save_collections_metadata(meta)
        return True
    except Exception:
        logger.exception(f"Failed to reindex collection: {collection_name}")
        return False


def get_collection_info(collection_name: str) -> Dict[str, str]:
    """Return metadata dict for a collection (filename, checksum, last_indexed)."""
    meta = load_collections_metadata()
    info = meta.get(collection_name)
    if isinstance(info, dict):
        return info
    return {"filename": str(info) if info else "", "checksum": "", "last_indexed": ""}


def delete_collection(collection_name: str) -> None:
    """Delete a named Chroma collection and remove it from metadata."""
    try:
        # Use chromadb client to delete the collection for reliability
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass
    except Exception:
        logger.exception(f"Failed to delete collection via chromadb client: {collection_name}")
    # Remove from metadata
    meta = load_collections_metadata()
    if collection_name in meta:
        meta.pop(collection_name, None)
        save_collections_metadata(meta)


def delete_all_collections() -> None:
    """Delete all Chroma collections listed in metadata and clear metadata."""
    meta = load_collections_metadata()
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        for coll in list(meta.keys()):
            try:
                client.delete_collection(name=coll)
            except Exception:
                logger.exception(f"Failed to delete collection via client: {coll}")
    except Exception:
        logger.exception("Failed to initialize chromadb client for delete_all_collections")
    save_collections_metadata({})


def list_collections() -> Dict[str, str]:
    """Return the metadata mapping of collection_name -> original filename."""
    meta = load_collections_metadata()
    return {k: v.get('filename', '') if isinstance(v, dict) else v for k, v in meta.items()}


def bootstrap_collections_from_uploads(upload_dir: Path) -> Dict[str, str]:
    """Scan upload_dir for PDF files and ensure each has a Chroma collection and metadata entry.

    Returns simplified mapping collection_name -> filename for UI.
    """
    meta = load_collections_metadata()
    changed = False
    if not upload_dir.exists():
        return {k: v.get('filename', '') if isinstance(v, dict) else v for k, v in meta.items()}

    for p in upload_dir.iterdir():
        if not p.is_file() or p.suffix.lower() != '.pdf':
            continue
        try:
            file_bytes = p.read_bytes()
            checksum = _compute_checksum_bytes(file_bytes)
        except Exception:
            checksum = ''

        # If checksum already recorded, ensure filename present
        found = None
        for coll, info in meta.items():
            if isinstance(info, dict) and info.get('checksum') and info.get('checksum') == checksum:
                found = coll
                break
        if found:
            if meta[found].get('filename') != p.name:
                meta[found]['filename'] = p.name
                meta[found]['last_indexed'] = datetime.utcnow().isoformat()
                changed = True
            continue

        # If metadata has filename match, try to reuse (compute disk checksum and store)
        coll_by_name = None
        for coll, info in meta.items():
            if isinstance(info, dict) and info.get('filename') == p.name:
                coll_by_name = coll
                break
        if coll_by_name:
            try:
                disk_checksum = _compute_checksum_bytes(p.read_bytes())
                meta[coll_by_name]['checksum'] = disk_checksum
                meta[coll_by_name]['last_indexed'] = datetime.utcnow().isoformat()
                changed = True
                continue
            except Exception:
                pass

        # Otherwise create a collection for this file
        try:
            coll_name = f"pdf_{checksum[:16]}" if checksum else None
            collection_name = create_vector_db_from_path(p, coll_name)
            meta[collection_name] = {"filename": str(p.name), "checksum": checksum, "last_indexed": datetime.utcnow().isoformat()}
            changed = True
        except Exception:
            logger.exception(f"Failed to bootstrap collection for {p}")
            continue

    if changed:
        save_collections_metadata(meta)

    return {k: v.get('filename', '') if isinstance(v, dict) else v for k, v in meta.items()}
