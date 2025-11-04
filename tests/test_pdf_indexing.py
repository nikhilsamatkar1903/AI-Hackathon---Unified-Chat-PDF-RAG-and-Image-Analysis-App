import tempfile
from pathlib import Path
import json
import os
import pytest

from components import pdf_indexing


def test_create_and_load_collection(monkeypatch, tmp_path):
    # Monkeypatch PERSIST_DIRECTORY and METADATA_PATH to a temp dir
    monkeypatch.setattr(pdf_indexing, 'PERSIST_DIRECTORY', str(tmp_path / 'vectors'))
    monkeypatch.setattr(pdf_indexing, 'METADATA_PATH', tmp_path / 'meta.json')

    # Create a small fake PDF file (we'll just write text and rely on UnstructuredPDFLoader to fail gracefully)
    pdf_path = tmp_path / 'test.pdf'
    pdf_path.write_bytes(b'%PDF-1.4\n%Fake PDF content\n')

    # Monkeypatch UnstructuredPDFLoader to return a fake document list
    class FakeDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {'source': str(pdf_path.name)}

    def fake_loader(path):
        class Loader:
            def __init__(self, path):
                pass
            def load(self):
                return [FakeDoc('Hello world')]
        return Loader(path)

    monkeypatch.setattr(pdf_indexing, 'UnstructuredPDFLoader', fake_loader)

    # Monkeypatch OllamaEmbeddings to a lightweight dummy that returns predictable vectors
    class DummyEmbeds:
        def __init__(self, model=None):
            pass
        def embed_documents(self, texts):
            # return list of lists
            return [[float(len(t))] for t in texts]
        def embed_query(self, text):
            return [float(len(text))]

    monkeypatch.setattr(pdf_indexing, 'OllamaEmbeddings', DummyEmbeds)

    # Monkeypatch Chroma.from_documents and Chroma to avoid heavy dependencies
    class DummyChroma:
        _storage = {}
        def __init__(self, persist_directory=None, collection_name=None, embedding_function=None):
            self.persist_directory = persist_directory
            self.collection_name = collection_name
            self.embedding_function = embedding_function
            DummyChroma._storage.setdefault(collection_name, {'docs': []})
        @classmethod
        def from_documents(cls, documents, embedding, persist_directory, collection_name):
            cls._storage[collection_name] = {'docs': [getattr(d, 'page_content', str(d)) for d in documents]}
            return cls(persist_directory=persist_directory, collection_name=collection_name, embedding_function=embedding)
        def similarity_search(self, query, k=3):
            # return fake docs as objects with page_content and metadata
            docs = []
            for content in DummyChroma._storage.get(self.collection_name, {}).get('docs', []):
                d = type('D', (), {})()
                d.page_content = content
                d.metadata = {'source': 'test.pdf'}
                docs.append(d)
            return docs[:k]
        def delete_collection(self):
            DummyChroma._storage.pop(self.collection_name, None)

    monkeypatch.setattr(pdf_indexing, 'Chroma', DummyChroma)

    # Now run the handle_uploaded_files path using a fake uploaded file object
    class FakeUpload:
        def __init__(self, name, content):
            self.name = name
            self._content = content
        def getvalue(self):
            return self._content

    upload = FakeUpload('test.pdf', b'%PDF-1.4...')
    meta = pdf_indexing.handle_uploaded_files([upload], tmp_path)
    assert isinstance(meta, dict)
    assert len(meta) == 1

    # Ensure metadata written to METADATA_PATH
    saved = json.loads((tmp_path / 'meta.json').read_text())
    assert len(saved) == 1

    # Test list_collections
    listed = pdf_indexing.list_collections()
    assert isinstance(listed, dict)
    assert len(listed) == 1

    # Test load_chroma_collection and similarity_search
    coll_name = list(saved.keys())[0]
    chroma = pdf_indexing.load_chroma_collection(coll_name)
    results = chroma.similarity_search('hello', k=2)
    assert results

    # Test delete_collection
    pdf_indexing.delete_collection(coll_name)
    listed_after = pdf_indexing.list_collections()
    assert coll_name not in listed_after


def test_delete_all_collections(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_indexing, 'PERSIST_DIRECTORY', str(tmp_path / 'vectors'))
    monkeypatch.setattr(pdf_indexing, 'METADATA_PATH', tmp_path / 'meta.json')

    # Prepare fake metadata
    meta = {'coll_a': 'a.pdf', 'coll_b': 'b.pdf'}
    (tmp_path / 'meta.json').write_text(json.dumps(meta))

    # Monkeypatch Chroma to record deletions
    class DummyChroma2:
        def __init__(self, persist_directory=None, collection_name=None, embedding_function=None):
            self.collection_name = collection_name
        def delete_collection(self):
            # Just pass
            pass

    monkeypatch.setattr(pdf_indexing, 'Chroma', DummyChroma2)

    pdf_indexing.delete_all_collections()
    after = pdf_indexing.list_collections()
    assert after == {}
