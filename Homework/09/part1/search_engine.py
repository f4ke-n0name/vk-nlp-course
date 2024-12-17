from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer

@dataclass
class Document:
   id: str
   title: str
   text: str
   embedding: Optional[np.ndarray] = None

@dataclass
class SearchResult:
   doc_id: str
   score: float
   title: str
   text: str

def load_documents(path: str) -> List[Document]:
   """Загрузка документов из json файла"""
   with open(path, 'r', encoding='utf-8') as f:
       data = json.load(f)
   return [
       Document(
           id=article['id'],
           title=article['title'],
           text=article['text'],
           embedding=None
       )
       for article in data['articles']
   ]

class Indexer:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)
        combined_texts = [doc.title + " " + doc.text for doc in documents]
        self.embeddings = self.model.encode(combined_texts, convert_to_numpy=True)
        for doc, embedding in zip(documents, self.embeddings):
            doc.embedding = embedding

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump((self.documents, self.embeddings), f)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.documents, self.embeddings = pickle.load(f)

class Searcher:
    def __init__(self, index_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.load(index_path)

    def load(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.documents, self.embeddings = pickle.load(f)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        similarities = np.dot(self.embeddings, query_embedding) / (np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding))
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        for idx in top_k_indices:
            results.append(SearchResult(
                doc_id=self.documents[idx].id,
                score=float(similarities[idx]),
                title=self.documents[idx].title,
                text=self.documents[idx].text
            ))
        return results
