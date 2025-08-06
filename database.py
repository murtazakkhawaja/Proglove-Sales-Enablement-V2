import json
import os
import pickle
from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingDatabase:
    def __init__(self, db_path="embeddings_db"):
        self.db_path = db_path
        self.embeddings_file = os.path.join(db_path, "embeddings.pkl")
        self.metadata_file = os.path.join(db_path, "metadata.json")
        self.embeddings = []
        self.metadata = []
        self._load_database()

    def _load_database(self):
        """Load existing embeddings and metadata"""
        os.makedirs(self.db_path, exist_ok=True)

        if os.path.exists(self.embeddings_file):
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)

        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def _save_database(self):
        """Save embeddings and metadata"""
        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(self.embeddings, f)

        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def clear_database(self):
        """Remove all stored embeddings and metadata"""
        self.embeddings = []
        self.metadata = []
        self._save_database()
        print("Database cleared successfully!")

    def clear(self):
        """for clearing the database, used by chatbot"""
        self.clear_database()

    def add_pdf_embeddings(self, pdf_name: str, embeddings_data: List[Dict]):
        """Add embeddings for a new PDF"""
        self.remove_pdf_embeddings(pdf_name)

        for i, item in enumerate(embeddings_data):
            self.embeddings.append(item['embedding'])
            self.metadata.append({
                'pdf_name': pdf_name,
                'chunk_id': i,
                'text': item['text'],
                'embedding_index': len(self.embeddings) - 1
            })

        self._save_database()
        print(f"Added {len(embeddings_data)} embeddings for '{pdf_name}'")

    def remove_pdf_embeddings(self, pdf_name: str):
        """Remove all embeddings for a specific PDF"""
        indices_to_remove = [i for i, meta in enumerate(self.metadata) if meta['pdf_name'] == pdf_name]

        for i in reversed(indices_to_remove):
            del self.embeddings[i]
            del self.metadata[i]

        for i, meta in enumerate(self.metadata):
            meta['embedding_index'] = i

        self._save_database()
        print(f"Removed embeddings for '{pdf_name}'")

    def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """Search for similar chunks based on cosine similarity"""
        if not self.embeddings:
            return []

        embeddings_array = np.array(self.embeddings)
        query_array = np.array(query_embedding).reshape(1, -1)

        similarities = cosine_similarity(query_array, embeddings_array)[0]

        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= threshold:
                results.append({
                    'metadata': self.metadata[i],
                    'similarity': similarity,
                    'text': self.metadata[i]['text']
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def get_all_pdf_names(self) -> List[str]:
        """Return a list of all unique PDF names in the database"""
        return list(set(meta['pdf_name'] for meta in self.metadata))

    def get_total_chunks(self) -> int:
        """Return total number of stored chunks"""
        return len(self.embeddings)

    def get_database_stats(self) -> Dict:
        """Optional, can return stats as a single dictionary"""
        return {
            'total_chunks': self.get_total_chunks(),
            'total_pdfs': len(self.get_all_pdf_names()),
            'pdf_names': self.get_all_pdf_names()
        }
