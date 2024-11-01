"""Basic vector database class."""

import numpy as np


class VectorStore:
    """A simple class to store and retrieve vectors."""

    def __init__(self):
        """Init method for VectorStore."""
        self.vector_data = {}

    def add_vector(self, text: str, vector: np.ndarray):
        """Add a element to the store.

        Args:
            text (str): input text to be stored.
            vector (np.ndarray): vector data representation

        """
        self.vector_data[text] = vector

    def find_similar_vectors(self, query_vector: np.ndarray, num_results: int = 1):
        """Find similar sentences.

        Args:
            query_vector (numpy.ndarray): query vector for similarity search
            num_results (int): number of similar vectors to return

        Returns:
            list: A list of (vector_id, similarity_score) tuples for the most similar
                vectors.

        """
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            results.append((vector_id, similarity))

        # top N results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]
