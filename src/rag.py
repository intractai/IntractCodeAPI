import logging
import threading
from typing import List

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding


logger = logging.getLogger(__name__)


class VectorStoreProvider:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, config: dict = None):
        # First check for the singleton instance existence without acquiring the lock
        if cls._instance is None:
            # Acquire the lock and check again to ensure no other thread created the instance
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    def __init__(self, config: dict):
        if VectorStoreProvider._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            VectorStoreProvider._instance = self

        # Initialize the model here
        self.config = config
        self._vector_stores: dict[str, VectorStoreIndex] = {}
        self._user_locks = {} # Locks for each user's model

    def _get_user_lock(self, username: str):
        """Get the lock for the user's model."""
        if username not in self._user_locks:
            with VectorStoreProvider._lock:
                if username not in self._user_locks:
                    self._user_locks[username] = threading.Lock()
        return self._user_locks[username]

    def create_new_vector_store(self):
        """Create a new model and model utilities."""
        transformations = [
            TokenTextSplitter(
                chunk_size = self.config.chunk_size,
                chunk_overlap = self.config.chunk_overlap,
            ),
            # TODO: Double check if this is needed
            OpenAIEmbedding(
                model = self.config.embed_model,
                dimensions = self.config.get('embed_dim'),
                max_retries = 4,
            ),
        ]

        return VectorStoreIndex(embed_model=self.config.embed_model, transformations=transformations)

    def get_vector_store(self, username: str) -> VectorStoreIndex:
        """Get the model and model utilities for the user."""
        with self._get_user_lock(username):
            if username not in self._vector_stores:
                self._vector_stores[username] = self.create_new_vector_store()
            return self._vector_stores[username]
    
    def add_documents(self, username: str, documents: List[str]):
        """Add documents to the RAG model of the given user."""
        vector_store = self.get_vector_store(username)
        documents = [Document(text=doc) for doc in documents]
        vector_store.update(documents)

    def delete_vector_store(self, username: str):
        """Delete the vector store for the user."""
        with self._get_user_lock(username):
            if username in self._vector_stores:
                del self._vector_stores[username]
            else:
                logger.warning(f"Tried to delete vector store for user {username}, but it does not exist.")
        
        with VectorStoreProvider._lock:
            if username in self._user_locks:
                del self._user_locks[username]
            else:
                logger.warning(f"Tried to delete lock for user {username}, but it does not exist.")


def get_vector_store(username: str):
    vs_provider = VectorStoreProvider.get_instance()
    return vs_provider.get_vector_store(username)
