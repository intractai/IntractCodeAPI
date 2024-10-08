import logging
import threading
from typing import List, Optional

from omegaconf import DictConfig
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.indices.base_retriever import BaseRetriever
# from llama_index.core.base_retriever import BaseRetriever
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
        embed_model = OpenAIEmbedding(
                model = self.config.embed_model,
                dimensions = self.config.get('embed_dim'),
                max_retries = 4,
            )

        transformations = [
            TokenTextSplitter(
                chunk_size = self.config.chunk_size,
                chunk_overlap = self.config.chunk_overlap,
            ),
            # TODO: Double check if this is needed
            # OpenAIEmbedding(
            #     model = self.config.embed_model,
            #     dimensions = self.config.get('embed_dim'),
            #     max_retries = 4,
            # ),
        ]

        return VectorStoreIndex(nodes=[], embed_model=embed_model, transformations=transformations)

    def get_vector_store(self, username: str) -> VectorStoreIndex:
        """Get the model and model utilities for the user."""
        with self._get_user_lock(username):
            if username not in self._vector_stores:
                self._vector_stores[username] = self.create_new_vector_store()
            return self._vector_stores[username]
    
    def add_documents(self, username: str, documents: List[str]):
        """Add documents to the RAG model of the given user."""
        vector_store = self.get_vector_store(username)
        self._add_documents_to_vs(vector_store, documents)

    def _add_documents_to_vs(self, vector_store: VectorStoreIndex, documents: List[str]):
        """Add documents to the vector store."""
        for doc in documents:
            vector_store.insert(Document(text=doc))

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


def retrieve_context(
        text: str,
        vector_store: Optional[VectorStoreIndex],
        retriever: Optional[BaseRetriever] = None,
        top_k: int = 1,
    ) -> List[str]:
    """Retrieve the context from the vector store.

    Args:
        text: The query text.
        vector_store: The vector store to retrieve the context from.
        top_k: The number of chunks to retrieve. Defaults to 1.

    Returns:
        str: The retrieved context.
    """
    assert vector_store is not None or retriever is not None, "Either vector_store or retriever must be provided."

    if not text:
        return None
    
    if retriever is None:
        retriever = vector_store.as_retriever(similarity_top_k=top_k)

    results = retriever.retrieve(text)
    return [result.node.get_content() for result in results]
