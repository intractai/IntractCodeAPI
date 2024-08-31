import abc
import torch
from typing import Union, Optional
from vllm import LLM
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, BatchEncoding
from llama_index.core import VectorStoreIndex

from src.training.data_formatting import prepare_input
from src.types import GenerateData



class ModelWrapper(abc.ABC):

    def __init__(
            self, 
            model: Union[torch.nn.Module, LLM], 
            tokenizer: PreTrainedTokenizer = None,
            **kwargs
        ):
        self._wrapped_model = model
        self._wrapped_tokenizer = tokenizer
    
    def __getattr__(self, name):
        # Delegate attribute access to the wrapped instance
        return getattr(self._wrapped_model, name)
    
    def __setattr__(self, name, value):
        if name == '_wrapped_model' or name == '_wrapped_tokenizer':
            super().__setattr__(name, value)
        else:
            setattr(self._wrapped_model, name, value)
    
    def __delattr__(self, name):
        delattr(self._wrapped_model, name)

    @abc.abstractmethod
    def generate_completion(
            self, 
            item: GenerateData,
            config: DictConfig,
            vector_store: Optional[VectorStoreIndex] = None,
        ) -> dict:
        pass

    def _prepare_input(
            self, 
            item: GenerateData,
            config: DictConfig,
            tokenizer: PreTrainedTokenizer,
            vector_store: Optional[VectorStoreIndex] = None,
        ) -> BatchEncoding:
        return prepare_input(item, config, tokenizer, vector_store)