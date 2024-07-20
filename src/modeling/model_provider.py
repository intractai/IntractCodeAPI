import abc
from argparse import Namespace
from concurrent.futures import CancelledError
import copy
from functools import partial
import logging
import os
import threading
from typing import NamedTuple

import torch

from src.modeling.model_loaders import StandardModelLoader, LoraModelLoader, QLoraModelLoader
from src.session import thread_hook

logger = logging.getLogger(__name__)


# Named tuple for model and model utilities
ModelTuple = NamedTuple(
    'ModelTuple',
    [('model', torch.nn.Module), ('model_utils', dict)]
)

class ModelProvider:
    _instance = None
    _lock = threading.Lock()
    _model_loaders = {
        'standard': StandardModelLoader,
        'lora': LoraModelLoader,
        'qlora': QLoraModelLoader
    }

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
        if ModelProvider._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelProvider._instance = self

        # Initialize the model here
        self.config = config
        self._models: dict[str, ModelTuple] = {}
        self._inference_models: dict[str, ModelTuple] = {}
        self._user_locks = {} # Locks for each user's model

        # New models are cloned from the base model, which is stored on the CPU
        # This drastically reduces load times for newly active users
        model_loader = ModelProvider._model_loaders[config.model_type](config)
        self._target_device = model_loader._determine_device()
        model, model_utils = model_loader.load_model(device=self._target_device)
        inf_model, inf_model_utils = model_loader.load_inference_model() #NOTE: VLLM only works in GPU mode for now
        self._base_model_tuple = ModelTuple(model, model_utils)
        self._base_inf_model_tuple = ModelTuple(inf_model, inf_model_utils)

    def _get_user_lock(self, username: str):
        """Get the lock for the user's model."""
        if username not in self._user_locks:
            with ModelProvider._lock:
                if username not in self._user_locks:
                    self._user_locks[username] = threading.Lock()
        return self._user_locks[username]

    def create_new_model_tuple(self) -> ModelTuple:
        """Create a new model tuple from the base model."""
        model = copy.deepcopy(self._base_model_tuple.model)
        model = model.to(self._target_device)
        # We also want to copy the tokenizer because HuggingFace tokenizer may
        # throw an error if you try to use them concurrently
        # Previously was having this problem:
        # https://github.com/huggingface/tokenizers/issues/537
        model_utils = copy.deepcopy(self._base_model_tuple.model_utils)
        return ModelTuple(model, model_utils)
    
    def create_new_inference_model_tuple(self) -> ModelTuple:
        """Create a new model tuple from the base model."""
        # model = copy.deepcopy(self._base_inf_model_tuple.model)
        model = self._base_inf_model_tuple.model
        # model = model.to(self._target_device)
        # We also want to copy the tokenizer because HuggingFace tokenizer may
        # throw an error if you try to use them concurrently
        # Previously was having this problem:
        # https://github.com/huggingface/tokenizers/issues/537
        # model_utils = copy.deepcopy(self._base_inf_model_tuple.model_utils)
        model_utils = self._base_inf_model_tuple.model_utils
        return ModelTuple(model, model_utils)
    
    def _register_preemption_hooks(self, model: torch.nn.Module, username: str):
        """Register hooks that preempt the model if a newer request is made.
        
        This allows us to cancel the current request if a new one is made,
        saving resources and preventing the server from being overwhelmed for
        requests that are no longer needed.
        
        Args:
            model (torch.Module): The model to register the hooks on.
            username (str): The username of the user.
        """
        for _, md in model.named_modules():
            md.register_forward_hook(
                partial(thread_hook, username))

    def get_model_tuple(self, username: str) -> ModelTuple:
        """Get the model and model utilities for the user."""
        with self._get_user_lock(username):
            if username not in self._models:
                self._models[username] = self.create_new_inference_model_tuple()
                model = self._models[username].model
                self._register_preemption_hooks(model, username)
            return self._models[username]
        
    def get_inference_model_tuple(self, username: str) -> ModelTuple: 
        """Get the inference model and model utilities for the user."""
        # TODO: think about how to use a single inference model in the case of multiple users
        with self._get_user_lock(username): # TODO: the inference and training lock should be different
            if username not in self._inference_models:
                self._inference_models[username] = self.create_new_inference_model_tuple()
                # TODO: figure out forward hook is required for vllm like models
                # model = self._inf_models[username].model
                # self._register_preemption_hooks(model, username) 

            return self._inference_models[username]
        
    def get_inference_model(self, username: str):
        return self.get_inference_model_tuple(username).model

    def get_model(self, username: str):
        return self.get_model_tuple(username).model

    def get_model_utils(self, username: str):
        return self.get_model_tuple(username).model_utils

    def update_model(self, username: str, model: torch.nn.Module):
        """Update the model for the user."""
        with self._get_user_lock(username):
            if username in self._models:
                self._models[username] = ModelTuple(model, self._models[username].model_utils)
            else:
                raise ValueError(f"Model for user {username} does not exist.")

    def delete_model(self, username: str):
        """Delete the model for the user."""
        with self._get_user_lock(username):
            if username in self._models:
                del self._models[username]
            else:
                logger.warning(f"Tried to delete model for user {username}, but it does not exist.")
        
        torch.cuda.empty_cache()

        with ModelProvider._lock:
            if username in self._user_locks:
                del self._user_locks[username]
            else:
                logger.warning(f"Tried to delete lock for user {username}, but it does not exist.")


