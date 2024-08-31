import copy
import logging
import os
import threading
import types
from typing import NamedTuple, Tuple, Union, Any, Optional

from huggingface_hub import snapshot_download
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM

from src.modeling.model_loaders.base import ModelLoader


logger = logging.getLogger(__name__)


def make_synchronous_func(tokenizer, lock, source_func):
    def new_func(*args, **kwargs):
        with lock:
            return source_func(*args, **kwargs)
    return new_func


class SynchronizedTokenizer():
    """
    A synchronized tokenizer class that creates partially synchronous tokenizers
    to prevent tokenizer concurrency issues.
    This could probably be made more efficient by allowing multiple encode / decodes
    at the same time, but just making sure that writes are synchronous with encodes /
    decodes. For now, the this seems efficient enough.
    """
        
    def from_tokenizer(tokenizer):
        tokenizer._lock = threading.RLock()

        # Take the functions of this class and copy them over to the tokenizer
        # This allows us to copy the tokenizer
        for func in ('__getstate__', '__setstate__', '__deepcopy__'):
            source_func = getattr(SynchronizedTokenizer, func)
            setattr(tokenizer, func, types.MethodType(source_func, tokenizer))

        # Make these function synchronous with delete and set var functions
        for func in ('__call__', 'encode', 'decode'):
            source_func = getattr(tokenizer, func)
            setattr(tokenizer, func, make_synchronous_func(tokenizer, tokenizer._lock, source_func))
        
        for func in ('__setattr__', '__delattr__'):
            source_func = getattr(tokenizer, func)
            setattr(tokenizer, func, make_synchronous_func(tokenizer, tokenizer._lock, source_func))
    
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove the lock from the state because it cannot be pickled.
        del state['_lock']
        return state

    def __setstate__(self, state):
        # Reinitialize the lock after unpickling.
        state['_lock'] = threading.RLock()
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        # Customize the deepcopy behavior to handle the lock.
        cls = self.__class__
        # Create a new instance without calling __init__ (to avoid creating another lock initially).
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_lock':
                # Initialize a new lock for the copied object.
                setattr(result, k, threading.RLock())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result




class StandardModelLoader(ModelLoader):

    def _load_vllm_model(self):
        model_name = self._config.model_name

        model = LLM(
                model=model_name,
                gpu_memory_utilization=0.4,
                enable_lora=False
            )
        
        return model, {}
    
    def _load_huggingface_model(self, device=None):
        global GLOBAL_MAIN_THREAD_ID

        model_dir = self._config.model_dir
        model_name = self._config.model_name

        device = device or self._determine_device()
        dtype = self._determine_model_type()
        use_flash_attention = self._determine_flash_attention()

        if use_flash_attention:
            attn_implementation = 'flash_attention_2'
            logger.info('Using flash attention')
        else:
            attn_implementation = None

        # Ideally we should check all the files, but for now just check one
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isfile(os.path.join(model_dir, 'config.json')):
            snapshot_download(model_name, local_dir=model_dir, local_dir_use_symlinks=False,
                            ignore_patterns=['*.msgpack', '*.h5'])

        os.environ['TOKENIZERS_PARALLELISM'] = \
            os.environ.get('TOKENIZERS_PARALLELISM', 'true')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_max_length = self._config.context_length
        tokenizer.truncation_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, attn_implementation=attn_implementation,
            device_map=device, torch_dtype=dtype)
        model.to(device)

        GLOBAL_MAIN_THREAD_ID = threading.get_ident()

        return model, {'tokenizer': SynchronizedTokenizer.from_tokenizer(tokenizer)}

    def load_model(self, device=None) -> Tuple[torch.nn.Module, dict]:
        return self._load_huggingface_model(device)

    def _load_inference_model(self, model_type: str) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        if model_type == 'huggingface':
            return self._load_huggingface_model()
        elif model_type == 'vllm':
            return self._load_vllm_model()
        else:
            ValueError(f'This model type is not supported: {model_type}')