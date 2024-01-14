from typing import Tuple
from argparse import Namespace
import os
import abc
import logging
from pathlib import Path

import torch
import bitsandbytes as bnb
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, BitsAndBytesConfig
from datasets import load_dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import threading
from concurrent.futures import  CancelledError


logger = logging.getLogger(__name__)

# Create a global variable to store the model
GLOBAL_GENERATE_THREAD_ID = None
GLOBAL_MAIN_THREAD_ID = None


def thread_hook(*args):
    current_thread_id = threading.get_ident()
    if GLOBAL_GENERATE_THREAD_ID is not None \
        and current_thread_id != GLOBAL_GENERATE_THREAD_ID \
        and current_thread_id != GLOBAL_MAIN_THREAD_ID:
        raise CancelledError("Cancelled by new request")
    
    
class ModelLoader(abc.ABC):
    """This class has the responsibility of providing the functionality to load the model and its utilities, including tokenizer.

    Args:
        cfg (Namespace): The configuration object.
    """

    def __init__(self, cfg: Namespace):
        self._cfg = cfg

    def _determine_model_type(self):
        if self._cfg.fp16:
            return torch.float16
        elif self._cfg.bf16:
            return torch.bfloat16
        else:
            return torch.float32
    
    def _determine_device(self):
        if self._cfg.device == 'cpu':
            return torch.device("cpu")
        elif self._cfg.device == 'gpu':
            assert torch.cuda.is_available(), "GPU is not available"
            return torch.device("cuda")
        
    def _find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit if self._cfg.bits == 4 else \
            (bnb.nn.Linear8bitLt if self._cfg.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)
        
    def _determine_flash_attention(self):
        return self._cfg.device == 'gpu' and self._cfg.fp16

    @abc.abstractmethod
    def load_model(self) -> Tuple[torch.nn.Module, dict]:
        pass


class StandardModelLoader(ModelLoader):

    def load_model(self) -> Tuple[torch.nn.Module, dict]:

        global GLOBAL_MAIN_THREAD_ID

        model_dir = self._cfg.model_dir
        model_name = self._cfg.model_name

        device = self._determine_device()
        dtype = self._determine_model_type()
        use_flash_attention = self._determine_flash_attention()

        if use_flash_attention:
            logger.info('Using flash attention')

        # Ideally we should check all the files, but for now just check one
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isfile(os.path.join(model_dir, 'config.json')):
            snapshot_download(model_name, local_dir=model_dir, local_dir_use_symlinks=False,
                            ignore_patterns=['*.msgpack', '*.h5'])

        os.environ['TOKENIZERS_PARALLELISM'] = \
            os.environ.get('TOKENIZERS_PARALLELISM', 'true')
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_max_length = self._cfg.context_length

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, use_flash_attention_2=use_flash_attention,
            device_map=device, torch_dtype=dtype)
        model.to(device)

        for _, md in model.named_modules():
            md.register_forward_hook(thread_hook)

        GLOBAL_MAIN_THREAD_ID = threading.get_ident()

        return model, {'tokenizer': tokenizer}

class LoraModelLoader(ModelLoader):
    
    def load_model(self) -> Tuple[torch.nn.Module, dict]:

        global GLOBAL_MAIN_THREAD_ID

        model_dir = self._cfg.model_dir
        model_name = self._cfg.model_name

        device = self._determine_device()
        dtype = self._determine_model_type()
        use_flash_attention = self._determine_flash_attention()

        if use_flash_attention:
            logger.info('Using flash attention')

        # Ideally we should check all the files, but for now just check one
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isfile(os.path.join(model_dir, 'config.json')):
            snapshot_download(model_name, local_dir=model_dir, local_dir_use_symlinks=False,
                            ignore_patterns=['*.msgpack', '*.h5'])

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_max_length = self._cfg.context_length

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, use_flash_attention_2=use_flash_attention,
            device_map=device, torch_dtype=dtype
            )

        modules = self._find_all_linear_names(model)
        logger.info('Modules to be fine-tuned: ', modules)
        config = LoraConfig(
                r=self._cfg.lora_r,
                lora_alpha=self._cfg.lora_alpha,
                target_modules=modules,
                lora_dropout=self._cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)
        model.to(device)

        for _, md in model.named_modules():
            md.register_forward_hook(thread_hook)

        GLOBAL_MAIN_THREAD_ID = threading.get_ident()

        return model, {'tokenizer': tokenizer}

class QLoraModelLoader(ModelLoader):

    def load_model(self) -> Tuple[torch.nn.Module, dict]:

        global GLOBAL_MAIN_THREAD_ID
            
        model_dir = self._cfg.model_dir
        model_name = self._cfg.model_name

        device = self._determine_device()
        dtype = self._determine_model_type()
        use_flash_attention = self._determine_flash_attention()

        if use_flash_attention:
            logger.info('Using flash attention')

        # Ideally we should check all the files, but for now just check one
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isfile(os.path.join(model_dir, 'config.json')):
            snapshot_download(model_name, local_dir=model_dir, local_dir_use_symlinks=False,
                            ignore_patterns=['*.msgpack', '*.h5'])

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_max_length = self._cfg.context_length

        bb_config = BitsAndBytesConfig(
                load_in_4bit=self._cfg.bits == 4,
                load_in_8bit=self._cfg.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=self._cfg.double_quant,
                bnb_4bit_quant_type=self._cfg.quant_type,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, use_flash_attention_2=use_flash_attention,
            # device_map=device, # it weirdly crashes if I add the device for QLORA
            torch_dtype=dtype, use_cache=False,
            load_in_4bit=self._cfg.bits == 4,
            load_in_8bit=self._cfg.bits == 8,
            config=bb_config
            )
        
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=self._cfg.gradient_checkpointing)

        modules = self._find_all_linear_names(model)
        logger.info('Modules to be fine-tuned: ', modules)
        config = LoraConfig(
                r=self._cfg.lora_r,
                lora_alpha=self._cfg.lora_alpha,
                target_modules=modules,
                lora_dropout=self._cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                )
        model = get_peft_model(model, config)
        model.to(device)

        GLOBAL_MAIN_THREAD_ID = threading.get_ident()

        return model, {'tokenizer': tokenizer}


class ModelProvider:
    _instance = None
    _lock = threading.Lock()
    _model_loaders = {
        'standard': StandardModelLoader,
        'lora': LoraModelLoader,
        'qlora': QLoraModelLoader
    }

    @classmethod
    def get_instance(cls, cfg: dict = None):
        # First check for the singleton instance existence without acquiring the lock
        if cls._instance is None:
            # Acquire the lock and check again to ensure no other thread created the instance
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self, cfg: dict):
        if ModelProvider._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelProvider._instance = self
            self._model, self._model_utils = self._load_model(cfg)
        
    def _load_model(self, cfg: dict):
        model_type = cfg.get('model_type', 'standard')
        model_loader_class = self._model_loaders.get(model_type)
        if model_loader_class is None:
            raise ValueError(f"Model type '{model_type}' is not supported.")
        model_loader = model_loader_class(cfg)
        return model_loader.load_model()

    @classmethod
    def register_model_loader(cls, key: str, loader_class):
        cls._model_loaders[key] = loader_class

    def get_model(self):
        return self._model
    
    def get_model_utils(self):
        return self._model_utils
    
    def update_model(self, model: torch.nn.Module):
        with self._lock:  
            self._model = model


def get_model():
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model()


def get_model_utils():
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model_utils()