import abc
from argparse import Namespace
from concurrent.futures import CancelledError
import logging
import os
import threading
from typing import Tuple

import bitsandbytes as bnb
from huggingface_hub import snapshot_download
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


logger = logging.getLogger(__name__)

# Create a global variable to store the model
GLOBAL_GENERATE_THREAD_ID = None
GLOBAL_FINETUNE_THREAD_ID = None
GLOBAL_MAIN_THREAD_ID = None

EOT_TOKEN = "<|EOT|>"
FIM_BEGIN_TOKEN = "<｜fim▁begin｜>" # 32016
FIM_HOLE_TOKEN = "<｜fim▁hole｜>" # 32015
FIM_END_TOKEN = "<｜fim▁end｜>" # 32017


def thread_hook(*args):
    current_thread_id = threading.get_ident()
    if GLOBAL_GENERATE_THREAD_ID is not None \
        and current_thread_id != GLOBAL_GENERATE_THREAD_ID \
        and current_thread_id != GLOBAL_FINETUNE_THREAD_ID \
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
            return torch.device('cpu')
        elif self._cfg.device == 'cuda':
            assert torch.cuda.is_available(), 'CUDA device is not available'
            return torch.device('cuda')
        else:
            raise Exception(f"Unknown device: {self._cfg.device}")

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
        if not self._cfg.use_flash_attention:
            return False

        flash_attention_compatible = \
            'cuda' in self._cfg.device and (self._cfg.fp16 or self._cfg.bf16)

        if not flash_attention_compatible:
            raise ValueError("Flash attention is only compatible with CUDA and FP16/BF16!")

        return True

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
        tokenizer.model_max_length = self._cfg.context_length
        tokenizer.truncation_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, attn_implementation=attn_implementation,
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
            logger.info("Using flash attention")

        # Ideally we should check all the files, but for now just check one
        model_dir = os.path.join(model_dir, model_name)
        if not os.path.isfile(os.path.join(model_dir, 'config.json')):
            snapshot_download(model_name, local_dir=model_dir, local_dir_use_symlinks=False,
                            ignore_patterns=['*.msgpack', '*.h5'])

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        tokenizer.model_max_length = self._cfg.context_length
        tokenizer.truncation_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, use_flash_attention_2=use_flash_attention,
            device_map=device, torch_dtype=dtype
            )

        modules = self._find_all_linear_names(model)
        logger.info(f"Modules to be fine-tuned: {modules}")
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
        tokenizer.truncation_side = 'left'

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

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=self._cfg.gradient_checkpointing)

        modules = self._find_all_linear_names(model)
        logger.info(f"Modules to be fine-tuned: {modules}")
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
        # Initialize the model here
        self._model, self._model_utils = \
            self._model_loaders[cfg.model_type](cfg).load_model()

    def get_model(self):
        return self._model

    def get_model_utils(self):
        return self._model_utils

    def update_model(self, model: torch.nn.Module):
        with self._lock:  # Acquire the lock for thread safety
            self._model = model

def get_model():
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model()


def get_model_utils():
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model_utils()