import abc
from argparse import Namespace
from typing import NamedTuple, Tuple, Union, Any, Optional

import bitsandbytes as bnb
import torch
from vllm import LLM

from src.modeling.model_wrappers import VLLMWrapper, HuggingFaceModelWrapper


class ModelLoader(abc.ABC):
    """This class has the responsibility of providing the functionality to load the model and its utilities, including tokenizer.

    Args:
        config (Namespace): The configuration object.
    """

    _model_wrappers = {
        'vllm': VLLMWrapper,
        'huggingface': HuggingFaceModelWrapper
    }
    
    def __init__(self, config: Namespace):
        self._config = config

    def _determine_model_type(self):
        if self._config.fp16:
            return torch.float16
        elif self._config.bf16:
            return torch.bfloat16
        else:
            return torch.float32

    def _determine_device(self):
        if self._config.device == 'cpu':
            return torch.device('cpu')
        elif self._config.device == 'cuda':
            assert torch.cuda.is_available(), 'CUDA device is not available'
            return torch.device('cuda')
        else:
            raise Exception(f"Unknown device: {self._config.device}")

    def _find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit if self._config.bits == 4 else \
            (bnb.nn.Linear8bitLt if self._config.bits == 8 else torch.nn.Linear)
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names: # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def _determine_flash_attention(self):
        if not self._config.use_flash_attention:
            return False

        flash_attention_compatible = \
            'cuda' in self._config.device and (self._config.fp16 or self._config.bf16)

        if not flash_attention_compatible:
            raise ValueError("Flash attention is only compatible with CUDA and FP16/BF16!")

        return True

    @abc.abstractmethod
    def load_model(self) -> Tuple[torch.nn.Module, dict]:
        pass

    def load_inference_model(self) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        model, utils = self._load_inference_model(self._config.inference_model_type)
        model = self._model_wrappers[self._config.inference_model_type](model, utils.get('tokenizer', None))
        return model, utils
    
    @abc.abstractmethod
    def _load_inference_model(self) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        pass