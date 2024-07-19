import abc
from argparse import Namespace
from concurrent.futures import CancelledError
import copy
from functools import partial
import logging
import os
import threading
import types
from typing import NamedTuple, Tuple, Union, Any, Optional

from vllm import LLM, SamplingParams
import bitsandbytes as bnb
from huggingface_hub import snapshot_download
from llama_index.core import VectorStoreIndex
from omegaconf import DictConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer
from transformers import BitsAndBytesConfig, BatchEncoding, GenerationConfig

from src.routers.generator import prepare_input, GenerateData
from src.users import SessionTracker

logger = logging.getLogger(__name__)

# Create a global variable to store the main thread ID
GLOBAL_MAIN_THREAD_ID = None

EOT_TOKEN = '<|EOT|>'
FIM_BEGIN_TOKEN = '<｜fim▁begin｜>' # 32016
FIM_HOLE_TOKEN = '<｜fim▁hole｜>' # 32015
FIM_END_TOKEN = '<｜fim▁end｜>' # 32017


def thread_hook(username, *args):
    current_thread_id = threading.get_ident()

    session_tracker = SessionTracker.get_instance()
    activity_threads = session_tracker.get_user_activity_threads(username)

    if current_thread_id not in activity_threads:
        raise CancelledError("Cancelled by new request")


def set_main_thread_id():
    global GLOBAL_MAIN_THREAD_ID
    GLOBAL_MAIN_THREAD_ID = threading.get_ident()


class ModelWrapper(abc.ABC):

    def __init__(self, model: Union[torch.nn.Module, LLM]):
        self._wrapped_model = model
        self._tokenizer = None
    
    def __getattr__(self, name):
        # Delegate attribute access to the wrapped instance
        return getattr(self._wrapped_model, name)
    
    def __setattr__(self, name, value):
        if name == '_wrapped_model':
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
            tokenizer: PreTrainedTokenizer,
            config: DictConfig,
            vector_store: Optional[VectorStoreIndex] = None,
        ) -> BatchEncoding:
        return prepare_input(item, config, tokenizer, vector_store)


class VLLMWrapper(ModelWrapper):

    _sampling_params = SamplingParams(temperature=0.75, top_k=5)
    
    def generate_completion(
            self, 
            item: GenerateData,
            config: DictConfig,
            vector_store: Optional[VectorStoreIndex] = None,
        ) -> dict:

        input_ids = self._prepare_input(item, config, self.get_tokenizer(), vector_store).input_ids.tolist()

        outputs = self.generate(prompt_token_ids=input_ids, sampling_params=self._sampling_params)

        output_text = outputs[0].outputs[0].text

        perplexity = 0 #TODO: calculate perplexity later
        return {
            'outputs': outputs,
            'output_text': output_text,
            'perplexity': perplexity,
        } 
    

class HuggingFaceModelWrapper(ModelWrapper):

    GENERATION_KWARGS = dict(
        temperature=0.7, do_sample=True, top_k=5,
        # num_beams=3, early_stopping=True, 
        # do_sample=True, temperature=1.1, top_k=3,
    )

    _generation_config = GenerationConfig(
        temperature=0.7, top_k=5, do_sample=True,
    )

    def generate_completion(
            self, 
            item: GenerateData,
            config: DictConfig,
            vector_store: Optional[VectorStoreIndex] = None,
        ) -> dict:

        tokenizer = get_tokenizer()

        inputs = self._prepare_input(item, config, tokenizer, vector_store).to(self.device)

        outputs = self.generate(
            **inputs, max_new_tokens=config.inference.max_gen_length,
            return_dict_in_generate=True, output_scores=True,
            **self.GENERATION_KWARGS,
        )

        out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
        logits = torch.stack(outputs.scores[-len(out_tokens):]).squeeze(1)
        probs = logits.softmax(dim=1).gather(1, out_tokens.unsqueeze(1))

        perplexity = torch.exp(-torch.sum(torch.log(probs)) / len(out_tokens))

        return {
            'outputs': outputs,
            'output_text': output_text,
            'perplexity': perplexity,
        }


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
        model, utils = self._load_inference_model()
        model = self._model_wrappers[self._config.inference_model_type](model)
        return model, utils
    
    @abc.abstractmethod
    def _load_inference_model(self) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        pass
       

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

    def load_model(self, device=None) -> Tuple[torch.nn.Module, dict]:

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
    
    def _load_inference_model(self) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        model_name = self._config.model_name

        model = LLM(
                model=model_name,
                gpu_memory_utilization=0.4,
                enable_lora=False
            )
        
        return model, {}


class LoraModelLoader(ModelLoader):

    def load_model(self) -> Tuple[torch.nn.Module, dict]:

        global GLOBAL_MAIN_THREAD_ID

        model_dir = self._config.model_dir
        model_name = self._config.model_name

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
        tokenizer.model_max_length = self._config.context_length
        tokenizer.truncation_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, use_flash_attention_2=use_flash_attention,
            device_map=device, torch_dtype=dtype
            )

        modules = self._find_all_linear_names(model)
        logger.info(f"Modules to be fine-tuned: {modules}")
        config = LoraConfig(
                r=self._config.lora_r,
                lora_alpha=self._config.lora_alpha,
                target_modules=modules,
                lora_dropout=self._config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        model = get_peft_model(model, config)
        model.to(device)

        for _, md in model.named_modules():
            md.register_forward_hook(thread_hook)

        GLOBAL_MAIN_THREAD_ID = threading.get_ident()

        return model, {'tokenizer': tokenizer}
    
    def _load_inference_model(self) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        model_name = self._config.model_name

        model = LLM(
                model=model_name,
                gpu_memory_utilization=0.4,
                enable_lora=True
            )
        
        return model, {}


class QLoraModelLoader(ModelLoader):

    def load_model(self) -> Tuple[torch.nn.Module, dict]:

        global GLOBAL_MAIN_THREAD_ID

        model_dir = self._config.model_dir
        model_name = self._config.model_name

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
        tokenizer.model_max_length = self._config.context_length
        tokenizer.truncation_side = 'left'

        bb_config = BitsAndBytesConfig(
                load_in_4bit=self._config.bits == 4,
                load_in_8bit=self._config.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_use_double_quant=self._config.double_quant,
                bnb_4bit_quant_type=self._config.quant_type,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, use_flash_attention_2=use_flash_attention,
            # device_map=device, # it weirdly crashes if I add the device for QLORA
            torch_dtype=dtype, use_cache=False,
            load_in_4bit=self._config.bits == 4,
            load_in_8bit=self._config.bits == 8,
            config=bb_config
            )

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=self._config.gradient_checkpointing)

        modules = self._find_all_linear_names(model)
        logger.info(f"Modules to be fine-tuned: {modules}")
        config = LoraConfig(
                r=self._config.lora_r,
                lora_alpha=self._config.lora_alpha,
                target_modules=modules,
                lora_dropout=self._config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                )
        model = get_peft_model(model, config)
        model.to(device)

        GLOBAL_MAIN_THREAD_ID = threading.get_ident()

        return model, {'tokenizer': tokenizer}
    
    def _load_inference_model(self) -> Tuple[Union[LLM, torch.nn.Module], dict]:
        ValueError('VLLM is not supporting QLORA yet!')


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
                self._models[username] = self.create_new_model_tuple()
                model = self._models[username].model
                self._register_preemption_hooks(model, username)
            return self._models[username]
        
    def get_inference_model_tuple(self, username: str) -> ModelTuple: 
        """Get the inference model and model utilities for the user."""
        # TODO: think about how to use a single inference model in the case of multiple users
        with self._get_user_lock(username): # TODO: the inference and training lock should be different
            if username not in self._inference_models:
                self._inference_models[username] = self.create_new_model_tuple()
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


def get_model(username: str):
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model(username)


def get_model_utils(username: str):
    model_provider = ModelProvider.get_instance()
    return model_provider.get_model_utils(username)


def get_tokenizer(username: str):
    model_utils = get_model_utils(username)
    return model_utils['tokenizer']


def get_inference_model(username: str):
    model_provider = ModelProvider.get_instance()
    return model_provider.get_inference_model(username)
