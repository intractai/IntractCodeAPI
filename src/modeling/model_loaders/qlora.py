import logging
import os
import threading
from typing import NamedTuple, Tuple, Union, Any, Optional

from huggingface_hub import snapshot_download
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

from src.modeling.model_loaders.base import ModelLoader


logger = logging.getLogger(__name__)


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
    
    def _load_inference_model(self) -> Tuple[torch.nn.Module, dict]:
        ValueError('VLLM is not supporting QLORA yet!')


