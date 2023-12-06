import os
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

# Create a global variable to store the model
GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None
GLOBAL_GENERATE_THREAD_ID = None

def _find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def thread_hook(*args):
    current_thread_id = threading.get_ident()
    if GLOBAL_GENERATE_THREAD_ID is not None and current_thread_id != GLOBAL_GENERATE_THREAD_ID:
        raise CancelledError("Cancelled by new request")

def intialize_model(model_name, local_dir, args):
    """Download a model from the HuggingFace Hub and save it to the local directory."""

    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.fp16 else torch.float32
    use_flash_attention = device == "cuda" and args.fp16

    # Ideally we should check all the files, but for now just check one
    if not os.path.isfile(os.path.join(local_dir, 'config.json')):
        snapshot_download(model_name, local_dir=local_dir, local_dir_use_symlinks=False,
                          ignore_patterns=['*.msgpack', '*.h5'])

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    tokenizer.model_max_length = args.context_length

    model = AutoModelForCausalLM.from_pretrained(
        local_dir, use_flash_attention_2=use_flash_attention,
        device_map=device, torch_dtype=dtype)
    model.to(device)
    for _, md in model.named_modules():
        md.register_forward_hook(thread_hook)
    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer

def initialize_lora_model(model_name: str, local_dir: Path, args) -> None:
    """Download a model from the HuggingFace Hub and save it to the local directory."""

    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    args.lora_r = 64
    args.lora_alpha = 16
    args.lora_dropout = 0.01
    args.bits = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.fp16 else torch.float32
    use_flash_attention = device == "cuda" and args.fp16

    # Ideally we should check all the files, but for now just check one
    if not os.path.isfile(os.path.join(local_dir, 'config.json')):
        snapshot_download(model_name, local_dir=local_dir, local_dir_use_symlinks=False,
                          ignore_patterns=['*.msgpack', '*.h5'])

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    tokenizer.model_max_length = args.context_length

    model = AutoModelForCausalLM.from_pretrained(
        local_dir, use_flash_attention_2=use_flash_attention,
        device_map=device, torch_dtype=dtype
        )
    
    modules = _find_all_linear_names(args, model)
    print('Modules to be fine-tuned: ', modules)
    config = LoraConfig(
              r=args.lora_r,
              lora_alpha=args.lora_alpha,
              target_modules=modules,
              lora_dropout=args.lora_dropout,
              bias="none",
              task_type="CAUSAL_LM",
          )
    model = get_peft_model(model, config)
    model.to(device)


    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer

def initialize_qlora_model(model_name: str, local_dir: Path, args) -> None:
    """Download a model from the HuggingFace Hub and save it to the local directory."""

    global GLOBAL_MODEL, GLOBAL_TOKENIZER

    args.lora_r = 64
    args.lora_alpha = 16
    args.lora_dropout = 0.01
    args.bits = 4
    args.double_quant = True
    args.quant_type = 'nf4'
    args.optim = 'paged_adamw_32bit' # this will be passed to the trainer
    args.gradient_checkpointing = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.fp16 else torch.float32
    use_flash_attention = device == "cuda" and args.fp16

    # Ideally we should check all the files, but for now just check one
    if not os.path.isfile(os.path.join(local_dir, 'config.json')):
        snapshot_download(model_name, local_dir=local_dir, local_dir_use_symlinks=False,
                          ignore_patterns=['*.msgpack', '*.h5'])

    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    tokenizer.model_max_length = args.context_length

    bb_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
    )

    model = AutoModelForCausalLM.from_pretrained(
        local_dir, use_flash_attention_2=use_flash_attention,
        # device_map=device, # it weirdly crashes if I add the device for QLORA
        torch_dtype=dtype, use_cache=False,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        config=bb_config
        )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    modules = _find_all_linear_names(args, model)
    print('Modules to be fine-tuned: ', modules)
    config = LoraConfig(
              r=args.lora_r,
              lora_alpha=args.lora_alpha,
              target_modules=modules,
              lora_dropout=args.lora_dropout,
              bias="none",
              task_type="CAUSAL_LM",
          )
    model = get_peft_model(model, config)
    model.to(device) 


    GLOBAL_MODEL = model
    GLOBAL_TOKENIZER = tokenizer

