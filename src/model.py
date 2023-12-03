import torch
import os 

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

#Create a global variable to store the model
current_model=None
current_tokenizer=None

def intialize_model(model_name, local_dir,args):
    """Download a model from the HuggingFace Hub and save it to the local directory.
    """
    global current_model,current_tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.fp16 else torch.float32
    use_flash_attention = device == "cuda" and args.fp16
    if not os.path.isfile(os.path.join(local_dir, 'config.json')): #Ideally we should check all the files, but for now just check one
        snapshot_download(model_name, local_dir=local_dir,local_dir_use_symlinks=False,
                        ignore_patterns=['*.msgpack','*.h5'])
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForCausalLM.from_pretrained(
        local_dir, use_flash_attention_2=use_flash_attention,
        device_map=device, torch_dtype=dtype)
    model.to(device)
    current_model=model
    current_tokenizer=tokenizer
