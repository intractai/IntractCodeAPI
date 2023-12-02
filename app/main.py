import argparse
import logging
import os
import time
from typing import Dict

from fastapi import FastAPI

from pydantic import BaseModel
from src import finetune
from src import model as model_mdl
from huggingface_hub import snapshot_download
import torch


logger = logging.getLogger("docker_agent")


def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run the API.")
    parser.add_argument("--full_precision", action="store_false", dest="fp16",
                        help="Run with fp32 intead of fp16.")
    parser.set_defaults(fp16=True)
    return parser.parse_known_args()


def main():
    args, _ = setup_arg_parser()

    global app, tokenizer, model,local_model_dir,device,dtype,use_flash_attention
    app = FastAPI()

    model_name = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-base")
    local_model_dir = os.getenv("LOCAL_MODEL_DIR", "./.model")
    logger.info(f"Using model {model_name}")
    model_mdl.intialize_model(model_name, local_model_dir, args)
    

main()


class GenerateData(BaseModel):
    input_text: str
    max_decode_length: int = 128

@app.put("/generate")
def generate(item: GenerateData):
    model=model_mdl.current_model
    tokenizer=model_mdl.current_tokenizer
    inputs = tokenizer(item.input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_length=item.max_decode_length, return_dict_in_generate=True, output_scores=True)
    out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
    output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)

    # Calculate score of sequence (avg probability of the tokens)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    score = torch.exp(transition_scores).mean().item()

    return {"generated_text": output_text, "score": score}


class ProjectFinetuneData(BaseModel):
    project_dict: Dict[str, str]

@app.post("/finetune/project")
def generate(item: ProjectFinetuneData):
    for file_name, file_code in item.project_dict.items():
        print(f">>> {file_name}\n\n{file_code}\n\n")

    time.sleep(1)

    #Launch the training job
    #Fetch all environeent variables that start with "FINETUNE_"
    env_vars = {k: v for k, v in os.environ.items() if k.startswith("FINETUNE_")}
    #Remove the prefix from the env vars, and lowercase them, and create a dict
    env_vars = {k.replace("FINETUNE_", "").lower(): v for k, v in env_vars.items()}

    finetune.train_supervised_projectdir(item.project_dict,
                                        output_dir=local_model_dir, 
                                         report_to='none',**env_vars)
    return {"result": "success"}

