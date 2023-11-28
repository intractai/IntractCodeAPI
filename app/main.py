from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch
import os
import logging

logger = logging.getLogger("docker_agent")

def main():
    global app, tokenizer, model
    app = FastAPI()

    model_name = os.getenv('MODEL_NAME', 'deepseek-ai/deepseek-coder-1.3b-base')
    logger.info(f"Using model {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,)

main()


class GenerateData(BaseModel):
    input_text: str
    max_decode_length: int = 128

@app.put("/generate")
def generate(item: GenerateData):
    inputs = tokenizer(item.input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=item.max_decode_length,
                             )
    return {"generated_text": tokenizer.decode(outputs[0])}

