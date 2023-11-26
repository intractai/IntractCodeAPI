from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py", trust_remote_code=True,)

class GenerateData(BaseModel):
    input_text: str

@app.put("/generate")
def generate(item: GenerateData):
    inputs = tokenizer(item.input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=128,
                             )
    return {"generated_text": tokenizer.decode(outputs[0])}