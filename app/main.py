import argparse
from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch
import os
import logging

logger = logging.getLogger("docker_agent")

def setup_arg_parser():
    parser = argparse.ArgumentParser(description="Run the API.")
    parser.add_argument("--full_precision", action="store_false", dest="fp16",
                        help="Run with fp32 intead of fp16.")
    parser.set_defaults(fp16=True)
    return parser.parse_known_args()

def main():
    args, _ = setup_arg_parser()

    global app, tokenizer, model
    app = FastAPI()

    model_name = os.getenv("MODEL_NAME", "deepseek-ai/deepseek-coder-1.3b-base")
    logger.info(f"Using model {model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.fp16 else torch.float32
    use_flash_attention = device == "cuda" and args.fp16

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, use_flash_attention_2=use_flash_attention,
        device_map=device, torch_dtype=dtype)
    model.to(device)


main()


class GenerateData(BaseModel):
    input_text: str
    max_decode_length: int = 128

@app.put("/generate")
def generate(item: GenerateData):
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