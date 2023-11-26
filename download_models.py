from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeGPT-small-py", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/CodeGPT-small-py", trust_remote_code=True)
