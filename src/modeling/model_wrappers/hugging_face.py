import torch
from typing import Optional

from llama_index.core import VectorStoreIndex
from omegaconf import DictConfig
from transformers import GenerationConfig, PreTrainedTokenizer

from src.modeling.model_wrappers.base import ModelWrapper


class HuggingFaceModelWrapper(ModelWrapper):

    _generation_config = GenerationConfig(
        temperature=0.7, top_k=5, do_sample=True,
    )

    def __init__(
            self, 
            model: torch.nn.Module, 
            tokenizer: PreTrainedTokenizer,
            **kwargs
        ):
        super().__init__(model, tokenizer)

    def generate_completion(
            self, 
            item: 'GenerateData',
            config: DictConfig,
            vector_store: Optional[VectorStoreIndex] = None,
        ) -> dict:

        tokenizer = self._wrapped_tokenizer

        inputs = self._prepare_input(item, config, tokenizer, vector_store).to(self.device)

        outputs = self.generate(
            **inputs, max_new_tokens=config.inference.max_gen_length,
            return_dict_in_generate=True, output_scores=True,
            generation_config=self._generation_config,
        )

        out_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        output_text = tokenizer.decode(out_tokens, skip_special_tokens=True)
        logits = torch.stack(outputs.scores[-len(out_tokens):]).squeeze(1)
        probs = logits.softmax(dim=1).gather(1, out_tokens.unsqueeze(1))

        perplexity = torch.exp(-torch.sum(torch.log(probs)) / len(out_tokens)).item()

        return {
            'outputs': outputs,
            'output_text': output_text,
            'perplexity': perplexity,
        }