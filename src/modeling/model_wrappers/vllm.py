from typing import NamedTuple, Tuple, Union, Any, Optional

from vllm import LLM, SamplingParams
from llama_index.core import VectorStoreIndex
from omegaconf import DictConfig

from src.modeling.model_wrappers.base import ModelWrapper
from src.types import GenerateData

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