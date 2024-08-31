import logging
import random
import sys
from typing import Optional
import warnings

import numpy as np
import torch

sys.path.append('../')
from src.modeling.model_provider import ModelProvider


def create_new_model_tuple(model_provider: ModelProvider):
    """Create a new model and tokenizer."""
    model, model_utils = model_provider.create_new_model_tuple()
    tokenizer = model_utils['tokenizer']
    return model, tokenizer


class OpenAIRequestLogFilter(logging.Filter):
    """Filter out the OpenAI request log."""
    def filter(self, record):
        return 'HTTP Request: POST https://api.openai.com' not in record.getMessage()
    

def configure_logging(logger):
    """Configure logging for the server."""
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # add log in the file
    handler = logging.FileHandler('../eval_log.txt')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    warnings.filterwarnings('ignore', message=".*Could not load referrer policy.*")
    trafilatura_logger = logging.getLogger('trafilatura')
    trafilatura_logger.setLevel(logging.INFO)
    lite_llm = logging.getLogger('LiteLLM')
    lite_llm.setLevel(logging.INFO)

    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.INFO)

    httpx_logger.addFilter(OpenAIRequestLogFilter())
    

def set_seed(seed: Optional[int] = None):
    """Set the random seed for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)