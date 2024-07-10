"""Data formatting utilities for model inference and training."""

from argparse import Namespace
from dataclasses import dataclass
import numpy as np
import logging
from typing import Optional, Sequence

from fastapi import APIRouter
from omegaconf import DictConfig
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedTokenizer

from src.modeling import FIM_BEGIN_TOKEN, FIM_HOLE_TOKEN, FIM_END_TOKEN


logger = logging.getLogger(__name__)
router = APIRouter()


# When doing FIM generation and truncation is required, this is the
# target fraction of the context that should be prior context
# (the rest is proceeding context)
GEN_PRIOR_CONTEXT_FRAC = 0.8
MAX_FP_TOKENS = 32 # Maximum tokens in file path
# Labels with -100 index are ignored in the loss calculation
# HuggingFace... document this stuff please
IGNORE_INDEX = -100
SHORT_CONTEXT_WARNING_THRESHOLD = 128
RAG_QUERY_FIM_HOLE_TOKEN = '<COMPLETION HOLE>'


@dataclass
class FIMTrainSample:
    """A single FIM sample for training."""
    preceeding_context: str
    proceeding_context: str
    target_text: str


def text_to_fim_train_sample(text: str):
    """Convert a string to a FIM sample with start, hole, and end strings.

    Args:
        text (str): The text to convert.

    Returns:
        FIMSample: The converted FIMSample object.
    """
    # A boundary can be = 0 (prefix will be empty)
    # a boundary can be = len(contents) (suffix will be empty)
    # The two boundaries can be equal (middle will be empty)
    boundaries = list(np.random.randint(low=0, high=len(text) + 1, size=2))
    boundaries.sort()

    # start = np.random.randint(low=0, high=len(contents))
    # mode_len = 32
    # max_len = 192
    # end = int(np.random.triangular(
    #     start + 1,
    #     start + mode_len,
    #     max(min(len(contents), max_len), start + mode_len)))
    # boundaries = [start, end]

    return FIMTrainSample(
        preceeding_context = text[:boundaries[0]],
        proceeding_context = text[boundaries[1]:],
        target_text = text[boundaries[0]:boundaries[1]],
    )


# Adapted from https://github.com/EleutherAI/gpt-neox/blob/
# FIM-clean/megatron/data/gpt2_dataset.py#L339
def prepare_fim_train_input(
        tokenizer: PreTrainedTokenizer,
        fp_tokens: Sequence[int],
        code_tokens: Optional[Sequence[int]] = None,
        fim_sample: Optional[FIMTrainSample] = None,
        additional_context_tokens: Optional[Sequence[int]] = None,
        truncate: bool = True):
    """
    Prepare a single training example for FIM.

    Either code_tokens or fim_sample are used to generate the problem.
    One or the other must be provided.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer.
        fp_tokens (Sequence[int]): The tokenized file path.
        code_tokens (Sequence[int]): The tokenized code.
        fim_sample (Optional[FIMTrainSample], optional):
            The FIM sample to use. If not provided, one will be generated from the code tokens.
            Defaults to None.
        additional_context_tokens (Optional[Sequence[int]], optional):
            Additional content retrieved from the vector store. Defaults to None.
        truncate (bool, optional):
            Whether or not to truncate the input. Defaults to True.

    Returns:
        Dict[str, Any]: The formatted training example.
    """
    additional_context_tokens = additional_context_tokens or []

    if truncate and code_tokens is not None:
        # -6 is for -3 FIM special tokens, BOS, EOS, and 1 for extra space
        code_tokens = code_tokens[:tokenizer.model_max_length - len(additional_context_tokens) - len(fp_tokens) - 6]

    assert code_tokens is not None or fim_sample is not None

    if fim_sample is None:
        contents = tokenizer.decode(code_tokens, skip_special_tokens=False)
        fim_sample = text_to_fim_train_sample(contents)

    prefix = np.array(tokenizer.encode(fim_sample.preceeding_context, add_special_tokens=False), dtype=np.int64)
    middle = np.array(tokenizer.encode(fim_sample.target_text, add_special_tokens=False), dtype=np.int64)
    suffix = np.array(tokenizer.encode(fim_sample.proceeding_context, add_special_tokens=False), dtype=np.int64)
    fp_tokens = np.array(fp_tokens, dtype=np.int64)
    additional_context_tokens = np.array(additional_context_tokens, dtype=np.int64)

    prefix_tok_id = tokenizer.vocab[FIM_BEGIN_TOKEN]
    middle_tok_id = tokenizer.vocab[FIM_HOLE_TOKEN]
    suffix_tok_id = tokenizer.vocab[FIM_END_TOKEN]

    # Here we truncate each given segment to fit the same length as it was before
    # A consequence is that we never reach the end of a file?
    # we should rather truncate at the context-level
    if truncate:
        # need to make same length as the input. Take the 3 sentinel tokens into account
        new_length = \
            suffix.shape[0] + additional_context_tokens.shape[0] \
            + fp_tokens.shape[0] + prefix.shape[0] + middle.shape[0] + 3
        diff = new_length - tokenizer.model_max_length
        if diff > 0: # too long
            # If there's no space to truncate the suffix:
            # stop and report it. atm i should have stopped this from happening
            if suffix.shape[0] <= diff:
                return None
            suffix = suffix[:suffix.shape[0] - diff]

    input_ids = np.concatenate([
        [tokenizer.bos_token_id],
        [prefix_tok_id], additional_context_tokens, fp_tokens, prefix,
        [middle_tok_id], suffix,
        [suffix_tok_id], middle,
        [tokenizer.eos_token_id]
    ])

    # Ignore the prompt tokens when calculating loss
    labels = np.concatenate([
        [tokenizer.bos_token_id],
        [IGNORE_INDEX] * (len(additional_context_tokens) + len(fp_tokens) + len(prefix) + 1),
        [IGNORE_INDEX] * (len(suffix) + 1),
        [IGNORE_INDEX], middle,
        [tokenizer.eos_token_id]
    ])

    return dict(
        input_ids = input_ids,
        labels = labels,
    )


def prepare_ntp_train_input(
        code_tokens: Sequence[int],
        fp_tokens: Sequence[int],
        tokenizer: PreTrainedTokenizer,
        additional_context_tokens: Optional[Sequence[int]] = None,
    ):
    """
    Prepare a single training example for next token prediction.

    Args:
        code_tokens (Sequence[int]): The tokenized code.
        fp_tokens (Sequence[int]): The tokenized file path.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        additional_context_tokens (Optional[Sequence[int]], optional):
            Additional content retrieved from the vector store. Defaults to None.

    Returns:
        Dict[str, Any]: The formatted training example.
    """
    additional_context_tokens = additional_context_tokens or []

    input_ids = [tokenizer.bos_token_id] + \
        additional_context_tokens + \
        fp_tokens + \
        code_tokens + \
        [tokenizer.eos_token_id] # TODO: FIX THIS, EOS SHOULD BE PUT INTO CODE BEFORE IT IS CHUNKED

    labels = [tokenizer.bos_token_id] + \
        len(additional_context_tokens) * [IGNORE_INDEX] + \
        len(fp_tokens) * [IGNORE_INDEX] + \
        code_tokens + \
        [tokenizer.eos_token_id]

    return dict(
        input_ids = input_ids,
        labels = labels,
    )


def format_ntp_inference_input(
        text: str,
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        file_path: Optional[str] = None,
        max_decode_length: int = 256,
        context_length: Optional[int] = None,
        retrieved_context: Optional[Sequence[str]] = None,
    ):
    """Format an input for next token prediction model inference.

    Args:
        text (str): The text to generate from.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        config (DictConfig): The config global config.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
        context_length (Optional[int], optional):
            Overrides context length from config when provided. Defaults to None.
        retrieved_context (Optional[Sequence[str]], optional):
            Additional context retrieved with the RAG retrieval system. Defaults to None.
    """

    if file_path:
        fp_tokens = tokenizer.encode(
            f'# {file_path}\n',
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = MAX_FP_TOKENS,
        )[0]
    else:
        fp_tokens = torch.tensor([], dtype=torch.long)

    if retrieved_context:
        context_template = config.rag.context_for_generation_template
        retrieved_context = [context_template.format(c) for c in retrieved_context]
        retrieved_context = '\n'.join(retrieved_context)
        retrieved_context_tokens = tokenizer.encode(
            retrieved_context,
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = config.rag.max_gen_context_length,
        )[0]
    else:
        retrieved_context_tokens = torch.tensor([], dtype=torch.long)

    context_length = context_length or config.model.context_length

    max_context_length = \
        context_length - len(retrieved_context_tokens) - len(fp_tokens) - max_decode_length
    
    if max_context_length < 0:
        raise ValueError(
            f"The context length {context_length} is too short to fit the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"and generated text ({max_decode_length})."
        )
    elif max_context_length <= SHORT_CONTEXT_WARNING_THRESHOLD:
        logger.warning(
            f"Only {max_context_length} tokens available for context after reserving space for the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"and generated text ({max_decode_length}). Consider increasing the context length."
        )

    context_tokens = tokenizer.encode(
        text,
        return_tensors = 'pt',
        add_special_tokens = False,
        truncation = True,
        max_length = max_context_length,
    )[0]

    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]), # TODO: FIX THIS, THERE SHOULDN'T ALWAYS BE A BOS TOKEN
        retrieved_context_tokens,
        fp_tokens,
        context_tokens,
    ]).unsqueeze(0)

    inputs = BatchEncoding({
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(tokenizer.pad_token_id).long(),
    })

    return inputs


def format_fim_inference_input(
        preceeding_text: str,
        proceeding_text: str,
        tokenizer: PreTrainedTokenizer,
        config: Namespace,
        file_path: Optional[str] = None,
        max_decode_length: int = 256,
        context_length: Optional[int] = None,
        retrieved_context: Optional[Sequence[str]] = None,
    ):
    """Format an input for FIM model inference.

    Args:
        preceeding_text (str): The text preceding the hole.
        proceeding_text (str): The text following the hole.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        config (Namespace): The config global config.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
        context_length (Optional[int], optional):
            Overrides context length from config when provided. Defaults to None.
        retrieved_context (Optional[Sequence[str]], optional):
            Additional context retrieved with the RAG retrieval system. Defaults to None.
    """
    if file_path:
        fp_tokens = tokenizer.encode(
            f'# {file_path}\n',
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = MAX_FP_TOKENS,
        )[0]
    else:
        fp_tokens = torch.tensor([], dtype=torch.long)

    if retrieved_context:
        context_template = config.rag.context_for_generation_template
        retrieved_context = [context_template.format(c) for c in retrieved_context]
        retrieved_context = '\n'.join(retrieved_context)
        retrieved_context_tokens = tokenizer.encode(
            retrieved_context,
            return_tensors = 'pt',
            add_special_tokens = False,
            truncation = True,
            max_length = config.rag.max_gen_context_length,
        )[0]
    else:
        retrieved_context_tokens = torch.tensor([], dtype=torch.long)

    prefix = tokenizer.encode(
        preceeding_text,
        return_tensors = 'pt',
        add_special_tokens = False
    )[0]
    suffix = tokenizer.encode(
        proceeding_text,
        return_tensors = 'pt',
        add_special_tokens = False
    )[0]

    # Get the length of the prior and proceeding context, then truncate
    # as necessary

    # The length of the prompt + generated response cannot be longer than
    # the max context length of the model
    # -4 is for the 4 FIM and BOS special tokens added to the prompt
    context_length = context_length or config.model.context_length
    n_special_tokens = 4
    max_context_length = \
        context_length - len(retrieved_context_tokens) - len(fp_tokens) - max_decode_length +  - n_special_tokens
    raw_text_length = len(prefix) + len(suffix)

    if max_context_length < 0:
        raise ValueError(
            f"The context length {context_length} is too short to fit the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"generated text ({max_decode_length}), and special tokens ({n_special_tokens})."
        )
    elif max_context_length <= SHORT_CONTEXT_WARNING_THRESHOLD:
        logger.warning(
            f"Only {max_context_length} tokens available for context after reserving space for the " + \
            f"retrieved context ({len(retrieved_context_tokens)}), file path ({len(fp_tokens)}), " + \
            f"generated text ({max_decode_length}), and special tokens ({n_special_tokens}). " + \
            "Consider increasing the context length."
        )

    # If the raw text is too long, truncate it
    if raw_text_length > max_context_length:
        max_prefix_length = int(max_context_length * GEN_PRIOR_CONTEXT_FRAC)
        max_suffix_length = max_context_length - max_prefix_length

        # If both the prefix and suffix are too long, truncate both
        if len(prefix) > max_prefix_length and len(suffix) > max_suffix_length:
            prefix = prefix[-max_prefix_length:]
            suffix = suffix[:max_suffix_length]

        # If just the prefix is too long, truncate it
        elif len(prefix) > max_prefix_length:
            target_length = max_context_length - len(suffix)
            prefix = prefix[-target_length:]

        # If just the suffix is too long, truncate it
        else:
            target_length = max_context_length - len(prefix)
            suffix = suffix[:target_length]

    prefix_tok_id = tokenizer.vocab[FIM_BEGIN_TOKEN]
    middle_tok_id = tokenizer.vocab[FIM_HOLE_TOKEN]
    suffix_tok_id = tokenizer.vocab[FIM_END_TOKEN]

    # Construct the final prompt
    input_ids = torch.cat([
        torch.tensor([tokenizer.bos_token_id]),
        torch.tensor([prefix_tok_id]), retrieved_context_tokens, fp_tokens, prefix,
        torch.tensor([middle_tok_id]), suffix,
        torch.tensor([suffix_tok_id]),
    ]).unsqueeze(0)

    inputs = BatchEncoding({
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(tokenizer.pad_token_id).long(),
    })

    return inputs


def format_inference_input(
        preceeding_text: str,
        tokenizer: PreTrainedTokenizer,
        config: Namespace,
        proceeding_text: Optional[str] = None,
        file_path: Optional[str] = None,
        max_decode_length: int = 256,
        context_length: Optional[int] = None,
        retrieved_context: Optional[Sequence[str]] = None,
    ):
    """Format an input for model inference.

    Args:
        preceeding_text (str): The text preceding the hole.
        tokenizer (PreTrainedTokenizer): The tokenizer.
        config (Namespace): The config global config.
        proceeding_text (Optional[str], optional):
            The text following the hole. Defaults to None.
        file_path (Optional[str], optional):
            The path to the file to generate from. Defaults to None.
        max_decode_length (int, optional):
            The maximum length of the generated sequence. Defaults to 256.
        context_length (Optional[int], optional):
            Overrides context length from config when provided. Defaults to None.
        retrieved_context (Optional[Sequence[str]], optional):
            The context retrieved from the vector store. Defaults to None.
    """

    if proceeding_text is None or not proceeding_text.strip():
        return format_ntp_inference_input(
            preceeding_text, tokenizer, config, file_path,
            max_decode_length, context_length, retrieved_context)
    
    return format_fim_inference_input(
        preceeding_text, proceeding_text, tokenizer, config,
        file_path, max_decode_length, context_length, retrieved_context,
    )


def format_rag_query(
        prior_context: str,
        proceeding_context: str = None,
        max_length: int = 512,
    ):
    """Format a RAG query for retrieval using the code to be completed.
    
    Args:
        prior_context (str): The text preceding the hole, or the full text if no hole.
        proceeding_context (str, optional): The text following the hole. Defaults to None.
        max_length (int, optional): The maximum length of the query in characters. Defaults to 512.
    """
    proceeding_context = proceeding_context or ''

    if proceeding_context:
        prior_context += RAG_QUERY_FIM_HOLE_TOKEN
    
    # Truncate in a way that we try to keep 80% of the prior context and 20% of the proceeding context
    if len(prior_context) + len(proceeding_context) > max_length:
        max_prefix_length = int(max_length * GEN_PRIOR_CONTEXT_FRAC)
        max_suffix_length = max_length - max_prefix_length

        # If both the prefix and suffix are too long, truncate both
        if len(prior_context) > max_prefix_length and len(proceeding_context) > max_suffix_length:
            prior_context = prior_context[:max_prefix_length]
            proceeding_context = proceeding_context[:max_suffix_length]

        # If just the prefix is too long, truncate it
        elif len(prior_context) > max_prefix_length:
            target_length = max_length - len(proceeding_context)
            prior_context = prior_context[-target_length:]

        # If just the suffix is too long, truncate it
        else:
            target_length = max_length - len(prior_context)
            proceeding_context = proceeding_context[:target_length]

    query_str = prior_context + proceeding_context

    return query_str
