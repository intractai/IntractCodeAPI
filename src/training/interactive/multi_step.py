import logging
from typing import Callable, List, Optional, Tuple

from accelerate import Accelerator
from accelerate.utils import gather_object
import torch
from transformers import PreTrainedTokenizer
from trl import PreTrainedModelWrapper


log = logging.getLogger(__name__)


def prepare_queries(
        tokenizer: PreTrainedTokenizer,
        instructions: List[str],
        code_blocks: List[str],
        exec_outputs: Optional[List[str]] = None,
        truncation: bool = False,
    ) -> Tuple[List[str], List[torch.Tensor]]:
    """Prepare queries for the agent and return the tokenized tensors.
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use
        instructions (list[str]): The instructions to the agent
        code_blocks (list[str]): The code blocks to the agent
        exec_outputs (list[str], optional): The outputs of the executed code. Defaults to None.
        truncation (bool, optional): Whether to truncate the queries. Defaults to True.

    Returns:
        Tuple[list[str], list[torch.Tensor]]: The queries and their tokenized tensors
    """
    query_ids = []
    query_text = []
    for i in range(len(instructions)):
        instruction = instructions[i]
        code = code_blocks[i]
        exec_output = exec_outputs[i] if exec_outputs is not None else None
    
        if exec_output is not None:
            query = f"{instruction}\n\n# Previous code\n```\n{code}\n```\n\n" + \
                f"# Output of previous code\n```\n{exec_output}\n```\n\n" + \
                f"The previous code was incorrect and needs to be fixed. Respond with only code."
        else:
            query = f"{instruction}\n\n# Starting code\n```\n{code}\n```\n\nRespond with only code."
        query_text.append(query)

        messages = [{'role': 'user', 'content': query}]
        query_ids.append(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors='pt', truncation=truncation,
        )[0])
        
    return query_text, query_ids


# Adapted from: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L507
def llm_generate_batched(
        model: PreTrainedModelWrapper,
        tokenizer: PreTrainedTokenizer,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        accelerator: Optional[str] = None,
        **generation_kwargs,
    ):
        """Generate responses from queries in batches.
        
        Args:
            model (PreTrainedModelWrapper): The model to use
            tokenizer (PreTrainedTokenizer): The tokenizer to use
            query_tensors (list[torch.Tensor]): The queries to the model
            length_sampler (Callable, optional): A function to sample the length of the response. Defaults to None.
            batch_size (int, optional): The batch size to use. Defaults to 4.
            return_prompt (bool, optional): Whether to return the prompt in the response. Defaults to True.
            pad_to_multiple_of (int, optional): The multiple to pad to. Defaults to None.
            remove_padding (bool, optional): Whether to remove the padding from the response. Defaults to True.
            accelerator (str, optional): The accelerator to use. If the accelerator uses multiple devices,
                inference will be parallelized. Defaults to None.

        Returns:
            list[torch.Tensor]: The responses from the model
        """

        if accelerator is None:
            return _llm_generate_batched(
                model, tokenizer, query_tensors, length_sampler, batch_size, return_prompt,
                pad_to_multiple_of, remove_padding, **generation_kwargs)
        else:
            return _llm_generate_batched_accelerate(
                model, tokenizer, query_tensors, accelerator, length_sampler, batch_size,
                return_prompt, pad_to_multiple_of, remove_padding, **generation_kwargs)


# Adapted from: https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L507
def _llm_generate_batched(
        model: PreTrainedModelWrapper,
        tokenizer: PreTrainedTokenizer,
        query_tensors: List[torch.Tensor],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        """Refer for `llm_generate_batched` function"""
        device = next(model.parameters()).device
        outputs = []

        padding_side_default = tokenizer.padding_side
        is_encoder_decoder = hasattr(model, 'is_encoder_decoder')
        if not is_encoder_decoder:
            tokenizer.padding_side = 'left'

        # in case we have fewer examples than bs
        batch_size = min(len(query_tensors), batch_size)

        for i in range(0, len(query_tensors), batch_size):
            if length_sampler is not None:
                generation_kwargs['max_new_tokens'] = length_sampler()

            # prevent overflow if query tensors are not even multiple of bs
            end_index = min(len(query_tensors), i + batch_size)

            batch = query_tensors[i:end_index]
            batch_mask = [torch.ones_like(element) for element in batch]
            inputs = {'input_ids': batch, 'attention_mask': batch_mask}

            padded_inputs = tokenizer.pad(
                inputs,
                padding = True,
                max_length = None,
                pad_to_multiple_of = pad_to_multiple_of,
                return_tensors = 'pt',
            ).to(device)

            # Note that this does not by default support multiple GPUs
            log.info(f"Generating responses for batch {i} to {end_index}")
            generations = model.generate(**padded_inputs, **generation_kwargs)

            for generation, mask in zip(generations, padded_inputs['attention_mask']):
                if not is_encoder_decoder:
                    output = generation[(1 - mask).sum() :]  # remove padding
                else:
                    output = generation

                if not return_prompt and not is_encoder_decoder:
                    output = output[(mask).sum():]  # remove prompt

                if remove_padding and tokenizer.eos_token_id in output:
                    pad_mask = output == tokenizer.eos_token_id
                    pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                    output = output[: pad_start + 1]  # keep the eos token at the end

                outputs.append(output)

        tokenizer.padding_side = padding_side_default
        return outputs


def _llm_generate_batched_accelerate(
        model: PreTrainedModelWrapper,
        tokenizer: PreTrainedTokenizer,
        query_tensors: List[torch.Tensor],
        accelerator: Accelerator,
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        pad_to_multiple_of: int = None,
        remove_padding: bool = True,
        **generation_kwargs,
    ):
        """Refer for `llm_generate_batched` function"""
        padding_side_default = tokenizer.padding_side
        is_encoder_decoder = hasattr(model, 'is_encoder_decoder')
        input_device = query_tensors[0].device
        if not is_encoder_decoder:
            tokenizer.padding_side = 'left'

        with accelerator.split_between_processes(query_tensors) as queries:
            log.debug(f"Process rank: {accelerator.process_index}, device: {accelerator.device}, # queries: {len(queries)}")

            outputs = []

            for i in range(0, len(queries), batch_size):
                # prevent overflow if query tensors are not even multiple of bs
                end_index = min(len(queries), i + batch_size)

                batch = queries[i:end_index]
                batch_mask = [torch.ones_like(element) for element in batch]
                inputs = {'input_ids': batch, 'attention_mask': batch_mask}

                padded_inputs = tokenizer.pad(
                    inputs,
                    padding = True,
                    max_length = None,
                    pad_to_multiple_of = pad_to_multiple_of,
                    return_tensors = 'pt',
                ).to(accelerator.device)

                # Generate responses
                log.info(f"Generating responses for batch {i} to {end_index}")
                generations = model.generate(**padded_inputs, **generation_kwargs)

                # Send back to original device
                generations = generations.to(input_device)

                for generation, mask in zip(generations, padded_inputs['attention_mask']):
                    if not is_encoder_decoder:
                        output = generation[(1 - mask).sum():] # remove padding
                    else:
                        output = generation

                    if not return_prompt and not is_encoder_decoder:
                        output = output[(mask).sum():]  # remove prompt

                    if remove_padding and tokenizer.eos_token_id in output:
                        pad_mask = output == tokenizer.eos_token_id
                        pad_start = torch.nonzero(pad_mask, as_tuple=False)[0, 0].item()
                        output = output[:pad_start + 1]  # keep the eos token at the end

                    outputs.append(output)

        gathered_outputs = gather_object(outputs)
        tokenizer.padding_side = padding_side_default
        return gathered_outputs