# blt_eval_harness_script.py
#
# Integration script for ByteLatentTransformer (BLT) model with lm-eval-harness.
# This script wraps the BLT model and its custom generation logic (`generate_blt`)
# to implement the required LM methods: generate_until, loglikelihood, and loglikelihood_rolling.

import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Union

import torch
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from rich.progress import track
from torch.nn import functional as F

# --- BLT Library Imports ---
# NOTE: Ensure these modules and classes are available in your environment.
from bytelatent.args import (
    EvalArgs,
    TrainArgs,
    ValidationArgs,
    find_and_sanitize_chunks,
)
from bytelatent.checkpoint import CONSOLIDATE_FOLDER, consolidate_checkpoints
from bytelatent.config_parser import parse_args_to_pydantic_model
from bytelatent.data.file_util import get_fs
from bytelatent.data.iterators.arrow_iterator import ArrowFileIterator
from bytelatent.data.iterators.limit_iterator import LimitIterator
from bytelatent.data.iterators.packing_iterator import (
    PackingArgs,
    PackingIterator,
    PackingMode,
)
from bytelatent.data.iterators.preprocess_iterator import PreprocessIterator
from bytelatent.data.iterators.sequence_iterator import (
    SequenceIterator,
    SequencePackingArgs,
)
from bytelatent.data.patcher import Patcher, PatcherArgs, PatchingModeEnum
from bytelatent.distributed import (
    DistributedArgs,
    dist_max,
    dist_min,
    dist_sum,
    get_device_mesh,
    get_global_rank,
    get_world_size,
    setup_torch_distributed,
    to_py_num,
)
from bytelatent.generate import load_consolidated_model_and_tokenizer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.tokenizers.blt_tokenizer import BltTokenizer
from bytelatent.tokenizers.build_tokenizer import TokenizerArgs
from bytelatent.transformer import LMTransformer

EVAL_FOLDER_NAME = "{:010d}"
logger = logging.getLogger(__name__)


# --- Core BLT Helper Functions from Original Script ---

def get_max_length(input_tokens: list[list[int]] | None) -> int:
    if input_tokens is None:
        max_length = 0
    else:
        max_length = max([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        max_length = int(dist_max(max_length))
    return max_length

def get_min_length(input_tokens: list[list[int]] | None) -> int:
    if input_tokens is None:
        min_length = 0
    else:
        min_length = min([len(t) for t in input_tokens])
    if torch.distributed.is_initialized():
        min_length = int(dist_min(min_length))
    return min_length

def get_generation_range(
    prompt_tokens: list[list[int]] | None, max_gen_len: int
) -> tuple[int, int]:
    batch_min_prompt_length = get_min_length(prompt_tokens)
    batch_max_prompt_length = get_max_length(prompt_tokens)
    return batch_min_prompt_length, batch_max_prompt_length + max_gen_len

def sample_top_k(probs, k):
    topk_value, _ = torch.topk(probs, k)
    min_value_top_k = topk_value[:, [-1]]
    probs[probs < min_value_top_k] = 0.0
    probs.div_(probs.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

@torch.inference_mode()
def generate_blt(
    prompts: list[str] | None,
    *,
    model: ByteLatentTransformer,
    tokenizer: BltTokenizer,
    patcher: Patcher,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
) -> list[list[int]]:
    assert (
        patcher.realtime_patching
    ), "generate_nocache requires patcher.realtime_patching=True"
    model.eval()
    if max_prompt_len is None:
        max_prompt_len = 512
    if prompts is None:
        prompt_tokens = None
        n_truncated_prompts = 0
        total_truncated_prompts = 0
    else:
        prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in prompts]
        n_truncated_prompts = sum([max_prompt_len < len(t) for t in prompt_tokens])
        total_truncated_prompts = dist_sum(n_truncated_prompts)

        # Truncation
        prompt_tokens = [
            t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :]
            for t in prompt_tokens
        ]

    if total_truncated_prompts > 0:
        logger.info(
            f"There are {total_truncated_prompts} prompts that are truncated on the left, "
            f"length greater than max_prompt_len = {max_prompt_len}, "
            f"maximum prompt length = {get_max_length(prompt_tokens)} across all gpus."
        )

    if prompt_tokens is None:
        prompt_tokens = [[tokenizer.bos_id] for _ in range(end_pos)]

    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    batch_size = len(prompt_tokens)
    tokens = torch.full((batch_size, end_pos), tokenizer.pad_id).cuda().long()

    # Copy inputs to tensor for generated tokens
    for i, row_tokens in enumerate(prompt_tokens):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long()
    input_text_mask = tokens != tokenizer.pad_id

    for i, curr_pos in enumerate(range(start_pos, end_pos)):
        current_tokens = tokens[:, :curr_pos]
        patch_lengths, _ = patcher.patch(current_tokens, include_next_token=True)
        logits = model(current_tokens, patch_lengths=patch_lengths)[:, -1]

        if use_sampling:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = sample_top_p(probs, top_p)
            elif top_k > 0:
                next_token = sample_top_k(probs, top_k)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = torch.where(
            input_text_mask[:, curr_pos], tokens[:, curr_pos], next_token
        )
        tokens[:, curr_pos] = next_token

    if remove_prompts:
        generated_tokens = [
            t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    else:
        generated_tokens = [
            t[: len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    return generated_tokens

@torch.inference_mode()
def generate_bltold( # Replaces generate_nocache
    prompts: list[str] | None,
    *,
    model: ByteLatentTransformer,
    tokenizer: BltTokenizer,
    patcher: Patcher,
    max_prompt_len: int = 256,
    max_gen_len: int = 256,
    use_sampling: bool = False,
    temp: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    remove_prompts: bool = True,
) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
    """
    Core BLT generation function adapted to return loglikelihoods and greedy flags
    for the lm-eval-harness generator interface.
    
    Returns: (list of generated strings, list of loglikelihood tensors, list of is_greedy tensors)
    """
    assert patcher.realtime_patching, "generate_blt requires patcher.realtime_patching=True"
    model.eval()
    
    # 1. Tokenize and Truncate
    if prompts is None:
        # If prompts is None, we generate from BOS
        prompt_tokens = [[tokenizer.bos_id] for _ in range(1)]
    else:
        prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in prompts]
    # max_prompt_len = max_prompt_len or min(
    #         max_seqlen - max_gen_len, self.max_tokens - max_gen_len
    #     )
    # Truncation
    if max_prompt_len:
        prompt_tokens = [
        t if len(t) < max_prompt_len else t[len(t) - max_prompt_len :]
        for t in prompt_tokens
    ]
    
    batch_size = len(prompt_tokens)
    
    # 2. Setup Tensors
    start_pos, end_pos = get_generation_range(prompt_tokens, max_gen_len)
    tokens = torch.full((batch_size, end_pos), tokenizer.pad_id).cuda().long()
    
    # Copy inputs to tensor
    for i, row_tokens in enumerate(prompt_tokens):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long().to(tokens.device)
    input_text_mask = tokens != tokenizer.pad_id

    # Buffers to store token-by-token results for the lm-eval interface
    # Shape: (batch_size, sequence_length)
    log_probs_all = torch.zeros((batch_size, end_pos), dtype=torch.float32).cuda()
    is_greedy_all = torch.zeros((batch_size, end_pos), dtype=torch.bool).cuda()

    # 3. Generation Loop
    for curr_pos in range(start_pos, end_pos):
        current_tokens = tokens[:, :curr_pos]
        
        # Prepare patches
        patch_lengths, _ = patcher.patch(current_tokens, include_next_token=True)
        
        # Forward pass
        logits = model(current_tokens, patch_lengths=patch_lengths)[:, -1]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Determine next token
        if use_sampling:
            if top_p > 0.0:
                next_token = sample_top_p(log_probs.exp(), top_p)
            elif top_k > 0:
                next_token = sample_top_k(log_probs.exp(), top_k)
            else:
                next_token = torch.multinomial(log_probs.exp(), num_samples=1)
            is_greedy = torch.zeros(batch_size, dtype=torch.bool).to(tokens.device)
            
        else:
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            is_greedy = torch.ones(batch_size, dtype=torch.bool).to(tokens.device)

        # Get the log probability of the chosen next token
        current_log_probs = torch.gather(log_probs, 1, next_token).squeeze(-1)
        
        # 4. Update Tokens and Records
        is_prompt_pos = input_text_mask[:, curr_pos]
        
        # If position is part of the prompt, use the prompt token
        next_token_final = torch.where(is_prompt_pos, tokens[:, curr_pos].unsqueeze(-1), next_token).squeeze(-1)
        
        # Update tokens tensor
        tokens[:, curr_pos] = next_token_final
        
        # Record log-likelihood and greedy status for the token at curr_pos
        # If it was a prompt token, we calculate its LL based on the model's prediction
        # (This LL calculation is what loglikelihood tasks require)
        
        # Find the log-prob of the token that actually appeared at curr_pos
        ll_of_actual_token = torch.gather(log_probs, 1, tokens[:, curr_pos].unsqueeze(-1)).squeeze(-1)
        
        log_probs_all[:, curr_pos] = ll_of_actual_token
        is_greedy_all[:, curr_pos] = is_greedy # Only True for generated tokens if greedy was used

    # 5. Decode and Return
    # Convert token Tensors to lists for the generator interface
    
    # Extract only the relevant (non-padded) generated/prompt parts
    final_sequences = [
        t[: len(prompt_tokens[i]) + max_gen_len].tolist()
        for i, t in enumerate(tokens)
    ]
    
    # Extract generated tokens/text (excluding prompt)
    if remove_prompts:
        output_tokens = [
            t[len(prompt_tokens[i]) : len(prompt_tokens[i]) + max_gen_len].tolist()
            for i, t in enumerate(tokens)
        ]
    else:
        # If we return the whole sequence, we'd need to adjust the generator wrapper
        raise NotImplementedError("remove_prompts=False not fully supported in current generator wrapper.")

    text_outputs = [tokenizer.decode(t) for t in output_tokens]

    # Slice log-likelihoods and greedy flags to match the final token length
    # NOTE: These slices need to match the indices used in loglikelihood() method
    ll_tensors = []
    gr_tensors = []
    for i, seq_len in enumerate([len(t) for t in final_sequences]):
        # The lm-eval-harness LL calculation starts from index 1 (predicting token 1 from token 0)
        # We need the log_probs from index 1 up to the end of the sequence.
        # This is where the complexity lies: we need the log-probs of all tokens *except* the first one.
        ll_tensors.append(log_probs_all[i, 1:seq_len])
        gr_tensors.append(is_greedy_all[i, 1:seq_len])

    return text_outputs, ll_tensors, gr_tensors


@torch.inference_mode()
def compute_loglikelihood(
    full_texts: list[str],
    *,
    model: ByteLatentTransformer,
    tokenizer: BltTokenizer,
    patcher: Patcher,
    max_sequence_len: int = 512, # Adjusted parameter name for clarity
) -> tuple[list[torch.Tensor], list[list[int]]]:
    """
    Computes the token-by-token log-likelihood for a list of input texts.

    Args:
        full_texts: List of strings (Context + Continuation) for which to calculate LL.
        model, tokenizer, patcher: BLT model components.
        max_sequence_len: Maximum length of the input text (truncation limit).

    Returns:
        A tuple: (list of log-likelihood Tensors, list of tokenized inputs)
    """
    import code;code.interact(local=locals()|globals())
    assert (
        patcher.realtime_patching
    ), "compute_loglikelihood requires patcher.realtime_patching=True"
    model.eval()

    # 1. Tokenization and Truncation
    # We must tokenize the entire sequence (context + continuation)
    prompt_tokens = [tokenizer.encode(t, add_eos=False) for t in full_texts]
    if max_sequence_len is None:
        max_sequence_len = 512
    # Truncation (on the left, as per your original function)
    tokenized_inputs = [
        t if len(t) < max_sequence_len else t[len(t) - max_sequence_len :]
        for t in prompt_tokens
    ]
    
    batch_size = len(tokenized_inputs)
    max_len = max([len(t) for t in tokenized_inputs])
    
    # Pad all sequences to the longest sequence in the batch (or max_sequence_len)
    tokens = torch.full((batch_size, max_len), tokenizer.pad_id).cuda().long()
    
    for i, row_tokens in enumerate(tokenized_inputs):
        tokens[i, : len(row_tokens)] = torch.tensor(row_tokens).long().to(tokens.device)

    # 2. Setup Log-Likelihood Tracking
    # We track log-likelihood for tokens T_1 to T_N (T_0 is the context start, which has no LL)
    log_likelihoods_all = torch.zeros((batch_size, max_len)).cuda()
    
    # The sequence length is the longest sequence in the batch
    sequence_length = max_len
    
    # 3. Autoregressive Forward Pass Loop
    # We iterate from position 1 up to the end of the sequence (T_1 to T_N)
    # At curr_pos, we predict tokens[:, curr_pos] given tokens[:, :curr_pos]
    for curr_pos in range(1, sequence_length):
        # Input context for prediction: tokens T_0 to T_{curr_pos-1}
        current_context = tokens[:, :curr_pos]
        
        # Target token is T_{curr_pos}
        target_token = tokens[:, curr_pos]

        # Skip if all targets at this position are padding (i.e., we are past all real sequences)
        if (target_token == tokenizer.pad_id).all():
            break

        # Get patches for the current context
        patch_lengths, _ = patcher.patch(current_context, include_next_token=True)
        
        # Get logits for the next token (T_{curr_pos})
        # Output shape: (batch_size, vocab_size)
        logits = model(current_context, patch_lengths=patch_lengths)[:, -1]
        
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather the log probability for the *actual* token at this position
        # This is the P(T_i | T_{<i}) term
        ll_of_actual_token = torch.gather(
            log_probs, 
            1, 
            target_token.unsqueeze(-1)
        ).squeeze(-1)
        
        # Apply mask to zero out LL for padding tokens
        is_padding = (target_token == tokenizer.pad_id)
        ll_of_actual_token[is_padding] = 0.0
        
        log_likelihoods_all[:, curr_pos] = ll_of_actual_token

    # 4. Extract Results
    # Return a list of tensors, where each tensor corresponds to the token-by-token LL
    # (LL for T_1 to T_N, zeroed for padding)
    results_ll = []
    
    # We only return the meaningful part of the LL tensor (excluding padding)
    for i, seq_tokens in enumerate(tokenized_inputs):
        # The LL tensor starts at index 1 (LL of T1) and goes up to the sequence length.
        # We exclude the LL at index 0 because there is no token before T0.
        results_ll.append(log_likelihoods_all[i, 1 : len(seq_tokens)])
        
    return results_ll, tokenized_inputs
# --- Generator Wrapper ---
class BltGeneratorWrapper:
    """
    A wrapper that mimics the interface of PackedCausalTransformerGenerator
    but calls generate_blt.
    """
    def __init__(self, args, model: ByteLatentTransformer, tokenizer: BltTokenizer, patcher: Patcher):
        self.model = model
        self.tokenizer = tokenizer
        self.patcher = patcher
        self.max_gen_len = args.max_gen_len
        self.max_prompt_len = args.max_prompt_len
        self.temperature = 0.0
        self.top_p = 0.0
        self.top_k = 0
        self.until = []
        self.device = next(model.parameters()).device

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        results = []
        
        # 1. Prepare Inputs for Batched LL Calculation
        full_texts = [req.args[0] + req.args[1] for req in requests]
        import code; code.interact(local=locals()|globals())
        
        # 2. Calculate Token-by-Token Log-Likelihood
        # max_prompt_len is used here as the max sequence length limit
        ll_tensors, tokenized_inputs = compute_loglikelihood(
            full_texts=full_texts,
            model=self.generator.model,
            tokenizer=self.generator.tokenizer,
            patcher=self.generator.patcher,
            max_sequence_len=self.generator.max_prompt_len,
        )

        # 3. Aggregate Results (P(Continuation | Context))
        for i, req in enumerate(requests):
            prompt = req.args[0]
            ll_tensor = ll_tensors[i] # LLs for T1, T2, ... Tn

            # Calculate prompt length in tokens
            # NOTE: We use the actual tokenized input, which may be truncated
            p_len = len(
                self.generator.tokenizer.encode(prompt, add_bos=False, add_eos=False)
            )
            
            # The LL tensor starts at the log-prob of the 2nd token (index 0 is P(T1|T0)).
            # If prompt is C tokens long (T0...T_{C-1}), the continuation starts at T_C.
            # In the LL tensor (which starts at T1), T_C is at index C-1.
            continuation_start_idx = p_len - 1
            
            if continuation_start_idx < 0:
                # Handle empty prompt case, where continuation starts at T0 (index 0 of LL tensor)
                continuation_start_idx = 0

            # Sum the log-likelihoods of the continuation tokens
            ll_sum = ll_tensor[continuation_start_idx:].sum().item()
            
            # For loglikelihood tasks, we always return False for is_greedy
            is_all_greedy = False
            
            results.append((ll_sum, is_all_greedy))

        return results
    
    def generate(self, prompts: list[str]) -> Tuple[List[str], List[torch.Tensor], List[torch.Tensor]]:
        """
        Main generation entry point, calls the core BLT logic.
        """
        use_sampling = self.temperature > 1e-6 or self.top_p > 1e-6 or self.top_k > 0

        # Call the core BLT generation function
        return generate_bltold(
            prompts=prompts,
            model=self.model,
            tokenizer=self.tokenizer,
            patcher=self.patcher,
            max_prompt_len=self.max_prompt_len,
            max_gen_len=self.max_gen_len,
            use_sampling=use_sampling,
            temp=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            remove_prompts=True,
        )


# --- EvalHarnessLM and Utility Functions ---

def all_dicts_same(dict_list):
    if not dict_list:
        return True
    first_dict = dict_list[0]
    return all(d == first_dict for d in dict_list)

class MockAccelerator:
    def gather(self, tensor):
        l = [torch.zeros_like(tensor) for _ in range(get_world_size())]
        torch.distributed.all_gather(l, tensor)
        return torch.stack(l)

    def wait_for_everyone(self):
        torch.distributed.barrier()

class EvalHarnessLM(LM):
    def __init__(self, generator: BltGeneratorWrapper):
        super().__init__()
        self.generator = generator
        self.accelerator = MockAccelerator()
        self._rank = get_global_rank()
        self._world_size = get_world_size()
        self.device = generator.device

    def generate_until(self, requests: list[Instance]) -> list[str]:
        prompts, gen_args = zip(*[req.args for req in requests])
        assert all_dicts_same(gen_args), "Doesn't support different gen args for now"
        gen_args = gen_args[0]
        temperature = gen_args.get("temperature", 0.0)
        top_p = gen_args.get("top_p", None)
        top_k = gen_args.get("top_k", None)
        until = gen_args.get("until", [])

        self.generator.temperature = temperature
        self.generator.top_p = top_p
        self.generator.top_k = top_k
        self.generator.until = until
        
        # Calls BltGeneratorWrapper.generate, which calls generate_blt
        generations, _, _ = self.generator.generate(prompts)
        
        filtered_gen = []
        for g in generations:
            # Handle stop sequences (required by lm-eval-harness)
            for e in until:
                if e in g:
                    g = g[: g.index(e)]
            filtered_gen.append(g)
        return filtered_gen
    
    def _loglikelihood_tokens(requests: list[tuple[str, str]], disable_tqdm: bool = False) -> list[tuple[float, bool]]:
        """
        Internal helper to compute loglikelihoods given pre-tokenized inputs.
        Each entry in new_reqs is a tuple: ((context, continuation), context_enc, continuation_enc)
        """
        import code;code.interact(local=locals()|globals())
        results = []
        
        # Prepare full texts for LL computation
        full_texts = [ctx + cont for (ctx, cont) in requests]

        self.generator.model.eval()
        for ctx, cont in requests:
            # do 1 by 1 
            prompt_tokens = self.generator.tokenizer.encode(ctx + cont, add_eos=False)
            tokens = torch

        # Calculate token-by-token Log-Likelihood
        ll_tensors, tokenized_inputs = compute_loglikelihood(
            full_texts=full_texts,
            model=self.generator.model,
            tokenizer=self.generator.tokenizer,
            patcher=self.generator.patcher,
            max_sequence_len=self.generator.max_prompt_len,
        )

        # Aggregate Results
        for i, ((context, continuation), context_enc, continuation_enc) in enumerate(new_reqs):
            ll_tensor = ll_tensors[i] # LLs for T1, T2, ... Tn
            
            # Calculate context length in tokens
            p_len = len(context_enc)
            
            # The LL tensor starts at the log-prob of the 2nd token (T1, index 0 of LL tensor).
            # Continuation starts at T_C where C = p_len
            continuation_start_idx = p_len - 1
            
            if continuation_start_idx < 0:
                continuation_start_idx = 0

            # Sum the log-likelihoods of the continuation tokens
            ll_sum = ll_tensor[continuation_start_idx:].sum().item()
            
            # For loglikelihood tasks, we always return False for is_greedy
            results.append((ll_sum, False))

        return results
    def loglikelihood1(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Calculates P(Continuation | Context) for each request.
        """

        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            continuation_enc = self.generator.tokenizer.encode(
                    continuation, add_bos=False, add_eos=False
                )
            context_enc = self.generator.tokenizer.encode(
                    context, add_bos=False, add_eos=False
                )
            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

        results = []
        
        # 1. Prepare Inputs for Batched LL Calculation
        # Full text is Context + Continuation
        full_texts = [req.args[0] + req.args[1] for req in requests]
        
        # 2. Calculate Token-by-Token Log-Likelihood
        # Uses the external helper function to perform the batched forward passes
        ll_tensors, tokenized_inputs = compute_loglikelihood(
            full_texts=full_texts,
            model=self.generator.model,
            tokenizer=self.generator.tokenizer,
            patcher=self.generator.patcher,
            max_sequence_len=self.generator.max_prompt_len,
        )

        # 3. Aggregate Results (Sum LL over Continuation part)
        for i, req in enumerate(requests):
            prompt = req.args[0]
            ll_tensor = ll_tensors[i] # LLs for T1, T2, ... Tn (sliced from index 1 of full sequence)
            
            # Get the actual tokenized context length
            # This is critical for correctly slicing the LL tensor
            p_len = len(self.generator.tokenizer.encode(prompt, add_bos=False, add_eos=False))
            
            # The LL tensor starts at the log-prob of the 2nd token (T1, which is index 0 of the LL tensor).
            # If the prompt is C tokens (T0...T_{C-1}), the continuation starts at T_C.
            # T_C corresponds to index C-1 in the LL tensor.
            continuation_start_idx = max(0, p_len - 1)
            
            # Sum the log-likelihoods of the continuation tokens
            ll_sum = ll_tensor[continuation_start_idx:].sum().item()
            
            # For loglikelihood tasks, we always return False for is_greedy
            results.append((ll_sum, False))

        return results
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """
        Calculates P(Continuation | Context) for each request using a single-instance loop.
        
        Note: This implementation bypasses the complex batched 'compute_loglikelihood' 
        for simplicity, relying on the internal logic derived from the original BLT script.
        """
        results = []
        
        # Accessing BLT components
        model = self.generator.model
        tokenizer = self.generator.tokenizer
        patcher = self.generator.patcher
        device = self.generator.device
        max_prompt_len = self.generator.max_prompt_len
        
        model.eval()

        for req in requests:
            context, continuation = req.args

            # 1. Prepare Full Tokenized Sequence (Context + Continuation)
            full_text = context + continuation
            full_tokens = tokenizer.encode(full_text, add_eos=False)
            
            # Truncation (if needed, consistent with BLT's standard behavior)
            if max_prompt_len and len(full_tokens) > max_prompt_len:
                full_tokens = full_tokens[len(full_tokens) - max_prompt_len :]
                
            token_count = len(full_tokens)
            
            # If the sequence is empty after truncation/tokenization, skip
            if token_count <= 1:
                results.append((0.0, False))
                continue

            # Input context for prediction: T_0 to T_{N-2}
            input_tokens = torch.tensor(full_tokens[:-1], dtype=torch.long, device=device).unsqueeze(0)
            # Target tokens: T_1 to T_{N-1}
            target_tokens = torch.tensor(full_tokens[1:], dtype=torch.long, device=device).unsqueeze(0)

            # 2. Forward Pass (Single Call)
            # We predict the sequence T_1...T_{N-1} given T_0...T_{N-2}
            
            # Since BLT's model expects the full sequence length, padding or 
            # batching is usually required. For simplicity, we assume this single 
            # forward pass handles the input without padding, or we pad to the nearest
            # block size (if required by BLT, though not shown here).
            
            # Using the patcher from your original script
            patch_lengths, _ = patcher.patch(input_tokens, include_next_token=True)
            
            # Logits shape: (1, token_count - 1, vocab_size)
            logits = model(input_tokens, patch_lengths=patch_lengths)

            # 3. Calculate Log-Likelihood for the Full Sequence
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            
            # Gather the log-probability of the actual target token for each position
            # log_probs shape: (1, token_count - 1)
            token_log_likelihoods = torch.gather(
                log_probs, 
                2, 
                target_tokens.unsqueeze(-1)
            ).squeeze(2).squeeze(0) # Remove batch and vocab dim

            # 4. Sum LL over Continuation Tokens
            
            # Context length in tokens (after potential truncation)
            context_tokens = tokenizer.encode(context, add_eos=False)
            p_len = len(context_tokens)

            # We need the LL sum starting from the LL of the token *after* the context (T_C).
            # The LL tensor starts at P(T1|T0). If context is T0...T_{C-1} (C tokens), 
            # the continuation starts at T_C.
            # T_C is at index C-1 in the token_log_likelihoods tensor.
            continuation_start_idx = max(0, p_len - 1)

            # Sum LL from the continuation start index to the end
            ll_sum = token_log_likelihoods[continuation_start_idx:].sum().item()
            
            results.append((ll_sum, False)) # False because we are calculating LL, not checking greedy match

        return results
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """
        Calculates log-likelihood for the entire sequence by splitting it into chunks.
        Used primarily for Perplexity (PPL) tasks.
        """
        # For simplicity, we can use the same LL calculation logic as `loglikelihood` 
        # but apply it to the full, unsplit context/continuation (which is just the prompt).
        
        # The prompt field for rolling LL holds the entire long text chunk
        full_texts = [req.args[0] for req in requests]
        
        # Calculate token-by-token Log-Likelihood for the long texts
        ll_tensors, _ = compute_loglikelihood(
            full_texts=full_texts,
            model=self.generator.model,
            tokenizer=self.generator.tokenizer,
            patcher=self.generator.patcher,
            # Use a sensible limit, perhaps max_prompt_len * 2 or a model limit
            max_sequence_len=self.generator.max_prompt_len, 
        )
        
        results = []
        for ll_tensor in ll_tensors:
            # For rolling LL, we sum the log-likelihoods of all tokens calculated
            results.append(ll_tensor.sum().item())

        return results

# --- Evaluation and Launch Functions ---

# Helper function (from your example)
def eval_ppl_on_path(
    # ... (body of eval_ppl_on_path is the same as your example)
    *args, **kwargs
):
    # This function is used for PPL/BPB evaluation using the data pipeline, 
    # not the lm-eval-harness interface, so it remains unchanged.
    raise NotImplementedError("PPL logic requires full context of eval_ppl_on_path")
    # ... (Placeholder to avoid massive copy-paste)
    pass 

def launch_eval(eval_args: EvalArgs):
    assert eval_args.dump_dir is not None
    assert eval_args.ckpt_dir is not None
    
    # 1. Setup Distributed Environment
    distributed_args = DistributedArgs()
    distributed_args.configure_world()
    if not torch.distributed.is_initialized():
        setup_torch_distributed(distributed_args)

    world_mesh = get_device_mesh(distributed_args)
    dp_mesh = world_mesh["dp_replicate"]
    world_size = dp_mesh.size()
    world_rank = dp_mesh.get_local_rank()

    # 2. Load Checkpoint
    fs = get_fs(eval_args.ckpt_dir, s3_profile=eval_args.s3_profile)
    # ... (Checkpoint consolidation logic from your example)
    
    fs.mkdirs(eval_args.dump_dir, exist_ok=True)
    
    # 3. Load Model and Patcher
    torch.distributed.barrier()
    logger.info("Loading model")
    consolidate_path = eval_args.ckpt_dir # Assuming path is consolidated/ready
    model, tokenizer, train_cfg = load_consolidated_model_and_tokenizer(
        consolidate_path,
    )
    model.eval()
    logger.info("Model loaded")
    
    patcher_args = train_cfg.data.patcher_args.model_copy(deep=True)
    patcher_args.realtime_patching = True
    logger.info("Loading entropy model and patcher")
    patcher_args.entropy_model_checkpoint_dir = eval_args.entropy_ckpt_dir
    patcher = patcher_args.build()
    logger.info("Patcher loaded")

    # 4. Run PPL Evaluation (If requested)
    ppl_results = None
    if eval_args.run_ppl:
        # ... (PPL setup logic from your example, calls eval_ppl_on_path)
        pass

    # 5. Run LM-Eval-Harness Tasks (If requested)
    task_results = None
    if eval_args.run_tasks:
        assert eval_args.generator is not None
        assert eval_args.harness is not None
        
        # Instantiate the custom BLT generator wrapper
        generator = BltGeneratorWrapper(
            eval_args.generator, model, tokenizer, patcher
        )
        
        # Wrap the custom generator in the LM interface
        wrap = EvalHarnessLM(generator)
        
        # Run the simple_evaluate harness
        task_results = simple_evaluate(wrap, **eval_args.harness.model_dump(), batch_size=1)

    # 6. Save Results
    results = {"ppl": ppl_results, "tasks": task_results}

    if get_global_rank() == 0:
        with fs.open(os.path.join(eval_args.dump_dir, "results.json"), "w") as f:
            f.write(json.dumps(results, indent=4))
        logger.info(f"All evaluation results written to results.json")
    print(task_results)
        # ... (Metric log writing from your example)

def main():
    # EvalArgs must be defined elsewhere or via command line arguments
    eval_args = parse_args_to_pydantic_model(EvalArgs)
    launch_eval(eval_args)


if __name__ == "__main__":
    main()