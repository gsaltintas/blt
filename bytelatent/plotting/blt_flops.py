# Copyright (c) Meta Platforms, Inc. and affiliates.
"""BLT-aware FLOPs/step estimator, corrected for patch-size scaling.

Why this exists
----------------
The FLOPS value logged during training (bytelatent/train.py, ~line 561) is:

    get_num_flop_per_token(model_param_count - vocab_size * dim_global,
                            n_layers_global, dim_global, args.data.seq_len)

i.e. a generic dense-transformer estimate (6N + attention) applied to the
*entire* model's parameter count, using the raw byte sequence length
(args.data.seq_len) for every layer. That's wrong for BLT specifically:

  - The local encoder/decoder run once per raw BYTE, so their cost scales
    with n_bytes (constant across compression ratios, since batch_size and
    seq_len in bytes are fixed for every T in this sweep).
  - The global/latent transformer runs once per PATCH, and under static
    patching, n_patches ~= n_bytes / patch_size (= n_bytes / T). So its cost
    should shrink roughly linearly as T grows.

Because the logged metric multiplies the *whole* model's params by the byte
seq_len regardless of which component they belong to, it can't see that
shrinkage -- which is why naive FLOPS looked nearly identical for T4 vs T8
even though the global transformer is doing ~2x less work per step at T8.

This script builds the real model per config (on the "meta" device, so no
weights/GPU are needed) to get exact per-component parameter counts, then
applies the 6N + attention FLOP formula separately to each component using
its own effective sequence length (n_bytes for local encoder/decoder,
n_patches for the global transformer). Local self-attention and cross-attention
are approximated as windowed (using local_attention_window_len /
cross_attn_window_encoder / cross_attn_window_decoder) rather than full O(seq^2),
matching how they're actually computed.

This must run in an environment with the full training deps (xformers, etc.)
-- e.g. on the cluster, not a laptop checkout.

Usage:
    python bytelatent/plotting/blt_flops.py \
        --base bytelatent/configs/toklens/data_optimal_fineweb/data_optimal_base_fineweb.yaml \
        --configs bytelatent/configs/toklens/data_optimal_fineweb/data_optimal_143M_T4.yaml \
                  bytelatent/configs/toklens/data_optimal_fineweb/data_optimal_143M_T8.yaml
"""
import argparse

import torch
from omegaconf import OmegaConf

from bytelatent.metrics import get_num_params
from bytelatent.model.blt import ByteLatentTransformer, ByteLatentTransformerArgs

ATTN_FLOPS_CONST = 3.5 * 4  # from attention_flops_per_token's flash-attn-benchmark formula


def load_model_args(config_paths: list[str]) -> ByteLatentTransformerArgs:
    """Merge yaml configs (later ones override earlier) and build the `model` sub-args."""
    merged = OmegaConf.merge(*[OmegaConf.load(p) for p in config_paths])
    model_cfg = OmegaConf.to_container(merged["model"], resolve=True)
    return ByteLatentTransformerArgs(**model_cfg)


def build_meta_model(args: ByteLatentTransformerArgs) -> ByteLatentTransformer:
    with torch.device("meta"):
        return ByteLatentTransformer(args)


def _attn_flops(n_layers: int, seq_len: int, window: int, dim: int, causal: bool) -> float:
    """Windowed (or full, if window >= seq_len) causal/bidirectional self-attention FLOPs."""
    eff = min(window, seq_len)
    return ATTN_FLOPS_CONST * n_layers * seq_len * eff * dim / (2 if causal else 1)


def corrected_flops_per_step(
    args: ByteLatentTransformerArgs, model: ByteLatentTransformer, n_bytes: int
) -> dict:
    n_patches = max(1, n_bytes // int(args.patch_size))

    p_enc = get_num_params(model.local_encoder) - args.vocab_size * args.dim_local_encoder
    p_dec = get_num_params(model.local_decoder) - args.vocab_size * args.dim_local_decoder
    p_global = get_num_params(model.global_transformer)

    enc_window = args.local_attention_window_len or n_bytes
    dec_window = args.local_attention_window_len or n_bytes

    flops_enc = 6 * p_enc * n_bytes + _attn_flops(
        args.n_layers_local_encoder, n_bytes, enc_window, args.dim_local_encoder, causal=True
    )
    flops_dec = 6 * p_dec * n_bytes + _attn_flops(
        args.n_layers_local_decoder, n_bytes, dec_window, args.dim_local_decoder, causal=True
    )
    flops_global = 6 * p_global * n_patches + _attn_flops(
        args.n_layers_global, n_patches, n_patches, args.dim_global, causal=True
    )

    flops_cross = 0.0
    if args.cross_attn_encoder:
        n_cross_layers = (
            args.n_layers_local_encoder if args.cross_attn_all_layers_encoder else 1
        )
        window = args.cross_attn_window_encoder or n_bytes
        # patches (queries) attend over a local window of bytes (keys/values); bidirectional.
        flops_cross += ATTN_FLOPS_CONST * n_cross_layers * n_patches * min(window, n_bytes) * args.dim_global
    if args.cross_attn_decoder:
        n_cross_layers = (
            args.n_layers_local_decoder if args.cross_attn_all_layers_decoder else 1
        )
        window = args.cross_attn_window_decoder or n_bytes
        # bytes (queries) attend over a local window of patches (keys/values); bidirectional.
        flops_cross += ATTN_FLOPS_CONST * n_cross_layers * n_bytes * min(window, n_patches) * args.dim_local_decoder

    total = flops_enc + flops_dec + flops_global + flops_cross
    return {
        "n_bytes": n_bytes,
        "n_patches": n_patches,
        "params_local_encoder": p_enc,
        "params_local_decoder": p_dec,
        "params_global": p_global,
        "flops_local_encoder": flops_enc,
        "flops_local_decoder": flops_dec,
        "flops_global": flops_global,
        "flops_cross_attn": flops_cross,
        "flops_total": total,
    }


def naive_flops_per_step(args: ByteLatentTransformerArgs, model: ByteLatentTransformer, n_bytes: int) -> float:
    """Reproduces the metric currently logged in train.py, for comparison."""
    total_params = get_num_params(model)
    non_embed = total_params - args.vocab_size * args.dim_global
    return 6 * non_embed + ATTN_FLOPS_CONST * args.n_layers_global * n_bytes * args.dim_global / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--configs", nargs="+", required=True, help="one or more T-specific configs")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=4096)
    args_cli = parser.parse_args()

    n_bytes = args_cli.batch_size * args_cli.seq_len

    rows = []
    for cfg_path in args_cli.configs:
        model_args = load_model_args([args_cli.base, cfg_path])
        model = build_meta_model(model_args)
        corrected = corrected_flops_per_step(model_args, model, n_bytes)
        naive = naive_flops_per_step(model_args, model, n_bytes)
        rows.append((cfg_path, model_args.patch_size, naive, corrected))

    print(f"{'config':<45}{'T':>4}{'naive FLOPs/step':>20}{'corrected FLOPs/step':>22}{'ratio':>10}")
    for cfg_path, t, naive, corrected in rows:
        c = corrected["flops_total"]
        print(f"{cfg_path:<45}{t:>4}{naive:>20.3e}{c:>22.3e}{c / naive:>10.3f}")


if __name__ == "__main__":
    main()
