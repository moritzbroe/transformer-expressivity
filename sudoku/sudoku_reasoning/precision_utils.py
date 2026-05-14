import argparse

import torch


def add_fp32_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Force full fp32 on GPU and disable TF32 instead of using the default mixed precision.",
    )
    return parser


def configure_fp32_mode(enabled: bool = False) -> None:
    if not enabled:
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def get_precision(force_fp32: bool = False):
    configure_fp32_mode(force_fp32)
    if force_fp32:
        if torch.cuda.is_available():
            print("Using fp32 precision on CUDA (TF32 disabled).")
        else:
            print("CUDA not available, using fp32.")
        return "fp32"
    if not torch.cuda.is_available():
        print("CUDA not available, using fp32.")
        return "fp32"
    elif torch.cuda.is_bf16_supported():
        print("Using bf16 precision.")
        return "bf16"
    else:
        print("Using fp16 precision.")
        return "fp16"


__all__ = ["add_fp32_arg", "configure_fp32_mode", "get_precision"]
