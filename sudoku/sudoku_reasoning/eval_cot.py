"""Evaluate CoT models using vLLM for efficient long-sequence generation."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import List, Sequence

from vllm import LLM, SamplingParams

from sudoku_reasoning.precision_utils import add_fp32_arg, configure_fp32_mode
from sudoku_reasoning.puzzle_utils import (
    build_prompt_tokens,
    extract_board_from_output,
    is_valid_solution,
    load_puzzles,
    parse_targets,
    sample_by_targets,
    sample_puzzles,
)
from sudoku_reasoning.tokenizer import SudokuTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CoT models with vLLM")
    add_fp32_arg(parser)
    parser.add_argument("checkpoint", help="Path to saved model directory")
    parser.add_argument(
        "--data-path",
        default="data/test.txt",
        help="Text file with one puzzle per line (default: data/test.txt)",
    )
    parser.add_argument(
        "--cot-lengths-path",
        default="data/test_cot_lengths.json",
        help="JSON file with precomputed CoT lengths (default: data/test_cot_lengths.json)",
    )
    # Sampling options (mutually exclusive)
    parser.add_argument("--count", type=int, default=None, help="Number of puzzles to randomly sample")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random sampling")
    parser.add_argument("--targets", type=str, default=None, help="Comma-separated target CoT lengths, e.g. 1024,2048,4096")
    parser.add_argument("--targets-logspace", nargs=3, metavar=("START", "END", "N"), help="Log-spaced targets: START END N")
    parser.add_argument("--per-target", type=int, default=100, help="Puzzles per target (default: 100)")
    # Generation options
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Maximum tokens to generate per puzzle")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-puzzle progress logs")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization (default: 0.9)")
    return parser.parse_args()


def analyze_generation(gen_ids: List[int], puzzle: Sequence[int], tokenizer: SudokuTokenizer) -> str:
    output_token_indices = [i for i, t in enumerate(gen_ids) if t == tokenizer.output_id]
    if not output_token_indices:
        return "no_output"
    if len(output_token_indices) > 1:
        return "trash"
    output_end_indices = [i for i, t in enumerate(gen_ids) if t == tokenizer.output_end_id]
    if not output_end_indices:
        return "incomplete"
    if len(output_end_indices) > 1:
        return "trash"
    if not output_end_indices[0] > output_token_indices[0]:
        return "trash"

    output_ids = gen_ids[output_token_indices[0] + 1 : output_end_indices[0]]
    if len(output_ids) % 2 != 0:
        return "trash"

    triples = []
    for i in range(len(output_ids) // 2):
        cell = tokenizer.decode_token(output_ids[2 * i])
        value = tokenizer.decode_token(output_ids[2 * i + 1])
        if not (isinstance(cell, tuple) and len(cell) == 2 and isinstance(value, int)):
            return "trash"
        triples.append((cell[0], cell[1], value))

    candidate = extract_board_from_output(triples)
    if candidate is None:
        return "trash"
    if not is_valid_solution(puzzle, candidate):
        return "invalid"
    return "ok"


def main():
    args = parse_args()
    configure_fp32_mode(args.fp32)
    puzzles = load_puzzles(args.data_path)
    if not puzzles:
        raise SystemExit("No puzzles found")
    tokenizer = SudokuTokenizer()

    cot_lengths = json.loads(Path(args.cot_lengths_path).read_text())

    # Determine sampling mode
    targets = parse_targets(args.targets, args.targets_logspace)
    sample_targets = None

    if targets:
        # Target-based sampling
        samples, sample_targets = sample_by_targets(puzzles, cot_lengths, targets, args.per_target)
    elif args.count is not None:
        # Random sampling
        sampled = sample_puzzles(args.data_path, args.count, seed=args.seed)
        samples = [(idx, p, cot_lengths["".join(str(c) for c in p)]) for idx, p in sampled]
    else:
        # All puzzles
        samples = [(idx, p, cot_lengths["".join(str(c) for c in p)]) for idx, p in enumerate(puzzles)]

    # Shuffle to mix short/long sequences for better vLLM batching
    indices = list(range(len(samples)))
    random.Random(args.seed).shuffle(indices)
    samples = [samples[i] for i in indices]
    if sample_targets:
        sample_targets = [sample_targets[i] for i in indices]

    llm = LLM(
        model=args.checkpoint,
        skip_tokenizer_init=True,
        dtype="float32" if args.fp32 else "auto",
        max_model_len=args.max_new_tokens + 256,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.output_end_id],
        temperature=args.temperature if args.do_sample else 0.0,
    )

    # Build all prompts
    prompt_ids = [tokenizer.tokens_to_ids(build_prompt_tokens(p, tokenizer)) for _, p, _ in samples]

    prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Track results
    solved = 0
    results = []  # (correct_cot, is_solved, target_or_none)
    for i, ((row_idx, puzzle, correct_cot), output, prompt) in enumerate(zip(samples, outputs, prompt_ids)):
        gen_ids = list(output.outputs[0].token_ids)
        total_cot = len(prompt) + len(gen_ids)
        status = analyze_generation(gen_ids, puzzle, tokenizer)
        is_solved = status == "ok"
        if is_solved:
            solved += 1
        target = sample_targets[i] if sample_targets else None
        results.append((correct_cot, is_solved, target))
        if not args.quiet:
            label = "OK" if is_solved else f"FAIL ({status})"
            print(f"row {row_idx}: {label} (total_cot: {total_cot}, correct_cot: {correct_cot})")

    print()

    # Print per-target statistics if targets were used
    if targets and sample_targets:
        print("Per-target results:")
        by_target = defaultdict(list)
        for correct_cot, is_solved, target in results:
            by_target[target].append((correct_cot, is_solved))

        for target in targets:
            entries = by_target[target]
            if not entries:
                continue
            lengths = [c for c, _ in entries]
            n_solved = sum(1 for _, s in entries if s)
            n_total = len(entries)
            acc = n_solved / n_total * 100 if n_total else 0
            print(
                f"  target={target}: n={n_total}, "
                f"min={min(lengths)}, max={max(lengths)}, "
                f"mean={mean(lengths):.1f}, median={median(lengths):.1f}, "
                f"solved={n_solved}/{n_total}, acc={acc:.2f}%"
            )
        print()

    print(f"Accuracy: {solved / len(samples) * 100:.2f}%")


if __name__ == "__main__":
    main()
