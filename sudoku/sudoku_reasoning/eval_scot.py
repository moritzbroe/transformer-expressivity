import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import GPTNeoXForCausalLM

from sudoku_reasoning.precision_utils import add_fp32_arg, get_precision
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
    parser = argparse.ArgumentParser(description="Evaluate Sudoku decoder checkpoints (generation)")
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
    parser.add_argument("--batch-size", type=int, default=8, help="Number of active puzzles per generation step")
    parser.add_argument("--max-new-tokens", type=int, default=4096, help="Maximum tokens to generate per puzzle")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature when --do-sample is set")
    parser.add_argument(
        "--max-segment-length",
        type=int,
        default=None,
        help="If a segment exceeds this length, it breaks off and the sample is marked as fail. Used for SCoT so one failed sample doesn't slow down the whole batch.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-puzzle progress logs")
    return parser.parse_args()


def analyze_generation(gen_ids, puzzle, tokenizer: SudokuTokenizer):
    gen_ids = list(gen_ids)
    output_token_indices = [idx for idx, tok_id in enumerate(gen_ids) if tok_id == tokenizer.output_id]
    if not output_token_indices:
        return "no_output"
    if len(output_token_indices) > 1:
        return "trash"
    output_end_token_indices = [idx for idx, tok_id in enumerate(gen_ids) if tok_id == tokenizer.output_end_id]
    if not output_end_token_indices:
        return "incomplete"
    if len(output_end_token_indices) > 1:
        return "trash"
    if not output_end_token_indices[0] > output_token_indices[0]:
        return "trash"
    output_token_ids = gen_ids[output_token_indices[0] + 1 : output_end_token_indices[0]]
    triples = []
    if len(output_token_ids) % 2 != 0:
        return "trash"

    for i in range(len(output_token_ids) // 2):
        cell_token_id = output_token_ids[2 * i]
        value_token_id = output_token_ids[2 * i + 1]
        cell_token = tokenizer.decode_token(cell_token_id)
        value_token = tokenizer.decode_token(value_token_id)
        if not (
            isinstance(cell_token, tuple)
            and isinstance(cell_token[0], int)
            and isinstance(cell_token[1], int)
            and isinstance(value_token, int)
        ):
            return "trash"
        triples.append((cell_token[0], cell_token[1], value_token))

    candidate = extract_board_from_output(triples)
    if candidate is None:
        return "trash"
    if not is_valid_solution(puzzle, candidate):
        return "invalid"
    return "ok"


def build_prompt_ids(puzzle: Sequence[int], tokenizer: SudokuTokenizer) -> List[int]:
    prompt_tokens = build_prompt_tokens(puzzle, tokenizer)
    return tokenizer.tokens_to_ids(prompt_tokens)


def eval_segments(
    model: GPTNeoXForCausalLM,
    samples: Sequence[Tuple[int, Sequence[int], int, Optional[int]]],
    tokenizer: SudokuTokenizer,
    batch_size: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    max_segment_length: Optional[int],
    log_progress: bool,
) -> List[Tuple[int, bool, Optional[int]]]:
    """Returns list of (correct_cot, is_solved, target) for each sample."""
    samples = list(samples)
    batch_size = max(batch_size, 1)

    model.eval()
    device = next(model.parameters()).device
    segment_cap = max_segment_length if max_segment_length and max_segment_length > 0 else max_new_tokens

    not_started = []
    active = []
    for row_idx, puzzle, cot_len, target in samples:
        not_started.append({"row_idx": row_idx, "puzzle": puzzle, "cot_len": cot_len, "target": target})

    results = []
    iteration = 0
    while len(not_started) > 0 or len(active) > 0:
        iteration += 1
        while len(active) < batch_size and len(not_started) > 0:
            row = not_started.pop(0)
            row_idx = row["row_idx"]
            puzzle = row["puzzle"]
            cot_len = row["cot_len"]
            target = row["target"]
            prompt = build_prompt_ids(puzzle, tokenizer)
            active.append(
                {
                    "row_idx": row_idx,
                    "puzzle": puzzle,
                    "tokens": prompt,
                    "generated": 0,
                    "segments": 0,
                    "total_tokens": 0,
                    "cot_len": cot_len,
                    "target": target,
                }
            )
        max_context = max(len(state["tokens"]) for state in active)
        input_ids = torch.full(
            (len(active), max_context),
            tokenizer.pad_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for i, state in enumerate(active):
            ctx = state["tokens"]
            ctx_len = len(ctx)
            input_ids[i, -ctx_len:] = torch.tensor(ctx, dtype=torch.long, device=device)
            attention_mask[i, -ctx_len:] = 1

        sequences = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=segment_cap,
            eos_token_id=[tokenizer.summary_end_id, tokenizer.output_end_id],
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_id,
        )

        next_active: List[dict] = []
        for state, generated in zip(active, sequences):
            state["segments"] += 1
            new_tokens = generated.tolist()[max_context:]
            while new_tokens and new_tokens[-1] == tokenizer.pad_id:
                new_tokens.pop()

            status: Optional[str] = None

            summary_start = None
            has_summary = False
            if new_tokens and new_tokens[-1] == tokenizer.summary_end_id:
                if new_tokens.count(tokenizer.summary_id) == 1:
                    summary_start = new_tokens.index(tokenizer.summary_id)
                    has_summary = True

            state["generated"] += len(new_tokens)
            state["total_tokens"] += len(state["tokens"]) + len(new_tokens)

            if not new_tokens:
                status = "segment_length_exceeded"
            elif tokenizer.output_end_id not in new_tokens and tokenizer.summary_end_id not in new_tokens:
                status = "segment_length_exceeded"

            if status is None:
                last_token = new_tokens[-1]
                if state["generated"] >= max_new_tokens and last_token != tokenizer.output_end_id:
                    status = "total_length_exceeded"
                elif last_token == tokenizer.output_end_id:
                    status = analyze_generation(new_tokens, state["puzzle"], tokenizer)
                elif last_token == tokenizer.summary_end_id:
                    if not has_summary or summary_start is None:
                        status = "trash"
                    else:
                        state["tokens"] = new_tokens[summary_start:]
                        if state["generated"] >= max_new_tokens:
                            status = "total_length_exceeded"
                        else:
                            next_active.append(state)
                        status = None
                else:
                    status = "trash"

            if status is None:
                continue

            is_solved = status == "ok"
            results.append((state["cot_len"], is_solved, state["target"]))

            if log_progress:
                label = "OK" if is_solved else f"FAIL ({status})"
                print(
                    f"row {state['row_idx']}: {label} "
                    f"(total_scot: {state['total_tokens']}, correct_cot: {state['cot_len']})"
                )

        if log_progress and iteration % 100 == 0:
            print(f"iteration {iteration}, segments: {[s['segments'] for s in active]}")
        active = next_active

    return results


def main():
    args = parse_args()
    puzzles = load_puzzles(args.data_path)
    if not puzzles:
        raise SystemExit("No puzzles found to evaluate")
    tokenizer = SudokuTokenizer()

    cot_lengths = json.loads(Path(args.cot_lengths_path).read_text())

    # Determine sampling mode
    targets = parse_targets(args.targets, args.targets_logspace)
    sample_targets = None

    if targets:
        # Target-based sampling
        samples, sample_targets = sample_by_targets(puzzles, cot_lengths, targets, args.per_target)
        # Add target to each sample
        samples = [(idx, p, cot, sample_targets[i]) for i, (idx, p, cot) in enumerate(samples)]
    elif args.count is not None:
        # Random sampling
        sampled = sample_puzzles(args.data_path, args.count, seed=args.seed)
        samples = [(idx, p, cot_lengths["".join(str(c) for c in p)], None) for idx, p in sampled]
    else:
        # All puzzles
        samples = [(idx, p, cot_lengths["".join(str(c) for c in p)], None) for idx, p in enumerate(puzzles)]

    precision = get_precision(force_fp32=args.fp32)
    dtype = (
        torch.bfloat16
        if precision == "bf16"
        else torch.float16
        if precision == "fp16"
        else torch.float32
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        dtype = torch.float32

    model = GPTNeoXForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype)
    model.config.pad_token_id = tokenizer.pad_id
    model.to(device)

    total = len(samples)

    results = eval_segments(
        model,
        samples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        log_progress=not args.quiet,
        max_segment_length=args.max_segment_length,
    )

    solved = sum(1 for _, is_solved, _ in results if is_solved)

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

    accuracy = solved / total if total else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
