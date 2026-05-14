from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

from sudoku_reasoning.tokenizer import SudokuTokenizer, Token

Coord = Tuple[int, int]
TraceEntry = Tuple[Coord, Sequence[int]]
Assignment = Tuple[Coord, int]
Payload = Union[int, Sequence[int]]


def to_tokens(part: Iterable[Tuple[Coord, Payload]], tokenizer: SudokuTokenizer) -> List[Token]:
    tokens: List[Token] = []
    for coord, payload in part:
        tokens.append(coord)
        if isinstance(payload, (list, tuple)):
            if payload:
                tokens.extend(payload)
            else:
                tokens.append(tokenizer.none_token)
        else:
            tokens.append(payload)
    return tokens


def build_input_tokens(inputs: Sequence[Assignment], tokenizer: SudokuTokenizer) -> List[Token]:
    return [tokenizer.input_token] + to_tokens(inputs, tokenizer) + [tokenizer.input_end_token]


def build_output_tokens(
    outputs: Sequence[Assignment] | None, tokenizer: SudokuTokenizer
) -> List[Token]:
    if outputs is None:
        return []
    return [tokenizer.output_token] + to_tokens(outputs, tokenizer) + [tokenizer.output_end_token]


def build_cot_trace_tokens(
    trace: Sequence[TraceEntry], tokenizer: SudokuTokenizer
) -> List[Token]:
    return to_tokens(trace, tokenizer)


def build_cot_tokens(
    inputs: Sequence[Assignment],
    trace: Sequence[TraceEntry],
    outputs: Sequence[Assignment] | None,
    tokenizer: SudokuTokenizer,
) -> List[Token]:
    return (
        build_input_tokens(inputs, tokenizer)
        + build_cot_trace_tokens(trace, tokenizer)
        + build_output_tokens(outputs, tokenizer)
    )


def build_cot_segment(
    inputs: Sequence[Assignment],
    trace: Sequence[TraceEntry],
    outputs: Sequence[Assignment] | None,
    tokenizer: SudokuTokenizer,
) -> dict:
    input_tokens = build_input_tokens(inputs, tokenizer)
    trace_tokens = build_cot_trace_tokens(trace, tokenizer)
    output_tokens = build_output_tokens(outputs, tokenizer)
    full_tokens = input_tokens + trace_tokens + output_tokens
    loss_mask = [0] * len(input_tokens) + [1] * (len(full_tokens) - len(input_tokens))
    return {"input_ids": tokenizer.tokens_to_ids(full_tokens), "loss_mask": loss_mask}
