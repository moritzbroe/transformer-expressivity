"""Tokenizer for standard 9x9 Sudoku grids."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union


Token = Union[str, Tuple[int, int], int]


GRID_SIZE = 9


class SudokuTokenizer:
    """Builds a vocabulary for 9x9 Sudoku."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.input_token = "<input>"
        self.input_end_token = "</input>"
        self.summary_token = "<summary>"
        self.summary_end_token = "</summary>"
        self.output_token = "<output>"
        self.output_end_token = "</output>"
        self.none_token = "<none>"
        self.special_tokens = [
            self.pad_token,
            self.input_token,
            self.input_end_token,
            self.summary_token,
            self.summary_end_token,
            self.output_token,
            self.output_end_token,
            self.none_token,
        ]
        self.coord_tokens: List[Tuple[int, int]] = [
            (r, c) for r in range(1, GRID_SIZE + 1) for c in range(1, GRID_SIZE + 1)
        ]
        self.digit_tokens: List[int] = list(range(1, GRID_SIZE + 1))
        self.vocab: List[Token] = self.special_tokens + self.coord_tokens + self.digit_tokens
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

        self.pad_id = self.token_to_id[self.pad_token]
        self.input_id = self.token_to_id[self.input_token]
        self.input_end_id = self.token_to_id[self.input_end_token]
        self.summary_id = self.token_to_id[self.summary_token]
        self.summary_end_id = self.token_to_id[self.summary_end_token]
        self.output_id = self.token_to_id[self.output_token]
        self.output_end_id = self.token_to_id[self.output_end_token]
        self.none_id = self.token_to_id[self.none_token]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def decode_token(self, tid: int) -> Token | None:
        return self.id_to_token.get(int(tid))

    def decode_ids(self, ids: Iterable[int]) -> str:
        parts: List[str] = []
        for tid in ids:
            tok = self.decode_token(tid)
            if tok is None:
                parts.append(f"<unk:{tid}>")
            elif isinstance(tok, tuple) and len(tok) == 2:
                parts.append(f"({tok[0]},{tok[1]})")
            else:
                parts.append(str(tok))
        return " ".join(parts)

    def tokens_to_ids(self, tokens: Sequence[Token]) -> List[int]:
        return [self.token_to_id[token] for token in tokens]


__all__ = ["SudokuTokenizer"]
