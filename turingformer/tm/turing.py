from typing import Dict, List, Optional, Tuple


class Tape:
    def __init__(self, blank: str, cells: List[str]) -> None:
        self.head = 0
        self.blank = blank
        self.cells = cells

    def read(self) -> str:
        return self.cells[self.head]

    def write(self, symbol: str) -> None:
        self.cells[self.head] = symbol

    def move(self, direction: str) -> None:
        if direction == "L":
            if self.head > 0:
                self.head -= 1
            return
        if direction == "R":
            self.head += 1
            if self.head == len(self.cells):
                self.cells.append(self.blank)
            return
        if direction == "S":
            return
        raise ValueError(f"bad move {direction}")


class Configuration:
    def __init__(self, state: str, tapes: List[Tape], blank: str) -> None:
        self.state = state
        self.tapes = tapes
        self.blank = blank

    def __str__(self) -> str:
        right = max(len(tape.cells) for tape in self.tapes) - 1
        positions = range(0, right + 1)

        lines = [f"state {self.state}"]
        for index, tape in enumerate(self.tapes):
            head_marks = ["^" if position == tape.head else " " for position in positions]
            symbols = [tape.cells[position] if position < len(tape.cells) else self.blank for position in positions]
            lines.append(" ".join(head_marks))
            lines.append(" ".join(symbols))
        return "\n".join(lines)


class MultiTapeTuringMachine:
    def __init__(
        self,
        num_tapes: int,
        transitions: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, Tuple[str, ...], Tuple[str, ...]]],
        initial_state: str,
        halting_state: str,
        input_vocabulary: List[str],
        blank: str = "_",
    ) -> None:
        # band vocabulary and states are inferred from transitions table
        if num_tapes <= 0:
            raise ValueError("num_tapes must be positive")
        if initial_state == halting_state:
            raise ValueError("initial_state and halting_state must be different")
        if not input_vocabulary:
            raise ValueError("input_vocabulary must be non-empty")
        if len(set(input_vocabulary)) != len(input_vocabulary):
            raise ValueError("input_vocabulary must not contain duplicates")
        if blank in input_vocabulary:
            raise ValueError("blank symbol must not be in input_vocabulary")

        for (state, read_symbols), (next_state, writes, moves) in transitions.items():
            if len(read_symbols) != num_tapes:
                raise ValueError(f"transition read symbols arity mismatch for state={state!r}")
            if len(writes) != num_tapes:
                raise ValueError(f"transition write symbols arity mismatch for state={state!r}")
            if len(moves) != num_tapes:
                raise ValueError(f"transition moves arity mismatch for state={state!r}")
            for mv in moves:
                if mv not in ("L", "S", "R"):
                    raise ValueError(f"bad move {mv!r} in transition for state={state!r}")

        self.num_tapes = num_tapes
        self.transitions = transitions
        self.initial_state = initial_state
        self.halting_state = halting_state
        self.blank = blank
        self.states = self._infer_states()
        self.input_vocabulary = input_vocabulary
        self.band_vocabulary = self._infer_band_vocabulary()
        band_vocab_set = set(self.band_vocabulary)
        assert blank in band_vocab_set, "blank symbol must be in band vocabulary"

    def _infer_states(self) -> List[str]:
        states: List[str] = [self.initial_state]
        for state, _ in self.transitions.keys():
            if state not in states:
                states.append(state)
        for next_state, _, _ in self.transitions.values():
            if next_state not in states:
                states.append(next_state)
        if self.halting_state not in states:
            states.append(self.halting_state)
        return states

    def _infer_band_vocabulary(self) -> List[str]:
        symbols = [self.blank] + self.input_vocabulary.copy()
        for _, read_symbols in self.transitions.keys():
            for symbol in read_symbols:
                if symbol not in symbols:
                    symbols.append(symbol)
        for _, writes, _ in self.transitions.values():
            for symbol in writes:
                if symbol not in symbols:
                    symbols.append(symbol)
        return list(symbols)

    def initial_configuration(self, input_word: str) -> Configuration:
        primary_cells = list(input_word)
        if not primary_cells:
            primary_cells = [self.blank]
        tapes = [Tape(self.blank, primary_cells)]
        for _ in range(self.num_tapes - 1):
            tapes.append(Tape(self.blank, [self.blank]))
        return Configuration(self.initial_state, tapes, self.blank)

    def step(self, configuration: Configuration, return_diff: bool = False):
        if configuration.state == self.halting_state:
            return (configuration, (configuration.state, tuple(t.read() for t in configuration.tapes), tuple("S" for _ in configuration.tapes))) if return_diff else configuration

        read_symbols = tuple(tape.read() for tape in configuration.tapes)
        next_state, writes, moves = self.transitions[(configuration.state, read_symbols)]

        for tape, symbol in zip(configuration.tapes, writes):
            tape.write(symbol)
        for tape, move in zip(configuration.tapes, moves):
            tape.move(move)

        configuration.state = next_state
        if return_diff:
            return configuration, (next_state, writes, moves)
        return configuration

    def run(self, input_word: str, max_steps: Optional[int] = None, log: bool = False) -> str:
        configuration = self.initial_configuration(input_word)
        steps = 0
        while True:
            if log:
                print(f"step={steps}")
                print(configuration)
                print("===============")
            if configuration.state == self.halting_state:
                break
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError("too many steps")
            configuration = self.step(configuration)
            steps += 1
        top = configuration.tapes[0].cells[:]
        while top and top[-1] == self.blank:
            top.pop()
        return "".join(top), steps

