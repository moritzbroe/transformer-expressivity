from __future__ import annotations

import random
from itertools import product
from typing import Dict, List, Tuple

from tm.turing import MultiTapeTuringMachine


def random_tm_and_input(
    *,
    rng: random.Random,
    num_tapes_choices: Tuple[int, ...] = (1, 2),
    num_states_range: Tuple[int, int] = (3, 6),
    sigma_choices: Tuple[str, ...] = ("a", "b", "c", "d"),
    sigma_size_range: Tuple[int, int] = (2, 3),
    input_len_range: Tuple[int, int] = (1, 12),
    blank: str = "_",
    halt_transition_prob: float = 0.05,
) -> Tuple[MultiTapeTuringMachine, str]:
    K = rng.choice(num_tapes_choices)

    min_states, max_states = num_states_range
    if min_states < 2 or max_states < min_states:
        raise ValueError("bad num_states_range")
    num_states = rng.randint(min_states, max_states)

    min_sigma, max_sigma = sigma_size_range
    if min_sigma < 1 or max_sigma < min_sigma or max_sigma > len(sigma_choices):
        raise ValueError("bad sigma_size_range")
    sigma_size = rng.randint(min_sigma, max_sigma)
    input_vocabulary = list(rng.sample(list(sigma_choices), k=sigma_size))

    band_vocabulary = input_vocabulary + [blank]

    states = [f"q{i}" for i in range(num_states)]
    initial_state = states[0]
    halting_state = "halt"
    if halting_state in states:
        raise RuntimeError("internal error: halting state name collision")

    moves = ("L", "S", "R")

    transitions: Dict[
        Tuple[str, Tuple[str, ...]],
        Tuple[str, Tuple[str, ...], Tuple[str, ...]],
    ] = {}
    for state in states:
        for read_symbols in product(band_vocabulary, repeat=K):
            if rng.random() < halt_transition_prob:
                next_state = halting_state
            else:
                next_state = rng.choice(states)
            writes = tuple(rng.choice(band_vocabulary) for _ in range(K))
            mv = tuple(rng.choice(moves) for _ in range(K))
            transitions[(state, read_symbols)] = (next_state, writes, mv)

    tm = MultiTapeTuringMachine(
        num_tapes=K,
        transitions=transitions,
        initial_state=initial_state,
        halting_state=halting_state,
        input_vocabulary=input_vocabulary,
        blank=blank,
    )

    min_len, max_len = input_len_range
    if min_len < 1 or max_len < min_len:
        raise ValueError("bad input_len_range")
    n = rng.randint(min_len, max_len)
    input_word = "".join(rng.choice(input_vocabulary) for _ in range(n))
    return tm, input_word

