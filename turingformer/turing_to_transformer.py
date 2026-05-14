from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from tm.turing import Configuration, MultiTapeTuringMachine
from transformer import MOVE_ENC_2D, Transformer, int_to_signed_binary_lsb

INP = "inp"
EINP = "einp"
OUTP = "outp"
EOUTP = "eoutp"
POS = "pos"
EPOS = "epos"
SUMM = "summ"
ESUMM = "esumm"


Move = Literal["L", "S", "R"]
RunToken = tuple[str, tuple[str, ...], tuple[Move, ...]]
PosToken = tuple[int, ...]  # each entry is -1 or 1
TapeEntry = tuple[str, bool]  # (symbol, is_head)
TapeToken = tuple[TapeEntry, ...]  # length K
Token = Hashable


@dataclass(frozen=True)
class TMRun:
    run_tokens: tuple[RunToken, ...]
    head_positions: tuple[tuple[int, ...], ...]  # positions after each step (length t_M)
    final_configuration: Configuration

    @property
    def num_steps(self) -> int:
        return len(self.run_tokens)


def _run_tm_with_diffs(
    tm: MultiTapeTuringMachine,
    input_word: str,
    *,
    max_steps: int | None,
) -> TMRun:
    cfg = tm.initial_configuration(input_word)
    run_tokens: list[RunToken] = []
    head_positions: list[tuple[int, ...]] = []
    steps = 0
    while cfg.state != tm.halting_state:
        if max_steps is not None and steps >= max_steps:
            raise RuntimeError("too many steps")
        cfg, diff = tm.step(cfg, return_diff=True)
        run_tokens.append(diff)
        head_positions.append(tuple(tape.head for tape in cfg.tapes))
        steps += 1
    return TMRun(tuple(run_tokens), tuple(head_positions), cfg)


def _tm_output(cfg: Configuration) -> str:
    top = cfg.tapes[0].cells[:]
    while top and top[-1] == cfg.blank:
        top.pop()
    return "".join(top)


def cot_token_sequence(
    tm: MultiTapeTuringMachine,
    *,
    r: int,
    input_word: str,
    max_steps: int | None = None,
) -> list[Token]:
    if r < 2 or r % 2 != 0:
        raise ValueError("r must be an even integer >= 2")

    run = _run_tm_with_diffs(tm, input_word, max_steps=max_steps)
    t = run.num_steps
    if t <= 0:
        raise ValueError("CoT token sequence is only defined for t_M(w) >= 1")

    output_word = _tm_output(run.final_configuration)

    tokens: list[Token] = [INP, *list(input_word), EINP]

    j_max = ((t + r - 1) // r) - 1
    for j in range(j_max):
        tokens.extend(run.run_tokens[j * r : (j + 1) * r])
        tokens.append(POS)

        heads = run.head_positions[(j + 1) * r - 1]
        if len(heads) != tm.num_tapes:
            raise RuntimeError("internal error: head position arity mismatch")
        for bit in range(r):
            tok: PosToken = tuple(1 if ((heads[k] >> bit) & 1) else -1 for k in range(tm.num_tapes))
            tokens.append(tok)
        tokens.append(EPOS)

    tokens.extend(run.run_tokens[j_max * r :])
    tokens.extend([OUTP, *list(output_word), EOUTP])
    return tokens


def cot_vocab(
    tm: MultiTapeTuringMachine,
) -> list[Token]:
    vocab: list[Token] = [INP, EINP, OUTP, EOUTP, POS, EPOS]
    vocab.extend(tm.input_vocabulary)

    for q in tm.states:
        for symbols in _cartesian_power(tm.band_vocabulary, tm.num_tapes):
            for moves in _cartesian_power(("L", "S", "R"), tm.num_tapes):
                vocab.append((q, symbols, moves))

    for bits in _cartesian_power((-1, 1), tm.num_tapes):
        vocab.append(bits)
    return vocab


def scot_vocab(
    tm: MultiTapeTuringMachine,
) -> list[Token]:
    vocab = cot_vocab(tm)
    vocab.extend([SUMM, ESUMM])
    vocab.extend(tm.states)

    tape_entries: list[TapeEntry] = []
    for sym in tm.band_vocabulary:
        tape_entries.append((sym, False))
        tape_entries.append((sym, True))
    for tok in _cartesian_power(tape_entries, tm.num_tapes):
        vocab.append(tok)
    return vocab


def _pos_block(
    tm: MultiTapeTuringMachine,
    head_positions: tuple[int, ...],
    *,
    r: int,
) -> list[Token]:
    if len(head_positions) != tm.num_tapes:
        raise ValueError("head_positions arity mismatch")
    out: list[Token] = [POS]
    for bit in range(r):
        tok: PosToken = tuple(1 if ((head_positions[k] >> bit) & 1) else -1 for k in range(tm.num_tapes))
        out.append(tok)
    out.append(EPOS)
    return out


def scot_token_segments(
    tm: MultiTapeTuringMachine,
    *,
    r: int,
    input_word: str,
    max_steps: int | None = None,
) -> list[list[Token]]:
    if r < 2 or r % 2 != 0:
        raise ValueError("r must be an even integer >= 2")

    cfg = tm.initial_configuration(input_word)
    if cfg.state == tm.halting_state:
        raise ValueError("SCoT token segments are only defined for t_M(w) >= 1")

    segments: list[list[Token]] = []
    prompt: list[Token] = [INP, *list(input_word), EINP]

    s_t = len(input_word)
    steps = 0

    while True:
        trace: list[Token] = []
        run_since_pos = 0
        length_cap = 3 * (len(prompt) - 1)

        while True:
            if max_steps is not None and steps >= max_steps:
                raise RuntimeError("too many steps")

            cfg, diff = tm.step(cfg, return_diff=True)
            next_state, writes, moves = diff
            trace.append((next_state, tuple(writes), tuple(moves)))
            steps += 1
            run_since_pos += 1

            heads = tuple(tape.head for tape in cfg.tapes)
            s_t = max(s_t, *[1 + h for h in heads])
            if s_t >= (1 << r) or any(h >= (1 << r) for h in heads):
                raise ValueError("head position or summary length cannot be represented with r bits")

            if cfg.state == tm.halting_state:
                break
            if len(trace) >= length_cap:
                break
            if run_since_pos >= r:
                trace.extend(_pos_block(tm, heads, r=r))
                run_since_pos = 0

        if cfg.state == tm.halting_state:
            output_word = _tm_output(cfg)
            if any(ch not in tm.input_vocabulary for ch in output_word):
                raise ValueError("TM output contains symbols not in the input vocabulary")
            summary: list[Token] = [OUTP, *list(output_word), EOUTP]
            segments.append([*prompt, *trace, *summary])
            break

        summary = [SUMM]
        for pos in range(s_t):
            cell: list[TapeEntry] = []
            for tape in cfg.tapes:
                sym = tape.cells[pos] if pos < len(tape.cells) else tm.blank
                cell.append((sym, tape.head == pos))
            summary.append(tuple(cell))
        summary.append(cfg.state)
        summary.append(ESUMM)

        segments.append([*prompt, *trace, *summary])
        prompt = summary

    return segments


def _cartesian_power(items: Sequence[Hashable], k: int) -> list[tuple[Hashable, ...]]:
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return [()]
    out: list[tuple[Hashable, ...]] = [()]
    for _ in range(k):
        out = [prefix + (x,) for prefix in out for x in items]
    return out


def turing_machine_to_transformer(
    tm: MultiTapeTuringMachine,
    *,
    r: int,
    summarized: bool = False,
    device: torch.device | str | None = None,
) -> Transformer:
    if r < 2 or r % 2 != 0:
        raise ValueError("r must be an even integer >= 2")

    K = tm.num_tapes
    d_Q = (len(tm.states) - 1).bit_length()
    d_Gamma = (len(tm.band_vocabulary) - 1).bit_length()
    vocab = scot_vocab(tm) if summarized else cot_vocab(tm)

    def enc_Q(q: str) -> tuple[float, ...]:
        idx = tm.states.index(q)
        return int_to_signed_binary_lsb(idx, d_Q)

    def enc_Gamma(sym: str) -> tuple[float, ...]:
        idx = tm.band_vocabulary.index(sym)
        return int_to_signed_binary_lsb(idx, d_Gamma)

    def enc_Delta(mv: Move) -> tuple[float, float]:
        a, b = MOVE_ENC_2D[mv]
        return (a, b)

    # Registers (first is positional embedding register ipos).
    registers: list[tuple[str, int]] = [("ipos", r)]
    registers.extend([(f"ibit{k}", 1) for k in range(1, K + 1)])
    registers.append(("istate", d_Q))
    registers.extend([(f"isymbol{k}", d_Gamma) for k in range(1, K + 1)])
    registers.extend([(f"imove{k}", 2) for k in range(1, K + 1)])
    registers.append(("iconst", 1))

    registers.extend([(f"ihpos{k}", r) for k in range(1, K + 1)])
    registers.extend([(f"ispos{k}", r) for k in range(1, K + 1)])

    registers.append(("iposoutp", r))
    registers.extend([("ibackpos", r), ("ibackposs", r)])
    registers.extend([(f"ibitsex{k}", 1) for k in range(1, K + 1)])
    registers.extend([(f"ibitsexx{k}", 1) for k in range(1, K + 1)])
    registers.append(("iposback", r))
    registers.extend([(f"ihposback{k}", r) for k in range(1, K + 1)])

    registers.extend([(f"ipossym{k}", r) for k in range(1, K + 1)])
    registers.extend([(f"imaxpos{k}", r) for k in range(1, K + 1)])
    registers.extend([(f"isymex{k}", d_Gamma) for k in range(1, K + 1)])

    registers.extend([(f"ihpospos{k}", r) for k in range(1, K + 1)])
    registers.append(("iposbacktwo", r))
    registers.extend([(f"inextposbit{k}", 1) for k in range(1, K + 1)])
    registers.append(("istateex", d_Q))

    registers.append(("inewstate", d_Q))
    registers.extend([(f"inewsym{k}", d_Gamma) for k in range(1, K + 1)])
    registers.extend([(f"inewmove{k}", 2) for k in range(1, K + 1)])
    registers.append(("inewsymsigma", d_Gamma))

    # Flags.
    flags: list[str] = [
        "finp",
        "feinp",
        "foutp",
        "feoutp",
        "fpos",
        "fepos",
        "frun",
        "fsigma",
        "fposbits",
        "fexistsoutp",
        "finput",
        "foutput",
        "fhalt",
        "flastrun",
        "fwritepos",
        "fwriteepos",
        "fexblank",
        "fwriteeoutp",
        "fwritesigma",
        "notinp",
    ]
    flags.extend([f"fextok{k}" for k in range(1, K + 1)])
    flags.extend([f"fexistshigh{k}" for k in range(1, K + 1)])

    if summarized:
        registers.append(("ipospromptend", r))
        registers.append(("ipossumm", r))
        registers.extend([("ifinalstate", d_Q), ("ifinalstateout", d_Q)])
        registers.extend([(f"ihposfinal{k}", r) for k in range(1, K + 1)])
        registers.extend([(f"inexttapesym{k}", d_Gamma) for k in range(1, K + 1)])
        registers.extend([(f"inexttapehead{k}", 1) for k in range(1, K + 1)])
        flags.extend(
            [
                "fsumm",
                "fesumm",
                "ftape",
                "fstate",
                "fexistseinp",
                "fexistsesumm",
                "ffinalsumm",
                "ftapeinit",
                "ftapefin",
                "flengthcap",
                "fwritesumm",
                "fnexttape",
            ]
        )
        flags.extend([f"fhead{k}" for k in range(1, K + 1)])
        flags.extend([f"fnexthead{k}" for k in range(1, K + 1)])
        flags.extend([f"fbiteq{s}" for s in range(r - 2)])

    t = Transformer(vocab=vocab, registers=registers, flags=flags, device=device)

    # Embedding helpers.
    def is_run_token(tok: Token) -> bool:
        return isinstance(tok, tuple) and len(tok) == 3 and isinstance(tok[0], str) and isinstance(tok[1], tuple) and isinstance(tok[2], tuple)

    def is_pos_token(tok: Token) -> bool:
        return isinstance(tok, tuple) and len(tok) == K and all(isinstance(x, int) and x in (-1, 1) for x in tok)

    run_tokens = [tok for tok in vocab if is_run_token(tok)]
    pos_tokens = [tok for tok in vocab if is_pos_token(tok)]

    # Delimiter flags.
    t.set_flag_embeddings("finp", {INP: 1.0})
    t.set_flag_embeddings("feinp", {EINP: 1.0})
    t.set_flag_embeddings("foutp", {OUTP: 1.0})
    t.set_flag_embeddings("feoutp", {EOUTP: 1.0})
    t.set_flag_embeddings("fpos", {POS: 1.0})
    t.set_flag_embeddings("fepos", {EPOS: 1.0})

    # Type flags.
    t.set_flag_embeddings("frun", {tok: 1.0 for tok in run_tokens})
    t.set_flag_embeddings("fsigma", {tok: 1.0 for tok in tm.input_vocabulary})
    t.set_flag_embeddings("fposbits", {tok: 1.0 for tok in pos_tokens})

    # notinp: explicit replacement for (1 - finp).
    t.set_flag_embeddings("notinp", {tok: 0.0 if tok == INP else 1.0 for tok in vocab})

    # Constant register.
    t.set_register_embeddings("iconst", {tok: [1.0] for tok in vocab})

    # Position token embeddings (ibitk).
    for k in range(1, K + 1):
        t.set_register_embeddings(f"ibit{k}", {tok: [float(tok[k - 1])] for tok in pos_tokens})  # type: ignore[index]

    # State embeddings (run tokens + einp).
    t.set_register_embeddings("istate", {tok: enc_Q(tok[0]) for tok in run_tokens} | {EINP: enc_Q(tm.initial_state)})  # type: ignore[index]

    # Symbol embeddings (run tokens and k=1 input/output tokens).
    for k in range(1, K + 1):
        mapping: dict[Token, list[float]] = {}
        for tok in run_tokens:
            writes = tok[1]
            mapping[tok] = enc_Gamma(writes[k - 1])  # type: ignore[index]
        if k == 1:
            for sym in tm.input_vocabulary:
                mapping[sym] = enc_Gamma(sym)
        t.set_register_embeddings(f"isymbol{k}", mapping)

    # Movement embeddings (run tokens).
    for k in range(1, K + 1):
        mapping: dict[Token, list[float]] = {}
        for tok in run_tokens:
            moves = tok[2]
            mapping[tok] = enc_Delta(moves[k - 1])  # type: ignore[index]
        t.set_register_embeddings(f"imove{k}", mapping)

    # Initial head positions for einp (all tapes) and outp (tape 1).
    zero_r = [-1.0] * r
    for k in range(1, K + 1):
        t.set_register_embeddings(f"ihpos{k}", {EINP: zero_r})
    t.set_register_embeddings("ihpos1", {OUTP: zero_r})

    # Halt flag for run tokens in halting state.
    t.set_flag_embeddings("fhalt", {tok: 1.0 for tok in run_tokens if tok[0] == tm.halting_state})  # type: ignore[index]

    # Layer indices.
    L_1 = r // 2 + 1
    L_2 = L_1 + r + 2
    L_3 = L_2 + r + 1
    _L_total = L_3 + 4

    # --- Layer 1: distinguish input/output, set input symbol positions, prepare epos extraction helpers.
    t.add_head(layer=1, q=["iconst"], k=["foutp"], v=["foutp"], out="fexistsoutp")
    t.mlp_general_map(layer=1, inputs=["fsigma", "fexistsoutp"], mapping={(1, 1): 1}, out="foutput")
    t.mlp_general_map(layer=1, inputs=["fsigma", "fexistsoutp"], mapping={(1, 0): 1}, out="finput")
    t.mlp_subtract_power_of_two(layer=1, inp="ipos", out="ispos1", k=0, when=["fsigma", ("fexistsoutp", 0)])
    t.mlp_copy(layer=1, src="ipos", dst="iposoutp", when=["foutp"])

    t.mlp_subtract_power_of_two(layer=1, inp="ipos", out="ibackpos", k=0)
    t.mlp_subtract_power_of_two(layer=1, inp="ipos", out="ibackposs", k=1)
    t.mlp_subtract_power_of_two(layer=1, inp="ipos", out="iposback", k=0)

    # --- Layer 2: broadcast outp position, prepare output offsets, and ipossymk.
    t.add_head(layer=2, q=["iconst", "foutp", "foutp"], k=["foutp", "finp", "finp"], v=["iposoutp"], out="iposoutp")
    t.mlp_copy(layer=2, src="ipos", dst="ihpos1", when=["foutput"])
    for k in range(1, K + 1):
        t.mlp_copy(layer=2, src="ipos", dst=f"ipossym{k}", when=["frun"])
    t.mlp_copy(layer=2, src="ipos", dst="ipossym1", when=["finput"])

    # Output token offset subtraction: layers 3..r+2.
    t.mlp_full_subtraction(layer=3, subtrahend="iposoutp", minuend_inplace="ihpos1", when=["foutput"])

    # --- Head position acquisition for epos: layers 2..L_1.
    for j in range(1, r // 2 + 1):
        layer = j + 1
        for k in range(1, K + 1):
            t.add_head(layer=layer, q=["ibackpos"], k=["ipos"], v=[f"ibit{k}"], out=f"ibitsex{k}")
            t.add_head(layer=layer, q=["ibackposs"], k=["ipos"], v=[f"ibit{k}"], out=f"ibitsexx{k}")

            hi_bit = r - 2 * j + 1
            lo_bit = r - 2 * j
            t.mlp_copy(layer=layer, src=f"ibitsex{k}", dst=(f"ihpos{k}", slice(hi_bit, hi_bit + 1)), when=["fepos"])
            t.mlp_copy(layer=layer, src=f"ibitsexx{k}", dst=(f"ihpos{k}", slice(lo_bit, lo_bit + 1)), when=["fepos"])
            t.mlp_zero(layer=layer, target=f"ibitsex{k}")
            t.mlp_zero(layer=layer, target=f"ibitsexx{k}")

        t.mlp_subtract_power_of_two_inplace(layer=layer, target="ibackpos", k=1)
        t.mlp_subtract_power_of_two_inplace(layer=layer, target="ibackposs", k=1)

    # --- Propagate head positions through run tokens: layers L_1+1 .. L_1+r+1.
    for j in range(1, r + 2):
        layer = L_1 + j
        for k in range(1, K + 1):
            t.add_head(layer=layer, q=["iposback"], k=["ipos"], v=[f"ihpos{k}"], out=f"ihposback{k}")
            t.mlp_add_head_movement(layer=layer, inp=f"ihposback{k}", move=f"imove{k}", out=f"ihpos{k}", when=["frun"])
            t.mlp_zero(layer=layer, target=f"ihpos{k}", when=["fpos"])
            t.mlp_copy(layer=layer, src=f"ihposback{k}", dst=f"ihpos{k}", when=["fpos"])
            if j <= r:
                t.mlp_zero(layer=layer, target=f"ihposback{k}")

    # Layer L_2: copy symbol positions for run tokens (isposk) from ihposbackk.
    for k in range(1, K + 1):
        t.mlp_zero(layer=L_2, target=f"ispos{k}", when=["frun"])
        t.mlp_copy(layer=L_2, src=f"ihposback{k}", dst=f"ispos{k}", when=["frun"])

    # --- Positional blocks setup at layer L_2.
    for k in range(1, K + 1):
        t.mlp_copy(layer=L_2, src=f"ihpos{k}", dst=f"ihpospos{k}", when=["fpos"])
        t.mlp_copy(layer=L_2, src=(f"ihpos{k}", slice(0, 1)), dst=f"inextposbit{k}", when=["fpos"])
    t.mlp_subtract_power_of_two(layer=L_2, inp="ipos", out="iposbacktwo", k=0)

    # --- fextok for each tape at layer L_2+1.
    for k in range(1, K + 1):
        t.add_head(
            layer=L_2 + 1,
            q=[f"ihpos{k}", *["iconst"] * (r - 1)],
            k=[f"ispos{k}", *["finp"] * (r - 1)],
            v=["notinp"],
            out=f"fextok{k}",
        )

    # --- Binary search for imaxposk: layers L_2+1 .. L_2+r.
    for j in range(0, r):
        layer = L_2 + j + 1
        b = r - 1 - j

        for k in range(1, K + 1):
            t.add_head(
                layer=layer,
                q=[
                    f"ihpos{k}",
                    *["iconst"] * (r + j),
                    "iconst",
                    (f"imaxpos{k}", slice(b + 1, r)),
                ],
                k=[
                    f"ispos{k}",
                    *["finp"] * (r + j),
                    (f"ipossym{k}", slice(b, r)),
                ],
                v=["notinp"],
                out=f"fexistshigh{k}",
            )
            t.mlp_general_map(
                layer=layer,
                inputs=[f"fexistshigh{k}", f"fextok{k}"],
                mapping={(1, 1): 1, (0, 1): -1},
                out=(f"imaxpos{k}", slice(b, b + 1)),
            )
            t.mlp_zero(layer=layer, target=f"fexistshigh{k}")

    # --- Symbol extraction and state extraction: layer L_3 (= L_2+r+1).
    for k in range(1, K + 1):
        t.add_head(
            layer=L_3,
            q=[f"imaxpos{k}", "iconst"],
            k=[f"ipossym{k}", "finp"],
            v=[f"isymbol{k}"],
            out=f"isymex{k}",
        )
        t.mlp_general_map(
            layer=L_3,
            inputs=[f"fextok{k}", "fpos", "fposbits", "finput"],
            mapping={(0, 0, 0, 0): tuple(enc_Gamma(tm.blank))},
            out=f"isymex{k}",
        )

    # Positional bit extraction for p=1..r-1: layers L_2+1 .. L_2+r-1.
    for p in range(1, r):
        layer = L_2 + p
        for k in range(1, K + 1):
            t.add_head(
                layer=layer,
                q=["iposbacktwo"],
                k=["ipos"],
                v=[(f"ihpospos{k}", slice(p, p + 1))],
                out=f"inextposbit{k}",
            )
        t.mlp_subtract_power_of_two_inplace(layer=layer, target="iposbacktwo", k=0)

    # Mark the last position token (r positions after POS) to output EPOS: layer L_2+r.
    t.add_head(layer=L_2 + r, q=["iposbacktwo"], k=["ipos"], v=["fpos"], out="fwriteepos")

    # In the same layer, decrement iposbacktwo by 2 so it points r+2 tokens back.
    t.mlp_subtract_power_of_two_inplace(layer=L_2 + r, target="iposbacktwo", k=1)

    # Mark r-th run tokens (for POS output) using iposbacktwo when it points r-1 back: layer L_2+r-1.
    t.add_head(layer=L_2 + r - 1, q=["iposbacktwo"], k=["ipos"], v=["frun"], out="flastrun")
    t.mlp_general_map(layer=L_2 + r - 1, inputs=["flastrun", "frun", "fhalt"], mapping={(1, 1, 0): 1}, out="fwritepos")

    # Extract state from the last run token for EPOS: layer L_3.
    t.add_head(layer=L_3, q=["iposbacktwo"], k=["ipos"], v=["istate"], out="istateex")
    t.mlp_zero(layer=L_3, target="istate", when=["fepos"])
    t.mlp_copy(layer=L_3, src="istateex", dst="istate", when=["fepos"])

    # --- Transition function and output logic: layers L_3+1 .. L_3+4.
    # Layer L_3+1: compute transition and detect blank for output termination.
    trans_layer = L_3 + 1
    input_idx: list[int] = []
    state_sl = t.register_slices["istate"]
    input_idx.extend(range(state_sl.start, state_sl.stop))
    for k in range(1, K + 1):
        sym_sl = t.register_slices[f"isymex{k}"]
        input_idx.extend(range(sym_sl.start, sym_sl.stop))

    if input_idx:
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        for (q, read_syms), (q_next, writes, moves) in tm.transitions.items():
            in_bits: list[float] = []
            in_bits.extend(enc_Q(q))
            for sym in read_syms:
                in_bits.extend(enc_Gamma(sym))

            w_in = torch.zeros(t.embedding_dim, dtype=torch.float32, device=t.device)
            for pos, val in enumerate(in_bits):
                w_in[input_idx[pos]] = float(val)

            # singleneuronmlp bias: -(sum_i |I_i|) + 1
            bias = -float(len(in_bits)) + 1.0

            w_out = torch.zeros(t.embedding_dim, dtype=torch.float32, device=t.device)
            # new state
            out_sl = t.register_slices["inewstate"]
            w_out[out_sl] = torch.tensor(enc_Q(q_next), dtype=torch.float32, device=t.device)
            # new symbols + moves
            for k in range(1, K + 1):
                sym_sl = t.register_slices[f"inewsym{k}"]
                w_out[sym_sl] = torch.tensor(enc_Gamma(writes[k - 1]), dtype=torch.float32, device=t.device)
                mv_sl = t.register_slices[f"inewmove{k}"]
                w_out[mv_sl] = torch.tensor(enc_Delta(moves[k - 1]), dtype=torch.float32, device=t.device)

            weight_in_rows.append(w_in)
            weight_out_rows.append(w_out)
            biases.append(bias)

        if weight_in_rows:
            t.add_mlp_neurons(
                layer=trans_layer,
                weight_in=torch.stack(weight_in_rows),
                bias=torch.tensor(biases, dtype=torch.float32, device=t.device),
                weight_out=torch.stack(weight_out_rows),
            )

    # fexblank := 1{isymex1 == blank}
    t.mlp_general_map(layer=trans_layer, inputs=["isymex1"], mapping={tuple(enc_Gamma(tm.blank)): 1}, out="fexblank")

    # Layer L_3+2: zero out inew* for POS/OUTP decisions, and set fwriteeoutp.
    layer = L_3 + 2
    for reg in ["inewstate", *[f"inewsym{k}" for k in range(1, K + 1)], *[f"inewmove{k}" for k in range(1, K + 1)]]:
        t.mlp_zero(layer=layer, target=reg, when=["fhalt"])
        t.mlp_zero(layer=layer, target=reg, when=["fwritepos"])
    t.mlp_general_map(layer=layer, inputs=["foutput", "foutp", "fexblank"], mapping={(1, 0, 1): 1, (0, 1, 1): 1}, out="fwriteeoutp")

    # Layer L_3+3: fwritesigma := 1{fexistsoutp=1 and fwriteeoutp=0}
    t.mlp_general_map(layer=L_3 + 3, inputs=["fexistsoutp", "fwriteeoutp"], mapping={(1, 0): 1}, out="fwritesigma")

    # Layer L_3+4: copy isymex1 -> inewsymsigma when fwritesigma.
    t.mlp_zero(layer=L_3 + 4, target="inewsymsigma", when=["fwritesigma"])
    t.mlp_copy(layer=L_3 + 4, src="isymex1", dst="inewsymsigma", when=["fwritesigma"])

    if summarized:
        def is_tape_token(tok: Token) -> bool:
            if not isinstance(tok, tuple) or len(tok) != K:
                return False
            for entry in tok:
                if not isinstance(entry, tuple) or len(entry) != 2:
                    return False
                sym, is_head = entry
                if not isinstance(sym, str) or not isinstance(is_head, bool):
                    return False
            return True

        tape_tokens = [tok for tok in vocab if is_tape_token(tok)]
        state_tokens = list(tm.states)

        # --- SCoT embeddings.
        t.set_flag_embeddings("fsumm", {SUMM: 1.0})
        t.set_flag_embeddings("fesumm", {ESUMM: 1.0})
        t.set_flag_embeddings("ftape", {tok: 1.0 for tok in tape_tokens})
        t.set_flag_embeddings("fstate", {q: 1.0 for q in state_tokens})
        for k in range(1, K + 1):
            t.set_flag_embeddings(f"fhead{k}", {tok: 1.0 for tok in tape_tokens if tok[k - 1][1]})  # type: ignore[index]

        t.set_register_embeddings("istate", {q: enc_Q(q) for q in state_tokens})
        for k in range(1, K + 1):
            t.set_register_embeddings(f"isymbol{k}", {tok: enc_Gamma(tok[k - 1][0]) for tok in tape_tokens})  # type: ignore[index]

        # --- Layer 1: segment structure flags + base token rewrite for initial SUMM.
        t.add_head(layer=1, q=["iconst"], k=["feinp"], v=["feinp"], out="fexistseinp")
        t.add_head(layer=1, q=["iconst"], k=["fesumm"], v=["fesumm"], out="fexistsesumm")

        t.mlp_general_map(
            layer=1,
            inputs=["fsumm", "fexistseinp", "fexistsesumm"],
            mapping={(1, 1, 0): 1, (1, 0, 1): 1, (1, 1, 1): 1},
            out="ffinalsumm",
        )
        t.mlp_general_map(layer=1, inputs=["fsumm", "fexistseinp", "fexistsesumm"], mapping={(1, 0, 0): 1}, out="finp")
        t.mlp_general_map(layer=1, inputs=["fsumm", "fexistseinp", "fexistsesumm"], mapping={(1, 0, 0): -1}, out="notinp")

        t.mlp_general_map(layer=1, inputs=["ftape", "fexistseinp", "fexistsesumm"], mapping={(1, 0, 0): 1}, out="ftapeinit")
        t.mlp_general_map(
            layer=1,
            inputs=["ftape", "fexistseinp", "fexistsesumm"],
            mapping={(1, 1, 0): 1, (1, 0, 1): 1, (1, 1, 1): 1},
            out="ftapefin",
        )

        t.mlp_copy(layer=1, src="ipos", dst="ipospromptend", when=["feinp"])
        t.mlp_copy(layer=1, src="ipos", dst="ipospromptend", when=["fesumm"])

        # --- Layer 2: broadcast prompt end position for length-cap detection.
        t.add_head(
            layer=2,
            q=["iconst", "iconst", "feinp", "feinp", "fesumm", "fesumm"],
            k=["feinp", "fesumm", "finp", "finp", "finp", "finp"],
            v=["ipospromptend"],
            out="ipospromptend",
        )

        # Initial summary tape tokens: set symbol positions + token positions so they act as base tape writes.
        for k in range(1, K + 1):
            t.mlp_subtract_power_of_two(layer=2, inp="ipos", out=f"ispos{k}", k=0, when=["ftapeinit"])
            t.mlp_copy(layer=2, src="ipos", dst=f"ipossym{k}", when=["ftapeinit"])

        # Length-cap marker bit comparisons: ipos[s+2] == ipospromptend[s] for s=0..r-3.
        for s in range(r - 2):
            t.mlp_general_map(
                layer=2,
                inputs=[("ipos", slice(s + 2, s + 3)), ("ipospromptend", slice(s, s + 1))],
                mapping={(-1, -1): 1, (1, 1): 1},
                out=f"fbiteq{s}",
            )

        # AND over all fbiteq flags, plus (ipos mod 4 == 0), plus (ipospromptend < 2^(r-2)),
        # to get flengthcap exactly at ipos == 4 * ipospromptend (no wraparound).
        w_in = torch.zeros(t.embedding_dim, dtype=torch.float32, device=t.device)
        for s in range(r - 2):
            w_in[t.flag_indices[f"fbiteq{s}"]] = 1.0
        ipos_sl = t.register_slices["ipos"]
        w_in[ipos_sl.start + 0] = -1.0
        w_in[ipos_sl.start + 1] = -1.0
        ipromptend_sl = t.register_slices["ipospromptend"]
        w_in[ipromptend_sl.start + (r - 2)] = -1.0
        w_in[ipromptend_sl.start + (r - 1)] = -1.0
        w_out = torch.zeros(t.embedding_dim, dtype=torch.float32, device=t.device)
        w_out[t.flag_indices["flengthcap"]] = 1.0
        t.add_mlp_neurons(
            layer=3,
            weight_in=w_in.unsqueeze(0),
            bias=torch.tensor([-float(r + 1)], dtype=torch.float32, device=t.device),
            weight_out=w_out.unsqueeze(0),
        )

        # Propagate flengthcap to all later tokens.
        t.add_head(layer=4, q=["iconst", "flengthcap", "flengthcap"], k=["flengthcap", "finp", "finp"], v=["flengthcap"], out="flengthcap")

        # fwritesumm: at run tokens once length cap is reached (OUTP still wins via fhalt).
        t.mlp_general_map(layer=4, inputs=["frun", "flengthcap", "fhalt"], mapping={(1, 1, 0): 1}, out="fwritesumm")

        # Ensure POS emission and run-token emissions are suppressed when we output SUMM.
        t.mlp_zero(layer=L_2 + r, target="fwritepos", when=["fwritesumm"])
        for reg in ["inewstate", *[f"inewsym{k}" for k in range(1, K + 1)], *[f"inewmove{k}" for k in range(1, K + 1)]]:
            t.mlp_zero(layer=L_3 + 2, target=reg, when=["fwritesumm"])
            t.mlp_zero(layer=L_3 + 2, target=reg, when=["fstate"])

        # --- Continuing from a summary: set head positions and state at ESUMM.
        for k in range(1, K + 1):
            t.add_head(layer=3, q=["fesumm", "fesumm", "iconst"], k=[f"fhead{k}", f"fhead{k}", "finp"], v=[f"ispos{k}"], out=f"ihpos{k}")
        t.add_head(layer=3, q=["fesumm", "fesumm", "iconst"], k=["fstate", "fstate", "finp"], v=["istate"], out="istate")

        # --- Final summary: broadcast SUMM position to compute (ipos - ipos_summ).
        t.mlp_copy(layer=2, src="ipos", dst="ipossumm", when=["ffinalsumm"])
        t.add_head(layer=3, q=["iconst", "ffinalsumm", "ffinalsumm"], k=["ffinalsumm", "finp", "finp"], v=["ipossumm"], out="ipossumm")

        # Prepare per-token summary offsets in ihpos{k} for final SUMM + final tape tokens.
        for k in range(1, K + 1):
            t.mlp_copy(layer=3, src="ipos", dst=f"ihpos{k}", when=["ffinalsumm"])
            t.mlp_copy(layer=3, src="ipos", dst=f"ihpos{k}", when=["ftapefin"])
            t.mlp_full_subtraction(layer=4, subtrahend="ipossumm", minuend_inplace=f"ihpos{k}", when=["ffinalsumm"])
            t.mlp_full_subtraction(layer=4, subtrahend="ipossumm", minuend_inplace=f"ihpos{k}", when=["ftapefin"])

        # Extract final state + head positions from the preceding run token at the final SUMM token.
        t.add_head(layer=L_2, q=["iposback"], k=["ipos"], v=["istate"], out="ifinalstate")
        for k in range(1, K + 1):
            t.add_head(layer=L_2, q=["iposback"], k=["ipos"], v=[f"ihpos{k}"], out=f"ihposfinal{k}")
            t.mlp_zero(layer=L_2 + 1, target=f"ihposfinal{k}", when=[("ffinalsumm", 0)])
        t.mlp_zero(layer=L_2 + 1, target="ifinalstate", when=[("ffinalsumm", 0)])

        # Broadcast the final state to all subsequent tokens.
        # Use the same query/key pattern as iposoutp to prevent the SUMM token from attending to itself.
        t.add_head(layer=L_2 + 2, q=["iconst", "ffinalsumm", "ffinalsumm"], k=["ffinalsumm", "finp", "finp"], v=["ifinalstate"], out="ifinalstate")

        # Head position marker flags for the next tape token: 1{(ipos-ipos_summ) == hpos_final^k}.
        for k in range(1, K + 1):
            t.add_head(
                layer=L_2 + 2,
                q=[f"ihpos{k}", *["iconst"] * (r - 1)],
                k=[f"ihposfinal{k}", *["finp"] * (r - 1)],
                v=["notinp"],
                out=f"fnexthead{k}",
            )

        # fnexttape := OR_k fextok{k} OR fnexthead{k}.
        sum_idx = [t.flag_indices[f"fextok{k}"] for k in range(1, K + 1)]
        sum_idx.extend([t.flag_indices[f"fnexthead{k}"] for k in range(1, K + 1)])
        w_in = torch.zeros(t.embedding_dim, dtype=torch.float32, device=t.device)
        for idx in sum_idx:
            w_in[idx] = 1.0
        w_out = torch.zeros(t.embedding_dim, dtype=torch.float32, device=t.device)
        w_out[t.flag_indices["fnexttape"]] = 1.0
        t.add_mlp_neurons(
            layer=L_2 + 3,
            weight_in=torch.stack([w_in, w_in]),
            bias=torch.tensor([0.0, -1.0], dtype=torch.float32, device=t.device),
            weight_out=torch.stack([w_out, -w_out]),
        )

        # Final summary token emissions: either another tape token (when fnexttape=1) or the state (when fnexttape=0).
        t.mlp_copy(layer=L_2 + 4, src="ifinalstate", dst="ifinalstateout", when=["ffinalsumm", ("fnexttape", 0)])
        t.mlp_copy(layer=L_2 + 4, src="ifinalstate", dst="ifinalstateout", when=["ftapefin", ("fnexttape", 0)])

        for k in range(1, K + 1):
            t.mlp_copy(layer=L_3 + 1, src=f"isymex{k}", dst=f"inexttapesym{k}", when=["ffinalsumm", "fnexttape"])
            t.mlp_copy(layer=L_3 + 1, src=f"isymex{k}", dst=f"inexttapesym{k}", when=["ftapefin", "fnexttape"])
            t.mlp_general_map(
                layer=L_3 + 1,
                inputs=[f"fnexthead{k}"],
                mapping={(0,): -1, (1,): 1},
                out=f"inexttapehead{k}",
                when=["ffinalsumm", "fnexttape"],
            )
            t.mlp_general_map(
                layer=L_3 + 1,
                inputs=[f"fnexthead{k}"],
                mapping={(0,): -1, (1,): 1},
                out=f"inexttapehead{k}",
                when=["ftapefin", "fnexttape"],
            )

        # --- SCoT unembeddings.
        t.set_flag_unembeddings("fwritesumm", {SUMM: 1.0})
        t.set_flag_unembeddings("fstate", {ESUMM: 1.0})

        t.set_register_unembeddings("ifinalstateout", {q: enc_Q(q) for q in state_tokens})
        for k in range(1, K + 1):
            t.set_register_unembeddings(f"inexttapesym{k}", {tok: enc_Gamma(tok[k - 1][0]) for tok in tape_tokens})  # type: ignore[index]
            t.set_register_unembeddings(f"inexttapehead{k}", {tok: [1.0 if tok[k - 1][1] else -1.0] for tok in tape_tokens})  # type: ignore[index]

    # --- Unembeddings.
    t.set_flag_unembeddings("fhalt", {OUTP: 1.0})
    t.set_flag_unembeddings("fwritepos", {POS: 1.0})

    t.set_register_unembeddings("inewstate", {tok: enc_Q(tok[0]) for tok in run_tokens})  # type: ignore[index]
    for k in range(1, K + 1):
        t.set_register_unembeddings(f"inewsym{k}", {tok: enc_Gamma(tok[1][k - 1]) for tok in run_tokens})  # type: ignore[index]
        t.set_register_unembeddings(f"inewmove{k}", {tok: enc_Delta(tok[2][k - 1]) for tok in run_tokens})  # type: ignore[index]

    t.set_flag_unembeddings("fwriteepos", {EPOS: 1.0})
    for k in range(1, K + 1):
        t.set_register_unembeddings(f"inextposbit{k}", {tok: [float(tok[k - 1])] for tok in pos_tokens})  # type: ignore[index]

    t.set_flag_unembeddings("fwriteeoutp", {EOUTP: 1.0})
    t.set_register_unembeddings("inewsymsigma", {sym: enc_Gamma(sym) for sym in tm.input_vocabulary})

    return t


def turing_machine_to_cot_transformer(
    tm: MultiTapeTuringMachine,
    *,
    r: int,
    device: torch.device | str | None = None,
) -> Transformer:
    return turing_machine_to_transformer(tm, r=r, summarized=False, device=device)


def turing_machine_to_scot_transformer(
    tm: MultiTapeTuringMachine,
    *,
    r: int,
    device: torch.device | str | None = None,
) -> Transformer:
    return turing_machine_to_transformer(tm, r=r, summarized=True, device=device)
