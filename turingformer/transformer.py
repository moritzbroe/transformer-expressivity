from __future__ import annotations

from collections.abc import Hashable
from typing import Mapping, Sequence

import torch
import torch.nn as nn

MOVE_ENC_2D: dict[str, tuple[float, float]] = {
    "L": (-1.0, -1.0),
    "S": (1.0, -1.0),
    "R": (1.0, 1.0),
}


def int_to_signed_binary_lsb(n: int, width: int) -> tuple[float, ...]:
    if width <= 0:
        raise ValueError("width must be positive")
    if n < 0 or n >= (1 << width):
        raise ValueError("n must fit within the specified width")
    bits: list[float] = []
    for _ in range(width):
        bits.append(1.0 if (n & 1) else -1.0)
        n >>= 1
    return tuple(bits)


class Hardmax(nn.Module):
    def __init__(self, dim: int = -1, warn_non_unique: bool = False):
        super().__init__()
        self.dim = dim
        self.warn_non_unique = warn_non_unique

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else x.dim() + self.dim
        if dim < 0 or dim >= x.dim():
            raise ValueError(f"Hardmax received invalid dim={self.dim} for input with {x.dim()} dimensions")

        max_vals = x.amax(dim=dim, keepdim=True)
        mask = x == max_vals
        counts = mask.sum(dim=dim, keepdim=True)
        if self.warn_non_unique and torch.any(counts > 1):
            raise RuntimeError("Hardmax encountered inputs with non-unique maxima.")
        return mask.to(x.dtype) / counts.clamp(min=1).to(x.dtype)


class MLP(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.activation = nn.ReLU()
        self.num_neurons = 0
        self.lin_1: nn.Linear | None = None
        self.lin_2: nn.Linear | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_neurons == 0:
            return torch.zeros_like(x)
        assert self.lin_1 is not None
        assert self.lin_2 is not None
        return self.lin_2(self.activation(self.lin_1(x)))

    def add_neurons(self, weight_in: torch.Tensor, bias: torch.Tensor, weight_out: torch.Tensor) -> None:
        weight_in = torch.as_tensor(weight_in, dtype=torch.float32)
        weight_out = torch.as_tensor(weight_out, dtype=torch.float32)
        bias = torch.as_tensor(bias, dtype=torch.float32)

        if weight_in.ndim != 2 or weight_out.ndim != 2:
            raise ValueError("weight_in and weight_out must be rank-2 tensors")
        if weight_in.shape != weight_out.shape:
            raise ValueError("weight_in and weight_out must have the same shape")
        num_new_neurons, emb_dim = weight_in.shape
        if emb_dim != self.embedding_dim:
            raise ValueError("weight matrices must have width equal to embedding_dim")
        if bias.numel() != num_new_neurons:
            raise ValueError("bias length must match number of neurons")

        if self.num_neurons == 0:
            self.lin_1 = nn.Linear(self.embedding_dim, num_new_neurons, bias=True)
            self.lin_1.weight.data.copy_(weight_in)
            self.lin_1.bias.data.copy_(bias)
            self.lin_2 = nn.Linear(num_new_neurons, self.embedding_dim, bias=False)
            self.lin_2.weight.data.copy_(weight_out.transpose(0, 1))
            self.num_neurons = num_new_neurons
            return

        assert self.lin_1 is not None
        assert self.lin_2 is not None

        old_lin_1_weight = self.lin_1.weight.data
        old_lin_1_bias = self.lin_1.bias.data
        old_lin_2_weight = self.lin_2.weight.data

        num_total = self.num_neurons + num_new_neurons
        new_lin_1 = nn.Linear(self.embedding_dim, num_total, bias=True)
        new_lin_1.weight.data.copy_(torch.cat([old_lin_1_weight, weight_in], dim=0))
        new_lin_1.bias.data.copy_(torch.cat([old_lin_1_bias, bias], dim=0))

        new_lin_2 = nn.Linear(num_total, self.embedding_dim, bias=False)
        new_lin_2.weight.data.copy_(torch.cat([old_lin_2_weight, weight_out.transpose(0, 1)], dim=1))

        self.lin_1 = new_lin_1
        self.lin_2 = new_lin_2
        self.num_neurons = num_total


class Transformer:
    def __init__(
        self,
        *,
        vocab: Sequence[Hashable],
        registers: Sequence[tuple[str, int]],
        flags: Sequence[str],
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        if dtype is not torch.float32:
            raise ValueError("only dtype=torch.float32 is supported (construction assumes ternary arithmetic)")
        if not vocab:
            raise ValueError("vocab must be non-empty")
        if not registers:
            raise ValueError("registers must be non-empty (the first register is reserved for positional encoding)")
        if any(w <= 0 for _, w in registers):
            raise ValueError("all register widths must be positive")

        reg_names = [name for name, _ in registers]
        if len(set(reg_names)) != len(reg_names):
            raise ValueError("register names must be unique")
        if len(set(flags)) != len(flags):
            raise ValueError("flag names must be unique")
        if set(reg_names) & set(flags):
            raise ValueError("register and flag names must be disjoint")

        self.vocab = list(vocab)
        self.token_to_id = {tok: i for i, tok in enumerate(self.vocab)}
        if len(self.token_to_id) != len(self.vocab):
            raise ValueError("vocab contains duplicates")

        self.registers = list(registers)
        self.flags = list(flags)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cpu")

        self.register_slices: dict[str, slice] = {}
        self.flag_indices: dict[str, int] = {}

        offset = 0
        for name, width in self.registers:
            self.register_slices[name] = slice(offset, offset + int(width))
            offset += int(width)
        for name in self.flags:
            self.flag_indices[name] = offset
            offset += 1

        self.embedding_dim = offset
        self.pos_register_name = self.registers[0][0]
        self.pos_width = self.registers[0][1]

        self._tok_embed = torch.zeros(
            (len(self.vocab), self.embedding_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self._tok_unembed = torch.zeros(
            (len(self.vocab), self.embedding_dim),
            dtype=self.dtype,
            device=self.device,
        )

        self._heads_by_layer: dict[float, list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]]] = {}
        self._mlps_by_layer: dict[float, MLP] = {}

        self._hardmax = Hardmax(dim=-1, warn_non_unique=False)

    def _resolve_item(self, item: str | tuple[str, slice]) -> list[int]:
        if isinstance(item, str):
            if item in self.register_slices:
                sl = self.register_slices[item]
                return list(range(sl.start, sl.stop))
            if item in self.flag_indices:
                return [self.flag_indices[item]]
            raise KeyError(f"unknown register/flag {item!r}")

        name, sl = item
        if name in self.flag_indices:
            raise ValueError("flags do not support slicing")
        if name not in self.register_slices:
            raise KeyError(f"unknown register {name!r}")
        reg_sl = self.register_slices[name]
        start, stop, step = sl.indices(reg_sl.stop - reg_sl.start)
        if step != 1:
            raise ValueError("only slice step=1 is supported")
        return list(range(reg_sl.start + start, reg_sl.start + stop))

    def _resolve_many(self, items: Sequence[str | tuple[str, slice]]) -> list[int]:
        idx: list[int] = []
        for it in items:
            idx.extend(self._resolve_item(it))
        return idx

    def _ensure_mlp(self, layer: float) -> MLP:
        layer = float(layer)
        mlp = self._mlps_by_layer.get(layer)
        if mlp is None:
            mlp = MLP(self.embedding_dim).to(device=self.device, dtype=self.dtype)
            self._mlps_by_layer[layer] = mlp
        return mlp

    def set_register_embeddings(self, register: str, embeddings: Mapping[Hashable, Sequence[float]]) -> None:
        if register == self.pos_register_name:
            raise ValueError("the first register is reserved for positional encoding and cannot be token-embedded")
        idx = self._resolve_item(register)
        width = len(idx)
        for tok, vec in embeddings.items():
            if tok not in self.token_to_id:
                raise KeyError(f"unknown token {tok!r}")
            if len(vec) != width:
                raise ValueError(f"embedding for {register!r} must have length {width}")
            self._tok_embed[self.token_to_id[tok], idx] = torch.as_tensor(vec, dtype=self.dtype, device=self.device)

    def set_flag_embeddings(self, flag: str, embeddings: Mapping[Hashable, float]) -> None:
        idx = self._resolve_item(flag)
        if len(idx) != 1:
            raise ValueError("internal error: flag resolved to non-scalar")
        col = idx[0]
        for tok, val in embeddings.items():
            if tok not in self.token_to_id:
                raise KeyError(f"unknown token {tok!r}")
            self._tok_embed[self.token_to_id[tok], col] = float(val)

    def set_register_unembeddings(self, register: str, unembeddings: Mapping[Hashable, Sequence[float]]) -> None:
        idx = self._resolve_item(register)
        width = len(idx)
        for tok, vec in unembeddings.items():
            if tok not in self.token_to_id:
                raise KeyError(f"unknown token {tok!r}")
            if len(vec) != width:
                raise ValueError(f"unembedding for {register!r} must have length {width}")
            self._tok_unembed[self.token_to_id[tok], idx] = torch.as_tensor(vec, dtype=self.dtype, device=self.device)

    def set_flag_unembeddings(self, flag: str, unembeddings: Mapping[Hashable, float]) -> None:
        idx = self._resolve_item(flag)
        if len(idx) != 1:
            raise ValueError("internal error: flag resolved to non-scalar")
        col = idx[0]
        for tok, val in unembeddings.items():
            if tok not in self.token_to_id:
                raise KeyError(f"unknown token {tok!r}")
            self._tok_unembed[self.token_to_id[tok], col] = float(val)

    def add_head(self, *, layer: float, q: Sequence[str | tuple[str, slice]], k: Sequence[str | tuple[str, slice]], v: Sequence[str | tuple[str, slice]], out: str | tuple[str, slice]) -> None:
        q_idx = tuple(self._resolve_many(list(q)))
        k_idx = tuple(self._resolve_many(list(k)))
        v_idx = tuple(self._resolve_many(list(v)))
        out_idx = tuple(self._resolve_item(out))

        if len(q_idx) != len(k_idx):
            raise ValueError("query and key dimensions must match")
        if len(v_idx) != len(out_idx):
            raise ValueError("value and out dimensions must match")

        layer = float(layer)
        self._heads_by_layer.setdefault(layer, []).append((q_idx, k_idx, v_idx, out_idx))

    def add_mlp_neurons(self, *, layer: float, weight_in: torch.Tensor, bias: torch.Tensor, weight_out: torch.Tensor) -> None:
        self._ensure_mlp(layer).add_neurons(weight_in=weight_in, bias=bias, weight_out=weight_out)

    def _parse_conditions(
        self,
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]],
    ) -> tuple[list[tuple[int, int]], int]:
        conds: list[tuple[int, int]] = []
        num_pos = 0
        for item in when:
            desired = 1
            spec = item
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], int):
                spec, desired = item
            if desired not in (0, 1):
                raise ValueError("conditional values must be 0 or 1")
            idx = self._resolve_item(spec)  # type: ignore[arg-type]
            if len(idx) != 1:
                raise ValueError("conditional items must be 1-dimensional (flags or single-bit register slices)")
            if desired == 0:
                resolved = spec[0] if isinstance(spec, tuple) else spec
                if resolved in self.register_slices:
                    raise ValueError("register-bit conditions do not support desired=0; use flags for 0/1 conditions")
            conds.append((idx[0], desired))
            if desired == 1:
                num_pos += 1
        return conds, num_pos

    def mlp_copy(
        self,
        *,
        layer: float,
        src: str | tuple[str, slice],
        dst: str | tuple[str, slice],
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        src_idx = self._resolve_item(src)
        dst_idx = self._resolve_item(dst)
        if len(src_idx) != len(dst_idx):
            raise ValueError("src and dst must have the same length")
        conds, num_pos = self._parse_conditions(when)

        mlp = self._ensure_mlp(layer)
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        for s_i, d_i in zip(src_idx, dst_idx):
            w_in_pos = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_out_pos = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_in_pos[s_i] = 1.0
            w_out_pos[d_i] = 1.0
            for cond_i, desired in conds:
                w_in_pos[cond_i] += 1.0 if desired == 1 else -1.0
            bias = -float(num_pos)
            weight_in_rows.append(w_in_pos)
            weight_out_rows.append(w_out_pos)
            biases.append(bias)

            w_in_neg = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_out_neg = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_in_neg[s_i] = -1.0
            w_out_neg[d_i] = -1.0
            for cond_i, desired in conds:
                w_in_neg[cond_i] += 1.0 if desired == 1 else -1.0
            weight_in_rows.append(w_in_neg)
            weight_out_rows.append(w_out_neg)
            biases.append(bias)

        mlp.add_neurons(torch.stack(weight_in_rows), torch.tensor(biases, dtype=self.dtype, device=self.device), torch.stack(weight_out_rows))

    def mlp_zero(
        self,
        *,
        layer: float,
        target: str | tuple[str, slice],
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        idx = self._resolve_item(target)
        conds, num_pos = self._parse_conditions(when)

        mlp = self._ensure_mlp(layer)
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        for i in idx:
            w_in_pos = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_out_pos = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_in_pos[i] = 1.0
            w_out_pos[i] = -1.0
            for cond_i, desired in conds:
                w_in_pos[cond_i] += 1.0 if desired == 1 else -1.0
            bias = -float(num_pos)
            weight_in_rows.append(w_in_pos)
            weight_out_rows.append(w_out_pos)
            biases.append(bias)

            w_in_neg = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_out_neg = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            w_in_neg[i] = -1.0
            w_out_neg[i] = 1.0
            for cond_i, desired in conds:
                w_in_neg[cond_i] += 1.0 if desired == 1 else -1.0
            weight_in_rows.append(w_in_neg)
            weight_out_rows.append(w_out_neg)
            biases.append(bias)

        mlp.add_neurons(torch.stack(weight_in_rows), torch.tensor(biases, dtype=self.dtype, device=self.device), torch.stack(weight_out_rows))

    def mlp_general_map(
        self,
        *,
        layer: float,
        inputs: Sequence[str | tuple[str, slice]],
        mapping: Mapping[tuple[int, ...], int | tuple[int, ...] | list[int]],
        out: str | tuple[str, slice],
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        input_idx = self._resolve_many(list(inputs))
        out_idx = self._resolve_item(out)
        conds, num_pos = self._parse_conditions(when)

        if not mapping:
            raise ValueError("mapping must be non-empty")

        first_out = next(iter(mapping.values()))
        out_dim = len(first_out) if isinstance(first_out, (tuple, list)) else 1
        if len(out_idx) != out_dim:
            raise ValueError("out spec must match mapping output dimension")

        mlp = self._ensure_mlp(layer)
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        for in_vals, out_vals_any in mapping.items():
            if len(in_vals) != len(input_idx):
                raise ValueError("mapping keys must match total input dimension")
            if not all(v in (-1, 0, 1) for v in in_vals):
                raise ValueError("mapping keys must be ternary (-1,0,1)")

            out_vals = out_vals_any if isinstance(out_vals_any, (tuple, list)) else (out_vals_any,)
            if len(out_vals) != out_dim:
                raise ValueError("mapping outputs must have consistent dimension")
            if not all(v in (-1, 0, 1) for v in out_vals):
                raise ValueError("mapping outputs must be ternary (-1,0,1)")

            in_weights = [-1.0 if v in (-1, 0) else 1.0 for v in in_vals]
            w_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            for pos, w in enumerate(in_weights):
                w_in[input_idx[pos]] = w
            for cond_i, desired in conds:
                w_in[cond_i] = 1.0 if desired == 1 else -1.0
            bias = -float(sum(1 for v in in_vals if v in (-1, 1)) + num_pos) + 1.0

            w_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            for j, out_v in enumerate(out_vals):
                w_out[out_idx[j]] = float(out_v)

            weight_in_rows.append(w_in)
            weight_out_rows.append(w_out)
            biases.append(bias)

        mlp.add_neurons(torch.stack(weight_in_rows), torch.tensor(biases, dtype=self.dtype, device=self.device), torch.stack(weight_out_rows))

    def mlp_subtract_power_of_two(
        self,
        *,
        layer: float,
        inp: str | tuple[str, slice],
        out: str | tuple[str, slice],
        k: int = 0,
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        src_idx = self._resolve_item(inp)
        dst_idx = self._resolve_item(out)
        if len(src_idx) != len(dst_idx):
            raise ValueError("inp and out must have the same length")
        if not (0 <= k < len(src_idx)):
            raise ValueError("k must be within the bit-width of inp")
        conds, num_pos = self._parse_conditions(when)

        self.mlp_copy(layer=layer, src=inp, dst=out, when=when)

        mlp = self._ensure_mlp(layer)
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        dim = len(src_idx)

        for bit in range(k, dim):
            base_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            base_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)

            for carry_bit in range(k, bit):
                base_in[src_idx[carry_bit]] = -1.0
                base_out[dst_idx[carry_bit]] = 1.0

            base_in[src_idx[bit]] = 1.0
            base_out[dst_idx[bit]] = -1.0

            for cond_i, desired in conds:
                base_in[cond_i] += 1.0 if desired == 1 else -1.0

            bias = -float((bit - k) + num_pos)
            weight_in_rows.extend([base_in, base_in])
            weight_out_rows.extend([base_out, base_out])
            biases.extend([bias, bias])

        if k > 0:
            for bit in range(k):
                base_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
                base_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)

                for hi in range(k, dim):
                    base_in[src_idx[hi]] = -1.0
                base_in[src_idx[bit]] = 1.0
                base_out[dst_idx[bit]] = -1.0

                for cond_i, desired in conds:
                    base_in[cond_i] += 1.0 if desired == 1 else -1.0

                bias = -float((dim - k) + num_pos)
                weight_in_rows.extend([base_in, base_in])
                weight_out_rows.extend([base_out, base_out])
                biases.extend([bias, bias])

        if weight_in_rows:
            mlp.add_neurons(torch.stack(weight_in_rows), torch.tensor(biases, dtype=self.dtype, device=self.device), torch.stack(weight_out_rows))

    def mlp_subtract_power_of_two_inplace(
        self,
        *,
        layer: float,
        target: str | tuple[str, slice],
        k: int,
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        idx = self._resolve_item(target)
        if not (0 <= k < len(idx)):
            raise ValueError("k must be within the bit-width of target")
        conds, num_pos = self._parse_conditions(when)

        mlp = self._ensure_mlp(layer)
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        dim = len(idx)

        for bit in range(k, dim):
            base_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            base_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)

            for carry_bit in range(k, bit):
                base_in[idx[carry_bit]] = -1.0
                base_out[idx[carry_bit]] = 1.0

            base_in[idx[bit]] = 1.0
            base_out[idx[bit]] = -1.0

            for cond_i, desired in conds:
                base_in[cond_i] += 1.0 if desired == 1 else -1.0

            bias = -float((bit - k) + num_pos)
            weight_in_rows.extend([base_in, base_in])
            weight_out_rows.extend([base_out, base_out])
            biases.extend([bias, bias])

        if k > 0:
            for bit in range(k):
                base_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
                base_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)

                for hi in range(k, dim):
                    base_in[idx[hi]] = -1.0
                base_in[idx[bit]] = 1.0
                base_out[idx[bit]] = -1.0

                for cond_i, desired in conds:
                    base_in[cond_i] += 1.0 if desired == 1 else -1.0

                bias = -float((dim - k) + num_pos)
                weight_in_rows.extend([base_in, base_in])
                weight_out_rows.extend([base_out, base_out])
                biases.extend([bias, bias])

        if weight_in_rows:
            mlp.add_neurons(torch.stack(weight_in_rows), torch.tensor(biases, dtype=self.dtype, device=self.device), torch.stack(weight_out_rows))

    def mlp_full_subtraction(
        self,
        *,
        layer: float,
        subtrahend: str | tuple[str, slice],
        minuend_inplace: str | tuple[str, slice],
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        if not isinstance(subtrahend, str):
            raise ValueError("subtrahend must be a register name (full subtraction needs bit access via slicing)")
        sub_idx = self._resolve_item(subtrahend)
        min_idx = self._resolve_item(minuend_inplace)
        if len(sub_idx) != len(min_idx):
            raise ValueError("subtrahend and minuend_inplace must have the same length")

        for bit in range(len(sub_idx)):
            self.mlp_subtract_power_of_two_inplace(
                layer=layer + bit,
                target=minuend_inplace,
                k=bit,
                when=[*when, (subtrahend, slice(bit, bit + 1))],
            )

    def mlp_add_head_movement(
        self,
        *,
        layer: float,
        inp: str | tuple[str, slice],
        move: str | tuple[str, slice],
        out: str | tuple[str, slice],
        when: Sequence[str | tuple[str, slice] | tuple[str | tuple[str, slice], int]] = (),
    ) -> None:
        src_idx = self._resolve_item(inp)
        dst_idx = self._resolve_item(out)
        move_idx = self._resolve_item(move)
        if len(src_idx) != len(dst_idx):
            raise ValueError("inp and out must have the same length")
        if len(move_idx) != 2:
            raise ValueError("move must be a 2D register slice (encoded as in MOVE_ENC_2D)")
        conds, num_pos = self._parse_conditions(when)

        self.mlp_zero(layer=layer, target=out, when=when)
        self.mlp_copy(layer=layer, src=inp, dst=out, when=when)

        mlp = self._ensure_mlp(layer)
        weight_in_rows: list[torch.Tensor] = []
        weight_out_rows: list[torch.Tensor] = []
        biases: list[float] = []

        inc0, inc1 = MOVE_ENC_2D["R"]
        dec0, dec1 = MOVE_ENC_2D["L"]

        for bit in range(len(src_idx)):
            base_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            base_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            for cond_i, desired in conds:
                base_in[cond_i] = 1.0 if desired == 1 else -1.0
            base_in[move_idx[0]] = inc0
            base_in[move_idx[1]] = inc1

            base_in[src_idx[bit]] = -1.0
            base_out[dst_idx[bit]] = 1.0
            for carry in range(bit):
                base_in[src_idx[carry]] = 1.0
                base_out[dst_idx[carry]] = -1.0
            bias = -float(num_pos + bit + 2)
            weight_in_rows.extend([base_in, base_in])
            weight_out_rows.extend([base_out, base_out])
            biases.extend([bias, bias])

        for bit in range(len(src_idx)):
            base_in = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            base_out = torch.zeros(mlp.embedding_dim, dtype=self.dtype, device=self.device)
            for cond_i, desired in conds:
                base_in[cond_i] = 1.0 if desired == 1 else -1.0
            base_in[move_idx[0]] = dec0
            base_in[move_idx[1]] = dec1

            base_in[src_idx[bit]] = 1.0
            base_out[dst_idx[bit]] = -1.0
            for carry in range(bit):
                base_in[src_idx[carry]] = -1.0
                base_out[dst_idx[carry]] = 1.0
            bias = -float(num_pos + bit + 2)
            weight_in_rows.extend([base_in, base_in])
            weight_out_rows.extend([base_out, base_out])
            biases.extend([bias, bias])

        mlp.add_neurons(torch.stack(weight_in_rows), torch.tensor(biases, dtype=self.dtype, device=self.device), torch.stack(weight_out_rows))

    def _embed_tokens(self, tokens: Sequence[Hashable]) -> torch.Tensor:
        ids = []
        for t in tokens:
            try:
                ids.append(self.token_to_id[t])
            except KeyError:
                raise KeyError(f"unknown token {t!r}") from None
        x = self._tok_embed[torch.tensor(ids, device=self.device)]

        pos_idx = self._resolve_item(self.pos_register_name)
        for i in range(len(tokens)):
            x[i, pos_idx] = torch.as_tensor(int_to_signed_binary_lsb(i, self.pos_width), dtype=self.dtype, device=self.device)
        return x

    def forward(self, tokens: Sequence[Hashable]) -> torch.Tensor:
        x = self._embed_tokens(tokens)
        if not self._heads_by_layer and not self._mlps_by_layer:
            return x

        layers = sorted(set(self._heads_by_layer.keys()) | set(self._mlps_by_layer.keys()))
        for layer in layers:
            heads = self._heads_by_layer.get(layer, [])
            if heads:
                delta = torch.zeros_like(x)
                for q_idx, k_idx, v_idx, out_idx in heads:
                    q = x[:, list(q_idx)]
                    k = x[:, list(k_idx)]
                    v = x[:, list(v_idx)]
                    scores = q @ k.t()
                    mask = torch.tril(torch.ones((scores.size(0), scores.size(1)), device=scores.device, dtype=torch.bool))
                    scores = scores.masked_fill(~mask, float("-inf"))
                    w = self._hardmax(scores)
                    out = w @ v
                    delta[:, list(out_idx)] += out
                x = x + delta

            mlp = self._mlps_by_layer.get(layer)
            if mlp is not None:
                x = x + mlp(x)
        return x

    def predict_next(self, tokens: Sequence[Hashable]) -> Hashable:
        if not tokens:
            raise ValueError("tokens must be non-empty")
        x = self.forward(tokens)
        last = x[-1]
        scores = self._tok_unembed @ last
        best = int(scores.argmax().item())
        return self.vocab[best]

    def predict_all(self, tokens: Sequence[Hashable], *, vocab_chunk_size: int = 1024) -> list[Hashable]:
        if vocab_chunk_size <= 0:
            raise ValueError("vocab_chunk_size must be positive")
        if not tokens:
            return []

        x = self.forward(tokens)  # (n, d)
        n = x.size(0)
        v = len(self.vocab)
        best_scores = torch.full((n,), float("-inf"), dtype=self.dtype, device=self.device)
        best_ids = torch.zeros((n,), dtype=torch.long, device=self.device)

        for start in range(0, v, vocab_chunk_size):
            end = min(v, start + vocab_chunk_size)
            sub = self._tok_unembed[start:end]  # (chunk, d)
            scores = x @ sub.t()  # (n, chunk)
            chunk_best_scores, chunk_best_idx = scores.max(dim=1)
            better = chunk_best_scores > best_scores
            best_scores[better] = chunk_best_scores[better]
            best_ids[better] = start + chunk_best_idx[better]

        return [self.vocab[i] for i in best_ids.tolist()]
