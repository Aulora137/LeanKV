"""Search space definition for LeanKV autoresearch (6 knobs)."""

import math
from dataclasses import dataclass
from itertools import product
from typing import List, Optional

from turboquant.lloyd_max import BITS_TO_LEVELS


@dataclass(frozen=True)
class QuantConfig:
    k_bits: float
    v_bits: float
    rotation: str
    group_size: int
    layer_policy: str
    use_qjl: bool
    seed: int = 42

    @property
    def name(self) -> str:
        qjl = "+QJL" if self.use_qjl else ""
        rot = "H" if self.rotation == "hadamard" else "RH"
        gs = f"g{self.group_size}"
        lp = {"uniform": "", "more_bits_later": "_LateB", "more_bits_first": "_EarlyB"}[self.layer_policy]
        return f"K{self.k_bits}V{self.v_bits}_{rot}_{gs}{lp}{qjl}"

    def _log2_levels(self, bits: float) -> float:
        n = BITS_TO_LEVELS.get(bits, round(2 ** bits))
        return math.log2(n)

    def effective_k_bits(self, head_dim: int) -> float:
        gs = min(self.group_size, head_dim)
        total = self._log2_levels(self.k_bits)
        if self.use_qjl:
            total += 1.0 + 32.0 / head_dim
        total += 32.0 / gs  # per-group scale
        return total

    def effective_v_bits(self, head_dim: int) -> float:
        gs = min(self.group_size, head_dim)
        return self._log2_levels(self.v_bits) + 32.0 / gs

    def total_bits_per_kv_pair(self, head_dim: int) -> float:
        return self.effective_k_bits(head_dim) + self.effective_v_bits(head_dim)

    def get_k_bits_per_layer(self, n_layers: int) -> List[float]:
        return _apply_layer_policy(self.k_bits, n_layers, self.layer_policy)

    def get_v_bits_per_layer(self, n_layers: int) -> List[float]:
        return _apply_layer_policy(self.v_bits, n_layers, self.layer_policy)


# Next higher supported bits for layer policy stepping
_NEXT_BITS = {2: 2.5, 2.5: 3, 3: 3.125, 3.125: 3.5, 3.5: 4, 4: 4}


def _apply_layer_policy(base_bits: float, n_layers: int, policy: str) -> List[float]:
    if policy == "uniform":
        return [base_bits] * n_layers
    half = n_layers // 2
    higher = _NEXT_BITS.get(base_bits, base_bits)
    if policy == "more_bits_later":
        return [base_bits] * half + [higher] * (n_layers - half)
    elif policy == "more_bits_first":
        return [higher] * half + [base_bits] * (n_layers - half)
    return [base_bits] * n_layers


K_BITS = [2, 2.5, 3, 3.125, 3.5, 4]
V_BITS = [2, 2.5, 3, 3.125, 3.5, 4]
ROTATIONS = ["hadamard", "randomized_hadamard"]
GROUP_SIZES = [16, 32, 64, 128]
LAYER_POLICIES = ["uniform", "more_bits_later", "more_bits_first"]
QJL_OPTIONS = [False, True]


def generate_search_space(seed: int = 42) -> List[QuantConfig]:
    configs = []
    for k_bits, v_bits, rotation, group_size, layer_policy, use_qjl in product(
        K_BITS, V_BITS, ROTATIONS, GROUP_SIZES, LAYER_POLICIES, QJL_OPTIONS
    ):
        configs.append(QuantConfig(
            k_bits=k_bits, v_bits=v_bits, rotation=rotation,
            group_size=group_size, layer_policy=layer_policy,
            use_qjl=use_qjl, seed=seed,
        ))
    return configs
