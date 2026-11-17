"""Small tensor-train surrogates for Boolean intervention response surfaces.

Each component mask bit selects one slice of a tensor-train core.  Contracting
the selected slices leaves a short latent state, which is read out into all
prompt contexts at once.  The model therefore represents interactions of every
order without explicitly enumerating the Boolean cube or its polynomial terms.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch


def count_tensor_train_parameters(
    component_count: int, output_count: int, rank: int
) -> int:
    """Return the exact trainable parameter count used by ``BinaryTensorTrain``."""
    if component_count < 2 or output_count < 1 or rank < 1:
        raise ValueError("invalid tensor-train dimensions")
    return (
        2 * rank
        + (component_count - 1) * 2 * rank * rank
        + rank * output_count
    )


class BinaryTensorTrain(torch.nn.Module):
    """An open-boundary tensor train with a shared multi-context readout."""

    def __init__(
        self,
        *,
        component_count: int,
        output_count: int,
        rank: int,
        component_order: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if component_count < 2 or output_count < 1 or rank < 1:
            raise ValueError("invalid tensor-train dimensions")
        if component_order is None:
            component_order = torch.arange(component_count)
        component_order = torch.as_tensor(component_order, dtype=torch.long)
        if sorted(component_order.tolist()) != list(range(component_count)):
            raise ValueError("component_order must be a permutation")
        self.component_count = component_count
        self.output_count = output_count
        self.rank = rank
        self.register_buffer("component_order", component_order.clone())

        self.first_core = torch.nn.Parameter(torch.empty(2, rank))
        self.middle_cores = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.empty(2, rank, rank))
                for _ in range(component_count - 1)
            ]
        )
        self.output_core = torch.nn.Parameter(torch.empty(rank, output_count))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize a stable near-identity contraction with broken symmetry."""
        torch.nn.init.normal_(self.first_core, mean=0.0, std=0.2)
        identity = torch.eye(
            self.rank,
            dtype=self.first_core.dtype,
            device=self.first_core.device,
        )
        for core in self.middle_cores:
            with torch.no_grad():
                core.copy_(identity.expand(2, -1, -1))
                core.add_(0.02 * torch.randn_like(core))
        torch.nn.init.normal_(self.output_core, mean=0.0, std=0.1)

    def latent_state(self, masks: torch.Tensor) -> torch.Tensor:
        """Contract all Boolean cores and return the final bond state."""
        if masks.ndim != 2 or masks.shape[1] != self.component_count:
            raise ValueError(
                f"masks must have shape (rows, {self.component_count})"
            )
        if not bool(torch.logical_or(masks == 0, masks == 1).all()):
            raise ValueError("masks must be binary")
        ordered = masks[:, self.component_order].to(dtype=torch.long)
        state = self.first_core[ordered[:, 0]]
        for component_index, core in enumerate(self.middle_cores, start=1):
            selected = core[ordered[:, component_index]]
            state = torch.einsum("bi,bij->bj", state, selected)
        return state

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Contract the tensor train and apply its multi-context readout."""
        return self.latent_state(masks) @ self.output_core


def build_cached_mask_split(
    mask_groups: Mapping[str, torch.Tensor], *, seed: int
) -> dict[str, torch.Tensor]:
    """Build the frozen response-blind split used by the cached E29.4 analysis."""
    required = {"low", "size4", "size8"}
    if set(mask_groups) != required:
        raise ValueError(f"mask_groups must have exactly the keys {sorted(required)}")
    low = mask_groups["low"]
    size4 = mask_groups["size4"]
    size8 = mask_groups["size8"]
    if low.shape != (96, 24) or size4.shape != (128, 24) or size8.shape != (128, 24):
        raise ValueError("cached mask groups have unexpected shapes")
    generator = torch.Generator().manual_seed(seed)
    size4_order = torch.randperm(128, generator=generator)
    size8_order = torch.randperm(128, generator=generator)
    return {
        "train_low": torch.arange(96),
        "train_size4": size4_order[:64],
        "validation_size4": size4_order[64:96],
        "test_size4": size4_order[96:],
        "train_size8": size8_order[:32],
        "validation_size8": size8_order[32:64],
        "test_size8": size8_order[64:],
    }


def _positive_square_root(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    clipped = eigenvalues.clamp_min(0.0)
    return (eigenvectors * clipped.sqrt().unsqueeze(0)) @ eigenvectors.mT


def schmidt_spectra(model: BinaryTensorTrain) -> tuple[torch.Tensor, ...]:
    """Return normalized core-only singular spectra across all depth cuts.

    These spectra measure information transported by the tensor-train state.
    """
    right_by_cut: dict[int, torch.Tensor] = {}
    right_gram = model.output_core @ model.output_core.mT
    for component_index in range(model.component_count - 1, 0, -1):
        core = model.middle_cores[component_index - 1]
        right_gram = sum(
            core[bit] @ right_gram @ core[bit].mT for bit in range(2)
        )
        right_by_cut[component_index - 1] = right_gram

    left_gram = model.first_core.mT @ model.first_core
    spectra = []
    for cut in range(model.component_count - 1):
        left_root = _positive_square_root(left_gram)
        balanced = left_root @ right_by_cut[cut] @ left_root
        squared = torch.linalg.eigvalsh(0.5 * (balanced + balanced.mT))
        singular_values = squared.clamp_min(0.0).sqrt().flip(0)
        norm = torch.linalg.vector_norm(singular_values)
        spectra.append(singular_values / norm.clamp_min(torch.finfo(norm.dtype).eps))
        if cut < model.component_count - 2:
            core = model.middle_cores[cut]
            left_gram = sum(
                core[bit].mT @ left_gram @ core[bit] for bit in range(2)
            )
    return tuple(spectra)


def mobius_latent_state(
    model: BinaryTensorTrain, masks: torch.Tensor
) -> torch.Tensor:
    """Return latent Möbius coefficients via a local tensor-core transform.

    For each Boolean mode, the value slices ``(G[0], G[1])`` become
    ``(G[0], G[1] - G[0])``.  This is an invertible change of local basis, so it
    preserves every tensor-train bond dimension while producing all subset
    Möbius coefficients.  Negating non-singleton coefficients gives the xINT
    convention of Vaidyanathan et al. (2026).
    """
    if masks.ndim != 2 or masks.shape[1] != model.component_count:
        raise ValueError(
            f"masks must have shape (rows, {model.component_count})"
        )
    if not bool(torch.logical_or(masks == 0, masks == 1).all()):
        raise ValueError("masks must be binary")
    ordered = masks[:, model.component_order].long()
    first_slices = torch.stack(
        (model.first_core[0], model.first_core[1] - model.first_core[0])
    )
    state = first_slices[ordered[:, 0]]
    for component_index, core in enumerate(model.middle_cores, start=1):
        transformed = torch.stack((core[0], core[1] - core[0]))
        state = torch.einsum(
            "bi,bij->bj", state, transformed[ordered[:, component_index]]
        )
    return state
