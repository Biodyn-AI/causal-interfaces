"""Build the paper figures directly from sealed score artifacts.

The script intentionally reads scored JSON files rather than reimplementing
any scientific scoring.  It can therefore be used in the lightweight
scoring-only reproduction path.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

COLORS = {
    "blue": "#2868B2",
    "orange": "#D97A1E",
    "green": "#238B57",
    "red": "#C13C37",
    "gray": "#6E7781",
    "light_gray": "#D8DEE4",
}


def _load(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE_DIR / filename,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        (FIGURE_DIR / filename).with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def _panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    # Keep the panel label in the title line so it cannot overlap the y-axis
    # label when the figure is resized for the journal page.
    ax.set_title(
        f"{letter}   {title}",
        loc="left",
        fontweight="bold",
        fontsize=9.5,
        pad=7,
    )


def make_micro_figure() -> None:
    summary = _load(
        "runs/e29_14a_position_lexical_interchange_v1/"
        "position_semantics_summary.json"
    )
    forward = summary["normalized_response_summaries"]["forward"]

    condition_keys = [
        "hub_all_positions_alpha1",
        "hub_differing_lexical_alpha1",
        "hub_shared_static_alpha1",
        "control_all_positions_alpha1",
    ]
    labels = ["MLP0\nall", "MLP0\nchanged", "MLP0\nunchanged", "Attn0\nall"]
    colors = [COLORS["blue"], COLORS["green"], COLORS["gray"], COLORS["orange"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), constrained_layout=True)
    ax = axes[0]
    rng = np.random.default_rng(29141)
    for index, (key, label, color) in enumerate(zip(condition_keys, labels, colors)):
        values = np.asarray(forward[key]["per_context"], dtype=float)
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(
            np.full_like(values, index, dtype=float) + jitter,
            values,
            s=15,
            color=color,
            alpha=0.6,
            linewidths=0,
        )
        ax.plot([index - 0.18, index + 0.18], [values.mean(), values.mean()], color="black", lw=2)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.35)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("Movement toward donor answer (0 to 1)")
    ax.set_ylim(-0.08, 1.09)
    _panel_title(ax, "a", "Changed tokens carry the early MLP effect")

    ax = axes[1]
    dose = summary["claims"]["graded_dose_response"]
    alpha = np.asarray(dose["alpha"], dtype=float)
    context_rows = np.stack(
        [
            np.zeros(12),
            forward["hub_all_positions_alpha025"]["per_context"],
            forward["hub_all_positions_alpha050"]["per_context"],
            forward["hub_all_positions_alpha075"]["per_context"],
            forward["hub_all_positions_alpha1"]["per_context"],
        ],
        axis=1,
    )
    for row in context_rows:
        ax.plot(alpha, row, color=COLORS["light_gray"], lw=0.8, zorder=1)
    ax.plot(alpha, dose["mean"], color=COLORS["blue"], marker="o", lw=2.2, ms=4.5, zorder=3)
    ax.plot(alpha, alpha, color=COLORS["gray"], lw=0.9, ls=":", label="linear reference")
    ax.set_xlabel("Patch strength")
    ax.set_ylabel("Movement toward donor answer (0 to 1)")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.08, 1.09)
    _panel_title(ax, "b", "Patch strength changes the output nonlinearly")
    ax.legend(loc="upper left")
    _save(fig, "fig_micro.pdf")


def make_scale_figure() -> None:
    singleton = _load(
        "runs/e29_14a_position_lexical_interchange_v1/"
        "position_semantics_summary.json"
    )
    bridge = _load(
        "runs/e29_14b_component_position_commutation_v1/"
        "component_position_bridge_summary.json"
    )
    ladder = _load(
        "runs/e29_14c_hub_off_scale_ladder_v2/scale_scope_summary.json"
    )

    scales = np.asarray([1, 24, 32, 36, 40, 44, 47])
    intervals = [
        singleton["claims"]["forward_lexical_localization"]["paired_absolute_gap_bootstrap"],
        bridge["claims"]["hub_off_scope_bridge_by_scale"]["confirmation_size24"]
        ["paired_absolute_gap_bootstrap"],
        bridge["claims"]["hub_off_scope_bridge_by_scale"]["confirmation_size32"]
        ["paired_absolute_gap_bootstrap"],
        *[
            ladder["claims"]["scope_equivalence_by_scale"][f"scale{scale}"]
            ["paired_absolute_gap_bootstrap"]
            for scale in [36, 40, 44, 47]
        ],
    ]
    point = np.asarray([item["point_estimate"] for item in intervals])
    upper = np.asarray([item["upper_95"] for item in intervals])

    endpoint_scales = np.asarray([24, 32, 36, 40, 44, 47])
    endpoint_distance = np.asarray(
        [
            bridge["claims"]["all_position_anchor_by_scale"]["confirmation_size24"]
            ["hub_off_endpoint_absolute_gap_bootstrap"]["point_estimate"],
            bridge["claims"]["all_position_anchor_by_scale"]["confirmation_size32"]
            ["hub_off_endpoint_absolute_gap_bootstrap"]["point_estimate"],
            *[
                ladder["scale_response_geometry"][f"scale{scale}"]["all_positions"]
                ["endpoint_absolute_distance_mean"]
                for scale in [36, 40, 44, 47]
            ],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.65), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(
        scales,
        point,
        # The prospective gate uses a one-sided 95% upper bound. Draw only
        # that registered bound rather than suggesting a two-sided interval.
        yerr=np.vstack([np.zeros_like(point), upper - point]),
        color=COLORS["green"],
        marker="o",
        ms=4.5,
        lw=1.8,
        capsize=2.5,
    )
    ax.axhline(0.03, color=COLORS["red"], lw=1.0, ls="--", label="registered limit")
    ax.set_xlabel("Number of patched components")
    ax.set_ylabel("Mean difference between patches")
    ax.set_xticks(scales)
    ax.set_ylim(-0.001, 0.033)
    ax.tick_params(axis="x", labelsize=7.5)
    _panel_title(ax, "a", "Changed-only matches all-position")
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
    )

    ax = axes[1]
    ax.plot(endpoint_scales, endpoint_distance, color=COLORS["orange"], marker="o", lw=2, ms=4.5)
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.35)
    ax.set_xlabel("Patched components (first MLP omitted)")
    ax.set_ylabel("Mean absolute distance from donor")
    ax.set_xticks(endpoint_scales)
    ax.set_ylim(-0.01, 0.30)
    _panel_title(ax, "b", "Agreement appears before the ceiling")
    _save(fig, "fig_scale.pdf")


def make_backup_figure() -> None:
    summary = _load(
        "runs/e29_14d_direct_backup_v1/direct_backup_summary.json"
    )
    analyses = summary["analyses_by_scope"]["all_positions"]

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.7), constrained_layout=True)
    for ax, key, title in [
        (axes[0], "add_one", "Add one component to small patched sets"),
        (axes[1], "dropout", "Remove one component from near-complete sets"),
    ]:
        effects = {int(component): value for component, value in analyses[key]["component_effect_means"].items()}
        layers = np.arange(24)
        # Component 1 (the layer-0 MLP hub) is deliberately absent from the
        # backup experiment, so render it as a gap instead of inventing zero.
        attention = np.asarray([effects.get(2 * int(layer), np.nan) for layer in layers])
        mlp = np.asarray([effects.get(2 * int(layer) + 1, np.nan) for layer in layers])
        ax.axhline(0.0, color="black", lw=0.7, alpha=0.35)
        ax.plot(layers, attention, color=COLORS["blue"], marker="o", ms=3.5, lw=1.4, label="attention")
        ax.plot(layers, mlp, color=COLORS["gray"], marker="s", ms=3, lw=1.0, label="MLP")
        late = np.arange(18, 22)
        ax.scatter(late, attention[late], s=34, facecolor=COLORS["red"], edgecolor="white", linewidth=0.6, zorder=4)
        ax.axvspan(17.5, 21.5, color=COLORS["red"], alpha=0.08, lw=0)
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Change in donor score")
        ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
        _panel_title(ax, "a" if key == "add_one" else "b", title)
    axes[0].legend(loc="upper left")
    _save(fig, "fig_backup.pdf")


def make_replication_figure() -> None:
    labels = _load("runs/e33_1_ioi_scope_replication_v1/labels.json")
    score = _load("runs/e33_1_ioi_scope_replication_v1/score.json")
    records = labels["context_records"]

    control_conditions = [
        "mlp0_all",
        "mlp0_lexical",
        "mlp0_shared_static",
        "attention0_lexical",
        "mlp0_lexical_energy_scrambled",
    ]
    control_labels = ["MLP0\nall", "MLP0\nchanged", "MLP0\nunchanged", "Attn0\nchanged", "MLP0\nscrambled"]
    scale_conditions = ["mlp0_lexical", "first12_lexical", "all24_lexical"]
    scales = np.asarray([1, 12, 24])

    def vector(direction: str, condition: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                record["directions"][direction][condition][metric]
                for record in records
            ],
            dtype=float,
        )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75), constrained_layout=True)
    ax = axes[0]
    rng = np.random.default_rng(33_101)
    direction_style = {
        "forward": (COLORS["blue"], "o", "forward"),
        "reverse": (COLORS["orange"], "^", "reverse"),
    }
    offsets = {"forward": -0.11, "reverse": 0.11}
    for direction, (color, marker, label) in direction_style.items():
        for index, condition in enumerate(control_conditions):
            values = vector(direction, condition, "normalized_transfer")
            jitter = rng.uniform(-0.035, 0.035, len(values))
            x = index + offsets[direction]
            ax.scatter(
                np.full_like(values, x) + jitter,
                values,
                s=10,
                color=color,
                marker=marker,
                alpha=0.32,
                linewidths=0,
            )
            ax.scatter(
                [x],
                [values.mean()],
                s=35,
                color=color,
                marker=marker,
                edgecolor="white",
                linewidth=0.6,
                zorder=4,
                label=label if index == 0 else None,
            )
    ax.axhline(0.0, color="black", lw=0.7, alpha=0.35)
    ax.set_xticks(range(len(control_labels)), control_labels)
    ax.tick_params(axis="x", labelsize=7.4)
    ax.set_ylabel("Movement toward donor answer (0 to 1)")
    ax.set_ylim(-0.12, 1.08)
    _panel_title(ax, "a", "Changed names carry the early MLP effect")
    ax.legend(loc="upper right", ncol=2, handletextpad=0.3, columnspacing=0.8)

    ax = axes[1]
    width = 0.28
    transfer_means = []
    fidelity_means = []
    for condition in scale_conditions:
        transfer_means.append(
            np.mean(
                [
                    vector(direction, condition, "normalized_transfer").mean()
                    for direction in ("forward", "reverse")
                ]
            )
        )
        fidelity_means.append(
            np.mean(
                [
                    vector(direction, condition, "js_fidelity").mean()
                    for direction in ("forward", "reverse")
                ]
            )
        )
    positions = np.arange(len(scales))
    ax.bar(
        positions - width / 2,
        transfer_means,
        width,
        color=COLORS["blue"],
        alpha=0.88,
        label="answer movement",
    )
    ax.bar(
        positions + width / 2,
        fidelity_means,
        width,
        color=COLORS["green"],
        alpha=0.88,
        label="full-distribution match",
    )
    for direction, (color, marker, _label) in direction_style.items():
        for index, condition in enumerate(scale_conditions):
            ax.scatter(
                index - width / 2,
                vector(direction, condition, "normalized_transfer").mean(),
                color="white",
                marker=marker,
                s=20,
                edgecolor=color,
                linewidth=0.9,
                zorder=4,
            )
            ax.scatter(
                index + width / 2,
                vector(direction, condition, "js_fidelity").mean(),
                color="white",
                marker=marker,
                s=20,
                edgecolor=color,
                linewidth=0.9,
                zorder=4,
            )
    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.45)
    ax.set_xticks(positions, ["MLP0", "first 12", "all 24"])
    ax.set_xlabel("Patched GPT-2 Small components")
    ax.set_ylabel("Match to donor (0 to 1)")
    ax.set_ylim(0.70, 1.02)
    _panel_title(ax, "b", "The rule works across all 24 components")
    ax.legend(loc="lower right", fontsize=7.5)
    _save(fig, "fig_replication.pdf")


def main() -> None:
    _style()
    make_micro_figure()
    make_scale_figure()
    make_backup_figure()
    make_replication_figure()


if __name__ == "__main__":
    main()
