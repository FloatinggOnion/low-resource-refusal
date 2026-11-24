"""Cross-size summary figures (aggregate charts only: no prompts, responses or labels).

  PYTHONPATH=src uv run python -m lrr.summary_figs outputs figures

Reads outputs/<tag>/analysis/results_numbers.json (primary judge) and, for Figure 4,
outputs/<tag>/analysis_judge_second/results_numbers.json (second judge).
English and the Yoruba cells come from the Yoruba run (<model>); Swahili and Hausa from <model>-swh-hau.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .analysis import COL, GRID, INK, INK2, LAB_COL, MUTED, SURF, style

SIZES = ("4B", "8B", "14B")
RUNS = {"4B": ("Qwen3-4B", "Qwen3-4B-swh-hau"), "8B": ("Qwen3-8B", "Qwen3-8B-swh-hau"), "14B": ("Qwen3-14B", "Qwen3-14B-swh-hau")}
CELLS = ("eng", "yor", "yor-stripped", "swh", "hau", "hau-stripped")
LABEL = {"eng": "English", "yor": "Yoruba", "yor-stripped": "Yoruba,\nstripped", "swh": "Swahili", "hau": "Hausa", "hau-stripped": "Hausa,\nstripped"}
# ordinal one-hue ramp for model size (light = small): ramp steps 250 / 450 / 650, all clear 2:1 on the surface
SIZE_COL = {"4B": "#86b6ef", "8B": "#2a78d6", "14B": "#104281"}


def load(root: Path, judge_dir: str = "analysis") -> dict:
    out = {}
    for size, (t_yor, t_sh) in RUNS.items():
        out[size] = {"yor": json.load(open(root / t_yor / judge_dir / "results_numbers.json")), "sh": json.load(open(root / t_sh / judge_dir / "results_numbers.json"))}
    return out


def src(runs: dict, size: str, cell: str) -> dict:
    return runs[size]["yor" if cell in ("eng", "yor", "yor-stripped") else "sh"]


def grouped_bars(ax, vals, errs=None, group_labels=None, width=0.26):
    """vals[size][i] over groups i; errs[size][i] = (lo, hi) or None."""
    n = len(next(iter(vals.values())))
    x = np.arange(n)
    for k, size in enumerate(SIZES):
        pos = x + (k - 1) * width
        ax.bar(pos, vals[size], width=width - 0.02, color=SIZE_COL[size], label=size)
        if errs:
            for p, e in zip(pos, errs[size]):
                if e:
                    ax.plot([p, p], e, color=INK2, lw=1)
    ax.set_xticks(x, group_labels, fontsize=8)


def fig1(runs, out: Path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, (key, title, chance) in zip(axes, (("belebele", "Belebele reading accuracy", 0.25), ("u1", "U1 source-matching accuracy", 0.25))):
        vals, errs = {}, {}
        for size in SIZES:
            vs, es = [], []
            for cell in CELLS:
                r = src(runs, size, cell)["C"][key][cell]
                e = r["acc"]
                vs.append(e["point"])
                es.append(tuple(e["ci95"]))
            vals[size], errs[size] = vs, es
        grouped_bars(ax, vals, errs, [LABEL[c] for c in CELLS])
        ax.axhline(chance, color=MUTED, lw=1)
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
        ax.set_title(f"{title}\ngrey line = chance, 25%", fontsize=10)
    axes[0].legend(frameon=False, title="Qwen3 size", fontsize=8, title_fontsize=8, loc="upper right")
    fig.suptitle("Comprehension (C): reading stays near chance in Yoruba and Hausa; Swahili improves with size", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_comprehension.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig2(runs, out: Path):
    style()
    fig, ax = plt.subplots(figsize=(10, 3.9))
    groups = ["eng", "eng-shuf", "yor", "yor-stripped", "swh", "hau", "hau-stripped"]
    labels = ["English", "English,\nwords shuffled", "Yoruba", "Yoruba,\nstripped", "Swahili", "Hausa", "Hausa,\nstripped"]
    vals, errs = {}, {}
    for size in SIZES:
        vs, es = [], []
        for g in groups:
            if g == "eng-shuf":
                vs.append(runs[size]["yor"]["H"]["shuffle_control"]["eng"]["auroc_shuffled_mean"])
                es.append(None)
            else:
                e = src(runs, size, g)["H"]["cells"][g]
                vs.append(e["auroc_chosen_layer"])
                es.append(tuple(e["ci95"]))
        vals[size], errs[size] = vs, es
    grouped_bars(ax, vals, errs, labels)
    bow = runs["4B"]["yor"]["H"]["surface_baselines"]["eng"]["bow_auroc"]
    ax.axhline(0.5, color=MUTED, lw=1)
    ax.axhline(bow, color=INK2, lw=1)
    ax.text(6.55, bow + 0.008, f"bag-of-words classifier on English, {bow:.2f}", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0.3, 1.05)
    ax.set_ylabel("AUROC, harmful vs benign\n(grey line = chance, 0.5)")
    ax.legend(frameon=False, title="Qwen3 size", fontsize=8, title_fontsize=8, loc="upper right", ncol=3)
    ax.set_title("Harm recognition (H): the English harm direction does not transfer, and most of its English signal survives word shuffling", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig2_harm_recognition.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig3(runs, out: Path):
    style()
    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    width, gap = 0.26, 0.5
    xs, tick_x, tick_l = [], [], []
    for i, cell in enumerate(CELLS):
        base = i * (3 * width + gap)
        for k, size in enumerate(SIZES):
            x = base + k * width
            lr = src(runs, size, cell)["R"]["label_rates"][f"{cell}/harm"]
            bottom = 0.0
            for lab in ("Refused", "Jailbroken", "Deflected"):
                v = lr[lab]["point"]
                ax.bar(x, v, bottom=bottom, width=width - 0.03, color=LAB_COL[lab], edgecolor=SURF, linewidth=1.2, label=lab if (i, k) == (0, 0) else None)
                bottom += v
            tick_x.append(x)
            tick_l.append(size)
        ax.text(base + width, -0.17, LABEL[cell].replace("\n", " "), ha="center", va="top", fontsize=9, color=INK)
    ax.set_xticks(tick_x, tick_l, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
    ax.set_ylabel("share of responses to harmful prompts")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
    ax.set_title("Refusal (R): English refuses; other languages mostly deflect, and with size more of it becomes non-refusing compliance", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig3_refusal_labels.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig4(runs1, runs2, out: Path):
    style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, runs, title in ((axes[0], runs1, "Primary judge: Mistral-Small-3.2-24B"), (axes[1], runs2, "Second judge: phi-4")):
        ends = {}
        for lab in ("Deflected", "Jailbroken", "Refused"):
            ys = [src(runs, s, "swh")["R"]["label_rates"]["swh/harm"][lab]["point"] for s in SIZES]
            ax.plot(range(3), ys, marker="o", ms=7, lw=2, color=LAB_COL[lab], mec=SURF, mew=2)
            ends[lab] = ys[-1]
        # direct labels at the right end, nudged apart so they never overlap
        order = sorted(ends, key=ends.get)
        pos, prev = {}, -1.0
        for lab in order:
            pos[lab] = max(ends[lab], prev + 0.075)
            prev = pos[lab]
        for lab in order:
            ax.text(2.08, pos[lab], f"{lab} {ends[lab]*100:.0f}%", va="center", fontsize=9, color=INK2)
        ax.set_xticks(range(3), SIZES)
        ax.set_xlim(-0.15, 3.0)
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1.01, 0.2), [f"{int(v*100)}%" for v in np.arange(0, 1.01, 0.2)])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Qwen3 size")
    axes[0].set_ylabel("share of responses, Swahili harmful prompts")
    fig.suptitle("Swahili: non-refusals move from Deflected to Jailbroken with size; the second judge sees a smaller shift and more Refused", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig4_swahili_by_judge.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig5(diag: Path, out: Path):
    """Cross-lingual retrieval by relative depth (last-token states; chance 0.5%)."""
    style()
    d = json.load(open(diag / "c4_last_token.json"))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.7), sharey=True)
    names = {"yor": "Yoruba", "yor-stripped": "Yoruba, stripped", "swh": "Swahili", "hau": "Hausa", "hau-stripped": "Hausa, stripped"}
    for ax, size in zip(axes, SIZES):
        for cell, r in d[size].items():
            t = np.array(r["top1"]) * 100
            ax.plot(np.linspace(0, 1, len(t)), t, color=COL[cell], lw=1.8, label=names[cell])
        ax.axhline(0.5, color=MUTED, lw=1)
        ax.set_title(f"Qwen3-{size}", fontsize=10)
        ax.set_xlabel("relative depth (0 = embeddings, 1 = output)")
        ax.set_ylim(0, 40)
    axes[0].set_ylabel("top-1 retrieval of the English counterpart (%)\n(grey line = chance, 0.5%)")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Cross-lingual alignment: Swahili aligns with English in the middle layers and more so with size; Yoruba and Hausa stay low", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig5_alignment_by_layer.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(root: Path, out: Path):
    out.mkdir(exist_ok=True, parents=True)
    runs, runs2 = load(root), load(root, "analysis_judge_second")
    fig1(runs, out)
    fig2(runs, out)
    fig3(runs, out)
    fig4(runs, runs2, out)
    if (root / "diagnostics" / "c4_last_token.json").exists():
        fig5(root / "diagnostics", out)
    print("wrote", sorted(p.name for p in out.glob("fig*.png")))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
