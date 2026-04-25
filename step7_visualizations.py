#!/usr/bin/env python3
"""
STEP 7 — Visualisations (all plots for the report)
Project: Empirical Distribution of Intervener Complexity

Produces all figures needed for the report (≤6 pages).
Each figure is saved as a high-resolution PDF + PNG.

FIGURES PRODUCED:
  Fig 1. Cross-linguistic POS heatmap       — Feature A, all 12 languages
  Fig 2. Aggregated POS bar chart           — Feature A, real vs baseline
  Fig 3. Arity distribution histograms      — Feature B, real vs baseline per language
  Fig 4. Subtree size distribution          — Feature C, real vs baseline per language
  Fig 5. Head-rate bar chart                — ICM metric, real vs baseline all languages
  Fig 6. Attachment type stacked bar        — Feature D, per language
  Fig 7. Word-order group comparison        — SOV vs SVO vs free, head-rate
  Fig 8. JSD heatmap (summary of tests)     — from step6 results
  Fig 9. Arc-distance vs head-rate scatter  — arc complexity vs intervener complexity

Usage:
    python3 step7_visualizations.py
    python3 step7_visualizations.py --results_dir ./data/results --show
"""

import csv
import json
import argparse
import collections
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# -----------------------------------------------------------------------
# Style — clean academic look suitable for the report
# -----------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
})

# Colour palette — distinct, print-friendly
REAL_COLOR     = "#2166AC"   # blue  — real data
BASELINE_COLOR = "#D6604D"   # red   — random baseline
COLORS_12 = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
    "#aec7e8","#ffbb78"
]

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

LANGUAGES = [
    "English", "German", "Spanish", "Russian", "Arabic",
    "Hindi", "Turkish", "Finnish", "Japanese", "Chinese",
    "Basque", "Ancient_Greek"
]

LANG_SHORT = {
    "English": "Eng", "German": "Deu", "Spanish": "Spa",
    "Russian": "Rus", "Arabic": "Ara", "Hindi": "Hin",
    "Turkish": "Tur", "Finnish": "Fin", "Japanese": "Jpn",
    "Chinese": "Zho", "Basque": "Eus", "Ancient_Greek": "Agr",
}

# UPOS tags — exclude PUNCT and SYM for cleaner plots (optional)
CONTENT_UPOS = ["NOUN","VERB","PRON","ADJ","ADV","ADP","AUX",
                 "CCONJ","DET","NUM","PART","PROPN","SCONJ","X"]

WORD_ORDER_GROUPS = {
    "SOV":  ["Hindi", "Turkish", "Finnish", "Japanese", "Basque"],
    "SVO":  ["English", "German", "Spanish", "Chinese"],
    "free": ["Russian", "Ancient_Greek"],
    "VSO":  ["Arabic"],
}

WO_COLORS = {"SOV": "#4393C3", "SVO": "#F4A582", "free": "#92C5DE", "VSO": "#D6604D"}


# -----------------------------------------------------------------------
# Data loaders
# -----------------------------------------------------------------------

def load_aggregated(aggregated_dir: Path) -> Dict:
    path = aggregated_dir / "aggregated_features.json"
    with open(path) as f:
        return json.load(f)


def load_baseline(aggregated_dir: Path) -> Dict:
    path = aggregated_dir / "baseline_features.json"
    with open(path) as f:
        return json.load(f)


def load_stat_results(results_dir: Path) -> List[Dict]:
    path = results_dir / "statistical_tests.json"
    with open(path) as f:
        return json.load(f)


def load_lang_csv(processed_dir: Path, language: str, column: str,
                  max_rows: int = 200000) -> List:
    """Load a single column from a language CSV (memory-efficient)."""
    path = processed_dir / f"{language.lower()}_interveners.csv"
    if not path.exists():
        return []
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                vals.append(int(row[column]))
            except (ValueError, KeyError):
                pass
    return vals


def load_baseline_csv(baseline_dir: Path, language: str, column: str,
                      max_rows: int = 200000) -> List:
    path = baseline_dir / f"{language.lower()}_baseline.csv"
    if not path.exists():
        return []
    vals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            try:
                vals.append(int(row[column]))
            except (ValueError, KeyError):
                pass
    return vals


# -----------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------

def get_lang_data(agg: Dict, language: str) -> Optional[Dict]:
    for d in agg["languages"]:
        if d and d["language"] == language:
            return d
    return None


def get_baseline_data(baseline: Dict, language: str) -> Optional[Dict]:
    for d in baseline["languages"]:
        if d and d["language"] == language:
            return d
    return None


def savefig(fig, out_dir: Path, name: str):
    for ext in ["pdf", "png"]:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        print(f"  Saved: {p.name}")
    plt.close(fig)


# -----------------------------------------------------------------------
# FIGURE 1 — Cross-linguistic POS heatmap
# -----------------------------------------------------------------------

def fig1_pos_heatmap(agg: Dict, out_dir: Path):
    """
    Heatmap: rows = languages, cols = UPOS tags
    Colour intensity = proportion of interveners with that POS.
    Shows at a glance which POS types dominate across languages.
    """
    langs   = [d["language"] for d in agg["languages"] if d]
    matrix  = []
    for d in agg["languages"]:
        if not d:
            continue
        props = d["pos_proportions"]
        matrix.append([props.get(tag, 0) for tag in CONTENT_UPOS])

    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(11, 5))

    cmap = LinearSegmentedColormap.from_list(
        "icm", ["#f7f7f7", "#2166AC"], N=256
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=mat.max())

    # Annotate cells with percentage
    for i in range(len(langs)):
        for j in range(len(CONTENT_UPOS)):
            val = mat[i, j]
            txt_col = "white" if val > mat.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_col)

    ax.set_xticks(range(len(CONTENT_UPOS)))
    ax.set_xticklabels(CONTENT_UPOS, rotation=40, ha="right")
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([LANG_SHORT.get(l, l) for l in langs])
    ax.set_xlabel("UPOS tag of intervening token")
    ax.set_ylabel("Language")
    ax.set_title(
        "Figure 1 — POS distribution of intervening tokens across 12 languages\n"
        "(proportion; darker = higher frequency)",
        pad=10
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Proportion of interveners", fontsize=8)

    fig.tight_layout()
    savefig(fig, out_dir, "fig1_pos_heatmap")


# -----------------------------------------------------------------------
# FIGURE 2 — Aggregated POS bar chart: real vs baseline
# -----------------------------------------------------------------------

def fig2_pos_bar_real_vs_baseline(agg: Dict, baseline: Dict, out_dir: Path):
    """
    Side-by-side bar chart of POS proportions:
    Real (blue) vs Random baseline (red) — cross-linguistic aggregate.
    """
    real_props = agg["all"]["pos_proportions"]
    base_props = baseline["all"]["pos_proportions"]

    tags  = CONTENT_UPOS
    x     = np.arange(len(tags))
    width = 0.38

    real_vals = [real_props.get(t, 0) for t in tags]
    base_vals = [base_props.get(t, 0) for t in tags]

    fig, ax = plt.subplots(figsize=(11, 4.5))

    bars_r = ax.bar(x - width/2, real_vals, width, label="Real",     color=REAL_COLOR,     alpha=0.85)
    bars_b = ax.bar(x + width/2, base_vals, width, label="Baseline", color=BASELINE_COLOR, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=35, ha="right")
    ax.set_ylabel("Proportion of interveners")
    ax.set_title(
        "Figure 2 — POS distribution of interveners: real vs random baseline (all 12 languages)\n"
        "Blue = attested linearisation; Red = random permutation baseline",
        pad=8
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))

    # Annotate bars > 5%
    for bar in list(bars_r) + list(bars_b):
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:.1%}", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    savefig(fig, out_dir, "fig2_pos_bar_real_vs_baseline")


# -----------------------------------------------------------------------
# FIGURE 3 — Arity distribution: real vs baseline, all languages grid
# -----------------------------------------------------------------------

def fig3_arity_grid(agg: Dict, baseline: Dict,
                    processed_dir: Path, baseline_dir: Path, out_dir: Path):
    """
    4×3 grid of arity histograms, one per language.
    Each subplot shows real (blue) vs baseline (red).
    Arity capped at 8 for readability.
    """
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 11), sharey=False)
    axes_flat = axes.flatten()

    MAX_ARITY = 8
    bins = np.arange(-0.5, MAX_ARITY + 1.5, 1)

    valid_langs = [d["language"] for d in agg["languages"] if d]

    for idx, lang in enumerate(valid_langs):
        ax = axes_flat[idx]

        real_vals = load_lang_csv(processed_dir, lang, "intervener_arity")
        base_vals = load_baseline_csv(baseline_dir, lang, "intervener_arity")

        real_clipped = [min(v, MAX_ARITY) for v in real_vals]
        base_clipped = [min(v, MAX_ARITY) for v in base_vals]

        ax.hist(real_clipped, bins=bins, density=True, alpha=0.7,
                color=REAL_COLOR,     label="Real",     rwidth=0.45,
                align="mid")
        ax.hist(base_clipped, bins=bins, density=True, alpha=0.7,
                color=BASELINE_COLOR, label="Baseline", rwidth=0.45,
                align="mid")

        short = LANG_SHORT.get(lang, lang)
        ld = get_lang_data(agg, lang)
        mean_real = ld["arity_stats"]["mean"] if ld else 0
        lb = get_baseline_data(baseline, lang)
        mean_base = lb["arity_mean"] if lb else 0

        ax.set_title(f"{short}  (μ_real={mean_real:.2f}, μ_base={mean_base:.2f})",
                     fontsize=8.5)
        ax.set_xlabel("Arity", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_xticks(range(MAX_ARITY + 1))
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplots
    for idx in range(len(valid_langs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Figure 3 — Arity distribution of intervening tokens per language\n"
        "Real (blue) vs random baseline (red). Arity capped at 8.",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig3_arity_grid")


# -----------------------------------------------------------------------
# FIGURE 4 — Subtree size distribution: real vs baseline, grid
# -----------------------------------------------------------------------

def fig4_subtree_grid(agg: Dict, baseline: Dict,
                      processed_dir: Path, baseline_dir: Path, out_dir: Path):
    """
    4×3 grid of subtree-size histograms per language.
    Subtree size capped at 10 for readability.
    """
    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 11), sharey=False)
    axes_flat = axes.flatten()

    MAX_SUB = 10
    bins = np.arange(0.5, MAX_SUB + 1.5, 1)

    valid_langs = [d["language"] for d in agg["languages"] if d]

    for idx, lang in enumerate(valid_langs):
        ax = axes_flat[idx]

        real_vals = load_lang_csv(processed_dir, lang, "intervener_subtree_size")
        base_vals = load_baseline_csv(baseline_dir, lang, "intervener_subtree_size")

        real_clipped = [min(v, MAX_SUB) for v in real_vals]
        base_clipped = [min(v, MAX_SUB) for v in base_vals]

        ax.hist(real_clipped, bins=bins, density=True, alpha=0.7,
                color=REAL_COLOR,     label="Real",     rwidth=0.45)
        ax.hist(base_clipped, bins=bins, density=True, alpha=0.7,
                color=BASELINE_COLOR, label="Baseline", rwidth=0.45)

        short = LANG_SHORT.get(lang, lang)
        ld = get_lang_data(agg, lang)
        mean_real = ld["subtree_stats"]["mean"] if ld else 0
        lb = get_baseline_data(baseline, lang)
        mean_base = lb["subtree_mean"] if lb else 0

        ax.set_title(f"{short}  (μ_real={mean_real:.2f}, μ_base={mean_base:.2f})",
                     fontsize=8.5)
        ax.set_xlabel("Subtree size", fontsize=7)
        ax.set_ylabel("Density", fontsize=7)
        ax.set_xticks(range(1, MAX_SUB + 1))
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7)

    for idx in range(len(valid_langs), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Figure 4 — Subtree size of intervening tokens per language\n"
        "Real (blue) vs random baseline (red). Size capped at 10.",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig4_subtree_grid")


# -----------------------------------------------------------------------
# FIGURE 5 — Head-rate bar chart: real vs baseline, all languages
# -----------------------------------------------------------------------

def fig5_head_rate_comparison(agg: Dict, baseline: Dict,
                               stat_results: List[Dict], out_dir: Path):
    """
    The KEY figure — shows the proportion of interveners that are
    syntactic heads (is_head=1) in real vs baseline.
    
    This directly tests the ICM hypothesis from Yadav et al. (2022).
    Languages where real << baseline support the hypothesis.
    Significance markers (* = p < α_corrected) added to bars.
    """
    valid_langs = [d["language"] for d in agg["languages"] if d]
    x = np.arange(len(valid_langs))
    width = 0.38

    real_rates = []
    base_rates = []
    for lang in valid_langs:
        ld = get_lang_data(agg, lang)
        lb = get_baseline_data(baseline, lang)
        real_rates.append(ld["head_rate"] if ld else 0)
        base_rates.append(lb["head_rate"] if lb else 0)

    # Significance lookup
    sig_map = {}
    for r in stat_results:
        sig_map[r["language"]] = r["head_rate_test"]["significant"]

    fig, ax = plt.subplots(figsize=(13, 5))

    bars_r = ax.bar(x - width/2, real_rates, width,
                    label="Real linearisation", color=REAL_COLOR, alpha=0.85)
    bars_b = ax.bar(x + width/2, base_rates, width,
                    label="Random baseline", color=BASELINE_COLOR, alpha=0.85)

    # Add significance stars
    for i, lang in enumerate(valid_langs):
        if sig_map.get(lang, False):
            ymax = max(real_rates[i], base_rates[i])
            ax.text(x[i], ymax + 0.012, "*", ha="center", fontsize=14,
                    color="#333333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [LANG_SHORT.get(l, l) for l in valid_langs], rotation=25, ha="right"
    )
    ax.set_ylabel("Head-rate (proportion of interveners that are heads)")
    ax.set_title(
        "Figure 5 — Intervener head-rate: real vs random baseline across 12 languages\n"
        "Head-rate = fraction of interveners with arity > 0 (syntactic heads).\n"
        "* = significantly lower real head-rate (Bonferroni-corrected p < 0.0042)",
        pad=8
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, max(base_rates) * 1.18)

    # Annotation: difference arrows
    for i in range(len(valid_langs)):
        diff = real_rates[i] - base_rates[i]
        mid_x = x[i]
        mid_y = (real_rates[i] + base_rates[i]) / 2
        ax.annotate("", xy=(mid_x - width/2, real_rates[i]),
                    xytext=(mid_x - width/2, base_rates[i] if base_rates[i] > 0 else 0.01),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
                    annotation_clip=True)

    fig.tight_layout()
    savefig(fig, out_dir, "fig5_head_rate_comparison")


# -----------------------------------------------------------------------
# FIGURE 6 — Attachment type stacked bar chart
# -----------------------------------------------------------------------

def fig6_attachment_stacked(agg: Dict, out_dir: Path):
    """
    Stacked bar chart showing attachment type distribution per language.
    head / dependent / external — which node does the intervener attach to?
    """
    valid_langs = [d["language"] for d in agg["languages"] if d]
    att_types = ["head", "dependent", "external"]
    att_colors = ["#4393C3", "#F4A582", "#92C5DE"]

    head_vals = []
    dep_vals  = []
    ext_vals  = []

    for lang in valid_langs:
        ld = get_lang_data(agg, lang)
        if not ld:
            head_vals.append(0); dep_vals.append(0); ext_vals.append(0)
            continue
        counts = ld["attachment_counts"]
        total = sum(counts.values()) or 1
        head_vals.append(counts.get("head",     0) / total)
        dep_vals.append( counts.get("dependent",0) / total)
        ext_vals.append( counts.get("external", 0) / total)

    x = np.arange(len(valid_langs))
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x, head_vals, label="Attached to head",       color=att_colors[0], alpha=0.9)
    ax.bar(x, dep_vals,  bottom=head_vals, label="Attached to dependent", color=att_colors[1], alpha=0.9)
    bottom2 = [h + d for h, d in zip(head_vals, dep_vals)]
    ax.bar(x, ext_vals,  bottom=bottom2,   label="External (neither)",    color=att_colors[2], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_SHORT.get(l, l) for l in valid_langs], rotation=25, ha="right")
    ax.set_ylabel("Proportion of interveners")
    ax.set_title(
        "Figure 6 — Attachment type of intervening tokens per language\n"
        "'Head-attached': intervener's head is the arc head. "
        "'Dep-attached': arc dependent. 'External': neither.",
        pad=8
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fig.tight_layout()
    savefig(fig, out_dir, "fig6_attachment_stacked")


# -----------------------------------------------------------------------
# FIGURE 7 — Word-order group comparison
# -----------------------------------------------------------------------

def fig7_word_order_comparison(agg: Dict, baseline: Dict, out_dir: Path):
    """
    Grouped bar chart comparing head-rate and mean arity
    between SOV, SVO, free, and VSO word-order groups.
    Tests the prediction that SOV (head-final) languages have
    more complex interveners than SVO (head-initial).
    """
    groups = ["SOV", "SVO", "VSO", "free"]
    group_members = {
        "SOV":  ["Hindi", "Turkish", "Finnish", "Japanese", "Basque"],
        "SVO":  ["English", "German", "Spanish", "Chinese"],
        "VSO":  ["Arabic"],
        "free": ["Russian", "Ancient_Greek"],
    }

    def group_avg(metric, group):
        vals = []
        for lang in group_members[group]:
            ld = get_lang_data(agg, lang)
            if ld:
                if metric == "head_rate":
                    vals.append(ld["head_rate"])
                elif metric == "mean_arity":
                    vals.append(ld["arity_stats"].get("mean", 0))
                elif metric == "mean_subtree":
                    vals.append(ld["subtree_stats"].get("mean", 0))
        return sum(vals) / len(vals) if vals else 0

    def base_group_avg(metric, group):
        vals = []
        for lang in group_members[group]:
            lb = get_baseline_data(baseline, lang)
            if lb:
                if metric == "head_rate":
                    vals.append(lb["head_rate"])
                elif metric == "mean_arity":
                    vals.append(lb["arity_mean"])
        return sum(vals) / len(vals) if vals else 0

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    metrics = [
        ("head_rate",   "Head-rate", "Proportion of interveners that are heads"),
        ("mean_arity",  "Mean arity", "Average arity of intervening tokens"),
        ("mean_subtree","Mean subtree size", "Average subtree size of interveners"),
    ]

    for ax, (metric, ylabel, desc) in zip(axes, metrics):
        real_vals = [group_avg(metric, g) for g in groups]
        base_vals = [base_group_avg(metric, g) if metric != "mean_subtree"
                     else 0 for g in groups]

        x = np.arange(len(groups))
        w = 0.35

        ax.bar(x - w/2, real_vals, w, color=REAL_COLOR,     alpha=0.85, label="Real")
        if any(v > 0 for v in base_vals):
            ax.bar(x + w/2, base_vals, w, color=BASELINE_COLOR, alpha=0.85, label="Baseline")

        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylabel(ylabel)
        ax.set_title(desc, fontsize=8.5)
        if metric == "head_rate":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.legend(fontsize=7)

        # Annotate bars
        for xi, rv in zip(x, real_vals):
            ax.text(xi - w/2, rv + 0.003, f"{rv:.2f}", ha="center",
                    fontsize=7, color=REAL_COLOR)

    fig.suptitle(
        "Figure 7 — Intervener complexity by word-order typology\n"
        "Averaged over languages in each group.",
        fontsize=10
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig7_word_order_comparison")


# -----------------------------------------------------------------------
# FIGURE 8 — JSD heatmap (statistical summary)
# -----------------------------------------------------------------------

def fig8_jsd_heatmap(stat_results: List[Dict], out_dir: Path):
    """
    Heatmap of Jensen-Shannon Divergence values for all tests.
    rows = languages, cols = feature (POS / arity / subtree)
    Shows at a glance where real distributions deviate most from baseline.
    """
    langs = [r["language"] for r in stat_results]
    features = ["POS", "Arity", "Subtree size"]

    matrix = []
    for r in stat_results:
        matrix.append([
            r["jsd_pos"]["jsd"],
            r["jsd_arity"]["jsd"],
            r["jsd_subtree"]["jsd"],
        ])

    mat = np.array(matrix)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = LinearSegmentedColormap.from_list("jsd", ["#f7f7f7", "#D6604D"], N=256)
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=min(mat.max(), 1.0))

    for i in range(len(langs)):
        for j in range(len(features)):
            ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if mat[i,j] > mat.max() * 0.6 else "black")

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features)
    ax.set_yticks(range(len(langs)))
    ax.set_yticklabels([LANG_SHORT.get(l, l) for l in langs])
    ax.set_xlabel("Feature")
    ax.set_ylabel("Language")
    ax.set_title(
        "Figure 8 — Jensen-Shannon Divergence\n"
        "Real vs baseline. Higher = more divergent from random.",
        pad=8
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("JSD (bits)", fontsize=8)

    fig.tight_layout()
    savefig(fig, out_dir, "fig8_jsd_heatmap")


# -----------------------------------------------------------------------
# FIGURE 9 — Arc distance vs head-rate scatter
# -----------------------------------------------------------------------

def fig9_arc_distance_vs_head_rate(processed_dir: Path, out_dir: Path):
    """
    Scatter plot: for each language, plot mean arc distance vs head-rate.
    Tests whether longer arcs tend to have more complex interveners.
    Each point is one language; labelled with language abbreviation.
    Colour-coded by word order group.
    """
    group_map = {}
    for g, langs in {
        "SOV":  ["Hindi","Turkish","Finnish","Japanese","Basque"],
        "SVO":  ["English","German","Spanish","Chinese"],
        "VSO":  ["Arabic"],
        "free": ["Russian","Ancient_Greek"],
    }.items():
        for l in langs:
            group_map[l] = g

    fig, ax = plt.subplots(figsize=(8, 6))

    plotted = []
    for lang in LANGUAGES:
        csv_path = processed_dir / f"{lang.lower()}_interveners.csv"
        if not csv_path.exists():
            continue

        arc_dists = []
        is_heads  = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= 100000:
                    break
                try:
                    arc_dists.append(int(row["arc_distance"]))
                    is_heads.append(int(row["is_head"]))
                except (ValueError, KeyError):
                    pass

        if not arc_dists:
            continue

        mean_dist = sum(arc_dists) / len(arc_dists)
        head_rate = sum(is_heads) / len(is_heads)

        group = group_map.get(lang, "unknown")
        color = WO_COLORS.get(group, "gray")

        ax.scatter(mean_dist, head_rate, s=80, color=color, zorder=3,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(
            LANG_SHORT.get(lang, lang),
            xy=(mean_dist, head_rate),
            xytext=(4, 3), textcoords="offset points",
            fontsize=8
        )
        plotted.append((group, color))

    # Legend for word-order groups
    seen = {}
    for g, c in plotted:
        if g not in seen:
            seen[g] = mpatches.Patch(color=c, label=g)
    ax.legend(handles=list(seen.values()), title="Word order", fontsize=8)

    ax.set_xlabel("Mean arc distance (tokens)")
    ax.set_ylabel("Head-rate (fraction of interveners that are heads)")
    ax.set_title(
        "Figure 9 — Mean arc distance vs intervener head-rate\n"
        "Each point = one language. Colour = word-order type.",
        pad=8
    )

    fig.tight_layout()
    savefig(fig, out_dir, "fig9_arc_vs_headrate_scatter")


# -----------------------------------------------------------------------
# FIGURE 10 — Bonus: POS top-3 per language radar/summary
# -----------------------------------------------------------------------

def fig10_top_pos_per_language(agg: Dict, out_dir: Path):
    """
    For each language, show the top 5 POS tags of interveners as a
    horizontal bar chart. Arranged in a 4×3 grid.
    Helps identify whether NOUN and ADV dominate as hypothesised.
    """
    valid = [(d["language"], d["pos_proportions"])
             for d in agg["languages"] if d]

    nrows, ncols = 4, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 12))
    axes_flat = axes.flatten()

    for idx, (lang, props) in enumerate(valid):
        ax = axes_flat[idx]
        sorted_tags = sorted(props.items(), key=lambda x: x[1], reverse=True)[:7]
        tags  = [t for t, _ in sorted_tags]
        vals  = [v for _, v in sorted_tags]
        colors = [COLORS_12[i % len(COLORS_12)] for i in range(len(tags))]

        bars = ax.barh(tags[::-1], vals[::-1], color=colors[::-1], alpha=0.85)

        for bar, val in zip(bars, vals[::-1]):
            ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{val:.1%}", va="center", fontsize=7)

        ax.set_title(f"{LANG_SHORT.get(lang, lang)} ({lang})", fontsize=8.5)
        ax.set_xlabel("Proportion", fontsize=7)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(labelsize=7)

    for idx in range(len(valid), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        "Figure 10 — Top 7 POS tags of intervening tokens per language\n"
        "Tests hypothesis: interveners are predominantly NOUN and ADV.",
        fontsize=10, y=1.01
    )
    fig.tight_layout()
    savefig(fig, out_dir, "fig10_top_pos_per_language")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate all report figures")
    parser.add_argument("--aggregated_dir", default="./data/aggregated")
    parser.add_argument("--processed_dir",  default="./data/processed")
    parser.add_argument("--baseline_dir",   default="./data/baseline")
    parser.add_argument("--results_dir",    default="./data/results")
    parser.add_argument("--out_dir",        default="./data/figures")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively (requires display)")
    args = parser.parse_args()

    aggregated_dir = Path(args.aggregated_dir)
    processed_dir  = Path(args.processed_dir)
    baseline_dir   = Path(args.baseline_dir)
    results_dir    = Path(args.results_dir)
    out_dir        = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 7 — Generating all figures")
    print("=" * 60)

    print("\n  Loading aggregated data...")
    agg     = load_aggregated(aggregated_dir)
    baseline = load_baseline(aggregated_dir)
    stat_results = load_stat_results(results_dir)

    print("\n  Fig 1 — POS heatmap...")
    fig1_pos_heatmap(agg, out_dir)

    print("  Fig 2 — POS bar: real vs baseline...")
    fig2_pos_bar_real_vs_baseline(agg, baseline, out_dir)

    print("  Fig 3 — Arity grid...")
    fig3_arity_grid(agg, baseline, processed_dir, baseline_dir, out_dir)

    print("  Fig 4 — Subtree size grid...")
    fig4_subtree_grid(agg, baseline, processed_dir, baseline_dir, out_dir)

    print("  Fig 5 — Head-rate comparison...")
    fig5_head_rate_comparison(agg, baseline, stat_results, out_dir)

    print("  Fig 6 — Attachment type stacked bar...")
    fig6_attachment_stacked(agg, out_dir)

    print("  Fig 7 — Word-order group comparison...")
    fig7_word_order_comparison(agg, baseline, out_dir)

    print("  Fig 8 — JSD heatmap...")
    fig8_jsd_heatmap(stat_results, out_dir)

    print("  Fig 9 — Arc distance vs head-rate scatter...")
    fig9_arc_distance_vs_head_rate(processed_dir, out_dir)

    print("  Fig 10 — Top POS per language...")
    fig10_top_pos_per_language(agg, out_dir)

    print("\n" + "=" * 60)
    print(f"  All 10 figures saved to {out_dir}/")
    print("  Each figure saved as both .pdf (for report) and .png (for preview)")
    print("=" * 60)
    print("\nNext: write the report using data/results/statistical_report.txt")
    print("      and figures from data/figures/")


if __name__ == "__main__":
    main()
