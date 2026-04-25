#!/usr/bin/env python3
"""
STEP 6 — Statistical tests: JSD + Mann-Whitney U
Project: Empirical Distribution of Intervener Complexity

This script tests whether real intervener distributions differ
significantly from the random-linearisation baseline.

TWO STATISTICAL TESTS (as required by the grading rubric):

1. JENSEN-SHANNON DIVERGENCE (a symmetric, bounded variant of KL)
   ─────────────────────────────────────────────────────────────
   Used for: comparing DISTRIBUTIONS (POS, arity counts, subtree counts)

   KL(P || Q) = Σ P(x) · log(P(x) / Q(x))
   JSD(P || Q) = 0.5·KL(P||M) + 0.5·KL(Q||M),  M = 0.5·(P+Q)

   With log base 2, JSD ∈ [0, 1].
     JSD = 0  → P and Q are identical
     JSD = 1  → P and Q are maximally different

2. MANN-WHITNEY U TEST (Wilcoxon rank-sum)
   ─────────────────────────────────────────
   Used for: comparing two SAMPLES (arity values, subtree sizes)

   H₀: real and baseline come from the same distribution
   H₁: real is stochastically LESS than baseline (one-tailed: 'less')
       — supports ICM if real values tend to be smaller.

   Significance threshold: α = 0.05, Bonferroni-corrected for 12 languages
   → α_corrected = 0.05 / 12 ≈ 0.0042.

PERFORMANCE NOTE
   Russian's processed CSV is ~435 MB; loading it whole would burn most
   of the user's free RAM. This script streams each CSV in a single pass
   and keeps a *reservoir sample* of arity/subtree values for the
   Mann-Whitney U test (which itself sub-samples to 50k anyway).

OUTPUT:
   ./data/results/statistical_tests.json   — machine-readable
   ./data/results/statistical_report.txt   — human-readable

Usage:
    python3 step6_statistical_tests.py
"""

import csv
import json
import math
import random
import argparse
import collections
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import scipy.stats as stats


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

LANGUAGES = [
    "English", "German", "Spanish", "Russian", "Arabic",
    "Hindi", "Turkish", "Finnish", "Japanese", "Chinese",
    "Basque", "Ancient_Greek"
]

ALL_UPOS = [
    "NOUN", "VERB", "PRON", "ADJ", "ADV", "ADP", "AUX",
    "CCONJ", "DET", "INTJ", "NUM", "PART", "PROPN",
    "PUNCT", "SCONJ", "SYM", "X"
]

ALPHA            = 0.05
BONFERRONI_N     = 12  # number of languages
ALPHA_CORRECTED  = ALPHA / BONFERRONI_N   # ≈ 0.0042

# Reservoir size for raw arity/subtree samples (MWU sub-samples to 50k
# inside ``mann_whitney_test`` so any value >= 50k preserves power).
RESERVOIR_CAP = 100_000

MAX_ARITY_BIN   = 10
MAX_SUBTREE_BIN = 15


# -----------------------------------------------------------------------
# Streaming summary of one CSV
# -----------------------------------------------------------------------

class StreamSummary:
    """Accumulates everything we need from a real or baseline CSV in a
    single pass. Stores raw arity / subtree values via reservoir
    sampling so the Mann-Whitney U test still has unbiased samples even
    when the CSV is gigabytes large.
    """

    def __init__(self, rng: random.Random):
        self.rng              = rng
        self.n                = 0
        self.pos_counts       = collections.Counter()
        self.arity_counts     = collections.Counter()
        self.subtree_counts   = collections.Counter()
        self.attachment_counts = collections.Counter()
        self.is_head_total    = 0
        self.arity_sample: List[int] = []
        self.subtree_sample: List[int] = []
        self._arity_seen      = 0
        self._subtree_seen    = 0

    def _reservoir(self, sample: List[int], seen_attr: str,
                   value: int) -> None:
        seen = getattr(self, seen_attr) + 1
        setattr(self, seen_attr, seen)
        if len(sample) < RESERVOIR_CAP:
            sample.append(value)
        else:
            j = self.rng.randint(0, seen - 1)
            if j < RESERVOIR_CAP:
                sample[j] = value

    def add_row(self, row: Dict) -> None:
        try:
            arity   = int(row.get("intervener_arity", 0) or 0)
            subtree = int(row.get("intervener_subtree_size", 0) or 0)
            is_head = int(row.get("is_head", 0) or 0)
        except ValueError:
            return
        self.n += 1
        self.pos_counts[row.get("intervener_upos", "X")] += 1
        self.arity_counts[min(arity, MAX_ARITY_BIN)] += 1
        self.subtree_counts[min(subtree, MAX_SUBTREE_BIN)] += 1
        self.attachment_counts[row.get("attachment_type", "external")] += 1
        self.is_head_total += is_head

        self._reservoir(self.arity_sample,   "_arity_seen",   arity)
        self._reservoir(self.subtree_sample, "_subtree_seen", subtree)

    # --- distribution accessors ----------------------------------------

    def pos_dist(self) -> List[float]:
        total = self.n or 1
        return [self.pos_counts.get(t, 0) / total for t in ALL_UPOS]

    def arity_dist(self) -> Tuple[List[int], List[float]]:
        keys = list(range(MAX_ARITY_BIN + 1))
        total = self.n or 1
        return keys, [self.arity_counts.get(k, 0) / total for k in keys]

    def subtree_dist(self) -> Tuple[List[int], List[float]]:
        keys = list(range(1, MAX_SUBTREE_BIN + 1))
        total = self.n or 1
        return keys, [self.subtree_counts.get(k, 0) / total for k in keys]

    def head_rate(self) -> float:
        return self.is_head_total / self.n if self.n else 0.0


def stream_csv(path: Path, rng: random.Random) -> StreamSummary:
    """One streaming pass over a CSV. Returns a populated StreamSummary."""
    summary = StreamSummary(rng)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            summary.add_row(row)
    return summary


# -----------------------------------------------------------------------
# Jensen-Shannon Divergence
# -----------------------------------------------------------------------

def kl_divergence(p: List[float], q: List[float], epsilon: float = 1e-10) -> float:
    """KL(P || Q) = Σ P(x) · log(P(x) / Q(x)).
    epsilon smoothing prevents log(0) when q(x)=0.
    """
    kl = 0.0
    for pi, qi in zip(p, q):
        qi_s = qi + epsilon
        if pi > 0:
            kl += pi * math.log(pi / qi_s)
    return kl


def jensen_shannon_divergence(p: List[float], q: List[float]) -> float:
    """Symmetric, bounded ∈ [0, 1] (using log base 2)."""
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    jsd_bits = jsd / math.log(2)
    return min(max(jsd_bits, 0.0), 1.0)


def compute_jsd(real: StreamSummary, base: StreamSummary, feature: str) -> Dict:
    """JSD between real and baseline distributions for one feature."""
    if feature == "pos":
        p_vals, q_vals = real.pos_dist(), base.pos_dist()
        label = "POS distribution"
    elif feature == "arity":
        _, p_vals = real.arity_dist()
        _, q_vals = base.arity_dist()
        label = "Arity distribution"
    elif feature == "subtree":
        _, p_vals = real.subtree_dist()
        _, q_vals = base.subtree_dist()
        label = "Subtree size distribution"
    else:
        raise ValueError(f"Unknown feature: {feature}")

    jsd = jensen_shannon_divergence(p_vals, q_vals)
    return {"feature": label, "jsd": round(jsd, 6)}


# -----------------------------------------------------------------------
# Mann-Whitney U test
# -----------------------------------------------------------------------

def mann_whitney_test(
    real_sample: List[int],
    baseline_sample: List[int],
    alternative: str = "less",
    max_sample: int = 50_000,
) -> Dict:
    """One-tailed Mann-Whitney U test (H₁: real < baseline).

    Effect size: r = |Z| / √N (Cohen 1992: 0.1 small, 0.3 medium, 0.5 large).
    """
    rng = random.Random(0)
    if len(real_sample) > max_sample:
        real_sample = rng.sample(real_sample, max_sample)
    if len(baseline_sample) > max_sample:
        baseline_sample = rng.sample(baseline_sample, max_sample)

    if not real_sample or not baseline_sample:
        return {"U": None, "p_value": None, "p_value_fmt": "n/a",
                "significant": False, "effect_size_r": None,
                "effect_label": "n/a", "n_real": 0, "n_baseline": 0,
                "interpretation": "Insufficient data"}

    u_stat, p_value = stats.mannwhitneyu(
        real_sample, baseline_sample, alternative=alternative
    )

    n1, n2 = len(real_sample), len(baseline_sample)
    mean_u = n1 * n2 / 2
    std_u  = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z_score = (u_stat - mean_u) / std_u if std_u > 0 else 0.0
    effect_r = abs(z_score) / math.sqrt(n1 + n2)

    return {
        "U":              round(float(u_stat), 2),
        "p_value":        float(p_value),
        "p_value_fmt":    f"{p_value:.2e}",
        "significant":    p_value < ALPHA_CORRECTED,
        "alpha_corrected": ALPHA_CORRECTED,
        "effect_size_r":  round(effect_r, 4),
        "effect_label":   ("large"  if effect_r >= 0.5 else
                           "medium" if effect_r >= 0.3 else
                           "small"  if effect_r >= 0.1 else "negligible"),
        "n_real":         n1,
        "n_baseline":     n2,
        "interpretation": (
            "Real < Baseline (supports ICM hypothesis)" if p_value < ALPHA_CORRECTED
            else "No significant difference from baseline"
        ),
    }


# -----------------------------------------------------------------------
# Two-proportion z-test for head-rate
# -----------------------------------------------------------------------

def compare_head_rates(real: StreamSummary, base: StreamSummary) -> Dict:
    """Two-proportion z-test on the head-rate (is_head=1 proportion).
    Lower real head-rate ⇒ language minimises complex interveners (ICM).
    """
    n_real = real.n
    n_base = base.n
    p_real = real.head_rate()
    p_base = base.head_rate()

    if n_real > 0 and n_base > 0:
        h_real = real.is_head_total
        h_base = base.is_head_total
        p_pool = (h_real + h_base) / (n_real + n_base)
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_real + 1 / n_base))
        if se > 0:
            z = (p_real - p_base) / se
            from scipy.stats import norm
            p = norm.cdf(z)  # one-tailed: H1 = real < baseline
        else:
            z, p = 0.0, 1.0
    else:
        z, p = 0.0, 1.0

    return {
        "head_rate_real":     round(p_real, 4),
        "head_rate_baseline": round(p_base, 4),
        "difference":         round(p_real - p_base, 4),
        "z_stat":             round(float(z), 4),
        "p_value":            float(p),
        "p_value_fmt":        f"{p:.2e}",
        "significant":        p < ALPHA_CORRECTED,
        "interpretation": (
            "Real head-rate significantly lower than baseline (supports ICM)"
            if p < ALPHA_CORRECTED
            else "Head-rate not significantly different from baseline"
        ),
    }


# -----------------------------------------------------------------------
# Per-language test battery
# -----------------------------------------------------------------------

def run_tests_for_language(
    language: str,
    processed_dir: Path,
    baseline_dir: Path,
    rng: random.Random,
) -> Optional[Dict]:
    real_csv     = processed_dir / f"{language.lower()}_interveners.csv"
    baseline_csv = baseline_dir  / f"{language.lower()}_baseline.csv"

    if not real_csv.exists():
        print(f"  [SKIP] {language}: real CSV not found ({real_csv})")
        return None
    if not baseline_csv.exists():
        print(f"  [SKIP] {language}: baseline CSV not found ({baseline_csv})")
        return None

    print(f"  [{language}] Streaming real CSV...")
    real = stream_csv(real_csv, rng)
    print(f"  [{language}] Streaming baseline CSV...")
    base = stream_csv(baseline_csv, rng)
    print(f"  [{language}] {real.n:,} real | {base.n:,} baseline observations")

    return {
        "language":   language,
        "n_real":     real.n,
        "n_baseline": base.n,

        # JSD — distributional comparison
        "jsd_pos":     compute_jsd(real, base, "pos"),
        "jsd_arity":   compute_jsd(real, base, "arity"),
        "jsd_subtree": compute_jsd(real, base, "subtree"),

        # Mann-Whitney U — median comparison
        "mwu_arity":   mann_whitney_test(real.arity_sample,
                                          base.arity_sample,
                                          alternative="less"),
        "mwu_subtree": mann_whitney_test(real.subtree_sample,
                                          base.subtree_sample,
                                          alternative="less"),

        # Two-proportion z-test on the primary ICM metric
        "head_rate_test": compare_head_rates(real, base),
    }


# -----------------------------------------------------------------------
# Report generation
# -----------------------------------------------------------------------

def generate_report(all_results: List[Dict]) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("  STATISTICAL TEST RESULTS")
    lines.append("  Project: Empirical Distribution of Intervener Complexity")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  Significance threshold : α = {ALPHA}")
    lines.append(f"  Bonferroni correction  : α_corrected = {ALPHA}/{BONFERRONI_N} "
                 f"= {ALPHA_CORRECTED:.4f}")
    lines.append(f"  MWU alternative        : H₁: real < baseline (one-tailed)")
    lines.append(f"  JSD range              : [0 = identical, 1 = maximally different]")
    lines.append("")

    # Head-rate
    lines.append("-" * 72)
    lines.append("  1. HEAD-RATE COMPARISON (primary ICM metric)")
    lines.append("     Proportion of interveners that are syntactic heads (arity > 0)")
    lines.append("     Lower real head-rate = language minimises complex interveners")
    lines.append("")
    lines.append(f"  {'Language':<16} {'Real':>8} {'Baseline':>9} "
                 f"{'Diff':>7} {'p-value':>12} {'Sig?':>6}")
    lines.append("  " + "-" * 62)
    for r in all_results:
        h = r["head_rate_test"]
        sig = "YES *" if h["significant"] else "no"
        lines.append(
            f"  {r['language']:<16} {h['head_rate_real']:>8.3f} "
            f"{h['head_rate_baseline']:>9.3f} {h['difference']:>7.3f} "
            f"{h['p_value_fmt']:>12} {sig:>6}"
        )

    # JSD
    lines.append("")
    lines.append("-" * 72)
    lines.append("  2. JENSEN-SHANNON DIVERGENCE (distributional difference)")
    lines.append("")
    lines.append(f"  {'Language':<16} {'JSD(POS)':>10} {'JSD(Arity)':>12} "
                 f"{'JSD(Subtree)':>14}")
    lines.append("  " + "-" * 54)
    for r in all_results:
        lines.append(
            f"  {r['language']:<16} "
            f"{r['jsd_pos']['jsd']:>10.4f} "
            f"{r['jsd_arity']['jsd']:>12.4f} "
            f"{r['jsd_subtree']['jsd']:>14.4f}"
        )

    # MWU arity
    lines.append("")
    lines.append("-" * 72)
    lines.append("  3. MANN-WHITNEY U TEST — ARITY")
    lines.append("     H₁: Real arity values tend to be LOWER than baseline")
    lines.append("")
    lines.append(f"  {'Language':<16} {'p-value':>12} {'Effect r':>10} "
                 f"{'Effect size':>13} {'Sig?':>6}")
    lines.append("  " + "-" * 60)
    for r in all_results:
        m = r["mwu_arity"]
        sig = "YES *" if m["significant"] else "no"
        eff_r = m['effect_size_r'] if m['effect_size_r'] is not None else 0
        lines.append(
            f"  {r['language']:<16} {m['p_value_fmt']:>12} "
            f"{eff_r:>10.4f} {m['effect_label']:>13} {sig:>6}"
        )

    # MWU subtree
    lines.append("")
    lines.append("-" * 72)
    lines.append("  4. MANN-WHITNEY U TEST — SUBTREE SIZE")
    lines.append("     H₁: Real subtree sizes tend to be LOWER than baseline")
    lines.append("")
    lines.append(f"  {'Language':<16} {'p-value':>12} {'Effect r':>10} "
                 f"{'Effect size':>13} {'Sig?':>6}")
    lines.append("  " + "-" * 60)
    for r in all_results:
        m = r["mwu_subtree"]
        sig = "YES *" if m["significant"] else "no"
        eff_r = m['effect_size_r'] if m['effect_size_r'] is not None else 0
        lines.append(
            f"  {r['language']:<16} {m['p_value_fmt']:>12} "
            f"{eff_r:>10.4f} {m['effect_label']:>13} {sig:>6}"
        )

    # Interpretation
    lines.append("")
    lines.append("=" * 72)
    lines.append("  INTERPRETATION SUMMARY")
    lines.append("=" * 72)

    sig_head_rate = [r["language"] for r in all_results
                     if r["head_rate_test"]["significant"]]
    sig_arity     = [r["language"] for r in all_results
                     if r["mwu_arity"]["significant"]]
    sig_subtree   = [r["language"] for r in all_results
                     if r["mwu_subtree"]["significant"]]

    lines.append(f"\n  Languages with significantly lower head-rate than baseline:")
    lines.append(f"    {sig_head_rate or ['None']} "
                 f"({len(sig_head_rate)}/{len(all_results)})")
    lines.append(f"\n  Languages with significantly lower arity than baseline:")
    lines.append(f"    {sig_arity or ['None']} "
                 f"({len(sig_arity)}/{len(all_results)})")
    lines.append(f"\n  Languages with significantly lower subtree size than baseline:")
    lines.append(f"    {sig_subtree or ['None']} "
                 f"({len(sig_subtree)}/{len(all_results)})")

    lines.append("")
    lines.append(f"  * Significance at Bonferroni-corrected α = {ALPHA_CORRECTED:.4f}")

    return "\n".join(lines)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run statistical tests")
    parser.add_argument("--processed_dir", default="./data/processed")
    parser.add_argument("--baseline_dir",  default="./data/baseline")
    parser.add_argument("--results_dir",   default="./data/results")
    parser.add_argument("--language",      default=None)
    parser.add_argument("--seed",          type=int, default=42)
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    baseline_dir  = Path(args.baseline_dir)
    results_dir   = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    langs = [args.language] if args.language else LANGUAGES
    rng = random.Random(args.seed)

    print("=" * 60)
    print("  Step 6 — Statistical Tests")
    print("=" * 60)

    all_results: List[Dict] = []
    for lang in langs:
        result = run_tests_for_language(lang, processed_dir, baseline_dir, rng)
        if result:
            all_results.append(result)

    # JSON
    json_path = results_dir / "statistical_tests.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Written: {json_path}")

    # Text report
    report = generate_report(all_results)
    report_path = results_dir / "statistical_report.txt"
    with open(report_path, "w") as f:
        f.write(report + "\n")
    print(f"  Written: {report_path}")
    print()
    print(report)

    print(f"\nNext: python3 step7_visualizations.py")


if __name__ == "__main__":
    main()
