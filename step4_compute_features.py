#!/usr/bin/env python3
"""
STEP 4 — Feature aggregation and distribution computation
Project: Empirical Distribution of Intervener Complexity

Reads the CSVs produced by step3_extract_interveners.py and computes:
  - POS distribution per language + cross-linguistically        (Feature A)
  - Arity distribution per language + cross-linguistically      (Feature B)
  - Subtree size distribution per language + cross-linguistically (Feature C)
  - Attachment type distribution                                 (Feature D)
  - Intervener head-rate (is_head proportion) per language
  - Arc-level summary statistics

Writes all aggregated tables to ./data/aggregated/ as CSV files
that steps 5, 6 (stats) and 7 (plots) will consume.

Usage:
    python3 step4_compute_features.py
    python3 step4_compute_features.py --processed_dir ./data/processed
"""

import os
import csv
import json
import argparse
import collections
import statistics
from pathlib import Path
from typing import Dict, List, Tuple


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

LANGUAGES = [
    "English", "German", "Spanish", "Russian", "Arabic",
    "Hindi", "Turkish", "Finnish", "Japanese", "Chinese",
    "Basque", "Ancient_Greek"
]

# All 17 Universal POS tags (UD tagset)
ALL_UPOS = [
    "NOUN", "VERB", "PRON", "ADJ", "ADV", "ADP", "AUX",
    "CCONJ", "DET", "INTJ", "NUM", "PART", "PROPN",
    "PUNCT", "SCONJ", "SYM", "X"
]

# Word-order groupings for cross-linguistic comparison
WORD_ORDER_GROUPS = {
    "SOV":  ["Hindi", "Turkish", "Finnish", "Japanese", "Basque"],
    "SVO":  ["English", "German", "Spanish", "Chinese"],
    "free": ["Russian", "Ancient_Greek"],
    "VSO":  ["Arabic"],
}

MAX_ARITY_SHOWN    = 10   # cap arity distribution display at 10+
MAX_SUBTREE_SHOWN  = 15   # cap subtree size display at 15+


# -----------------------------------------------------------------------
# CSV reader (no pandas dependency as fallback — but we have pandas)
# -----------------------------------------------------------------------

def load_language_csv(path: Path) -> List[Dict]:
    """Load a per-language intervener CSV into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Cast numeric columns
            for col in ("arc_distance", "num_interveners",
                        "intervener_arity", "intervener_subtree_size",
                        "is_head", "sent_length",
                        "head_idx", "dep_idx", "intervener_idx"):
                try:
                    row[col] = int(row[col])
                except (ValueError, KeyError):
                    row[col] = 0
            rows.append(row)
    return rows


# -----------------------------------------------------------------------
# Feature A — POS distribution
# -----------------------------------------------------------------------

def compute_pos_distribution(rows: List[Dict]) -> Dict[str, int]:
    """Count frequency of each UPOS tag across intervener observations."""
    counts = collections.Counter()
    for r in rows:
        counts[r["intervener_upos"]] += 1
    return dict(counts)


def compute_pos_proportions(counts: Dict[str, int]) -> Dict[str, float]:
    """Convert raw counts to proportions (sum to 1.0)."""
    total = sum(counts.values())
    if total == 0:
        return {tag: 0.0 for tag in ALL_UPOS}
    return {tag: counts.get(tag, 0) / total for tag in ALL_UPOS}


# -----------------------------------------------------------------------
# Feature B — Arity distribution
# -----------------------------------------------------------------------

def compute_arity_distribution(rows: List[Dict]) -> Dict[int, int]:
    """
    Count frequency of each arity value (0, 1, 2, ..., MAX_ARITY_SHOWN+).
    Arity is capped at MAX_ARITY_SHOWN to keep histograms readable.
    """
    counts = collections.Counter()
    for r in rows:
        arity = min(r["intervener_arity"], MAX_ARITY_SHOWN)
        counts[arity] += 1
    return dict(counts)


def arity_summary_stats(rows: List[Dict]) -> Dict:
    """Mean, median, mode, % leaves (arity=0), % heads (arity>0)."""
    arities = [r["intervener_arity"] for r in rows]
    if not arities:
        return {}
    total = len(arities)
    leaves = sum(1 for a in arities if a == 0)
    return {
        "mean":        round(statistics.mean(arities), 4),
        "median":      statistics.median(arities),
        "mode":        statistics.mode(arities),
        "pct_leaves":  round(leaves / total * 100, 2),
        "pct_heads":   round((total - leaves) / total * 100, 2),
        "n":           total,
    }


# -----------------------------------------------------------------------
# Feature C — Subtree size distribution
# -----------------------------------------------------------------------

def compute_subtree_distribution(rows: List[Dict]) -> Dict[int, int]:
    """
    Count frequency of each subtree size (1, 2, ..., MAX_SUBTREE_SHOWN+).
    """
    counts = collections.Counter()
    for r in rows:
        size = min(r["intervener_subtree_size"], MAX_SUBTREE_SHOWN)
        counts[size] += 1
    return dict(counts)


def subtree_summary_stats(rows: List[Dict]) -> Dict:
    """Mean, median, % singletons (size=1)."""
    sizes = [r["intervener_subtree_size"] for r in rows]
    if not sizes:
        return {}
    total = len(sizes)
    singletons = sum(1 for s in sizes if s == 1)
    return {
        "mean":           round(statistics.mean(sizes), 4),
        "median":         statistics.median(sizes),
        "pct_singletons": round(singletons / total * 100, 2),
        "n":              total,
    }


# -----------------------------------------------------------------------
# Feature D — Attachment type distribution
# -----------------------------------------------------------------------

def compute_attachment_distribution(rows: List[Dict]) -> Dict[str, int]:
    counts = collections.Counter()
    for r in rows:
        counts[r["attachment_type"]] += 1
    return dict(counts)


# -----------------------------------------------------------------------
# Intervener head-rate (core ICM metric from Yadav et al. 2022)
# -----------------------------------------------------------------------

def compute_head_rate(rows: List[Dict]) -> float:
    """
    Proportion of interveners that are syntactic heads (arity > 0).
    This is the key measure in Yadav et al. (2022):
      ICM = minimise the number of intervening HEADS.
    A lower head-rate means the language conforms more strongly to ICM.
    """
    if not rows:
        return 0.0
    return sum(r["is_head"] for r in rows) / len(rows)


# -----------------------------------------------------------------------
# Arc-level summary
# -----------------------------------------------------------------------

def compute_arc_stats(rows: List[Dict]) -> Dict:
    """
    Stats about the arcs themselves (not the interveners):
      - distribution of arc distance
      - distribution of num_interveners per arc
    We deduplicate to one row per arc first.
    """
    seen_arcs = {}
    for r in rows:
        key = (r["sent_id"], r["head_idx"], r["dep_idx"])
        if key not in seen_arcs:
            seen_arcs[key] = {
                "arc_distance":    r["arc_distance"],
                "num_interveners": r["num_interveners"],
            }

    arcs = list(seen_arcs.values())
    distances = [a["arc_distance"] for a in arcs]
    n_interv  = [a["num_interveners"] for a in arcs]

    if not distances:
        return {}

    return {
        "num_arcs":          len(arcs),
        "mean_arc_distance": round(statistics.mean(distances), 4),
        "median_arc_distance": statistics.median(distances),
        "mean_num_interveners": round(statistics.mean(n_interv), 4),
        "arc_dist_distribution": dict(
            collections.Counter(min(d, 20) for d in distances)
        ),
    }


# -----------------------------------------------------------------------
# Main aggregation routine
# -----------------------------------------------------------------------

def aggregate_language(language: str, processed_dir: Path) -> Dict:
    """
    Load a language's CSV and compute all feature distributions.
    Returns a dict of everything needed for stats + plotting.
    """
    csv_path = processed_dir / f"{language.lower()}_interveners.csv"
    if not csv_path.exists():
        print(f"  [SKIP] {language}: file not found at {csv_path}")
        return None

    print(f"  [{language}] Loading {csv_path.name} ...")
    rows = load_language_csv(csv_path)
    print(f"  [{language}] {len(rows):,} intervener observations")

    result = {
        "language":           language,
        "n_observations":     len(rows),
        "pos_counts":         compute_pos_distribution(rows),
        "pos_proportions":    compute_pos_proportions(compute_pos_distribution(rows)),
        "arity_counts":       compute_arity_distribution(rows),
        "arity_stats":        arity_summary_stats(rows),
        "subtree_counts":     compute_subtree_distribution(rows),
        "subtree_stats":      subtree_summary_stats(rows),
        "attachment_counts":  compute_attachment_distribution(rows),
        "head_rate":          round(compute_head_rate(rows), 4),
        "arc_stats":          compute_arc_stats(rows),
        # Store raw arity and subtree lists for statistical tests
        "arity_raw":          [r["intervener_arity"] for r in rows],
        "subtree_raw":        [r["intervener_subtree_size"] for r in rows],
    }
    return result


def aggregate_all(all_rows: List[Dict]) -> Dict:
    """Aggregate across all languages combined."""
    print(f"  [ALL] Aggregating {len(all_rows):,} total observations ...")
    return {
        "language":           "ALL",
        "n_observations":     len(all_rows),
        "pos_counts":         compute_pos_distribution(all_rows),
        "pos_proportions":    compute_pos_proportions(compute_pos_distribution(all_rows)),
        "arity_counts":       compute_arity_distribution(all_rows),
        "arity_stats":        arity_summary_stats(all_rows),
        "subtree_counts":     compute_subtree_distribution(all_rows),
        "subtree_stats":      subtree_summary_stats(all_rows),
        "attachment_counts":  compute_attachment_distribution(all_rows),
        "head_rate":          round(compute_head_rate(all_rows), 4),
        "arc_stats":          compute_arc_stats(all_rows),
        "arity_raw":          [r["intervener_arity"] for r in all_rows],
        "subtree_raw":        [r["intervener_subtree_size"] for r in all_rows],
    }


# -----------------------------------------------------------------------
# Write output tables
# -----------------------------------------------------------------------

def write_pos_table(lang_results: List[Dict], out_dir: Path):
    """Write POS proportion table: rows=languages, cols=UPOS tags."""
    path = out_dir / "pos_proportions_table.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["language"] + ALL_UPOS + ["n_observations"])
        for r in lang_results:
            if r is None:
                continue
            props = r["pos_proportions"]
            writer.writerow(
                [r["language"]] +
                [round(props.get(tag, 0.0), 4) for tag in ALL_UPOS] +
                [r["n_observations"]]
            )
    print(f"  Written: {path.name}")


def write_arity_table(lang_results: List[Dict], out_dir: Path):
    """Write arity summary stats table."""
    path = out_dir / "arity_summary_table.csv"
    cols = ["language", "n", "mean", "median", "mode", "pct_leaves", "pct_heads", "head_rate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in lang_results:
            if r is None:
                continue
            s = r["arity_stats"]
            writer.writerow([
                r["language"], s.get("n", 0),
                s.get("mean", 0), s.get("median", 0),
                s.get("mode", 0), s.get("pct_leaves", 0),
                s.get("pct_heads", 0), r["head_rate"]
            ])
    print(f"  Written: {path.name}")


def write_subtree_table(lang_results: List[Dict], out_dir: Path):
    """Write subtree size summary stats table."""
    path = out_dir / "subtree_summary_table.csv"
    cols = ["language", "n", "mean", "median", "pct_singletons"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in lang_results:
            if r is None:
                continue
            s = r["subtree_stats"]
            writer.writerow([
                r["language"], s.get("n", 0),
                s.get("mean", 0), s.get("median", 0),
                s.get("pct_singletons", 0)
            ])
    print(f"  Written: {path.name}")


def write_aggregated_json(lang_results: List[Dict], all_result: Dict, out_dir: Path):
    """
    Write the full aggregated data as JSON.
    Steps 5 (baseline), 6 (stats), and 7 (plots) all load this file.
    The raw arity and subtree lists are excluded (too large) — those
    come from the per-language CSVs directly.
    """
    # Strip raw lists before serializing (they're huge)
    def strip_raw(r):
        if r is None:
            return None
        out = {k: v for k, v in r.items() if not k.endswith("_raw")}
        return out

    data = {
        "languages":  [strip_raw(r) for r in lang_results if r is not None],
        "all":        strip_raw(all_result),
        "word_order_groups": WORD_ORDER_GROUPS,
    }
    path = out_dir / "aggregated_features.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path.name}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Aggregate intervener features")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--out_dir",       default="data/aggregated")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    out_dir       = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Step 4 — Feature Aggregation")
    print("=" * 60)

    # Per-language aggregation
    lang_results = []
    all_rows     = []

    for lang in LANGUAGES:
        result = aggregate_language(lang, processed_dir)
        lang_results.append(result)

        # Also accumulate all rows for cross-linguistic aggregate
        csv_path = processed_dir / f"{lang.lower()}_interveners.csv"
        if csv_path.exists():
            rows = load_language_csv(csv_path)
            all_rows.extend(rows)

    # Cross-linguistic aggregate
    print("\n  Computing cross-linguistic aggregate...")
    all_result = aggregate_all(all_rows)

    # Write output tables
    print("\n  Writing output tables...")
    valid = [r for r in lang_results if r is not None]
    valid_with_all = valid + [all_result]

    write_pos_table(valid_with_all, out_dir)
    write_arity_table(valid_with_all, out_dir)
    write_subtree_table(valid_with_all, out_dir)
    write_aggregated_json(lang_results, all_result, out_dir)

    # Print summary to terminal
    print("\n" + "=" * 60)
    print(f"  {'Language':<16} {'N interveners':>14} {'Head rate':>10} {'Mean arity':>11} {'Mean subtree':>13}")
    print("  " + "-" * 56)
    for r in valid:
        print(f"  {r['language']:<16} {r['n_observations']:>14,} "
              f"{r['head_rate']:>10.3f} "
              f"{r['arity_stats'].get('mean', 0):>11.3f} "
              f"{r['subtree_stats'].get('mean', 0):>13.3f}")
    print("  " + "-" * 56)
    r = all_result
    print(f"  {'ALL':<16} {r['n_observations']:>14,} "
          f"{r['head_rate']:>10.3f} "
          f"{r['arity_stats'].get('mean', 0):>11.3f} "
          f"{r['subtree_stats'].get('mean', 0):>13.3f}")

    print(f"\nDone. Outputs in {out_dir}/")
    print("Next: python3 step5_baseline_generator.py")


if __name__ == "__main__":
    main()
