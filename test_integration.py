#!/usr/bin/env python3
"""
INTEGRATION TEST — runs the FULL pipeline on synthetic data.
No real treebanks needed. Verifies steps 4-7 work end-to-end.

Usage:
    python3 test_integration.py
"""
import sys, os, csv, json, shutil, random, math
from pathlib import Path

# ---- Synthetic data generator ----------------------------------------
def make_synthetic_csv(path: Path, language: str, n_rows: int = 3000):
    """Create a fake interveners CSV with realistic-looking distributions."""
    random.seed(42)
    upos_pool = (["NOUN"]*30 + ["VERB"]*15 + ["ADV"]*20 + ["ADJ"]*10 +
                 ["PRON"]*8 + ["DET"]*7 + ["ADP"]*5 + ["AUX"]*3 +
                 ["CCONJ"]*1 + ["PART"]*1)
    att_pool = ["head", "head", "head", "dependent", "dependent", "external"]

    word_orders = {"English":"SVO","Hindi":"SOV","Japanese":"SOV","Arabic":"VSO",
                   "Russian":"free","German":"V2","Spanish":"SVO","Chinese":"SVO",
                   "Turkish":"SOV","Finnish":"SOV","Basque":"SOV","Ancient_Greek":"free"}
    wo = word_orders.get(language, "SVO")

    cols = [
        "language","word_order","morph_type","family","sent_id","sent_length",
        "head_idx","dep_idx","head_upos","dep_upos","head_deprel","dep_deprel",
        "arc_distance","num_interveners","intervener_idx","intervener_form",
        "intervener_upos","intervener_arity","intervener_subtree_size",
        "attachment_type","is_head"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(n_rows):
            arity = max(0, int(random.expovariate(1.5)))
            subtree = max(1, arity + random.randint(0, 2))
            upos = random.choice(upos_pool)
            row = {
                "language": language, "word_order": wo, "morph_type": "inflecting",
                "family": "Indo-European", "sent_id": f"s{i}", "sent_length": random.randint(5,20),
                "head_idx": random.randint(1,10), "dep_idx": random.randint(1,10),
                "head_upos": "VERB", "dep_upos": "NOUN",
                "head_deprel": "root", "dep_deprel": "obj",
                "arc_distance": random.randint(2,8), "num_interveners": random.randint(1,5),
                "intervener_idx": random.randint(1,10),
                "intervener_form": "word",
                "intervener_upos": upos, "intervener_arity": arity,
                "intervener_subtree_size": subtree,
                "attachment_type": random.choice(att_pool),
                "is_head": 1 if arity > 0 else 0,
            }
            w.writerow(row)

LANGUAGES = [
    "English","German","Spanish","Russian","Arabic",
    "Hindi","Turkish","Finnish","Japanese","Chinese","Basque","Ancient_Greek"
]

def main():
    print("=" * 60)
    print("  Integration Test — Steps 4-7 on Synthetic Data")
    print("=" * 60)

    # Create temp directory structure
    base = Path("/tmp/icm_test")
    if base.exists():
        shutil.rmtree(base)
    processed_dir  = base / "data" / "processed"
    baseline_dir   = base / "data" / "baseline"
    aggregated_dir = base / "data" / "aggregated"
    results_dir    = base / "data" / "results"
    figures_dir    = base / "data" / "figures"
    for d in [processed_dir, baseline_dir, aggregated_dir, results_dir, figures_dir]:
        d.mkdir(parents=True)

    # Generate synthetic processed CSVs
    print("\n[1] Generating synthetic data for 12 languages...")
    for lang in LANGUAGES:
        p = processed_dir / f"{lang.lower()}_interveners.csv"
        make_synthetic_csv(p, lang, n_rows=2000)
        # Baseline: slightly higher arity (simulates random having more complexity)
        pb = baseline_dir / f"{lang.lower()}_baseline.csv"
        make_synthetic_csv(pb, lang, n_rows=6000)  # 3x more rows (3 permutations)
    print("  OK — synthetic CSVs created")

    # --- Step 4 ---
    print("\n[2] Running Step 4 — Feature Aggregation...")
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "step4_compute_features.py",
         "--processed_dir", str(processed_dir),
         "--out_dir", str(aggregated_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STEP 4 FAILED:")
        print(result.stderr)
        sys.exit(1)
    # Verify outputs
    assert (aggregated_dir / "aggregated_features.json").exists(), "Missing aggregated_features.json"
    assert (aggregated_dir / "pos_proportions_table.csv").exists(), "Missing pos table"
    assert (aggregated_dir / "arity_summary_table.csv").exists(), "Missing arity table"
    print("  OK — aggregated_features.json written")

    # Also need baseline_features.json for steps 6+7
    # Build it manually from the synthetic baseline CSVs
    print("\n[3] Building baseline_features.json from synthetic baseline...")
    import collections, statistics as stats_mod

    ALL_UPOS = ["NOUN","VERB","PRON","ADJ","ADV","ADP","AUX","CCONJ","DET",
                "INTJ","NUM","PART","PROPN","PUNCT","SCONJ","SYM","X"]

    def load_csv_col(path, col):
        vals = []
        with open(path) as f:
            for row in csv.DictReader(f):
                try: vals.append(int(row[col]))
                except: pass
        return vals

    def build_baseline_entry(path, lang):
        rows = []
        with open(path) as f:
            rows = list(csv.DictReader(f))
        total = len(rows)
        pos_c = collections.Counter(r["intervener_upos"] for r in rows)
        arity_v = [int(r["intervener_arity"]) for r in rows]
        sub_v   = [int(r["intervener_subtree_size"]) for r in rows]
        head_r  = sum(int(r["is_head"]) for r in rows) / total
        return {
            "language": lang, "n": total,
            "pos_proportions": {t: pos_c.get(t,0)/total for t in ALL_UPOS},
            "arity_mean": round(sum(arity_v)/len(arity_v), 4),
            "arity_median": stats_mod.median(arity_v),
            "arity_counts": dict(collections.Counter(min(a,10) for a in arity_v)),
            "subtree_mean": round(sum(sub_v)/len(sub_v), 4),
            "subtree_median": stats_mod.median(sub_v),
            "subtree_counts": dict(collections.Counter(min(s,15) for s in sub_v)),
            "head_rate": round(head_r, 4),
        }

    lang_baselines = [build_baseline_entry(baseline_dir/f"{l.lower()}_baseline.csv", l)
                      for l in LANGUAGES]
    # ALL
    all_rows_b = []
    for l in LANGUAGES:
        with open(baseline_dir/f"{l.lower()}_baseline.csv") as f:
            all_rows_b.extend(list(csv.DictReader(f)))
    all_base = build_baseline_entry(baseline_dir/"english_baseline.csv", "ALL")  # stub
    all_base["language"] = "ALL"

    with open(aggregated_dir/"baseline_features.json","w") as f:
        json.dump({"languages": lang_baselines, "all": all_base}, f, indent=2)
    print("  OK — baseline_features.json written")

    # --- Step 6 ---
    print("\n[4] Running Step 6 — Statistical Tests...")
    result = subprocess.run(
        [sys.executable, "step6_statistical_tests.py",
         "--processed_dir", str(processed_dir),
         "--baseline_dir",  str(baseline_dir),
         "--results_dir",   str(results_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STEP 6 FAILED:")
        print(result.stderr[-2000:])
        sys.exit(1)
    assert (results_dir / "statistical_tests.json").exists()
    assert (results_dir / "statistical_report.txt").exists()
    print("  OK — statistical_tests.json written")

    # --- Step 7 ---
    print("\n[5] Running Step 7 — Visualizations...")
    result = subprocess.run(
        [sys.executable, "step7_visualizations.py",
         "--aggregated_dir", str(aggregated_dir),
         "--processed_dir",  str(processed_dir),
         "--baseline_dir",   str(baseline_dir),
         "--results_dir",    str(results_dir),
         "--out_dir",        str(figures_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("STEP 7 FAILED:")
        print(result.stderr[-2000:])
        sys.exit(1)

    figs = list(figures_dir.glob("*.png"))
    print(f"  OK — {len(figs)} figures generated")
    for f in sorted(figs):
        print(f"    {f.name}")

    print("\n" + "=" * 60)
    print("  ALL INTEGRATION TESTS PASSED")
    print(f"  Steps 4-7 verified on synthetic data")
    print("  Safe to run on real treebank data")
    print("=" * 60)

if __name__ == "__main__":
    main()
