#!/usr/bin/env python3
"""
STEP 5 — Random linearisation baseline generator
Project: Empirical Distribution of Intervener Complexity

This script generates the NULL DISTRIBUTION for our statistical tests
by randomising word order within each sentence while keeping the
dependency tree structure fixed.

METHODOLOGY (from Yadav et al. 2022, Section 3):
  For each sentence with n tokens:
    1. Keep the tree (all arcs head→dependent) exactly as annotated
    2. Randomly permute the linear positions of all tokens
    3. Re-extract interveners using the same formula:
         interveners(h,d) = {k : min(pos(h),pos(d)) < k < max(pos(h),pos(d))}
    4. Record the same features A,B,C,D for each intervener

  This gives us the EXPECTED distribution under random word order —
  i.e., what intervener profiles would look like if languages did NOT
  minimise intervener complexity.

  If real distributions differ significantly from random → languages
  actively minimise intervener complexity (supporting ICM hypothesis).

PERFORMANCE NOTES (this rewrite vs. the original):
  Random linearisation produces arcs with much longer expected distances
  than real word order, so each permuted sentence yields O(n^2) intervener
  observations on average. With 3 permutations and the Russian SynTagRus
  corpus (~870k sentences), the original script wrote a 2.5 GB CSV and
  then attempted to load it back into memory, killing the process.

  This version:
    - caps sentences per language (default: 20 000) — plenty for stable
      distributional estimates because each sentence yields many observations,
    - uses 1 permutation per sentence by default (configurable),
    - aggregates the distributions IN A SINGLE PASS while writing the CSV,
      so we never re-read the (potentially huge) baseline file from disk,
    - stores raw arity/subtree values via reservoir sampling (capped at
      200k per language) — this is exactly what step 6's Mann-Whitney U
      test consumes (it sub-samples to 50k anyway).

OUTPUT:
  ./data/baseline/<language>_baseline.csv      — per-language baseline CSV
  ./data/aggregated/baseline_features.json    — aggregated, for steps 6 + 7

Usage:
    python3 step5_baseline_generator.py
    python3 step5_baseline_generator.py --max_sentences 5000 --n_permutations 1
    python3 step5_baseline_generator.py --language English
"""

import csv
import json
import math
import random
import argparse
import collections
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from conllu_parser import parse_conllu_file

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

LANGUAGES = [
    "English", "German", "Spanish", "Russian", "Arabic",
    "Hindi", "Turkish", "Finnish", "Japanese", "Chinese",
    "Basque", "Ancient_Greek"
]

TREEBANKS = {
    "English":       "UD_English-EWT",
    "German":        "UD_German-GSD",
    "Spanish":       "UD_Spanish-GSD",
    "Russian":       "UD_Russian-SynTagRus",
    "Arabic":        "UD_Arabic-PADT",
    "Hindi":         "UD_Hindi-HDTB",
    "Turkish":       "UD_Turkish-IMST",
    "Finnish":       "UD_Finnish-TDT",
    "Japanese":      "UD_Japanese-GSD",
    "Chinese":       "UD_Chinese-GSD",
    "Basque":        "UD_Basque-BDT",
    "Ancient_Greek": "UD_Ancient_Greek-PROIEL",
}

TYPOLOGY = {
    "English":       {"word_order": "SVO",  "morph_type": "analytic"},
    "German":        {"word_order": "V2",   "morph_type": "inflecting"},
    "Spanish":       {"word_order": "SVO",  "morph_type": "pro-drop"},
    "Russian":       {"word_order": "free", "morph_type": "inflecting"},
    "Arabic":        {"word_order": "VSO",  "morph_type": "root-pattern"},
    "Hindi":         {"word_order": "SOV",  "morph_type": "inflecting"},
    "Turkish":       {"word_order": "SOV",  "morph_type": "agglutinative"},
    "Finnish":       {"word_order": "SOV",  "morph_type": "agglutinative"},
    "Japanese":      {"word_order": "SOV",  "morph_type": "agglutinative"},
    "Chinese":       {"word_order": "SVO",  "morph_type": "isolating"},
    "Basque":        {"word_order": "SOV",  "morph_type": "agglutinative"},
    "Ancient_Greek": {"word_order": "free", "morph_type": "inflecting"},
}

ALL_UPOS = [
    "NOUN", "VERB", "PRON", "ADJ", "ADV", "ADP", "AUX",
    "CCONJ", "DET", "INTJ", "NUM", "PART", "PROPN",
    "PUNCT", "SCONJ", "SYM", "X"
]

MAX_ARITY_SHOWN   = 10
MAX_SUBTREE_SHOWN = 15

# Reservoir size for raw arity/subtree samples (consumed by step 6's MWU,
# which itself sub-samples to 50k). 200k is comfortably sufficient.
RESERVOIR_CAP = 200_000

# Skip extremely long sentences during baseline generation. A 200-token
# sentence under random linearisation produces O(n^2) ~ 40k intervener
# rows per permutation, which adds nothing to the distribution beyond
# noise but blows up the output file.
MAX_SENT_LEN_FOR_BASELINE = 80

CSV_COLUMNS = [
    "language", "word_order", "sent_id", "sent_length",
    "head_idx_perm", "dep_idx_perm",
    "arc_distance", "num_interveners",
    "intervener_upos", "intervener_arity",
    "intervener_subtree_size", "attachment_type", "is_head",
    "permutation_id",
]


# -----------------------------------------------------------------------
# Reservoir sampler — keeps a uniform sample of size <= cap
# -----------------------------------------------------------------------

class Reservoir:
    """Vitter's Algorithm R — keep a uniform random sample of cap items
    from an unbounded stream without storing the whole stream."""

    def __init__(self, cap: int, rng: random.Random):
        self.cap = cap
        self.rng = rng
        self.buf: List[int] = []
        self.seen = 0

    def add(self, value: int) -> None:
        self.seen += 1
        if len(self.buf) < self.cap:
            self.buf.append(value)
        else:
            j = self.rng.randint(0, self.seen - 1)
            if j < self.cap:
                self.buf[j] = value


# -----------------------------------------------------------------------
# Streaming aggregator — computes all step-4-style summaries online
# -----------------------------------------------------------------------

class BaselineAggregator:
    """Accumulates POS / arity / subtree counts for one language without
    ever loading the full baseline CSV back into memory."""

    def __init__(self, language: str, rng: random.Random):
        self.language     = language
        self.n            = 0
        self.pos_counts   = collections.Counter()
        self.arity_counts = collections.Counter()
        self.subtree_counts = collections.Counter()
        self.attachment_counts = collections.Counter()
        self.is_head_total = 0

        # Running sums for means
        self.arity_sum   = 0
        self.subtree_sum = 0

        # Reservoir samples for medians + Mann-Whitney U in step 6
        self.arity_sample   = Reservoir(RESERVOIR_CAP, rng)
        self.subtree_sample = Reservoir(RESERVOIR_CAP, rng)

    def add(self, row: Dict) -> None:
        self.n += 1
        upos    = row["intervener_upos"]
        arity   = row["intervener_arity"]
        subtree = row["intervener_subtree_size"]

        self.pos_counts[upos] += 1
        self.arity_counts[min(arity, MAX_ARITY_SHOWN)] += 1
        self.subtree_counts[min(subtree, MAX_SUBTREE_SHOWN)] += 1
        self.attachment_counts[row["attachment_type"]] += 1
        self.is_head_total += row["is_head"]

        self.arity_sum   += arity
        self.subtree_sum += subtree

        self.arity_sample.add(arity)
        self.subtree_sample.add(subtree)

    def to_dict(self) -> Dict:
        if self.n == 0:
            return {"language": self.language, "n": 0}

        arity_mean   = self.arity_sum   / self.n
        subtree_mean = self.subtree_sum / self.n
        arity_med    = (statistics.median(self.arity_sample.buf)
                        if self.arity_sample.buf else 0)
        subtree_med  = (statistics.median(self.subtree_sample.buf)
                        if self.subtree_sample.buf else 0)
        head_rate    = self.is_head_total / self.n

        return {
            "language":        self.language,
            "n":               self.n,
            "pos_proportions": {tag: self.pos_counts.get(tag, 0) / self.n
                                for tag in ALL_UPOS},
            "arity_counts":    dict(self.arity_counts),
            "arity_mean":      round(arity_mean, 4),
            "arity_median":    arity_med,
            "subtree_counts":  dict(self.subtree_counts),
            "subtree_mean":    round(subtree_mean, 4),
            "subtree_median":  subtree_med,
            "attachment_counts": dict(self.attachment_counts),
            "head_rate":       round(head_rate, 4),
        }


# -----------------------------------------------------------------------
# Core: apply a random permutation to a sentence
# -----------------------------------------------------------------------

def permute_sentence(sentence, rng: random.Random) -> Dict[int, int]:
    """Apply a single uniform random permutation to a sentence.

    Returns a dict mapping original_idx → new_position (both 1-based).
    The tree structure is unchanged; only linear positions are shuffled.
    Every ordering of n tokens is equally likely (matches Yadav et al.
    2022 and Liu et al. 2017).
    """
    n = len(sentence.tokens)
    positions = list(range(1, n + 1))
    rng.shuffle(positions)
    orig_indices = [t.idx for t in sentence.tokens]
    return {orig: new for orig, new in zip(orig_indices, positions)}


def yield_baseline_rows(sentence, pos_map: Dict[int, int],
                        language: str, perm_id: int):
    """Generator: yield one dict per (arc, intervener) pair for the
    permuted linearisation. All tree-derived quantities (arity, subtree,
    UPOS, attachment) come from the ORIGINAL tree because the tree is
    unchanged — only linear positions are.
    """
    tok_map = {t.idx: t for t in sentence.tokens}
    typo    = TYPOLOGY.get(language, {})
    rev_map = {v: k for k, v in pos_map.items()}
    sent_id = sentence.sent_id or ""
    sent_len = len(sentence.tokens)

    for dep_tok in sentence.tokens:
        if dep_tok.head <= 0:
            continue

        head_orig = dep_tok.head
        dep_orig  = dep_tok.idx

        head_pos = pos_map.get(head_orig)
        dep_pos  = pos_map.get(dep_orig)
        if head_pos is None or dep_pos is None:
            continue

        arc_distance = abs(head_pos - dep_pos)
        if arc_distance <= 1:
            continue

        left  = min(head_pos, dep_pos)
        right = max(head_pos, dep_pos)

        intervener_orig_ids = [rev_map[p] for p in range(left + 1, right)
                               if p in rev_map]
        if not intervener_orig_ids:
            continue

        head_tok = tok_map.get(head_orig)
        if head_tok is None:
            continue

        num_interveners = len(intervener_orig_ids)

        for k_orig in intervener_orig_ids:
            k_tok = tok_map.get(k_orig)
            if k_tok is None:
                continue

            arity   = sentence.arity(k_orig)
            subtree = sentence.subtree_size(k_orig)

            # Attachment uses the ORIGINAL tree heads — the tree is fixed
            if k_tok.head == head_orig:
                attachment = "head"
            elif k_tok.head == dep_orig:
                attachment = "dependent"
            else:
                attachment = "external"

            is_head = 1 if arity > 0 else 0

            yield {
                "language":                language,
                "word_order":              typo.get("word_order", ""),
                "sent_id":                 sent_id,
                "sent_length":             sent_len,
                "head_idx_perm":           head_pos,
                "dep_idx_perm":            dep_pos,
                "arc_distance":            arc_distance,
                "num_interveners":         num_interveners,
                "intervener_upos":         k_tok.upos,
                "intervener_arity":        arity,
                "intervener_subtree_size": subtree,
                "attachment_type":         attachment,
                "is_head":                 is_head,
                "permutation_id":          perm_id,
            }


# -----------------------------------------------------------------------
# Per-language baseline processing — single streaming pass
# -----------------------------------------------------------------------

def process_language_baseline(
    language: str,
    treebank_dir: Path,
    out_dir: Path,
    n_permutations: int = 1,
    max_sentences: Optional[int] = 20_000,
    seed: int = 42,
) -> Optional[Dict]:
    """For each sentence in the language's treebank:
      - apply ``n_permutations`` random word-order permutations
      - write the resulting (arc, intervener) rows to a per-language CSV
      - simultaneously update the streaming aggregator
    Returns the aggregated dict (already includes all stats step 6/7 need),
    or ``None`` if the treebank is missing.
    """
    conllu_files = sorted(treebank_dir.glob("*.conllu"))
    if not conllu_files:
        print(f"  [SKIP] {language}: no .conllu files in {treebank_dir}")
        return None

    rng = random.Random(seed)
    agg = BaselineAggregator(language, rng)

    out_path = out_dir / f"{language.lower()}_baseline.csv"
    sentences = 0
    observations = 0
    skipped_long = 0

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        done = False
        for conllu_path in conllu_files:
            if done:
                break
            for sentence in parse_conllu_file(str(conllu_path)):
                if max_sentences is not None and sentences >= max_sentences:
                    done = True
                    break

                if len(sentence.tokens) > MAX_SENT_LEN_FOR_BASELINE:
                    skipped_long += 1
                    continue

                for perm_id in range(n_permutations):
                    pos_map = permute_sentence(sentence, rng)
                    for row in yield_baseline_rows(
                        sentence, pos_map, language, perm_id
                    ):
                        writer.writerow(row)
                        agg.add(row)
                        observations += 1

                sentences += 1
                if sentences % 5000 == 0:
                    print(f"  [{language}] {sentences:,} sentences "
                          f"| {observations:,} baseline observations")

    print(f"  [{language}] Done — {sentences:,} sentences "
          f"({skipped_long:,} skipped len>{MAX_SENT_LEN_FOR_BASELINE}), "
          f"{observations:,} observations → {out_path.name}")
    return agg.to_dict()


# -----------------------------------------------------------------------
# Cross-linguistic aggregator (combines per-language aggregates)
# -----------------------------------------------------------------------

def combine_aggregates(per_lang: List[Dict]) -> Dict:
    """Merge the dict-form aggregates from each language into a global
    distribution. We can do this analytically because each per-language
    aggregate carries its observation count `n`, count tables, and means.
    Medians of the union are approximated using the union of the
    reservoir samples is *not* possible here (we already discarded raw
    samples); instead we use the count-table to compute the median.
    """
    keep = [d for d in per_lang if d and d.get("n", 0) > 0]
    if not keep:
        return {"language": "ALL", "n": 0}

    n_total       = sum(d["n"] for d in keep)
    pos_props_all = {tag: 0.0 for tag in ALL_UPOS}
    for d in keep:
        for tag in ALL_UPOS:
            pos_props_all[tag] += d["pos_proportions"].get(tag, 0.0) * d["n"]
    pos_props_all = {tag: v / n_total for tag, v in pos_props_all.items()}

    arity_counts_all   = collections.Counter()
    subtree_counts_all = collections.Counter()
    attach_counts_all  = collections.Counter()
    for d in keep:
        arity_counts_all.update(
            {int(k): v for k, v in d.get("arity_counts", {}).items()}
        )
        subtree_counts_all.update(
            {int(k): v for k, v in d.get("subtree_counts", {}).items()}
        )
        attach_counts_all.update(d.get("attachment_counts", {}))

    arity_mean = sum(d["arity_mean"]   * d["n"] for d in keep) / n_total
    sub_mean   = sum(d["subtree_mean"] * d["n"] for d in keep) / n_total
    head_rate  = sum(d["head_rate"]    * d["n"] for d in keep) / n_total

    def median_from_counts(counts: Dict[int, int]) -> int:
        if not counts:
            return 0
        items = sorted(counts.items())
        total = sum(c for _, c in items)
        cum = 0
        target = total / 2
        for v, c in items:
            cum += c
            if cum >= target:
                return v
        return items[-1][0]

    return {
        "language":          "ALL",
        "n":                 n_total,
        "pos_proportions":   pos_props_all,
        "arity_counts":      dict(arity_counts_all),
        "arity_mean":        round(arity_mean, 4),
        "arity_median":      median_from_counts(arity_counts_all),
        "subtree_counts":    dict(subtree_counts_all),
        "subtree_mean":      round(sub_mean, 4),
        "subtree_median":    median_from_counts(subtree_counts_all),
        "attachment_counts": dict(attach_counts_all),
        "head_rate":         round(head_rate, 4),
    }


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate random linearisation baseline")
    parser.add_argument("--data_dir",        default="./data/raw")
    parser.add_argument("--out_dir",         default="./data/baseline")
    parser.add_argument("--aggregated_dir",  default="./data/aggregated")
    parser.add_argument("--n_permutations",  type=int, default=1,
                        help="Random permutations per sentence (default: 1)")
    parser.add_argument("--max_sentences",   type=int, default=20_000,
                        help="Sentences per language; -1 for unlimited (default: 20000)")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--language",        default=None,
                        help="Process one language only")
    args = parser.parse_args()

    data_dir       = Path(args.data_dir)
    out_dir        = Path(args.out_dir)
    aggregated_dir = Path(args.aggregated_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregated_dir.mkdir(parents=True, exist_ok=True)

    langs = [args.language] if args.language else LANGUAGES
    max_sents = None if args.max_sentences is not None and args.max_sentences < 0 \
                else args.max_sentences

    print("=" * 60)
    print("  Step 5 — Random Linearisation Baseline")
    print(f"  Languages                : {len(langs)}")
    print(f"  Permutations per sentence: {args.n_permutations}")
    print(f"  Max sentences per language: "
          f"{'unlimited' if max_sents is None else f'{max_sents:,}'}")
    print(f"  Seed                     : {args.seed}")
    print("=" * 60)

    lang_baselines: List[Dict] = []

    for lang in langs:
        if lang not in TREEBANKS:
            print(f"  [WARN] Unknown language: {lang} (skipping)")
            continue
        treebank_dir = data_dir / TREEBANKS[lang]
        if not treebank_dir.exists():
            print(f"  [MISS] {lang}: {treebank_dir} not found")
            continue

        agg = process_language_baseline(
            language=lang,
            treebank_dir=treebank_dir,
            out_dir=out_dir,
            n_permutations=args.n_permutations,
            max_sentences=max_sents,
            seed=args.seed,
        )
        if agg is not None:
            lang_baselines.append(agg)

    print("\n  Aggregating cross-linguistic baseline...")
    all_baseline_agg = combine_aggregates(lang_baselines)

    # Persist for steps 6 + 7
    json_out = {
        "languages": lang_baselines,
        "all":       all_baseline_agg,
    }
    json_path = aggregated_dir / "baseline_features.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Written: {json_path}")

    # Summary table
    print("\n" + "=" * 60)
    print(f"  {'Language':<16} {'N baseline':>12} {'Head rate':>10} {'Mean arity':>11}")
    print("  " + "-" * 52)
    for b in lang_baselines:
        print(f"  {b['language']:<16} {b['n']:>12,} "
              f"{b['head_rate']:>10.3f} {b['arity_mean']:>11.3f}")
    print("  " + "-" * 52)
    b = all_baseline_agg
    if b.get("n", 0) > 0:
        print(f"  {'ALL':<16} {b['n']:>12,} "
              f"{b['head_rate']:>10.3f} {b['arity_mean']:>11.3f}")

    print(f"\nDone. Outputs in {out_dir}/")
    print("Next: python3 step6_statistical_tests.py")


if __name__ == "__main__":
    main()
