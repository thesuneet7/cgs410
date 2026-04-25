#!/usr/bin/env python3
"""
STEP 3 — Extract intervening nodes from dependency arcs
Project: Empirical Distribution of Intervener Complexity
Authors: Aditya, Anya, Devansh, Nitika, Pubali, Sidhant, Srishti, Suneet

This is the CORE SCRIPT for Steps 1–3. It:
  1. Loads all 12 UD treebanks from ./data/raw/
  2. For every dependency arc (h → d) where distance > 1:
       - Identifies all intervening tokens k: min(h,d) < k < max(h,d)
  3. For each intervener, extracts four features:
       A. UPOS (part-of-speech)
       B. Arity (number of direct syntactic children)
       C. Subtree size (number of tokens dominated)
       D. Attachment type (head-attached / dep-attached / external)
  4. Writes per-language CSV files to ./data/processed/
  5. Writes a combined cross-linguistic CSV to ./data/processed/all_languages.csv

METHODOLOGY REFERENCE:
  Yadav, H., Mittal, S., & Husain, S. (2022). A reappraisal of dependency
  length minimization as a linguistic universal. Open Mind, 6, 147–168.
  https://doi.org/10.1162/opmi_a_00060

  Intervener = any token k where min(pos(h), pos(d)) < k < max(pos(h), pos(d))
  Intervener Complexity = number of syntactic heads in the intervening region
  (This script records ALL interveners; the head-counting is done in Step 4)

USAGE:
  python3 step3_extract_interveners.py
  python3 step3_extract_interveners.py --data_dir ./data/raw --out_dir ./data/processed
  python3 step3_extract_interveners.py --language English  # single language test
  python3 step3_extract_interveners.py --max_sentences 1000  # quick test run
"""

import os
import sys
import csv
import time
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Iterator

# Import our custom parser (must be in the same directory)
from conllu_parser import parse_conllu_file, Sentence, Token

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# All 12 languages from our proposal (language_name → treebank_folder)
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

# Language typology metadata — used for cross-linguistic analysis
# (word order and morphological type from our proposal)
TYPOLOGY = {
    "English":       {"word_order": "SVO",  "morph_type": "analytic",      "family": "Indo-European"},
    "German":        {"word_order": "V2",   "morph_type": "inflecting",    "family": "Indo-European"},
    "Spanish":       {"word_order": "SVO",  "morph_type": "pro-drop",      "family": "Indo-European"},
    "Russian":       {"word_order": "free", "morph_type": "inflecting",    "family": "Indo-European"},
    "Arabic":        {"word_order": "VSO",  "morph_type": "root-pattern",  "family": "Afro-Asiatic"},
    "Hindi":         {"word_order": "SOV",  "morph_type": "inflecting",    "family": "Indo-European"},
    "Turkish":       {"word_order": "SOV",  "morph_type": "agglutinative", "family": "Turkic"},
    "Finnish":       {"word_order": "SOV",  "morph_type": "agglutinative", "family": "Uralic"},
    "Japanese":      {"word_order": "SOV",  "morph_type": "agglutinative", "family": "Japonic"},
    "Chinese":       {"word_order": "SVO",  "morph_type": "isolating",     "family": "Sino-Tibetan"},
    "Basque":        {"word_order": "SOV",  "morph_type": "agglutinative", "family": "Language_isolate"},
    "Ancient_Greek": {"word_order": "free", "morph_type": "inflecting",    "family": "Indo-European"},
}

# Output CSV columns (one row per intervener observation)
CSV_COLUMNS = [
    "language",         # e.g. "English"
    "word_order",       # e.g. "SVO"
    "morph_type",       # e.g. "analytic"
    "family",           # e.g. "Indo-European"
    "sent_id",          # sentence identifier from CoNLL-U
    "sent_length",      # total tokens in sentence
    "head_idx",         # 1-based position of arc head
    "dep_idx",          # 1-based position of arc dependent
    "head_upos",        # UPOS of the arc head
    "dep_upos",         # UPOS of the arc dependent
    "head_deprel",      # dependency relation of the arc (arc label)
    "dep_deprel",       # dependency relation of dependent to its own head
    "arc_distance",     # |head_idx - dep_idx|  (dependency length in words)
    "num_interveners",  # count of tokens between h and d
    "intervener_idx",   # 1-based position of THIS intervener
    "intervener_form",  # surface form (useful for debugging)
    "intervener_upos",  # UPOS of intervener  ← Feature A
    "intervener_arity", # number of direct children  ← Feature B
    "intervener_subtree_size",  # subtree token count  ← Feature C
    "attachment_type",  # head / dependent / external  ← Feature D
    "is_head",          # 1 if intervener is itself a syntactic head (arity > 0)
]


# -----------------------------------------------------------------------
# Core extraction logic
# -----------------------------------------------------------------------

def extract_interveners_from_sentence(
    sentence: Sentence,
    language: str
) -> List[Dict]:
    """
    Extract all intervener observations from a single sentence.

    For each dependency arc (h → d) where |pos(h) - pos(d)| > 1:
        Find all tokens k where min(h, d) < k < max(h, d)
        Record features A, B, C, D for each intervener k

    Args:
        sentence: A parsed Sentence object
        language: Language name (for output rows)

    Returns:
        List of dicts, one per (arc, intervener) pair. If a sentence has
        3 arcs each with 2 interveners, this returns 6 rows.

    KEY FORMULA (from Yadav et al. 2022, extended in our proposal):
        For arc head h at position i, dependent d at position j:
            interveners = {k : min(i,j) < k < max(i,j)}
    """
    rows = []
    tokens = sentence.tokens
    n = len(tokens)

    # Build a fast lookup: token_idx → token object
    # (tokens are already 1-indexed and contiguous after our parser)
    tok_map = {t.idx: t for t in tokens}

    typo = TYPOLOGY.get(language, {})

    for dep_tok in tokens:
        # Skip root token (head = 0, no arc to extract)
        if dep_tok.head <= 0:
            continue

        head_idx = dep_tok.head
        dep_idx = dep_tok.idx

        # Skip self-loops (should not occur in valid UD, but defensive check)
        if head_idx == dep_idx:
            continue

        arc_distance = abs(head_idx - dep_idx)

        # Only process non-adjacent arcs (adjacent arcs have no interveners)
        if arc_distance <= 1:
            continue

        # Determine the intervening span
        left = min(head_idx, dep_idx)
        right = max(head_idx, dep_idx)
        # Interveners are strictly between left and right (exclusive)
        intervener_indices = list(range(left + 1, right))

        if not intervener_indices:
            continue  # defensive; shouldn't happen when arc_distance > 1

        head_tok = tok_map.get(head_idx)
        if head_tok is None:
            continue  # malformed sentence

        num_interveners = len(intervener_indices)

        for k_idx in intervener_indices:
            k_tok = tok_map.get(k_idx)
            if k_tok is None:
                continue  # gap in token sequence (shouldn't happen)

            # --- Feature A: UPOS ---
            intervener_upos = k_tok.upos

            # --- Feature B: Arity ---
            intervener_arity = sentence.arity(k_idx)

            # --- Feature C: Subtree size ---
            intervener_subtree = sentence.subtree_size(k_idx)

            # --- Feature D: Attachment type ---
            attachment = sentence.attachment_type(k_idx, head_idx, dep_idx)

            # Whether this intervener is itself a head (arity > 0)
            # This is the key measure in Yadav et al. (2022):
            # "Intervener Complexity = number of intervening HEADS"
            is_head = 1 if intervener_arity > 0 else 0

            row = {
                "language":                language,
                "word_order":              typo.get("word_order", "unknown"),
                "morph_type":              typo.get("morph_type", "unknown"),
                "family":                  typo.get("family", "unknown"),
                "sent_id":                 sentence.sent_id or "",
                "sent_length":             n,
                "head_idx":                head_idx,
                "dep_idx":                 dep_idx,
                "head_upos":               head_tok.upos,
                "dep_upos":                dep_tok.upos,
                "head_deprel":             head_tok.deprel,
                "dep_deprel":              dep_tok.deprel,
                "arc_distance":            arc_distance,
                "num_interveners":         num_interveners,
                "intervener_idx":          k_idx,
                "intervener_form":         k_tok.form,
                "intervener_upos":         intervener_upos,
                "intervener_arity":        intervener_arity,
                "intervener_subtree_size": intervener_subtree,
                "attachment_type":         attachment,
                "is_head":                 is_head,
            }
            rows.append(row)

    return rows


# -----------------------------------------------------------------------
# Language-level processing
# -----------------------------------------------------------------------

def get_conllu_files(treebank_dir: Path) -> List[Path]:
    """
    Return all .conllu files in a treebank directory, sorted.
    We use ALL splits (train + dev + test) for maximum corpus coverage.
    The paper uses full corpora for distributional analysis.
    """
    files = sorted(treebank_dir.glob("*.conllu"))
    if not files:
        raise FileNotFoundError(f"No .conllu files found in {treebank_dir}")
    return files


def process_language(
    language: str,
    treebank_dir: Path,
    out_dir: Path,
    max_sentences: Optional[int] = None,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Process all .conllu files for one language.
    Writes a CSV file per language and returns summary statistics.

    Args:
        language:        Language name
        treebank_dir:    Path to the treebank folder (contains .conllu files)
        out_dir:         Output directory for CSV files
        max_sentences:   If set, only process this many sentences (for testing)
        logger:          Logger instance

    Returns:
        Stats dict with counts of sentences, arcs, and interveners processed.
    """
    log = logger or logging.getLogger(__name__)
    out_path = out_dir / f"{language.lower()}_interveners.csv"

    log.info(f"[{language}] Starting processing → {out_path.name}")
    t_start = time.time()

    conllu_files = get_conllu_files(treebank_dir)
    log.info(f"[{language}] Found {len(conllu_files)} .conllu file(s): "
             f"{[f.name for f in conllu_files]}")

    stats = {
        "language": language,
        "sentences_processed": 0,
        "tokens_total": 0,
        "arcs_with_interveners": 0,
        "intervener_observations": 0,
    }

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for conllu_path in conllu_files:
            log.info(f"  [{language}] Parsing {conllu_path.name} ...")

            for sentence in parse_conllu_file(str(conllu_path)):
                if max_sentences and stats["sentences_processed"] >= max_sentences:
                    break

                rows = extract_interveners_from_sentence(sentence, language)

                if rows:
                    writer.writerows(rows)
                    # Count unique arcs (arcs that had at least one intervener)
                    unique_arcs = len(set(
                        (r["head_idx"], r["dep_idx"]) for r in rows
                    ))
                    stats["arcs_with_interveners"] += unique_arcs
                    stats["intervener_observations"] += len(rows)

                stats["sentences_processed"] += 1
                stats["tokens_total"] += len(sentence.tokens)

                # Progress log every 10,000 sentences
                if stats["sentences_processed"] % 10000 == 0:
                    log.info(f"  [{language}] {stats['sentences_processed']:,} sentences "
                             f"| {stats['intervener_observations']:,} intervener observations")

            if max_sentences and stats["sentences_processed"] >= max_sentences:
                log.info(f"  [{language}] Reached max_sentences={max_sentences}, stopping.")
                break

    elapsed = time.time() - t_start
    log.info(
        f"[{language}] Done in {elapsed:.1f}s | "
        f"{stats['sentences_processed']:,} sents | "
        f"{stats['arcs_with_interveners']:,} arcs | "
        f"{stats['intervener_observations']:,} intervener observations"
    )
    return stats


# -----------------------------------------------------------------------
# Combined output
# -----------------------------------------------------------------------

def combine_language_csvs(out_dir: Path, languages: List[str]) -> Path:
    """
    Merge all per-language CSVs into one combined file.
    This is what your teammates' analysis scripts will load.
    """
    combined_path = out_dir / "all_languages.csv"

    with open(combined_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for lang in languages:
            lang_csv = out_dir / f"{lang.lower()}_interveners.csv"
            if not lang_csv.exists():
                continue
            with open(lang_csv, "r", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)

    return combined_path


def write_summary_report(stats_list: List[Dict], out_dir: Path):
    """Write a human-readable summary of the extraction run."""
    report_path = out_dir / "extraction_summary.txt"

    total_sents = sum(s["sentences_processed"] for s in stats_list)
    total_arcs = sum(s["arcs_with_interveners"] for s in stats_list)
    total_obs = sum(s["intervener_observations"] for s in stats_list)

    lines = [
        "=" * 65,
        "  Intervener Extraction Summary",
        "=" * 65,
        "",
        f"{'Language':<16} {'Sentences':>10} {'Arcs':>10} {'Interveners':>12}",
        "-" * 52,
    ]

    for s in stats_list:
        lines.append(
            f"{s['language']:<16} "
            f"{s['sentences_processed']:>10,} "
            f"{s['arcs_with_interveners']:>10,} "
            f"{s['intervener_observations']:>12,}"
        )

    lines += [
        "-" * 52,
        f"{'TOTAL':<16} {total_sents:>10,} {total_arcs:>10,} {total_obs:>12,}",
        "",
        "Output files:",
        f"  Per-language CSVs : ./data/processed/<language>_interveners.csv",
        f"  Combined CSV      : ./data/processed/all_languages.csv",
        f"  This summary      : ./data/processed/extraction_summary.txt",
        "",
        "Column descriptions:",
        "  intervener_upos        — Feature A: POS of intervening token",
        "  intervener_arity       — Feature B: direct children count",
        "  intervener_subtree_size— Feature C: full subtree size",
        "  attachment_type        — Feature D: head/dependent/external",
        "  is_head                — 1 if arity > 0 (key metric from Yadav 2022)",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print(report)

    with open(report_path, "w") as f:
        f.write(report + "\n")


# -----------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract intervening nodes from UD dependency treebanks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (all 12 languages)
  python3 step3_extract_interveners.py

  # Quick test on English only
  python3 step3_extract_interveners.py --language English --max_sentences 500

  # Custom data directory
  python3 step3_extract_interveners.py --data_dir /path/to/UD --out_dir /path/to/out
        """
    )
    parser.add_argument(
        "--data_dir",
        default="./data/raw",
        help="Directory containing UD treebank folders (default: ./data/raw)"
    )
    parser.add_argument(
        "--out_dir",
        default="./data/processed",
        help="Output directory for CSV files (default: ./data/processed)"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Process only this language (e.g. English). Default: all 12."
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=None,
        help="Max sentences per language (for testing; default: no limit)"
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO)"
    )
    args = parser.parse_args()

    # --- Logging setup ---
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which languages to process
    if args.language:
        if args.language not in TREEBANKS:
            logger.error(
                f"Unknown language '{args.language}'. "
                f"Choose from: {list(TREEBANKS.keys())}"
            )
            sys.exit(1)
        languages_to_process = {args.language: TREEBANKS[args.language]}
    else:
        languages_to_process = TREEBANKS

    logger.info("=" * 60)
    logger.info("  Intervener Complexity Extraction Pipeline")
    logger.info(f"  Languages: {list(languages_to_process.keys())}")
    logger.info(f"  Data dir:  {data_dir}")
    logger.info(f"  Out dir:   {out_dir}")
    if args.max_sentences:
        logger.info(f"  [TEST MODE] max_sentences = {args.max_sentences}")
    logger.info("=" * 60)

    all_stats = []
    failed = []

    for lang, treebank_name in languages_to_process.items():
        treebank_dir = data_dir / treebank_name

        if not treebank_dir.exists():
            logger.error(
                f"[{lang}] Treebank directory not found: {treebank_dir}\n"
                f"  Run step1_download_treebanks.sh first, or check --data_dir"
            )
            failed.append(lang)
            continue

        try:
            stats = process_language(
                language=lang,
                treebank_dir=treebank_dir,
                out_dir=out_dir,
                max_sentences=args.max_sentences,
                logger=logger,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"[{lang}] FAILED: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            failed.append(lang)

    # Combine all language CSVs into one
    if all_stats:
        logger.info("Merging per-language CSVs into all_languages.csv ...")
        combined = combine_language_csvs(out_dir, [s["language"] for s in all_stats])
        logger.info(f"Combined CSV written: {combined}")

    # Print and save summary
    write_summary_report(all_stats, out_dir)

    if failed:
        logger.warning(f"Failed languages: {failed}")
        logger.warning("Re-run with --log_level DEBUG to see details.")

    logger.info("Step 3 complete. Hand ./data/processed/ to your teammates.")


if __name__ == "__main__":
    main()
