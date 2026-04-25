#!/usr/bin/env python3
"""
STEP 2 — Verify downloaded treebanks
Project: Empirical Distribution of Intervener Complexity

Run this AFTER step1_download_treebanks.sh to confirm:
  - All 12 treebank directories exist
  - Each treebank has at least one .conllu file
  - The .conllu files are parseable (non-empty, correct format)
  - Quick stats: sentence count, token count per language

Usage:
    python3 step2_verify_data.py
    python3 step2_verify_data.py --data_dir /path/to/your/data/raw
"""

import os
import sys
import argparse
from pathlib import Path


# -----------------------------------------------------------------------
# Configuration — the 12 treebanks from our proposal
# -----------------------------------------------------------------------
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


def count_sentences_and_tokens(filepath):
    """
    Quick scan of a .conllu file.
    Returns (num_sentences, num_tokens) without full parsing.
    Tokens are non-comment, non-empty lines with numeric IDs only
    (we skip multi-word tokens like 1-2 and empty nodes like 1.1).
    """
    sentences = 0
    tokens = 0
    in_sentence = False

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if in_sentence:
                    sentences += 1
                    in_sentence = False
            elif line.startswith("#"):
                continue
            else:
                # Check if this is a regular token (numeric ID, not 1-2 or 1.1)
                fields = line.split("\t")
                if fields and fields[0].isdigit():
                    tokens += 1
                    in_sentence = True

    # Handle file that doesn't end with blank line
    if in_sentence:
        sentences += 1

    return sentences, tokens


def verify_conllu_format(filepath, sample_size=5):
    """
    Check that the first `sample_size` sentences look like valid CoNLL-U.
    Returns (is_valid, error_message).
    """
    required_fields = 10  # CoNLL-U has exactly 10 tab-separated fields per token line

    sentence_count = 0
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                sentence_count += 1
                if sentence_count >= sample_size:
                    break
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) != required_fields:
                return False, (
                    f"Expected {required_fields} tab-separated fields, "
                    f"got {len(fields)}: '{line[:60]}'"
                )
    return True, "OK"


def main():
    parser = argparse.ArgumentParser(description="Verify UD treebank data")
    parser.add_argument(
        "--data_dir",
        default="./data/raw",
        help="Directory containing UD treebank folders (default: ./data/raw)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("=" * 65)
    print("  Treebank Verification Report")
    print("=" * 65)

    all_ok = True
    total_sentences = 0
    total_tokens = 0
    results = []

    for lang, treebank_name in TREEBANKS.items():
        treebank_dir = data_dir / treebank_name

        # Check directory exists
        if not treebank_dir.exists():
            print(f"\n[MISSING] {lang:15s} — directory not found: {treebank_dir}")
            all_ok = False
            continue

        # Find all .conllu files
        conllu_files = sorted(treebank_dir.glob("*.conllu"))
        if not conllu_files:
            print(f"\n[EMPTY]   {lang:15s} — no .conllu files in {treebank_dir}")
            all_ok = False
            continue

        # Count sentences and tokens across all splits
        lang_sentences = 0
        lang_tokens = 0
        file_details = []

        for f in conllu_files:
            # Determine split from filename
            name = f.stem
            if "train" in name:
                split = "train"
            elif "dev" in name:
                split = "dev"
            elif "test" in name:
                split = "test"
            else:
                split = "unknown"

            is_valid, err = verify_conllu_format(f)
            if not is_valid:
                print(f"\n[FORMAT ERROR] {lang} / {f.name}: {err}")
                all_ok = False
                continue

            sents, toks = count_sentences_and_tokens(f)
            lang_sentences += sents
            lang_tokens += toks
            file_details.append((split, f.name, sents, toks))

        total_sentences += lang_sentences
        total_tokens += lang_tokens
        results.append((lang, treebank_name, lang_sentences, lang_tokens, file_details))

    # Pretty-print results table
    print(f"\n{'Language':<16} {'Treebank':<30} {'Sentences':>10} {'Tokens':>10}")
    print("-" * 70)

    for lang, treebank_name, sents, toks, files in results:
        status = "[OK]" if sents > 0 else "[WARN]"
        print(f"{lang:<16} {treebank_name:<30} {sents:>10,} {toks:>10,}  {status}")
        for split, fname, s, t in files:
            print(f"  {'':14} {fname:<30} {s:>10,} {t:>10,}  ({split})")

    print("-" * 70)
    print(f"{'TOTAL':<16} {'':30} {total_sentences:>10,} {total_tokens:>10,}")
    print()

    if all_ok and len(results) == len(TREEBANKS):
        print("[SUCCESS] All 12 treebanks verified. Ready for Step 3.")
        print()
        print("Next: python3 step3_extract_interveners.py")
    else:
        missing = len(TREEBANKS) - len(results)
        print(f"[WARNING] {missing} treebank(s) missing or invalid.")
        print("Re-run step1_download_treebanks.sh and check the data directory.")
        sys.exit(1)


if __name__ == "__main__":
    main()
