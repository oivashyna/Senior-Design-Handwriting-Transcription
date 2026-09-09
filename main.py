"""
main.py — Evaluation test runner.

Scores model transcriptions against ground truth and prints a
side-by-side comparison report. Makes no API calls.

USAGE:
    python main.py                                      # all documents
    python main.py --document consensus_text_97731267.txt  # one document

FILE MATCHING:
    Ground truth and transcription files must share the same filename.
    The script automatically discovers all model folders in
    test_data/transcribed/ and scores any model that has a matching file.

    test_data/
      ground_truth/
        document_001.txt        (or .xml)
      transcribed/
        Claude/
          document_001.txt      scored
        GPT/
          document_001.txt      scored
        Gemini/
                                skipped — no matching file

NOTE:
    For full benchmarking including API calls and token cost tracking,
    see benchmark.py (not yet implemented).
"""
import argparse
import os
import sys
from pathlib import Path

# Add the project root to the path so evaluation/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.scorer import evaluate_lines
from evaluation.xml_parser import extract_text_from_file

# PATHS
TEST_DATA_DIR = Path(__file__).parent / "test_data"
GROUND_TRUTH_DIR = TEST_DATA_DIR / "ground_truth"
TRANSCRIBED_DIR = TEST_DATA_DIR / "transcribed"

# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".xml", ".alto"}


# HELPERS
def load_text(file_path: Path) -> str:
    return extract_text_from_file(str(file_path))


def find_model_transcriptions(document_stem: str) -> dict:
    matches = {}
    if not TRANSCRIBED_DIR.exists():
        return matches

    for model_dir in sorted(TRANSCRIBED_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for ext in SUPPORTED_EXTENSIONS:
            candidate = model_dir / f"{document_stem}{ext}"
            if candidate.exists():
                matches[model_dir.name] = candidate
                break  # found this model's file, move to next model

    return matches


def find_ground_truth_documents() -> list:
    if not GROUND_TRUTH_DIR.exists():
        return []
    return [
        f for f in sorted(GROUND_TRUTH_DIR.iterdir())
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

# REPORT PRINTING
def print_summary_table(document_name: str, results: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  DOCUMENT: {document_name}")
    print("=" * 70)

    # Header
    print(f"  {'Model':<20} {'CER':>8} {'WER':>8} {'BLEU':>8} {'Similarity':>10} {'Flagged':>10} {'Total Lines':>12}")
    print("  " + "-" * 74)

    # One row per model
    for model_name, result in sorted(results.items()):
        # Aggregate CER/WER/BLEU across all matched lines
        matched = [
            l for l in result["line_results"]
            if l["status"] in ("ok", "flagged")
        ]
        if matched:
            avg_cer = sum(l["cer"] for l in matched) / len(matched)
            avg_wer = sum(l["wer"] for l in matched) / len(matched)
            avg_bleu = sum(l["bleu"] for l in matched) / len(matched)
            avg_sim = sum(l["similarity"] for l in matched) / len(matched)
        else:
            avg_cer = avg_wer = avg_bleu = avg_sim = 0.0

        flagged = result["flagged_lines"]
        total = result["total_lines"]

        print(f"  {model_name:<20} "
              f"{avg_cer*100:>7.1f}% "
              f"{avg_wer*100:>7.1f}% "
              f"{avg_bleu:>8.3f} "
              f"{avg_sim:>8.3f} "
              f"{flagged:>10} "
              f"{total:>12}")

    print("=" * 70)


def print_flagged_lines(model_name: str, result: dict) -> None:
    problems = [
        l for l in result["line_results"]
        if l["status"] != "ok"
    ]
    if not problems:
        return

    print(f"\n  --- {model_name}: flagged lines ---")

    for entry in problems:
        status = entry["status"]

        if status == "flagged":
            line_num = entry["ref_line_nums"][0]
            print(f"\n  Line {line_num}  CER={entry['cer']*100:.1f}%  [FLAGGED]")
            print(f"    reference:  {entry['reference']}")
            if "detail" in entry:
                print(f"    prediction: {entry['detail']['highlighted_text']}")
                for issue in entry["detail"]["issues"]:
                    if issue["type"] == "substitute":
                        print(f"      - wrong: \"{issue['wrong']}\" "
                              f"(should be \"{issue['reference']}\")")
                    elif issue["type"] == "insert":
                        print(f"      - extra word: \"{issue['extra']}\"")
                    elif issue["type"] == "delete":
                        print(f"      - missing: \"{issue['missing']}\"")
            else:
                print(f"    prediction: {entry['prediction']}")

        elif status == "missing":
            line_num = entry["ref_line_nums"][0]
            print(f"\n  Line {line_num}  [MISSING]")
            print(f"    reference: {entry['reference']}")
            print(f"    {entry['note']}")

        elif status == "extra":
            print(f"\n  [EXTRA LINE]")
            print(f"    prediction: {entry['prediction']}")
            print(f"    {entry['note']}")


def print_no_data_warning() -> None:
    """Prints a helpful message when no test data is found."""
    print("\n  No test data found.")
    print(f"\n  Expected structure:")
    print(f"    {TEST_DATA_DIR}/")
    print(f"      ground_truth/")
    print(f"        document_001.txt  (or .xml)")
    print(f"      transcribed/")
    print(f"        Claude/")
    print(f"          document_001.txt")
    print(f"        GPT/")
    print(f"          document_001.txt")
    print(f"\n  Add ground truth and transcription files to get started.")


# MAIN
def run(document_filter: str = None) -> None:
    """
    Main evaluation loop.

    Finds all ground truth documents, matches them to model transcriptions.
    """
    ground_truth_docs = find_ground_truth_documents()

    if not ground_truth_docs:
        print_no_data_warning()
        return

    if document_filter:
        ground_truth_docs = [
            d for d in ground_truth_docs
            if d.name == document_filter
        ]
        if not ground_truth_docs:
            print(f"\n  Document not found in ground_truth/: {document_filter}")
            return

    evaluated_count = 0

    for gt_path in ground_truth_docs:
        document_stem = gt_path.stem
        document_name = gt_path.name

        # Find all model transcriptions for this document
        model_files = find_model_transcriptions(document_stem)

        if not model_files:
            print(f"\n  Skipping {document_name} — "
                  f"no model transcriptions found in transcribed/")
            continue

        # Load ground truth
        try:
            reference_text = load_text(gt_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"\n  Error loading ground truth {document_name}: {e}")
            continue

        # Score each model
        results = {}
        for model_name, transcription_path in model_files.items():
            try:
                prediction_text = load_text(transcription_path)
                results[model_name] = evaluate_lines(reference_text, prediction_text)
            except (FileNotFoundError, ValueError) as e:
                print(f"\n  Error loading {model_name}/{document_name}: {e}")
                continue

        if not results:
            continue

        # Print summary table
        print_summary_table(document_name, results)

        # Print flagged line detail for each model
        for model_name, result in sorted(results.items()):
            print_flagged_lines(model_name, result)

        evaluated_count += 1

    if evaluated_count == 0:
        print("\n  No documents could be evaluated. "
              "Check that ground truth and transcription files exist "
              "and have matching filenames.")
    else:
        print(f"\n  Evaluated {evaluated_count} document(s).\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HTR Evaluation — score transcriptions against ground truth."
    )
    parser.add_argument(
        "--document",
        type=str,
        default=None,
        help="Evaluate a specific document only (e.g. document_001.txt). "
             "If omitted, all documents in ground_truth/ are evaluated."
    )
    args = parser.parse_args()
    run(document_filter=args.document)