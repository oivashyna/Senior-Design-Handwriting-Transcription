"""
Handwriting Transcription System — Evaluation & Benchmarking

WHY THIS EXISTS
  Whole-document CER can hide a badly-wrong single line -- we proved this earlier:
  3 clearly-corrupted lines out of 39 barely moved the document-level score. FR-3.1
  explicitly asks for flagging specific segments, not whole documents. This scores
  and flags each LINE individually instead.

THE REAL PROBLEM THIS HAS TO HANDLE
  Reference lines and prediction lines won't always line up 1:1. A model might:
    - merge two source lines into one output line
    - split one source line into two
    - skip a line entirely
    - invent an extra line that isn't in the source at all
  Zipping reference[i] with prediction[i] will break when any of the above happens.
  This uses difflib sequence alignment at the line level to find the actual correspondence
  first, then score within that.


"""

import re
import difflib
import jiwer
import sacrebleu


FLAG_THRESHOLD_CER = 0.10

# NORMALIZE ACCURACY
def normalize_for_accuracy(text: str) -> str:
    """Matches the research department's established convention. 
    (from Historical Image Transcription Script — Technical Documentation)
    
    Scores here are directly comparable to theirs:
      1. Convert to Lowercase 
      2. Remove content [bracketed content]      -- annotations like [which], [illegible]
      3. Remove content <angle bracket content>  -- e.g. <unclear>
      4. Remove standalone ^carets^      -- superscript/interlinear markers
      5. Collapse multiple whitespace to a single space
      6. Strip leading/trailing whitespace

    The prompt instructs the model to ADD these annotations without changing the underlying text.
    Since ground truth is not annotated, we remove them before scoring to avoid penalizing the model
    for following prompt instructions.
    """
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\^", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# EVALUATION TRANSCRIPTION
def evaluate_transcription(reference: str, prediction: str, normalize: bool = False) -> dict:
    """Set normalize=True to score using the research department's convention 
    (annotation removal, case-insensitive) 
    """
    if normalize:
        reference = normalize_for_accuracy(reference)
        prediction = normalize_for_accuracy(prediction)
    wer = jiwer.wer(reference, prediction)
    cer = jiwer.cer(reference, prediction)
    bleu = sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0
    return {"cer": cer, "wer": wer, "bleu": bleu}

 
def evaluate_both(reference: str, prediction: str) -> dict:
    """Returns both raw and normalized scores side by side.
    """
    return {
        "raw": evaluate_transcription(reference, prediction, normalize=False),
        "normalized": evaluate_transcription(reference, prediction, normalize=True),
    }


def flag_for_review(cer: float) -> bool:
    return cer >= FLAG_THRESHOLD_CER


def highlight_errors(reference: str, prediction: str) -> dict:
    result = jiwer.process_words(reference, prediction)
    alignment = result.alignments[0]
    hyp_words = result.hypotheses[0]
    ref_words = result.references[0]
    marked_words = list(hyp_words)
    issues = []

    for chunk in alignment:
        if chunk.type == "substitute":
            wrong = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            correct = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            issues.append({"type": "substitute", "wrong": wrong, "reference": correct})
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                marked_words[i] = f">>{marked_words[i]}<<"
        elif chunk.type == "insert":
            extra = " ".join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
            issues.append({"type": "insert", "extra": extra})
            for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                marked_words[i] = f">>{marked_words[i]}<<"
        elif chunk.type == "delete":
            missing = " ".join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
            issues.append({"type": "delete", "missing": missing})

    return {"highlighted_text": " ".join(marked_words), "issues": issues}



# LINE ALIGNMENT
def align_lines(reference_lines: list, prediction_lines: list) -> list:
    """Aligns reference and prediction lines using sequence matching (same
    technique as a code diff), so merged/split/missing/extra lines are handled
    correctly instead of assuming a 1:1 correspondence.

    Returns a list of aligned units, each one of:
      {"type": "matched", "ref_line_nums": [...], "reference": str, "prediction": str}
      {"type": "missing",  "ref_line_nums": [...], "reference": str}   -- in ref, not in prediction
      {"type": "extra",    "prediction": str}                          -- in prediction, not in ref
    """
    matcher = difflib.SequenceMatcher(None, reference_lines, prediction_lines, autojunk=False)
    units = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                units.append({
                    "type": "matched",
                    "ref_line_nums": [i1 + offset + 1],
                    "reference": reference_lines[i1 + offset],
                    "prediction": prediction_lines[j1 + offset],
                })
        elif tag == "replace":
            # Line counts may differ within this block (merge/split case).
            # Pair what we can 1:1, treat any leftover as missing/extra.
            n = min(i2 - i1, j2 - j1)
            for offset in range(n):
                units.append({
                    "type": "matched",
                    "ref_line_nums": [i1 + offset + 1],
                    "reference": reference_lines[i1 + offset],
                    "prediction": prediction_lines[j1 + offset],
                })
            for offset in range(n, i2 - i1):
                units.append({
                    "type": "missing",
                    "ref_line_nums": [i1 + offset + 1],
                    "reference": reference_lines[i1 + offset],
                })
            for offset in range(n, j2 - j1):
                units.append({
                    "type": "extra",
                    "prediction": prediction_lines[j1 + offset],
                })
        elif tag == "delete":
            for offset in range(i2 - i1):
                units.append({
                    "type": "missing",
                    "ref_line_nums": [i1 + offset + 1],
                    "reference": reference_lines[i1 + offset],
                })
        elif tag == "insert":
            for offset in range(j2 - j1):
                units.append({
                    "type": "extra",
                    "prediction": prediction_lines[j1 + offset],
                })

    return units


# LINE-LEVEL EVALUATION
def evaluate_lines(reference_text: str, prediction_text: str) -> dict:
    """Scores each line individually instead of the whole document at once.
    Returns per-line results plus a summary."""
    reference_lines = [l for l in reference_text.split("\n") if l.strip()]
    prediction_lines = [l for l in prediction_text.split("\n") if l.strip()]

    units = align_lines(reference_lines, prediction_lines)

    line_results = []
    flagged_count = 0

    for unit in units:
        if unit["type"] == "matched":
            result = evaluate_transcription(unit["reference"], unit["prediction"], normalize=True)
            flagged = flag_for_review(result["cer"])
            entry = {
                "status": "flagged" if flagged else "ok",
                "ref_line_nums": unit["ref_line_nums"],
                "reference": unit["reference"],
                "prediction": unit["prediction"],
                "cer": result["cer"],
                "wer": result["wer"],
                "bleu": result["bleu"],
            }
            if flagged:
                flagged_count += 1
                entry["detail"] = highlight_errors(unit["reference"], unit["prediction"])
            line_results.append(entry)

        elif unit["type"] == "missing":
            flagged_count += 1
            line_results.append({
                "status": "missing",
                "ref_line_nums": unit["ref_line_nums"],
                "reference": unit["reference"],
                "note": "This reference line has no corresponding prediction line "
                        "-- likely skipped, or merged into an adjacent line.",
            })

        elif unit["type"] == "extra":
            flagged_count += 1
            line_results.append({
                "status": "extra",
                "prediction": unit["prediction"],
                "note": "This prediction line has no corresponding reference line "
                        "-- possible hallucinated content, or a split line.",
            })

    return {
        "line_results": line_results,
        "total_lines": len(line_results),
        "flagged_lines": flagged_count,
        "flagged_fraction": flagged_count / len(line_results) if line_results else 0,
    }


# DISPLAY
def print_line_results(result: dict) -> None:
    print(f"=== Line-level results: {result['flagged_lines']} of {result['total_lines']} "
          f"lines flagged ({result['flagged_fraction']*100:.0f}%) ===\n")

    for entry in result["line_results"]:
        if entry["status"] == "ok":
            continue  # only show problems
        elif entry["status"] == "flagged":
            line_num = entry["ref_line_nums"][0]
            print(f"Line {line_num}  CER={entry['cer']*100:.1f}%  [FLAGGED]")
            print(f"  reference:  {entry['reference']}")
            print(f"  prediction: {entry['detail']['highlighted_text']}")
            for issue in entry["detail"]["issues"]:
                if issue["type"] == "substitute":
                    print(f"    - wrong word: \"{issue['wrong']}\" (should be \"{issue['reference']}\")")
                elif issue["type"] == "insert":
                    print(f"    - extra word: \"{issue['extra']}\"")
                elif issue["type"] == "delete":
                    print(f"    - missing word(s): \"{issue['missing']}\"")
            print()
        elif entry["status"] == "missing":
            line_num = entry["ref_line_nums"][0]
            print(f"Line {line_num}  [MISSING FROM PREDICTION]")
            print(f"  reference: {entry['reference']}")
            print(f"  {entry['note']}\n")
        elif entry["status"] == "extra":
            print(f"[EXTRA LINE -- not in reference]")
            print(f"  prediction: {entry['prediction']}")
            print(f"  {entry['note']}\n")


# SELF-TEST — proves alignment handles merged/missing/extra lines correctly,
# not just the clean 1:1 case.
def _self_test():
    reference_lines = [
        "My dear friend, I hope this letter finds you well.",
        "The harvest this year has been poor due to the drought.",
        "Please write back as soon as you are able.",
        "I remain, your faithful servant.",
    ]

    # Simulate real failure modes: line 2 has a wrong word, line 3 is MISSING
    # entirely (model skipped it), and there's an EXTRA invented line at the end.
    prediction_lines = [
        "My dear friend, I hope this letter finds you well.",
        "The harvest this year has been poor the drought.",
        "I remain, your faithful servant.",
        "Written on a fine spring morning in the countryside.",
    ]

    reference_text = "\n".join(reference_lines)
    prediction_text = "\n".join(prediction_lines)

    result = evaluate_lines(reference_text, prediction_text)
    print_line_results(result)

def _self_test_normalized():

    reference_text = "hath [has] ^come^ Severall [several] greate [great] men"
    prediction_text = "hath come Severall greate men"

    result = evaluate_lines(reference_text, prediction_text)
    line = result["line_results"][0]
    print(f"Normalized check")
    print(f"Reference: (without annotations): {reference_text!r}")
    print(f"prediction: (plain text): {prediction_text!r}")
    print(f"Status: {line['status']} CER: {line['cer']*100:.1f}%  (expect status=ok,CER=0.0%)")

if __name__ == "__main__":
    _self_test()
    print()
    _self_test_normalized()