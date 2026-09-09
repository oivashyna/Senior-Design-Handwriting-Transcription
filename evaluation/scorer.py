from evaluation.normalizer import normalize_for_accuracy, filter_editorial_lines
from evaluation.flagging import flag_for_review
from evaluation.aligner import align_lines
from evaluation.highlighter import highlight_errors
import rapidfuzz.fuzz as fuzz
import jiwer
import sacrebleu

# LINE-LEVEL EVALUATION
def evaluate_lines(reference_text: str, prediction_text: str) -> dict:
    """Scores each line individually instead of the whole document at once.
    Returns per-line results plus a summary."""
    reference_text = filter_editorial_lines(reference_text)
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
                "similarity": result["similarity"],
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
    similarity = fuzz.ratio(prediction, reference) / 100.0
    return {"cer": cer, "wer": wer, "bleu": bleu, "similarity": similarity}

def evaluate_both(reference: str, prediction: str) -> dict:
    """Returns both raw and normalized scores side by side.
    """
    return {
        "raw": evaluate_transcription(reference, prediction, normalize=False),
        "normalized": evaluate_transcription(reference, prediction, normalize=True),
    }