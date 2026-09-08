import jiwer 

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

