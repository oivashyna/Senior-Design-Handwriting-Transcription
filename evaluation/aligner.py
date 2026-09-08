#LINE ALIGNMENT
import difflib

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