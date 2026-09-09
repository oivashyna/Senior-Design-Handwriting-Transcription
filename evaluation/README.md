## Overview

This module scores the quality of handwritten text transcriptions produced by LLM models. It compares model outputs against a verified human transcription (Ground truth) and returns CER, WER, and BLEU scores at the line level, flagging lines that fall below accepted quality threshold.

---

## How It Works

A short explanation of the logic/flow — not per-line, but the overall
approach: what goes in, what happens, what comes out.

---

## Usage

from evaluation import <filename>

Minimal working example.

---

## Files

`normalizer.py`
Cleans text before scoring.
Converts to lowercase, removes annotation markers i.e. (`[illegible]`,`<unclear>`, `^caret^`) that where added by the prompt and collapses whitespace. It matches Paola's `image_transcriber.py` so that the scores are directly comparable.

`aligner.py`
Matches reference and prediction lines using sequence alignment (difflib). Handles real model failure modes, merged, split, skipped, and hallucinated extra lines currently.

`scorer.py`
The core scoring logic. Takes reference and prediction text, returns CER, WER, BLEU, and similarity per lines plus a summary. This is the main file other modules import.

`flagging.py`
Determine whether lines needs human review based on its CER score. Default threshold is 10%.

`highlighter.py`
Shows specific words in a prediction were wrong, substitutions, insertions, and deletions by comparing against the reference.

`handler.py`
AWS lambda entry point.

`xml_parser.py`
This extracts plain text from ALTO XML and PAGE XML files. This is an extra file that is required for scoring if file is XML.

---

## Metrics

| Metric | Measures                      | Range   | Better is |
| ------ | ----------------------------- | ------- | --------- |
| CER    | Character Error Rate          | 0%+     | Lower     |
| WER    | Word Error Rate               | 0%+     | Lower     |
| BLEU   | N-gram overlap with reference | 0.0–1.0 | Higher    |

### CER quality scale (established benchmarks):

| CER       | Quality   |
| --------- | --------- |
| 0%        | Perfect   |
| Below 5%  | Excellent |
| 5–15%     | Good      |
| 15–30%    | Fair      |
| Above 30% | Poor      |

### WER

| WER       | Quality   |
| --------- | --------- |
| 0%        | Perfect   |
| Below 10% | Excellent |
| 10–25%    | Good      |
| 25–50%    | Fair      |
| Above 50% | Poor      |

### BLEU and Similarity

No established HTR-specific quality benchmarks exist for these metrics in the literature. BLEU above 0.6 generally indicates good n-gram overlap. Similarity above 0.8 indicates high structural resemblance. Both are reported alongside CER for completeness but CER remains the primary metric.

---

## Supported input format

| Format     | Extension       | Source                                             |
| ---------- | --------------- | -------------------------------------------------- |
| Plain text | `.txt`          | Model output (Claude, GPT, Gemini, CHURRO, olmOCR) |
| ALTO XML   | `.xml`, `.alto` | Transkribus export                                 |
| PAGE XML   | `.xml`          | eScriptorium export                                |
| CHURRO XML | `.txt`, `.xml`  | CHURRO HistoricalDocument format                   |

Use `extract_text_from_file()` from `xml_parser.py` it handles all three formats automatically.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

---

## Quick Start

```python
from evaluation.scorer import evaluate_lines
from evaluation.xml_parser import extract_text_from_file

reference  = extract_text_from_file("ground_truth/038.txt")
prediction = extract_text_from_file("transcribed/Churro/038.txt")

result = evaluate_lines(reference, prediction)
print(result["flagged_lines"])
print(result["total_lines"])
```

---

## Usage

Run evaluation against all documents in `test_data/`:

```bash
python main.py
```

Run evaluation against a specific document only:

```bash
python main.py --document 038.txt
```

The script automatically discovers all model folders in `test_data/transcribed/` and scores any model that has a matching filename. No configuration needed when adding new models or documents.

---

## Notes on Test Results

### Structural Fidelity

CER scores reflect structural fidelity as well as character accuracy.
Models that reorder content (example placing marginal notes at the
end of output when they appear mid-document in the original) will score
higher CER. This is intentional: require transcription to
preserve document structure as-is.

### Ground Truth Format

Current test GT files use scholarly diplomatic transcription with
editorial annotations e.g. `[that]`, `[none]`, `[deletion]...[/deletion]`.
These are stripped before scoring but their presence can still inflate
CER slightly. Scores will be more accurate once CHDR produces GT in
their own format.

### File Naming Convention

Ground truth and transcription files are matched by filename only.
Files must refer to the same physical document the scorer has no
way to detect a content mismatch. Always verify correspondence before
interpreting scores.

Example:
ground_truth/038.txt
transcribed/Churro/038.txt ← scored
transcribed/Churro/055.txt ← skipped, no GT match

---

## Known Limitations

- All models perform worse on non-English text
- Digits and proper names are consistently the hardest to transcribe correctly
- No ground truth path (LLM-as-judge) is planned but not yet implemented
- XML confidence scores are parsed but not currently used by the scorer
- GT format not yet standardized current test data uses scholarly annotations that may not reflect CHDR's final gold standard convention
