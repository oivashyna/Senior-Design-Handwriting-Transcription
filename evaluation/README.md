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
The core scoring logic. Takes reference and prediction text, returns CER, WER, BLEU per lines plus a summary. This is the main file other modules import.

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

**CER quality scale (handwriting only, not printed text):**

| CER       | Meaning       |
| --------- | ------------- |
| Below 3%  | Excellent     |
| 3–5%      | Very good     |
| 5–8%      | Good          |
| 8–10%     | Decent        |
| 10–15%    | Barely usable |
| Above 15% | Unusable      |

---

## Supported input format

| Format     | Extension       | Source                                             |
| ---------- | --------------- | -------------------------------------------------- |
| Plain text | `.txt`          | Model output (Claude, GPT, Gemini, CHURRO, olmOCR) |
| ALTO XML   | `.xml`, `.alto` | Transkribus export                                 |
| PAGE XML   | `.xml`          | eScriptorium export                                |

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

reference  = extract_text_from_file("ground_truth/letter_001.xml")
prediction = extract_text_from_file("transcribed/Claude/letter_001.txt")

result = evaluate_lines(reference, prediction)
print(result["flagged_lines"])
print(result["total_lines"])
```

---
