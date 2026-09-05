"""
HTS prompt generator

Usage:
    uv run python generate_prompt.py
    uv run python generate_prompt.py --model gpt --layout tabular
    uv run python generate_prompt.py --model gemini --style diplomatic --shot zero
    uv run python generate_prompt.py --model claude --shot two        # examples auto-filled w/ placeholders
"""

import argparse
import copy

from renderer import render_clean

HTS_TEMPLATE = """\
You are an expert in {{TRANSCRIPTION_STYLE_LABEL}} transcription of historical
handwritten and printed documents across many languages, scripts, and time periods.

Your task: transcribe ALL text visible on the page image {{FIDELITY_CLAUSE}},
in the correct reading order.

{{#IF LANGUAGE_HINT or ERA_HINT}}
DOCUMENT CONTEXT
{{#IF LANGUAGE_HINT}}- The document is believed to be in: {{LANGUAGE_HINT}}{{/IF}}
{{#IF ERA_HINT}}- Approximate period: {{ERA_HINT}}{{/IF}}
- Use this context to inform character recognition only. Do NOT use it to
  generate text you cannot actually see on the page.
{{/IF}}

READING ORDER & LAYOUT
{{LAYOUT_BLOCK}}
- Include headers, titles, marginalia, captions, page numbers, and footnotes.
- Insert a blank line between distinct text blocks (paragraphs, columns, entries).

{{FIDELITY_BLOCK}}

- Preserve the original language. Do NOT translate.
- Skip purely decorative or non-text elements (illustrations, seals, stamps)
  without describing them.

UNCERTAINTY & MISSING TEXT
- If a word or region is illegible, output {{ILLEGIBLE_MARKER}} rather than guessing.
- Never invent, complete, or infer text you cannot actually read. Do not fill
  gaps with plausible-sounding content.
- If the entire page has no legible text, output exactly: [no legible text]

{{#IF SHOT_MODE != zero}}
EXAMPLES
Below {{SHOT_COUNT_PHRASE}} with correct transcriptions in the target style.
Study how each transcription preserves the source exactly and follows reading order.

{{#FOREACH example IN EXAMPLES}}
Example {{example.n}} image: [attached example image {{example.n}}]
Correct transcription:
<{{OUTPUT_TAG}}>
{{example.gold_transcription}}
</{{OUTPUT_TAG}}>
{{/FOREACH}}
Now transcribe the following page in the same way, applying the same fidelity
to the actual text you see. Do not copy content from the examples.
{{/IF}}

{{MODEL_PATCH_BLOCK}}

{{#IF ANTI_ERROR}}
Even if the page is difficult to read, you must return your best transcription.
Do not return a refusal or an apology. Transcribe as much as you can and mark
unreadable spans as {{ILLEGIBLE_MARKER}}.
{{/IF}}

OUTPUT FORMAT
- Output ONLY the transcription between <{{OUTPUT_TAG}}> and </{{OUTPUT_TAG}}> tags.
- No commentary, notes, or explanation before or after.
"""

BASE_DEFAULTS = {
    "TRANSCRIPTION_STYLE": "diplomatic",
    "MODEL_FAMILY": "claude",
    "PAGE_LAYOUT": "auto",
    "ILLEGIBLE_MARKER": "[illegible]",
    "OUTPUT_TAG": "transcription",
    "LANGUAGE_HINT": "",
    "ERA_HINT": "",
    "COLUMN_ORDER": "left-to-right",
    "REFINE_PASS": False,
    "IMAGE_MAX_DIM": 2500,
    "MAX_TOKENS": 20000,
    "REASONING_BUDGET": "2048 tokens",
    "EXAMPLE_SOURCE": "same_batch",
    "N": 0,
    "EXAMPLES": [],
}

# DERIVED-BLOCK TABLES

LAYOUT_BLOCKS = {
    "auto": "- Determine the natural reading order before transcribing. For multiple\n"
    "  columns, transcribe each column fully, top to bottom, in the order a\n"
    "  human reader would follow.",
    "single_column": "- The page is a single column of text. Transcribe top to bottom.",
    "multi_column": "- The page has multiple columns. Transcribe the FIRST column completely\n"
    "  top-to-bottom before starting the next. Column order is {{COLUMN_ORDER}}\n"
    "  (left-to-right unless specified).",
    "tabular": "- The page is a table/register. Transcribe row by row, top to bottom.\n"
    "  Within each row, transcribe cells in column order and separate cells\n"
    "  with ' | '. Transcribe the header row(s) first.",
    "vertical_east_asian": "- The text is written vertically, top-to-bottom, with columns ordered\n"
    "  RIGHT to LEFT. Do not read it as horizontal text.",
    "rtl_script": "- The script reads right-to-left (e.g., Arabic, Hebrew). Preserve\n"
    "  right-to-left reading order; note marginal or interlinear notes separately.",
    "epistolary": "- This is a letter/postcard. Text may begin on the right side or in an\n"
    "  unconventional position; identify where the writing starts before transcribing.",
    "marginalia_heavy": "- The page contains substantial marginal/interlinear annotations.\n"
    "  Transcribe the main text block first, then marginalia, each prefaced with [margin].",
}

STYLE_SPECS = {
    "diplomatic": {
        "label": "diplomatic",
        "clause": "EXACTLY as it appears — do not modernize, correct, translate, or interpret",
        "block": "DIPLOMATIC FIDELITY — transcribe exactly what is on the page\n"
        "- Preserve original spelling exactly, including archaic, obsolete, or\n"
        "  non-standard forms. Do NOT correct or modernize.\n"
        "- Preserve archaic characters as written: keep long-S (\u017f), historical\n"
        "  glyph variants, and original diacritics. Do NOT substitute modern equivalents.\n"
        "- Keep abbreviations as written — do NOT expand them.\n"
        "- Preserve original punctuation, capitalization, line-break hyphens, and\n"
        "  ligatures as they appear.",
    },
    "light_normalization": {
        "label": "faithful",
        "clause": "faithfully, with minimal readability cleanup",
        "block": "FAITHFUL TRANSCRIPTION — minimal readability cleanup only\n"
        "- You MAY convert long-S (\u017f) to s, expand ligatures, join line-break\n"
        "  hyphens, and normalize fraction glyphs.\n"
        "- You MUST NOT alter spellings, names, dates, numbers, or abbreviations.",
    },
    "modernized": {
        "label": "readable",
        "clause": "as clean modern text",
        "block": "MODERNIZED TRANSCRIPTION\n"
        "- Normalize spelling and punctuation to modern conventions.\n"
        "- Preserve wording, names, and numbers exactly.",
    },
}

MODEL_PATCHES = {
    "gemini": "CRITICAL: Do not silently modernize spelling or substitute modern characters\n"
    "for archaic ones. When in doubt, preserve what is written.",
    "gpt": "You must always produce a transcription. Do not refuse.",
    "claude": "Be sure to transcribe the header, title, and any text at the very top of the\n"
    "page — do not skip it. Begin from the topmost text.",
    "open_weight_small": "Transcribe only text that is actually present in the image. Do not repeat\n"
    "sentences. Do not generate placeholder text.",
    "fine_tuned": "",
}

SHOT_PHRASES = {
    "zero": "",
    "one": "is one example page",
    "two": "are two example pages",
    "few_n": "are {{N}} example pages",
}

MODEL_DEFAULTS = {
    "gemini": {
        "SHOT_MODE": "zero",
        "INPUT_GRANULARITY": "whole_page",
        "ANTI_ERROR": False,
        "TEMPERATURE": 0.0,
    },
    "gpt": {
        "SHOT_MODE": "two",
        "INPUT_GRANULARITY": "line_by_line",
        "ANTI_ERROR": True,
        "TEMPERATURE": 0.0,
    },
    "claude": {
        "SHOT_MODE": "two",
        "INPUT_GRANULARITY": "whole_page",
        "ANTI_ERROR": False,
        "TEMPERATURE": 0.0,
    },
    "open_weight_small": {
        "SHOT_MODE": "two",
        "INPUT_GRANULARITY": "whole_page",
        "ANTI_ERROR": True,
        "TEMPERATURE": 0.0,
    },
}


def _placeholder_examples(shot_mode, n):
    count = {"one": 1, "two": 2, "few_n": n}.get(shot_mode, 0)
    return [
        {
            "n": i + 1,
            "gold_transcription": f"[gold transcription of example page {i + 1}]",
        }
        for i in range(count)
    ]


def build_config(overrides):
    """Merge base + model defaults + overrides, then compute derived blocks."""
    cfg = copy.deepcopy(BASE_DEFAULTS)

    # 1. Model family sets its method defaults (before user overrides win).
    family = overrides.get("MODEL_FAMILY", cfg["MODEL_FAMILY"])
    cfg.update(MODEL_DEFAULTS.get(family, {}))
    cfg["MODEL_FAMILY"] = family

    # 2. User overrides win over model defaults.
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    # 3. Derive text blocks from the categorical variables.
    style = STYLE_SPECS.get(cfg["TRANSCRIPTION_STYLE"], STYLE_SPECS["diplomatic"])
    cfg["TRANSCRIPTION_STYLE_LABEL"] = style["label"]
    cfg["FIDELITY_CLAUSE"] = style["clause"]
    cfg["FIDELITY_BLOCK"] = style["block"]

    cfg["LAYOUT_BLOCK"] = LAYOUT_BLOCKS.get(cfg["PAGE_LAYOUT"], LAYOUT_BLOCKS["auto"])
    cfg["MODEL_PATCH_BLOCK"] = MODEL_PATCHES.get(family, "")

    phrase = SHOT_PHRASES.get(cfg["SHOT_MODE"], "")
    cfg["SHOT_COUNT_PHRASE"] = phrase.replace("{{N}}", str(cfg.get("N", 0)))

    # 4. If few-shot and no examples supplied, fill placeholders so the block renders.
    if cfg["SHOT_MODE"] != "zero" and not cfg["EXAMPLES"]:
        cfg["EXAMPLES"] = _placeholder_examples(cfg["SHOT_MODE"], cfg.get("N", 0))

    return cfg


def generate(overrides):
    """Return {'prompt': str, 'sampling': dict, 'config': dict}."""
    cfg = build_config(overrides)
    prompt = render_clean(HTS_TEMPLATE, cfg)
    sampling = {
        "temperature": cfg["TEMPERATURE"],
        "max_tokens": cfg["MAX_TOKENS"],
        "reasoning_budget": cfg["REASONING_BUDGET"],
        "image_max_dim": cfg["IMAGE_MAX_DIM"],
    }
    return {"prompt": prompt, "sampling": sampling, "config": cfg}


def _parse_args():
    p = argparse.ArgumentParser(description="Generate an HTS transcription prompt.")
    p.add_argument("--model", dest="MODEL_FAMILY", choices=list(MODEL_DEFAULTS))
    p.add_argument("--layout", dest="PAGE_LAYOUT", choices=list(LAYOUT_BLOCKS))
    p.add_argument("--style", dest="TRANSCRIPTION_STYLE", choices=list(STYLE_SPECS))
    p.add_argument("--shot", dest="SHOT_MODE", choices=list(SHOT_PHRASES))
    p.add_argument("--lang", dest="LANGUAGE_HINT")
    p.add_argument("--era", dest="ERA_HINT")
    p.add_argument("--n", dest="N", type=int)
    return {k: v for k, v in vars(p.parse_args()).items() if v is not None}


if __name__ == "__main__":
    result = generate(_parse_args())
    print("=" * 72)
    print("SAMPLING:", result["sampling"])
    print(
        "MODEL:",
        result["config"]["MODEL_FAMILY"],
        "| SHOT:",
        result["config"]["SHOT_MODE"],
        "| LAYOUT:",
        result["config"]["PAGE_LAYOUT"],
        "| GRANULARITY:",
        result["config"]["INPUT_GRANULARITY"],
    )
    print("=" * 72)
    print()
    print(result["prompt"])
