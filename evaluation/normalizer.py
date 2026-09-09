# NORMALIZE ACCURACY
import re
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

# FILTER EDITORIAL LINES
def filter_editorial_lines(text: str) -> str:
    """
    Removes lines that are likely editorial notes, not part of the transcription.

    Removes:
        - Lines that are entirely bracketed e.g. (ffsH: [Francis Howgill]...)
        - Lines that are purely a page number (just digits)
    """

    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r'\d+', stripped):
            continue  # Skip lines that are (page numbers)

        if stripped.startswith("(") and stripped.endswith(")"):
            inner = stripped[1:-1]
            cleaned = re.sub(r'\[.*?\]', '', inner)
            cleaned = re.sub(r'&\w*:?', '', cleaned)
            cleaned = re.sub(r'\s+', '', cleaned)
            if not cleaned:
                continue
        filtered.append(line)
    return "\n".join(filtered)
