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