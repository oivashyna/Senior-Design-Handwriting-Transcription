import xml.etree.ElementTree as ET
from pathlib import Path

_ALTO_NAMESPACES = [
    "http://www.loc.gov/standards/alto/ns-v4#", # ALTO 4
    "http://www.loc.gov/standards/alto/ns-v3#", # ALTO 3
    "http://www.loc.gov/standards/alto/ns-v2#", # ALTO 2
    "http://schema.css-gmbh.com/ALTO", # older css variant
    "", # rare cases with no namespace
]

# PAGE XML namespace
_PAGE_NAMESPACE = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
_PAGE_NAMESPACE_ALT = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

# Auto specific parser
def extract_text_from_alto(xml_path: str) -> str:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"ALTO XML file not found: {xml_path}")
 
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in {xml_path}: {e}")
 
    # Try each known ALTO namespace until we find TextLine elements.
    lines = []
    for ns in _ALTO_NAMESPACES:
        prefix = f"{{{ns}}}" if ns else ""
        text_lines = root.findall(f".//{prefix}TextLine")
 
        if not text_lines:
            continue  # wrong namespace, try next
 
        for text_line in text_lines:
            # Collect all String elements on this line
            words = []
            for string_elem in text_line.findall(f"{prefix}String"):
                content = string_elem.get("CONTENT", "")
                if content:
                    words.append(content)
 
            # HYP element marks a hyphenated word continuation — append without space
            # SP elements are explicit spaces — already handled by joining with " "
            line_text = " ".join(words).strip()
            if line_text:
                lines.append(line_text)
 
        break  # found the right namespace, stop trying
 
    if not lines:
        raise ValueError(
            f"No transcribed text found in {xml_path}. "
            "File may use an unsupported ALTO namespace or contain only layout data."
        )
 
    return "\n".join(lines)


# PAGE specific parser
def extract_text_from_page(xml_path: str) -> str:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"PAGE XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in {xml_path}: {e}")
 
    lines = []
 
    # Try both known PAGE namespace versions
    for ns in [_PAGE_NAMESPACE, _PAGE_NAMESPACE_ALT, ""]:
        prefix = f"{{{ns}}}" if ns else ""
        text_lines = root.findall(f".//{prefix}TextLine")
 
        if not text_lines:
            continue
 
        for text_line in text_lines:
            # Prefer line-level TextEquiv/Unicode (eScriptorium HTR output style)
            line_text = ""
            text_equiv = text_line.find(f"{prefix}TextEquiv")
            if text_equiv is not None:
                unicode_elem = text_equiv.find(f"{prefix}Unicode")
                if unicode_elem is not None and unicode_elem.text:
                    line_text = unicode_elem.text.strip()
 
            # Fallback: assemble from Word elements
            if not line_text:
                words = []
                for word in text_line.findall(f".//{prefix}Word"):
                    word_equiv = word.find(f"{prefix}TextEquiv")
                    if word_equiv is not None:
                        unicode_elem = word_equiv.find(f"{prefix}Unicode")
                        if unicode_elem is not None and unicode_elem.text:
                            words.append(unicode_elem.text.strip())
                line_text = " ".join(words).strip()
 
            if line_text:
                lines.append(line_text)
 
        break  # found the right namespace
 
    if not lines:
        raise ValueError(
            f"No transcribed text found in {xml_path}. "
            "File may use an unsupported PAGE namespace or contain only layout data."
        )
 
    return "\n".join(lines)

#Churro specific parser
def extract_text_from_churro(xml_path: str) -> str:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"CHURRO XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in {xml_path}: {e}")

    ns = ""
    tag = root.tag
    if "}" in tag:
        ns = tag.split("}")[0].strip("{")
    prefix = f"{{{ns}}}" if ns else ""

    def get_line_text(line_elem) -> str:
        parts = []
        if line_elem.text:
            parts.append(line_elem.text)
        for child in line_elem:
            child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if child_tag == "Illegible":
                parts.append("[illegible]")
            elif child_tag in ("Emphasis", "Addition"):
                if child.text:
                    parts.append(child.text)
            else:
                if child.text:
                    parts.append(child.text)
            if child.tail:
                parts.append(child.tail)
        return "".join(parts).strip()

    lines = []
    body = root.find(f".//{prefix}Body")
    if body is None:
        raise ValueError(f"No <Body> element found in {xml_path}.")

    for element in body:
        tag = element.tag.split("}")[-1] if "}" in element.tag else element.tag
    
    if tag in ("Paragraph", "MarginalNote", "DateLine"):
        for line_elem in element.findall(f"{prefix}Line"):
            text = get_line_text(line_elem)
            if text:
                lines.append(text)

    if not lines:
        raise ValueError(f"No transcribed lines found in {xml_path}.")

    return "\n".join(lines)


# Auto detects format
def extract_text_from_xml(xml_path: str) -> str:
    path = Path(xml_path)
    if not path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
 
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML in {xml_path}: {e}")
 
    # Strip namespace from tag for detection
    tag = root.tag.split("}")[-1].lower() if "}" in root.tag else root.tag.lower()
 
    if "alto" in tag:
        return extract_text_from_alto(xml_path)
    elif "pcgts" in tag or "page" in tag:
        return extract_text_from_page(xml_path)
    elif "historicaldocument" in tag:
        return extract_text_from_churro(xml_path)
    else:
        for parser in (extract_text_from_alto, 
                        extract_text_from_page,
                        extract_text_from_churro
                        ):
            try:
                return parser(xml_path)
            except ValueError:
                continue
        raise ValueError(
            f"Could not detect XML format for {xml_path}. "
            "Supported: ALTO XML, PAGE XML, CHURRO HistoricalDocument."
        )

# Generic entry point for any supported file type
def extract_text_from_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
 
    suffix = path.suffix.lower()

    if suffix in {".xml", ".alto"}:
        return extract_text_from_xml(file_path)
    elif suffix == ".txt":
        content = path.read_text(encoding="utf-8") #if the txt file contains XML content
        if content.strip().startswith("<"):
            return extract_text_from_xml(file_path)
        else:
            return content
    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            "Supported formats: .txt, .xml, .alto"
        )
