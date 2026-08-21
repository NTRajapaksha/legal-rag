import re
import pdfplumber
import pytesseract
from PIL import Image

MIN_PAGE_CHARS = 50

def extract_columns_aware(page):
    """
    Extracts text from a page taking into account potential multi-column layouts.
    Uses horizontal clustering of words.
    """
    words = page.extract_words(
        x_tolerance=3,
        y_tolerance=3,
        keep_blank_chars=True
    )
    
    if not words:
        return ""
        
    # Find x0 clusters (simplified column detection)
    x0_values = [w["x0"] for w in words]
    min_x = min(x0_values)
    max_x = max(x0_values)
    width = max_x - min_x
    
    # If there is a large gap in x0 values, we might have columns.
    # We can rely on extract_text() with layout=True for most well-formed PDFs.
    # However, to explicitly handle columns safely, pdfplumber's extract_text is generally ok if we don't force strict reading order without it.
    
    # Using layout=True often preserves the column structure visually, 
    # but for text flow, we just use standard extract_text for now unless we see tables.
    # If the user specifically needs robust 2-column extraction, we cluster by x0.
    
    # Let's keep it simple and robust using pdfplumber's default which is mostly good enough,
    # but we sort by top and then x0 if we want manual flow.
    # Actually, the guide says: "Use page.extract_words() and cluster by x-coordinate — if two distinct x-clusters exist for a page, extract each column separately (top-to-bottom) before concatenating, rather than trusting default reading order."
    
    # Simple column clustering:
    # Sort words by top first
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    
    # Group words into potential columns based on x0
    mid_x = min_x + (width / 2)
    col1 = []
    col2 = []
    
    # Check if we clearly have two columns (many words strictly on left and right)
    # This is a heuristic.
    for w in sorted_words:
        if w["x0"] < mid_x:
            col1.append(w)
        else:
            col2.append(w)
            
    # Re-sort each column top-to-bottom, left-to-right
    col1 = sorted(col1, key=lambda w: (w["top"], w["x0"]))
    col2 = sorted(col2, key=lambda w: (w["top"], w["x0"]))
    
    def words_to_text(word_list):
        if not word_list: return ""
        lines = []
        current_line = []
        current_top = word_list[0]["top"]
        
        for w in word_list:
            if abs(w["top"] - current_top) > 5: # New line
                lines.append(" ".join(w["text"] for w in current_line))
                current_line = [w]
                current_top = w["top"]
            else:
                current_line.append(w)
        if current_line:
            lines.append(" ".join(w["text"] for w in current_line))
        return "\n".join(lines)
        
    text1 = words_to_text(col1)
    text2 = words_to_text(col2)
    
    # If both columns have significant text, we likely have a 2-column layout.
    if len(text1) > 200 and len(text2) > 200:
        return text1 + "\n\n" + text2
    else:
        # Fallback to default
        return page.extract_text(layout=False) or ""

def strip_footer_noise(text: str) -> str:
    # Remove lines like "Source: 11/12/2019" or bare page numbers
    lines = text.split("\n")
    cleaned = []
    
    source_pattern = re.compile(r"^Source:.*\d{1,2}/\d{1,2}/\d{4}$", re.IGNORECASE)
    page_num_pattern = re.compile(r"^\s*\d+\s*$")
    
    for line in lines:
        if source_pattern.match(line):
            continue
        if page_num_pattern.match(line):
            continue
        cleaned.append(line)
        
    return "\n".join(cleaned)

def ocr_fallback(page) -> str:
    # Requires pytesseract and poppler installed on the host/container
    try:
        img = page.to_image(resolution=200)
        text = pytesseract.image_to_string(img.original)
        return text
    except Exception as e:
        print(f"OCR failed for page {page.page_number}: {e}")
        return ""

def extract_pdf_text(path: str) -> list[dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = extract_columns_aware(page)
            text = strip_footer_noise(text)
            
            if len(text.strip()) < MIN_PAGE_CHARS:
                text = ocr_fallback(page) or ""
                unextracted = not text.strip()
            else:
                unextracted = False
                
            pages.append({
                "page_number": i + 1,
                "text": text,
                "unextracted": unextracted,
                # We also need to extract lines with formatting for chunking.
                # However, the architecture says chunking happens after or alongside.
                # Let's just extract standard layout elements if needed, but since chunking
                # needs char-level font info, it might be better if chunking reads the PDF directly,
                # or we store the lines here. For simplicity, we can let chunking open the PDF.
            })
    return pages
