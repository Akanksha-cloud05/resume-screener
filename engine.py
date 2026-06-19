# Resume Screener Backend

import fitz  # PyMuPDF for PDF parsing
import re
import math
import numpy as np
#from sentence_transformers import SentenceTransformer

# 1. PDF Text Extractor

def extract_text_from_pdf(pdf_file):
    """
    Extract text from a PDF file object.
    Returns none if the file is scanned/image-based or unreadable.
    Note: PyMuPDF works great on standard text PDFs but returns empty strings on scanned resumes - handled by the None check below.
    """
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
            doc.close()

            # Clean up hyphenation artifacts from LatTex-compiled PDFs
            # e.g. "ma-\nchine learn-\ning" -> "machine learning"
            # Noticed this was killing skill detection on formtted resumes
            text = text.replace('-\n', '').replace('\n', ' ')
            return text if text.strip() else None
    except Exception:
        return None
    
