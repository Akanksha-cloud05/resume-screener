# Resume Screener Backend

import fitz  # PyMuPDF for PDF parsing
import re
import math
import numpy as np
#from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. AI Model (Sentence-BERT with TF-IDF Fallback)
# BERT gives semantic similarity but requires PyTorch.
# On Windows, PyTorch sometimes fails with a DLL error.
# Fallback to TF-IDF ensures the app always works.

AI_READY = False
try:
    from sentence_transformers import SentenceTransformer
    #Using MiniLM - lighter and 5x faster than MPNet
    model = SentenceTransformer('all-MiniLM-L6-v2')
    AI_READY = True
    print("BERT loaded.")
except Exception as e:
    print(f"BERT failed: {e}. Using TF-IDF fallback.")
    AI_READY = False


# 2. PDF Text Extractor

def extract_text_from_pdf(file_object):
    """
    Extract text from a PDF file object.
    Returns None if the file is scanned/image-based or unreadable.
    Note: PyMuPDF works great on standard text PDFs but returns empty strings on scanned resumes - handled by the None check below.
    """
    try:
        # Read the uploaded file bytes (Streamlit's file uploader)
        pdf_bytes = file_object.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Clean up hyphenation artifacts from LaTex-compiled PDFs
        # e.g. "ma-\nchine learn-\ning" -> "machine learning"
        # Noticed this was killing skill detection on formatted resumes
        text = text.replace('-\n', '').replace('\n', ' ')
             
        # Normalize whitespace for consistent AI embedding input
        text = re.sub(r'\s+', ' ', text)

        # Return None for scanned/empty PDFs(matches docstrings)
        return text if text.strip() else None
    except Exception:
        return None
    
    
# 3. Similarity Scoring Engine

def calculate_match_score(job_description, resume_text):
    """
    Calculate similarity between Job Description and Resume.
    Returns percentage (0 to 100). Always returns a float, never None
    """
    try:
        if not job_description or not resume_text:
            return 0.0
    
        # Method 1: Sentence-BERT (if available)
        if AI_READY and model is not None:
            try:
                embedding = model.encode([job_description, resume_text])
                score = cosine_similarity([embedding[0]], [embedding[1]])[0][0]
                result = round(float(score) * 100, 1)
                return result if result is not None else 0.0
            except Exception:
                pass  # Fall through to TF-IDF

        # Method 2: TF-IDF Fallback (bulletproof, no GPU required)
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        result = round(float(score) * 100, 1)
        return result if result is not None else 0.0
    except Exception:
    # If anything else fails , return 0.0
        return 0.0 