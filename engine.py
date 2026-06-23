# Resume Screener Backend
# ENGINE VERSION 2 - FIXED compare_skills
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
#Safety net: test torch first to prevent Streamlit terminal spam
AI_READY = False
bert_model = None

try:
    import torch
    from sentence_transformers import SentenceTransformer
    # Using MiniLM — lighter and 5x faster than MPNet on CPU
    bert_model = SentenceTransformer('all-MiniLM-L6-v2')
    AI_READY = True
    print("[INIT] BERT loaded successfully.")
except Exception as e:
    print(f"[INIT] BERT failed: {e}. Using TF-IDF fallback.")
    AI_READY = False

# ------------------------------------------------------------
# PDF Parser 
# ------------------------------------------------------------

def extract_text_from_pdf(file_object):
    """Extract text from PDF. Returns None for scanned/image PDFs."""
    try:
        pdf_bytes = file_object.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        
        text = text.replace("\n", " ")
        text = " ".join(text.split())
        
        return text if text.strip() else None
    except Exception:
        return None

# ------------------------------------------------------------
# Similarity Scoring 
# ------------------------------------------------------------

def calculate_match_score(job_description, resume_text):
    try:
        if not job_description or not resume_text:
            return 0.0
        
        # Method 1: BERT
        if AI_READY and bert_model is not None:
            try:
                embeddings = bert_model.encode([job_description, resume_text])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                result = round(float(score) * 100, 2)
                return result if result is not None else 0.0
            except Exception:
                pass
        
        # Method 2: TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # FIX: Initialize final_score before the if block
        final_score = score * 100  # Default: no penalty
        
        word_count = len(resume_text.split())
        if word_count < 50:
            penalty = 0.5
            final_score = score * penalty * 100
        elif word_count > 1000:
            penalty = 0.9
            final_score = score * penalty * 100
        
        return round(final_score, 2)
        
    except Exception:
        return 0.0
    
# Skill Extraction
# Hardcoded because I didn't have time to train NER
# TODO: This misses phrases like "Deep Learning" vs "DeepLearning"
# Also abbreviations like "ML" won't match "Machine Learning"
# Will fix with synonyms later

# Skill Extraction (Day 4)
SKILLS_DATABASE = [
    'python', 'java', 'c++', 'javascript', 'sql', 'mysql', 'postgresql',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
    'tensorflow', 'pytorch', 'pandas', 'numpy',
    'machine learning', 'deep learning', 'nlp', 'data analysis',
    'react', 'django', 'flask', 'spark', 'hadoop',
    'tableau', 'power bi', 'statistics', 'excel',
    'ml', 'dl', 'scikit-learn', 'sklearn', 'data science',
    'artificial intelligence', 'ai', 'computer vision', 'cv'
]

def extract_skills(text):
    text_lower = text.lower()
    matched = []
    for skill in SKILLS_DATABASE:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(skill)
    return list(set(matched))

def compare_skills(jd_text, resume_text):
    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)
    matched = list(set(jd_skills) & set(resume_skills))
    missing = list(set(jd_skills) - set(resume_skills))  # <-- MUST BE MINUS
    return matched, missing
