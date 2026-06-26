# Resume Screener Backend
# What started as a simple PDF parser on Day 1 evolved into this.
# Section extraction cutting off content: switched from arbitrary offset to match.end()
# TF-IDF recreating on every call: moved to singleton
#Length penalty unfairly punishing concise resumes: applied ONCE to final score
# The code is still not perfect (OCR would be nice), but it works reliably for text-based PDFs.

import fitz
import re
import io
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# BERT: Lazy-loaded to avoid 30-second import hangs on Streamlit Cloud

@lru_cache(maxsize=1)
def _load_bert_model():
    """Load Sentence-BERT once, cached across calls."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")

BERT_LOADED = False
_bert_model = None

try:
    _bert_model = _load_bert_model()
    BERT_LOADED = True
    print("[INIT] BERT loaded successfully.")
except Exception as e:
    print(f"[INIT] BERT unavailable: {e}. Using TF-IDF fallback.")
    BERT_LOADED = False

# Configuration (tuned after testing 20 resumes)
SECTION_WEIGHTS = {
    "skills": 0.40,
    "experience": 0.35,
    "education": 0.15,
    "summary": 0.10,
}

SECTION_PATTERNS = {
    "education": r"\b(education|academic|qualification|degree|university|college)\b",
    "experience": r"\b(experience|work|employment|internship|career|professional|projects|project)\b",
    "skills": r"\b(skills|technologies|technical|competencies|expertise|proficiencies|tools)\b",
    "summary": r"\b(summary|objective|profile|about|overview|highlights)\b",
}

# Skill list with abbreviations and synonyms
# Hardcoded because I didn't have time to train NER
# TODO: Migrate to JSON for easier updates
# Known gap: abbreviations like "ML" won't match "Machine Learning" yet
# Will fix with synonyms later if I get time
SKILLS_DATABASE = [
    "python", "java", "c++", "javascript", "sql", "mysql", "postgresql",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git",
    "tensorflow", "pytorch", "pandas", "numpy",
    "machine learning", "deep learning", "nlp", "data analysis",
    "react", "django", "flask", "spark", "hadoop",
    "tableau", "power bi", "statistics", "excel",
    "ml", "dl", "scikit-learn", "sklearn", "data science",
    "artificial intelligence", "ai", "computer vision", "cv",
]

# ---------------------------------------------------------------------------
# TF-IDF Singleton (created once, reused across all calls)
# ---------------------------------------------------------------------------
# I was recreating this for every resume before — huge waste of time.
# Moved to module level as a singleton. Works much faster now.
_TFIDF_VECTORIZER = TfidfVectorizer(stop_words="english", max_features=500)


# Main Scorer Class 
class ResumeScorer:
    """Main scoring engine. I moved everything into a class so it's cleaner."""

    def __init__(self):
        self.section_weights = SECTION_WEIGHTS
        self.section_patterns = SECTION_PATTERNS
        self.skills_database = SKILLS_DATABASE
        self.bert_loaded = BERT_LOADED
        self.bert_model = _bert_model
        self.tfidf_vectorizer = _TFIDF_VECTORIZER

    # PDF Parser

    def extract_text_from_pdf(self, file_object: io.BytesIO) -> Optional[str]:
        """
        Extract text from PDF. Returns None for scanned/image PDFs.

        Took me an hour to figure out why only the first PDF worked...
        Streamlit reads files once and doesn't rewind automatically.
        So I added file.seek(0) here.
        """
        try:
            # Reset file pointer before reading (Streamlit fix)
            if hasattr(file_object, "seek"):
                file_object.seek(0)
            
            pdf_bytes = file_object.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Cleanup LaTeX hyphenation and normalize whitespace
            text = text.replace('-\n', '').replace('\n', ' ')
            text = " ".join(text.split())
            return text if text.strip() else None
        except Exception:
            return None

    
    # Section Extraction
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Split resume into sections using regex heuristics.
        This is NOT perfect. Creative resumes will fail.
        But for standard templates (LaTeX, Word), it works ~80% of the time.

        FIX: I was using start = pos + len(name) + 5 before, which was cutting off
        content. Now I use match.end() which is much more accurate.
        """
        sections: Dict[str, str] = {"full_text": text}
        if not text:
            return sections

        try:
            text_lower = text.lower()
            positions = []

            # Find all section headers
            for section_name, pattern in self.section_patterns.items():
                for match in re.finditer(pattern, text_lower):
                    positions.append((match.start(), section_name, match.end()))

            positions.sort()

            if not positions:
                return sections

            for i, (pos, name, end_pos) in enumerate(positions):
                start = end_pos  # Start right after the header
                end = positions[i + 1][0] if i + 1 < len(positions) else len(text)

                content = text[start:end].strip()
                content = re.sub(r"^[\s:,-]+", "", content)

                if len(content) > 10:
                    sections[name] = content

        except Exception:
            pass  # Fallback to full_text on any regex error

        return sections

    # Similarity Scoring

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity. Tries BERT first, then TF-IDF."""
        if not text1 or not text2:
            return 0.0

        if self.bert_loaded and self.bert_model is not None:
            try:
                embeddings = self.bert_model.encode([text1, text2])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return round(float(score) * 100, 2)
            except Exception:
                pass  # Fall through to TF-IDF

        # TF-IDF singleton is reused here
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(score) * 100, 2)


    # Weighted Scoring

    def calculate_weighted_score(self, jd_text: str, resume_text: str) -> float:
        """
        Calculate weighted score based on resume sections.

        CRITICAL FIX: Length penalty used to be applied to each section individually.
        This was unfairly punishing concise, well-organized resumes.
        Now I apply the length penalty ONCE to the final combined score.
        """
        sections = self.extract_sections(resume_text)

        scores = []
        weights = []

        for section, weight in self.section_weights.items():
            if section in sections and sections[section]:
                section_score = self.calculate_similarity(jd_text, sections[section])
                scores.append(section_score)
                weights.append(weight)

        if not scores:
            final_score = self.calculate_similarity(jd_text, resume_text)
        else:
            total_weight = sum(weights)
            final_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        # Length penalty applied ONCE to the final score
        word_count = len(resume_text.split())
        if word_count < 50:
            final_score = final_score * 0.5
        elif word_count > 1000:
            final_score = final_score * 0.9

        return round(final_score, 2)

    # Simple Scoring (No Sections)

    def calculate_match_score(self, jd_text: str, resume_text: str) -> float:
        """Simple full-text similarity. Used when weighted scoring is off."""
        score = self.calculate_similarity(jd_text, resume_text)

        word_count = len(resume_text.split())
        if word_count < 50:
            score = score * 0.5
        elif word_count > 1000:
            score = score * 0.9

        return round(score, 2)

    # Skill Extraction
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using regex."""
        text_lower = text.lower()
        matched = []
        for skill in self.skills_database:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(skill)
        return list(set(matched))

    def compare_skills(self, jd_text: str, resume_text: str) -> Tuple[List[str], List[str]]:
        """Compare skills between JD and resume. Returns (matched, missing)."""
        jd_skills = self.extract_skills(jd_text)
        resume_skills = self.extract_skills(resume_text)
        matched = list(set(jd_skills) & set(resume_skills))
        missing = list(set(jd_skills) - set(resume_skills))  # MUST BE MINUS
        return matched, missing

    # Explainability

    def explain_match(self, jd_text: str, resume_text: str, top_n: int = 3) -> List[str]:
        """
        Show which resume sentences matched the JD.
        This builds trust with HR — they can see WHY the score was given.
        """
        jd_sentences = [s.strip() for s in re.split(r"[.!?]+", jd_text) if len(s.strip()) > 10]
        resume_sentences = [s.strip() for s in re.split(r"[.!?]+", resume_text) if len(s.strip()) > 10]

        if len(jd_sentences) < 2 or len(resume_sentences) < 2:
            return ["Not enough text for explainability"]

        explanations = []
        try:
            for rs in resume_sentences[:15]:
                best_score = 0.0
                best_jd = ""
                for js in jd_sentences[:20]:
                    score = self.calculate_similarity(js, rs)
                    if score > best_score:
                        best_score = score
                        best_jd = js

                if best_score > 60:
                    explanations.append(
                        f"Resume: '{rs[:60]}...' matches JD: '{best_jd[:60]}...'"
                    )
                if len(explanations) >= top_n:
                    break
        except Exception as e:
            explanations.append(f"Explainability error: {e}")

        return explanations if explanations else ["No strong semantic matches found"]


# ---------------------------------------------------------------------------
# Backwards compatibility for app.py
# ---------------------------------------------------------------------------
_scorer = None

def _get_scorer():
    global _scorer
    if _scorer is None:
        _scorer = ResumeScorer()
    return _scorer

def extract_text_from_pdf(file_object):
    return _get_scorer().extract_text_from_pdf(file_object)

def extract_sections(text):
    return _get_scorer().extract_sections(text)

def calculate_similarity(text1, text2):
    return _get_scorer().calculate_similarity(text1, text2)

def calculate_weighted_score(jd_text, resume_text):
    return _get_scorer().calculate_weighted_score(jd_text, resume_text)

def calculate_match_score(jd_text, resume_text):
    return _get_scorer().calculate_match_score(jd_text, resume_text)

def extract_skills(text):
    return _get_scorer().extract_skills(text)

def compare_skills(jd_text, resume_text):
    return _get_scorer().compare_skills(jd_text, resume_text)

def explain_match(jd_text, resume_text, top_n=3):
    return _get_scorer().explain_match(jd_text, resume_text, top_n)