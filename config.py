"""
config.py — Central configuration for AI Resume Screener
Internship Project

I moved all hardcoded values here on Day 8.
Should have done this on Day 1 honestly — makes tuning so much easier.
"""

import os
import json

# Streamlit UI Configuration
STREAMLIT_CONFIG = {
    "chart_height": 350,
    "expander_border": True,
    "skill_tag_limit": 8,
    "progress_bar_update_interval": 1,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SKILLS_PATH = os.path.join(BASE_DIR, "assets", "skills.json")

# Section weights tuned after testing ~20 resumes manually
# Skills weighted highest because tech screening is skills-first
SECTION_WEIGHTS = {
    "skills": 0.40,
    "experience": 0.35,
    "education": 0.15,
    "summary": 0.10,
}

# Score thresholds for recruiter recommendations
# BERT mode: scores range 50-90% for relevant candidates -> thresholds 80/60/40
# TF-IDF mode: blended scores range 30-65% for relevant candidates -> thresholds 65/50/35
THRESHOLDS = {
    "strong_match": 80,
    "potential_match": 60,
    "weak_match": 40,
}

# TF-IDF mode thresholds (used when BERT is unavailable)
# Calibrated based on testing: 9/10 skills matched = ~58% blended score
THRESHOLDS_TFIDF = {
    "strong_match": 65,
    "potential_match": 45,
    "weak_match": 30,
}

# Length penalty thresholds
# Very short resumes (<50 words) lack context for semantic similarity
# Very long resumes (>1000 words) may contain irrelevant padding
LENGTH_PENALTY = {
    "short_limit": 50,
    "short_penalty": 0.5,
    "long_limit": 1000,
    "long_penalty": 0.9,
    "minimum_words": 10,  # Below this, we reject entirely
}

BERT_MODEL_NAME = "all-MiniLM-L6-v2"
TFIDF_MAX_FEATURES = 500
TFIDF_STOP_WORDS = "english"

SECTION_PATTERNS = {
    "education": r"\b(education|academic|qualification|degree|university|college)\b",
    "experience": r"\b(experience|work|employment|internship|career|professional|work experience|professional experience)\b",
    "projects": r"\b(projects|project|portfolio)\b",
    "skills": r"\b(skills|technologies|technical|competencies|expertise|proficiencies|tools)\b",
    "summary": r"\b(summary|objective|profile|about|overview|highlights)\b",
}

def load_skills() -> dict:
    """
    Load skills from JSON file. Falls back to minimal defaults if file missing.
    JSON lives in assets/skills.json so it's easy to update without touching code.
    """
    try:
        with open(SKILLS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] skills.json not found at {SKILLS_PATH}. Using minimal defaults.")
        return {
            "technical": {
                "python": ["python", "pandas", "numpy", "django", "flask"],
                "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn"],
                "sql": ["sql", "mysql", "postgresql"],
                "aws": ["aws", "amazon web services"],
            },
            "soft": {
                "communication": ["communication", "presentation"],
            },
        }
def validate_config():
    """Validates configuration parameters on system initialization."""
    import logging
    import os
    logger = logging.getLogger(__name__)
    
    try:
        # Validate core mathematical weights
        weights_sum = sum(SECTION_WEIGHTS.values())
        assert abs(weights_sum - 1.0) < 0.01, f"Section weights sum to {weights_sum}, must equal 1.0"
        
        # Validate scoring boundaries
        assert THRESHOLDS["weak_match"] < THRESHOLDS["potential_match"] < THRESHOLDS["strong_match"], "BERT thresholds must be sequential"
        assert THRESHOLDS_TFIDF["weak_match"] < THRESHOLDS_TFIDF["potential_match"] < THRESHOLDS_TFIDF["strong_match"], "TF-IDF thresholds must be sequential"
        
        # Validate metric limits
        assert 0 <= LENGTH_PENALTY["short_penalty"] <= 1, "short_penalty multiplier must be between 0 and 1"
        assert 0 <= LENGTH_PENALTY["long_penalty"] <= 1, "long_penalty multiplier must be between 0 and 1"
        assert LENGTH_PENALTY["short_limit"] < LENGTH_PENALTY["long_limit"], "Short content word limit must be lower than long content limit"
        
        # Confirm underlying JSON dependency exists
        assert os.path.exists("assets/skills.json"), "Required reference file 'assets/skills.json' not found"
        
        logger.info("System configuration verification complete. All parameters valid.")
        return True
        
    except AssertionError as e:
        logger.error(f"Configuration structural failure: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected configuration processing error: {e}")
        raise

# Auto-execute checks when module is compiled
validate_config()