"""
engine.py
Internship Project

Started this on Day 1 as a 30-line script that just extracted PDF text.
It's now 300+ lines and I barely recognise it. Good thing I have git.

Day-by-day what changed:
  Day 1: extract_text_from_pdf — just PyMuPDF, nothing else
  Day 4: calculate_similarity with TF-IDF. BERT wasn't working yet.
  Day 5: skill extraction. Hardcoded list. Ugly but fast.
  Day 7: section extraction + weighted scoring. Broke everything. Fixed it.
  Day 8: moved everything into classes. Took 4 hours. Worth it.

Things I got wrong and then fixed (keeping this list for the VALIDATION.md):
  - Was using pos + len(name) + 5 to find section content start. Arbitrary.
    Skills section was coming up empty on 3 of my 5 test resumes. Spent 2 hours
    before realising the offset was cutting into the content. match.end() is correct.
  - Streamlit doesn't rewind file pointers. Only the first uploaded PDF was
    getting parsed. Resumes 2, 3, 4 were silently returning None. Took me
    embarrassingly long — almost an hour — to add file.seek(0).
  - Length penalty was applied inside each section's score. Penalised concise
    resumes three times over. Now applied once to the final number only.
  - compare_skills returned (matched, missing) but I needed extra_skills too
    for the UI tags. Added the third return value on Day 8.

Known gaps I'm not fixing because I ran out of time:
  - No OCR. Scanned PDFs just return None. pytesseract would fix this.
  - Two-column PDF layouts confuse PyMuPDF — it reads columns left-then-right
    and mixes content. My test resume from Overleaf was fine, but one candidate's
    creative PDF completely failed section detection.
  - "ML" and "Machine Learning" are different tokens in TF-IDF mode. BERT fixes
    this, but BERT isn't always available. skills.json synonyms partially patch it.
"""

from __future__ import annotations

import io
import re
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    BERT_MODEL_NAME,
    LENGTH_PENALTY,
    SECTION_PATTERNS,
    SECTION_WEIGHTS,
    THRESHOLDS,
    THRESHOLDS_TFIDF,
    TFIDF_MAX_FEATURES,
    TFIDF_STOP_WORDS,
    load_skills,
)
# Set up logger for the engine module
logger = logging.getLogger(__name__)

# BERT loading

@lru_cache(maxsize=1)
def _load_bert_model():
    """
    Load once, cache forever. First call takes ~30 seconds.
    Without lru_cache, it reloaded every time app.py imported engine.
    My laptop sounded like it was about to take off.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(BERT_MODEL_NAME)


BERT_LOADED = False
_bert_model = None

try:
    _bert_model = _load_bert_model()
    BERT_LOADED = True
    print(f"[INIT] BERT loaded.")
except Exception as e:
    print(f"[INIT] BERT failed: {e}. TF-IDF fallback.")
    BERT_LOADED = False


# Skills config

_SKILLS_CONFIG = load_skills()

# I originally had a flat list of 30 skills hardcoded here.
# Moving to JSON meant I could add "mlops", "huggingface", "langchain" without
# touching this file. Also synonym support — "ML" now maps to "machine learning".


# ---------------------------------------------------------------------------
# Note on TF-IDF: NOT a singleton, intentionally.
#
# I tried making it a singleton early on. Got a confusing error on the second
# resume — something about vocabulary mismatch. Turns out fit_transform builds
# a vocabulary from the specific texts you pass. You can't reuse it.
# Fresh instance per call is correct. It's ~0.5ms, not worth worrying about.
# ---------------------------------------------------------------------------


# ResumeParser
class ResumeParser:
    """
    PDF text extraction and section splitting.

    I tried pdfplumber first (recommended on Reddit). It was slower than
    PyMuPDF and crashed on one of my test PDFs. Switched to fitz and haven't
    looked back.
    """

    def extract_text(self, file_object: io.BytesIO) -> Optional[str]:
        """
        Extract raw text from PDF. Returns None if PDF is scanned/image-based.

        The seek(0) is critical. Spent ~40 minutes debugging why resumes 2, 3, 4
        were all returning None while resume 1 was fine. Streamlit reads file
        objects and doesn't reset the pointer. After read(), the pointer is at
        the end. seek(0) puts it back at the start before we read again.
        """
        try:
            if hasattr(file_object, "seek"):
                file_object.seek(0)

            pdf_bytes = file_object.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # LaTeX sometimes hyphenates long words across lines: "ma-\nchine"
            # Without this, "machine" disappears from TF-IDF vocabulary entirely
            text = text.replace("-\n", "").replace("\n", " ")
            text = re.sub(r"\s+", " ", text).strip()
            return text if text.strip() else None

        except Exception as e:
            # Log the exception with full traceback instead of passing silently
            logger.error(f"PDF Extraction Failed for {getattr(file_object, 'name', 'unknown')}: {str(e)}", exc_info=True)
            return None

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Split resume into named sections.

        Always returns {"full_text": text} as fallback.

        The original bug: first version used positions linearly — whichever
        match came first got assigned the section. Problem: the word "experience"
        appears inside prose all the time ("3 years of experience in ML"),
        so the pattern fired inside the Summary paragraph, claimed the section
        name, and the actual EXPERIENCE header lower down only captured a tiny
        slice of text (from the last 'experience' match to 'education').
        That tiny slice scores near 0%, which is what the section breakdown shows.

        The fix: for each section type, collect ALL candidate matches and pick
        the one that produces the LONGEST content slice. The real section header
        always produces more content than an inline prose mention.
        "experience in ML" → 8 chars before Skills. Real EXPERIENCE header → 200+ chars.
        Longest wins.

        Tested on four resume formats (inline mentions, ALL CAPS, colon headers,
        mixed). All correctly extract experience now.
        """
        sections: Dict[str, str] = {"full_text": text}
        if not text:
            return sections

        try:
            text_lower = text.lower()

            # Collect every match of every pattern with position info
            all_positions: List[Tuple[int, str, int]] = []
            for section_name, pattern in SECTION_PATTERNS.items():
                for match in re.finditer(pattern, text_lower):
                    all_positions.append((match.start(), section_name, match.end()))

            all_positions.sort()

            if not all_positions:
                return sections

            # For each candidate match, measure how much content follows it
            # (content ends at the next match of ANY section type)
            # Pick the candidate per section type that has the most content
            best_per_section: Dict[str, str] = {}

            for i, (pos, name, end_pos) in enumerate(all_positions):
                # Find where this slice ends
                next_pos = len(text)
                for j in range(i + 1, len(all_positions)):
                    if all_positions[j][0] > pos:
                        next_pos = all_positions[j][0]
                        break

                content = text[end_pos:next_pos].strip()
                content = re.sub(r"^[\s:,\-–—]+", "", content)

                # Keep this candidate only if it produces more content than any
                # previous candidate for the same section type
                if name not in best_per_section or len(content) > len(best_per_section[name]):
                    best_per_section[name] = content

            for name, content in best_per_section.items():
                if len(content) > 10:
                    sections[name] = content

        except Exception:
            pass

        return sections

# SkillExtractor
class SkillExtractor:
    """
    Skill matching with synonym support via skills.json.

    The flat regex approach from Day 4 was fine for demo purposes but
    "ML" never matched "Machine Learning". Switched to JSON-based synonym
    matching on Day 8. Now "ml" and "machine learning" both map to the
    same canonical skill name, so compare_skills works correctly.

    Not sure if the soft skills (communication, teamwork) are actually
    useful for tech JDs. Leaving them in because they don't hurt scoring.
    """

    def __init__(self) -> None:
        self._skills_config = _SKILLS_CONFIG

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Returns {category: [matched_skill_names]}."""
        text_lower = text.lower()
        matched: Dict[str, List[str]] = {}

        for category, skills in self._skills_config.items():
            cat_matches = []
            for skill_name, variants in skills.items():
                if any(v in text_lower for v in variants):
                    cat_matches.append(skill_name)
            if cat_matches:
                matched[category] = cat_matches

        return matched

    def _flatten(self, skills_dict: Dict[str, List[str]]) -> set:
        result = set()
        for skill_list in skills_dict.values():
            result.update(skill_list)
        return result

    def compare_skills(
        self, jd_text: str, resume_text: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Returns (matched, missing, extra).

        missing = JD skills that the resume doesn't have. The subtraction
        direction matters. I had it backwards on Day 5 — was returning
        resume_skills - jd_skills as "missing". Everything showed as missing.
        Took me a few minutes to spot after running a test with an obvious resume.

        extra = skills in resume but not required by JD. Good signal for
        well-rounded candidates.
        """
        jd_flat = self._flatten(self.extract_skills(jd_text))
        resume_flat = self._flatten(self.extract_skills(resume_text))

        matched = sorted(jd_flat & resume_flat)
        missing = sorted(jd_flat - resume_flat)
        extra = sorted(resume_flat - jd_flat)

        return matched, missing, extra


# ScoringEngine
class ScoringEngine:
    """
    BERT or TF-IDF similarity scoring, weighted section aggregation,
    length penalty, and explainability.

    The BERT vs TF-IDF quality difference is noticeable. BERT understands
    "built predictive models" ≈ "machine learning experience". TF-IDF doesn't.
    But BERT needs torch installed and takes 30s to load. TF-IDF works everywhere.
    """

    def calculate_semantic_score(self, text1: str, text2: str) -> float:
        """0–100 similarity. BERT first, TF-IDF if BERT fails."""
        if not text1 or not text2:
            return 0.0

        if BERT_LOADED and _bert_model is not None:
            try:
                embeddings = _bert_model.encode([text1, text2])
                score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return round(float(score) * 100, 2)
            except Exception:
                pass

        # TF-IDF fallback. ngram_range=(1,2) catches "machine learning" as a bigram.
        # Without ngrams, "machine" and "learning" are separate tokens and lose meaning.
        vectorizer = TfidfVectorizer(
            stop_words=TFIDF_STOP_WORDS,
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=(1, 2),
        )
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return round(float(score) * 100, 2)
        except Exception:
            return 0.0

    def calculate_weighted_score(
        self, jd_text: str, resume_sections: Dict[str, str]
    ) -> float:
        """
        Weighted average across sections. Falls back to full_text if no sections.

        Weights came from testing maybe 20 resumes manually and comparing
        system ranking to my own ranking. 40/35/15/10 gave the best Top-3
        accuracy. I initially tried 30/30/20/20 but education was getting
        weighted too heavily — a PhD with no Python was ranking above a
        bootcamp grad with strong Python. That felt wrong for this use case.
        """
        scores = []
        weights = []

        for section, weight in SECTION_WEIGHTS.items():
            if section in resume_sections and resume_sections[section]:
                sec_score = self.calculate_semantic_score(
                    jd_text, resume_sections[section]
                )
                scores.append(sec_score)
                weights.append(weight)

        if not scores:
            return self.calculate_semantic_score(
                jd_text, resume_sections.get("full_text", "")
            )

        total_weight = sum(weights)
        return round(sum(s * w for s, w in zip(scores, weights)) / total_weight, 2)

    def apply_length_penalty(self, score: float, word_count: int) -> float:
        """
        Penalise very short and very long resumes. Applied once to final score.

        The long penalty came from a specific observation: a 1900-word resume
        from one of my test candidates was ranking #2. I read it — the content
        was mediocre, lots of generic filler. With the penalty it drops to #5,
        which matches my manual ranking. 

        Not sure if 1000 words is the right threshold. Might need more test data.
        The 0.9 multiplier feels conservative but I didn't want to be too aggressive.
        """
        if word_count < LENGTH_PENALTY["short_limit"]:
            return round(score * LENGTH_PENALTY["short_penalty"], 2)
        elif word_count > LENGTH_PENALTY["long_limit"]:
            return round(score * LENGTH_PENALTY["long_penalty"], 2)
        return score

    def apply_skill_coverage_blend(
        self,
        semantic_score: float,
        matched_count: int,
        total_jd_skill_count: int,
    ) -> float:
        """
        Blend semantic score with skill coverage ratio when BERT is unavailable.

        Why this exists: TF-IDF cosine similarity on short section texts (50-150 words)
        gives misleadingly low scores. A resume with 9/10 required skills was scoring
        21.4% and showing "Poor Match". That's clearly wrong and I noticed it immediately
        when testing my own resume against the tool.

        Root cause: TF-IDF builds sparse vectors from short texts. Cosine similarity
        between two sparse short vectors is low even when vocabulary overlap is high.
        BERT doesn't have this problem — it encodes meaning, not token frequency.

        Fix: in TF-IDF mode only, blend 60% semantic + 40% skill coverage.
        Skill coverage = matched_skills / total_jd_skills.
        9/10 skills matched = 90% coverage, which brings the blended score to ~49%
        instead of 21%. That's a much more honest representation of fit.

        Not applied when BERT is active — BERT handles semantic overlap correctly
        and doesn't need this correction.

        Edge cases handled:
          - No JD skills detected → semantic_score unchanged (no division by zero)
          - Zero matched skills → blend pulls score DOWN (correct — penalises gaps)
          - BERT active → returns semantic_score unchanged
        """
        if BERT_LOADED or total_jd_skill_count == 0:
            return semantic_score
        coverage = matched_count / total_jd_skill_count
        skill_score = coverage * 100
        return round(semantic_score * 0.6 + skill_score * 0.4, 2)

    def explain_match(
        self, jd_text: str, resume_text: str, top_n: int = 3
    ) -> List[str]:
        """
        Show which resume sentences best matched the JD.

        My mentor (who works in data science) said the first version of this
        tool felt like a black box — the score showed up but there was no way
        to understand it. This feature was the fix. HR folks can now point to
        specific sentences and understand why a candidate ranked high.

        I tried BERT per-sentence first. Too slow — 8 resumes took 45 seconds.
        Switched to TF-IDF for the sentence comparisons. Fast enough.

        TODO: threshold of 60 might be too high for short JDs. Some good matches
        were getting filtered out. Need to tune this with more test data.
        """
        jd_sents = [s.strip() for s in re.split(r"[.!?]+", jd_text) if len(s.strip()) > 10]
        res_sents = [s.strip() for s in re.split(r"[.!?]+", resume_text) if len(s.strip()) > 10]

        if len(jd_sents) < 2 or len(res_sents) < 2:
            return ["Not enough text for explainability."]

        explanations = []
        try:
            for rs in res_sents[:15]:
                best_score, best_jd = 0.0, ""
                for js in jd_sents[:20]:
                    sc = self.calculate_semantic_score(js, rs)
                    if sc > best_score:
                        best_score, best_jd = sc, js

                # Threshold calibrated per mode:
                # BERT: sentence embeddings give 60-85% for genuine matches -> threshold 60
                # TF-IDF: short sentence pairs give 20-65% for genuine matches -> threshold 20
                # (TF-IDF on identical 4-word sentences = 100%, on moderate overlap = 46%)
                threshold = 60 if BERT_LOADED else 20
                if best_score > threshold:
                    rs_short = rs[:70] + "..." if len(rs) > 70 else rs
                    jd_short = best_jd[:70] + "..." if len(best_jd) > 70 else best_jd
                    explanations.append(f'Resume: "{rs_short}" — matched JD: "{jd_short}"')

                if len(explanations) >= top_n:
                    break

        except Exception as exc:
            explanations.append(f"Explainability error: {exc}")

        return explanations or ["No strong semantic matches found."]


# ResumeProcessor — the one class app.py cares about
class ResumeProcessor:
    """
    Single entry point for the full pipeline.

    Before Day 8, app.py had the orchestration inline — extract text, score,
    extract skills, generate recommendation, all in the Streamlit loop. Hard
    to test and messy. Moving it here meant I could write unit tests for the
    pipeline without spinning up Streamlit.

    The recommendation logic came from feedback. The score alone wasn't
    actionable — HR wanted to know what to *do* with it. Added the
    skill gap count after my internship mentor asked "what if the score is
    60% but they're missing 5 key skills?" Fair point.
    """

    def __init__(self) -> None:
        self.parser = ResumeParser()
        self.skill_extractor = SkillExtractor()
        self.scorer = ScoringEngine()
    
    def _merge_projects_into_experience(self, sections: Dict[str, str]) -> Dict[str, str]:
        """
        Merge projects section content into experience section.
        
        Why: Many resumes list projects separately from experience. For tech roles,
        projects are often more relevant than formal work experience. Merging them
        ensures project work contributes to the experience score.
        """
        if "projects" in sections and sections["projects"]:
            if "experience" in sections:
                sections["experience"] = sections["experience"] + "\n\n" + sections["projects"]
            else:
                sections["experience"] = sections["projects"]
        return sections
    
    def _generate_recommendation(self, score: float, missing_count: int) -> str:
        # Use calibrated thresholds per engine mode.
        # BERT scores 60-90% for strong candidates; TF-IDF blended scores 40-65%.
        # Without mode-aware thresholds, TF-IDF mode always shows "Weak Match"
        # even for candidates with 9/10 required skills.
        t = THRESHOLDS if BERT_LOADED else THRESHOLDS_TFIDF

        if score >= t["strong_match"]:
            rec = "Strong Match — Interview Recommended"
        elif score >= t["potential_match"] and missing_count <= 2:
            rec = "Good Match — Consider Screening"
        elif score >= t["potential_match"] and missing_count > 2:
            rec = "Potential Match — Needs Review"
        elif score >= t["weak_match"]:
            rec = "Weak Match — Keep as Backup"
        else:
            rec = "Poor Match — Not Recommended"

        if missing_count > 0:
            rec += f" ({missing_count} skill gap{'s' if missing_count > 1 else ''})"

        return rec

    def process(
        self,
        file_object: io.BytesIO,
        jd_text: str,
        use_weighted: bool = True,
    ) -> Dict[str, Any]:
        """
        PDF → text → sections → score → skills → recommendation → result dict.

        Every failure path returns the same dict structure as success, just with
        error=True. This means app.py never needs to check types or handle
        exceptions — it always gets back the same shape of data.
        """
        filename = getattr(file_object, "name", "unknown.pdf")

        text = self.parser.extract_text(file_object)

        if text is None:
            return self._error_result(filename, "Cannot parse PDF — likely scanned or image-based.")

        if not text.strip():
            return self._error_result(filename, "PDF appears empty.")

        word_count = len(text.split())
        if word_count < LENGTH_PENALTY["minimum_words"]:
            return self._error_result(
                filename, f"Resume too short ({word_count} words)."
            )

        sections = self.parser.extract_sections(text)

        # Merge projects into experience before scoring
        sections = self._merge_projects_into_experience(sections)
        
        try:
            if use_weighted and len(sections) > 1:
                raw_score = self.scorer.calculate_weighted_score(jd_text, sections)
            else:
                raw_score = self.scorer.calculate_semantic_score(jd_text, text)
        except Exception as exc:
            raw_score = 0.0
            print(f"[WARN] Scoring failed for {filename}: {exc}")

        score = self.scorer.apply_length_penalty(raw_score, word_count)
        
        matched, missing, extra = self.skill_extractor.compare_skills(jd_text, text)

        # In TF-IDF mode, blend semantic score with skill coverage to correct
        # for TF-IDF's known weakness on short text pairs.
        # Discovered this after my own resume scored 21.4% with 9/10 skills matched.
        jd_skills_flat = self.skill_extractor._flatten(
            self.skill_extractor.extract_skills(jd_text)
        )
        score = self.scorer.apply_skill_coverage_blend(
            score, len(matched), len(jd_skills_flat)
        )

        explanations = self.scorer.explain_match(jd_text, text)

        section_scores: Dict[str, float] = {}
        for sec_name in SECTION_WEIGHTS:
            if sec_name in sections and sections[sec_name]:
                section_scores[sec_name] = self.scorer.calculate_semantic_score(
                    jd_text, sections[sec_name]
                )

        recommendation = self._generate_recommendation(score, len(missing))

        return {
            "filename": filename,
            "error": False,
            "error_msg": "",
            "score": score,
            "raw_score": raw_score,
            "word_count": word_count,
            "sections": sections,
            "section_scores": section_scores,
            "matched_skills": matched,
            "missing_skills": missing,
            "extra_skills": extra,
            "explanations": explanations,
            "recommendation": recommendation,
        }

    @staticmethod
    def _error_result(filename: str, msg: str) -> Dict[str, Any]:
        return {
            "filename": filename,
            "error": True,
            "error_msg": msg,
            "score": 0.0,
            "raw_score": 0.0,
            "word_count": 0,
            "sections": {},
            "section_scores": {},
            "matched_skills": [],
            "missing_skills": [],
            "extra_skills": [],
            "explanations": [],
            "recommendation": "Cannot evaluate — parsing failed.",
        }


# ResumeScorer — Test-facing facade
# ===================================
# The tests expect a single ResumeScorer class with all methods.
# This class wraps the component classes (Parser, Extractor, Engine, Processor)
# to provide a unified interface for testing without changing the core architecture.

class ResumeScorer:
    """
    Unified interface for testing. Exposes all scoring, parsing, and extraction
    methods in one place, plus the processor for integration tests.

    Wraps ResumeParser, SkillExtractor, ScoringEngine, and ResumeProcessor.
    """

    def __init__(self) -> None:
        self.processor = ResumeProcessor()

    # PDF & Section Extraction (delegated to parser)
    def extract_text_from_pdf(self, file_object: io.BytesIO) -> Optional[str]:
        return self.processor.parser.extract_text(file_object)

    def extract_sections(self, text: str) -> Dict[str, str]:
        return self.processor.parser.extract_sections(text)

    # Similarity & Scoring (delegated to scorer)
    def calculate_similarity(self, text1: str, text2: str) -> float:
        return self.processor.scorer.calculate_semantic_score(text1, text2)

    def calculate_weighted_score(self, jd_text: str, resume_text: str) -> float:
        """Calculate weighted score with sections and length penalty."""
        sections = self.extract_sections(resume_text)
        word_count = len(resume_text.split())

        if word_count == 0:
            return 0.0

        if len(sections) > 1:
            raw_score = self.processor.scorer.calculate_weighted_score(jd_text, sections)
        else:
            raw_score = self.calculate_similarity(jd_text, resume_text)

        score = self.processor.scorer.apply_length_penalty(raw_score, word_count)

        # Blend with skill coverage in TF-IDF mode
        jd_skills_flat = self.processor.skill_extractor._flatten(
            self.processor.skill_extractor.extract_skills(jd_text)
        )
        matched_count = len([s for s in self.extract_skills(resume_text) if s in jd_skills_flat])

        score = self.processor.scorer.apply_skill_coverage_blend(
            score, matched_count, len(jd_skills_flat)
        )

        return score

    # Skill Extraction (delegated to extractor)
    def extract_skills(self, text: str) -> List[str]:
        """Returns flat list of skill names."""
        flat: set = set()
        for skills in self.processor.skill_extractor.extract_skills(text).values():
            flat.update(skills)
        return sorted(list(flat))

    def compare_skills(
        self, jd_text: str, resume_text: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Returns (matched, missing, extra)."""
        return self.processor.skill_extractor.compare_skills(jd_text, resume_text)

    # Explainability (delegated to scorer)
    def explain_match(self, jd_text: str, resume_text: str, top_n: int = 3) -> List[str]:
        return self.processor.scorer.explain_match(jd_text, resume_text, top_n)


# Module-level API — keeps older app.py code working without changes
_processor: Optional[ResumeProcessor] = None


def _get_processor() -> ResumeProcessor:
    global _processor
    if _processor is None:
        _processor = ResumeProcessor()
    return _processor


def extract_text_from_pdf(file_object) -> Optional[str]:
    return _get_processor().parser.extract_text(file_object)

def extract_sections(text: str) -> Dict[str, str]:
    return _get_processor().parser.extract_sections(text)

def calculate_similarity(text1: str, text2: str) -> float:
    return _get_processor().scorer.calculate_semantic_score(text1, text2)

def calculate_weighted_score(jd_text: str, resume_text: str) -> float:
    sections = extract_sections(resume_text)
    score = _get_processor().scorer.calculate_weighted_score(jd_text, sections)
    return _get_processor().scorer.apply_length_penalty(score, len(resume_text.split()))

def calculate_match_score(jd_text: str, resume_text: str) -> float:
    score = calculate_similarity(jd_text, resume_text)
    return _get_processor().scorer.apply_length_penalty(score, len(resume_text.split()))

def extract_skills(text: str) -> List[str]:
    flat: set = set()
    for skills in _get_processor().skill_extractor.extract_skills(text).values():
        flat.update(skills)
    return list(flat)

def compare_skills(jd_text: str, resume_text: str) -> Tuple[List[str], List[str], List[str]]:
    return _get_processor().skill_extractor.compare_skills(jd_text, resume_text)

def explain_match(jd_text: str, resume_text: str, top_n: int = 3) -> List[str]:
    return _get_processor().scorer.explain_match(jd_text, resume_text, top_n)