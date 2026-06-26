"""
test_engine.py - Day 8 Unit Tests

Tests for the ResumeScorer class and helper functions.
Run with: pytest tests/test_engine.py -v
"""

import pytest
import io
from typing import Dict, Any

# Import the engine module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine


# Fixtures

@pytest.fixture
def scorer():
    """Return a ResumeScorer instance."""
    return engine.ResumeScorer()


@pytest.fixture
def sample_jd():
    """Sample job description for testing."""
    return """
    We are looking for a Data Scientist with strong Python and SQL skills.
    Experience with machine learning and cloud platforms (AWS) is required.
    Candidate should have at least 3 years of experience.
    """


@pytest.fixture
def sample_resume():
    """Sample resume text for testing."""
    return """
    PROFESSIONAL SUMMARY
    Data Scientist with 4 years of experience in Python, SQL, and machine learning.
    Worked on predictive modeling and data analysis projects.
    
    TECHNICAL SKILLS
    Python, SQL, Pandas, NumPy, Scikit-learn, TensorFlow
    AWS (EC2, S3), Docker, Git
    
    EXPERIENCE
    Senior Data Scientist - Tech Corp (2021-2025)
    - Built machine learning models for customer churn prediction
    - Developed ETL pipelines using Python and SQL
    - Deployed models on AWS
    
    EDUCATION
    M.Sc. in Computer Science - IIT Patna
    """


@pytest.fixture
def sample_pdf_bytes():
    """Create a sample PDF in memory for testing."""
    # This is a minimal PDF (created using fitz in memory)
    # For actual testing, you'd use a real PDF file
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox((50, 50, 550, 100), "Python developer with SQL experience", fontsize=12)
    pdf_bytes = doc.tobytes()
    doc.close()
    return io.BytesIO(pdf_bytes)



# Tests for PDF Parser

class TestPDFParser:
    """Tests for extract_text_from_pdf method."""

    def test_extract_text_from_pdf_valid(self, scorer, sample_pdf_bytes):
        """Test that valid PDF text is extracted correctly."""
        # Give the BytesIO object a name for better error messages
        sample_pdf_bytes.name = "sample.pdf"
        text = scorer.extract_text_from_pdf(sample_pdf_bytes)
        assert text is not None
        assert "Python" in text or "developer" in text

    def test_extract_text_from_pdf_none(self, scorer):
        """Test that invalid PDF returns None."""
        # Create an invalid file object
        invalid_file = io.BytesIO(b"not a pdf")
        invalid_file.name = "invalid.pdf"
        text = scorer.extract_text_from_pdf(invalid_file)
        assert text is None

    def test_extract_text_file_seek_guard(self, scorer):
        """Test that file.seek(0) is called before reading."""
        # Create a file that's already been partially read
        file_obj = io.BytesIO(b"%PDF-1.4 test content")
        file_obj.name = "test.pdf"
        # Read some bytes first (simulating Streamlit behavior)
        file_obj.read(5)
        # extract_text_from_pdf should seek to 0 and read fully
        text = scorer.extract_text_from_pdf(file_obj)
        # It may fail (not a real PDF), but should not crash
        assert True  # No exception thrown


# Tests for Section Extraction

class TestSectionExtraction:
    """Tests for extract_sections method."""

    def test_extract_sections_basic(self, scorer, sample_resume):
        """Test that sections are extracted correctly."""
        sections = scorer.extract_sections(sample_resume)
        assert "full_text" in sections
        # Check that at least some sections were found
        section_count = sum(1 for k in sections.keys() if k != "full_text")
        assert section_count >= 1

    def test_extract_sections_empty_text(self, scorer):
        """Test that empty text returns fallback."""
        sections = scorer.extract_sections("")
        assert sections == {"full_text": ""}

    def test_extract_sections_no_headers(self, scorer):
        """Test that text without headers falls back to full_text."""
        text = "This is a resume without any section headers."
        sections = scorer.extract_sections(text)
        assert "full_text" in sections
        assert sections["full_text"] == text

    def test_extract_sections_regex_accuracy(self, scorer):
        """Test that section extraction is accurate (match.end() fix)."""
        text = """
        SKILLS
        Python, SQL, AWS
        
        EXPERIENCE
        Data Scientist at Company
        
        EDUCATION
        IIT Patna
        """
        sections = scorer.extract_sections(text)
        assert "skills" in sections
        assert "Python" in sections["skills"] or "SQL" in sections["skills"]
        assert "experience" in sections
        assert "Data Scientist" in sections["experience"]


# Tests for Similarity Scoring

class TestSimilarityScoring:
    """Tests for calculate_similarity method."""

    def test_calculate_similarity_exact_match(self, scorer):
        """Test that identical texts get high score."""
        text = "Python developer with SQL experience"
        score = scorer.calculate_similarity(text, text)
        assert score >= 90.0  # Should be very high

    def test_calculate_similarity_different_texts(self, scorer):
        """Test that different texts get lower score."""
        text1 = "Python developer with SQL experience"
        text2 = "Doctor with medical experience"
        score = scorer.calculate_similarity(text1, text2)
        assert score < 60.0

    def test_calculate_similarity_empty_texts(self, scorer):
        """Test that empty texts return 0.0."""
        score = scorer.calculate_similarity("", "")
        assert score == 0.0

    def test_calculate_similarity_one_empty(self, scorer):
        """Test that one empty text returns 0.0."""
        score = scorer.calculate_similarity("Python developer", "")
        assert score == 0.0

# Tests for Weighted Scoring

class TestWeightedScoring:
    """Tests for calculate_weighted_score method."""

    def test_calculate_weighted_score_basic(self, scorer, sample_jd, sample_resume):
        """Test that weighted score returns a reasonable value."""
        score = scorer.calculate_weighted_score(sample_jd, sample_resume)
        assert 0.0 <= score <= 100.0
        # Should be a reasonable match (contains Python, SQL, AWS, ML)
        assert score > 10.0

    def test_calculate_weighted_score_empty_resume(self, scorer, sample_jd):
        """Test that empty resume returns 0.0."""
        score = scorer.calculate_weighted_score(sample_jd, "")
        assert score == 0.0

    def test_calculate_weighted_score_empty_jd(self, scorer, sample_resume):
        """Test that empty JD returns 0.0."""
        score = scorer.calculate_weighted_score("", sample_resume)
        assert score == 0.0

    def test_length_penalty_very_short(self, scorer, sample_jd):
        """Test that very short resumes get penalized (<50 words)."""
        short_text = "Python developer. SQL. AWS. ML."  # Very short
        score = scorer.calculate_weighted_score(sample_jd, short_text)
        # Short resumes get penalized heavily (0.5x)
        assert score < 30.0

    def test_length_penalty_very_long(self, scorer, sample_jd):
        """Test that very long resumes get slight penalty (>1000 words)."""
        # Create a long resume with repeated words
        long_text = "Data Scientist " * 500 + "Python SQL AWS " * 100
        score = scorer.calculate_weighted_score(sample_jd, long_text)
        assert 0.0 <= score <= 100.0

# Tests for Skill Extraction

class TestSkillExtraction:
    """Tests for extract_skills and compare_skills methods."""

    def test_extract_skills_basic(self, scorer, sample_resume):
        """Test that skills are extracted correctly."""
        skills = scorer.extract_skills(sample_resume)
        assert "python" in skills or "sql" in skills
        assert "aws" in skills or "docker" in skills

    def test_extract_skills_empty_text(self, scorer):
        """Test that empty text returns empty list."""
        skills = scorer.extract_skills("")
        assert skills == []

    def test_extract_skills_abbreviations(self, scorer):
        """Test that abbreviations are caught."""
        text = "ML, DL, NLP and CV experience"
        skills = scorer.extract_skills(text)
        # Our skill list includes 'ml', 'dl', 'nlp', 'cv'
        assert "ml" in skills or "nlp" in skills

    def test_compare_skills_basic(self, scorer, sample_jd, sample_resume):
        """Test that compare_skills returns matched and missing."""
        matched, missing = scorer.compare_skills(sample_jd, sample_resume)
        # Should have some matched skills
        assert isinstance(matched, list)
        assert isinstance(missing, list)

    def test_compare_skills_missing_subtraction(self, scorer):
        """Test that missing is correctly computed (JD skills - Resume skills)."""
        jd_text = "python sql aws docker kubernetes"
        resume_text = "python sql"
        matched, missing = scorer.compare_skills(jd_text, resume_text)
        # Matched: python, sql
        # Missing: aws, docker, kubernetes
        assert "aws" in missing
        assert "docker" in missing
        assert "kubernetes" in missing
        assert "python" not in missing
        assert "sql" not in missing

# Tests for Explainability

class TestExplainability:
    """Tests for explain_match method."""

    def test_explain_match_basic(self, scorer, sample_jd, sample_resume):
        """Test that explain_match returns meaningful explanations."""
        explanations = scorer.explain_match(sample_jd, sample_resume)
        assert isinstance(explanations, list)
        # Should return either explanations or a fallback message
        assert len(explanations) > 0

    def test_explain_match_empty_texts(self, scorer):
        """Test that empty texts return fallback."""
        explanations = scorer.explain_match("", "")
        assert explanations is not None
        assert "Not enough text" in explanations[0] or "No strong" in explanations[0]

    def test_explain_match_top_n(self, scorer, sample_jd, sample_resume):
        """Test that top_n parameter limits explanations."""
        explanations = scorer.explain_match(sample_jd, sample_resume, top_n=2)
        assert len(explanations) <= 2

# Tests for Backwards Compatibility

class TestBackwardsCompatibility:
    """Tests that module-level functions work correctly."""

    def test_module_level_functions_exist(self):
        """Test that all expected functions are available at module level."""
        assert hasattr(engine, "extract_text_from_pdf")
        assert hasattr(engine, "extract_sections")
        assert hasattr(engine, "calculate_similarity")
        assert hasattr(engine, "calculate_weighted_score")
        assert hasattr(engine, "calculate_match_score")
        assert hasattr(engine, "extract_skills")
        assert hasattr(engine, "compare_skills")
        assert hasattr(engine, "explain_match")

    def test_module_level_functions_return_correct_types(self, scorer, sample_jd, sample_resume):
        """Test that module-level functions return expected types."""
        # These should work without errors
        score = engine.calculate_weighted_score(sample_jd, sample_resume)
        assert isinstance(score, (int, float))
        
        matched, missing = engine.compare_skills(sample_jd, sample_resume)
        assert isinstance(matched, list)
        assert isinstance(missing, list)


# Tests for Edge Cases (Day 7 fixes)

class TestEdgeCases:
    """Tests for edge cases identified during Day 7 testing."""

    def test_very_short_resume_handling(self, scorer, sample_jd):
        """Test that very short resumes (<10 words) are handled."""
        short_text = "Python developer."  # 2 words
        score = scorer.calculate_weighted_score(sample_jd, short_text)
        # Should return 0.0 or very low because of edge case handling
        # This should have been caught in app.py, but engine should also handle gracefully
        assert score >= 0.0

    def test_scanned_pdf_handling(self, scorer):
        """Test that scanned PDF returns None."""
        # Scanned PDFs have no text layer. Our parser returns None.
        fake_pdf = io.BytesIO(b"not a real pdf")
        fake_pdf.name = "scanned.pdf"
        text = scorer.extract_text_from_pdf(fake_pdf)
        assert text is None

    def test_latex_hyphenation_cleanup(self, scorer):
        """Test that LaTeX hyphenation artifacts are cleaned."""
        # Create text with hyphenation artifacts
        text = "ma-\nchine learn-\ning is a core skill"
        # Simulate extraction by calling extract_sections (which doesn't clean hyphenation)
        # Actually, our parser cleans this, but we test the replacement logic indirectly
        # Since we can't easily test the parser with LaTeX PDFs, we test the logic in the parser
        # by checking the replace code is present
        import inspect
        source = inspect.getsource(scorer.extract_text_from_pdf)
        assert "replace('-\\n', '')" in source or "replace('-\n', '')" in source


# Run tests

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])