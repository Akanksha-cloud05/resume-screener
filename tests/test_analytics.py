"""
test_analytics.py - Day 8 Analytics Tests
"""

import pytest

import analytics
from engine import ResumeProcessor


@pytest.fixture
def sample_results():
    """Sample results for testing analytics functions."""
    return [
        {
            "filename": "candidate1.pdf",
            "score": 85.5,
            "error": False,
            "word_count": 450,
            "matched_skills": ["python", "sql", "aws", "docker"],
            "missing_skills": ["kubernetes"]
        },
        {
            "filename": "candidate2.pdf",
            "score": 72.3,
            "error": False,
            "word_count": 380,
            "matched_skills": ["python", "sql"],
            "missing_skills": ["aws", "docker", "kubernetes"]
        },
        {
            "filename": "candidate3.pdf",
            "score": 45.0,
            "error": False,
            "word_count": 280,
            "matched_skills": ["python"],
            "missing_skills": ["sql", "aws", "docker", "kubernetes"]
        },
        {
            "filename": "candidate4.pdf",
            "score": 25.0,
            "error": False,
            "word_count": 120,
            "matched_skills": [],
            "missing_skills": ["python", "sql", "aws", "docker", "kubernetes"]
        }
    ]


class TestAnalyticsCharts:
    """Tests for chart generation functions."""

    def test_create_score_distribution(self, sample_results):
        """Test that score distribution chart is created."""
        fig = analytics.create_score_distribution(sample_results)
        assert fig is not None
        # Should return a Plotly Figure
        assert hasattr(fig, "update_layout")

    def test_create_score_distribution_empty(self):
        """Test that empty results return a figure."""
        fig = analytics.create_score_distribution([])
        assert fig is not None

    def test_create_skill_gap_chart(self, sample_results):
        """Test that skill gap chart is created."""
        fig = analytics.create_skill_gap_chart(sample_results)
        assert fig is not None

    def test_create_skill_gap_chart_empty(self):
        """Test that empty results return a figure."""
        fig = analytics.create_skill_gap_chart([])
        assert fig is not None

    def test_create_score_comparison(self, sample_results):
        """Test that score comparison chart is created."""
        fig = analytics.create_score_comparison(sample_results)
        assert fig is not None

    def test_create_score_comparison_empty(self):
        """Test that empty results return a figure."""
        fig = analytics.create_score_comparison([])
        assert fig is not None


class TestBiasMetrics:
    """Tests for bias detection functions."""

    def test_calculate_bias_metrics(self, sample_results):
        """Test that bias metrics are calculated correctly."""
        bias = analytics.calculate_bias_metrics(sample_results)
        assert "correlation" in bias
        assert "avg_length" in bias
        assert "interpretation" in bias
        assert isinstance(bias["correlation"], (int, float))

    def test_calculate_bias_metrics_insufficient_data(self, sample_results):
        """Test that insufficient data returns error."""
        results = [sample_results[0]]  # Only 1 result
        bias = analytics.calculate_bias_metrics(results)
        assert "error" in bias
        assert "at least 3 valid resumes" in bias["error"].lower()

    def test_calculate_bias_metrics_correlation_bounds(self, sample_results):
        """Test that correlation is between -1 and 1."""
        bias = analytics.calculate_bias_metrics(sample_results)
        assert -1.0 <= bias["correlation"] <= 1.0


class TestSkillRadar:
    """Tests for skill radar chart function."""

    def test_create_skill_radar(self, sample_results):
        """Test that skill radar chart is created."""
        fig = analytics.create_skill_radar(sample_results[0])
        assert fig is not None

    def test_create_skill_radar_error_result(self):
        """Test that error results return a figure."""
        error_result = {
            "filename": "error.pdf",
            "error": True,
            "matched_skills": []
        }
        fig = analytics.create_skill_radar(error_result)
        assert fig is not None