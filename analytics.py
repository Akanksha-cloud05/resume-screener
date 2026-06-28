"""
analytics.py — Analytics and Visualization Module
Internship Project

This file contains all the charting and analytics functions used by the dashboard.
I kept these separate from app.py because it was getting too long to scroll through.

What's here:
  - Score distribution histogram (EDA Week 6)
  - Skill gap bar chart (top missing skills)
  - Candidate comparison chart (ranked bar chart)
  - Bias detection (Pearson correlation — Statistics Week 7)
  - Section breakdown radar chart (per candidate)

The functions are pure — no Streamlit calls inside. That makes them easy to test
and reuse if I ever want to generate reports outside the app.
"""

from typing import Any, Dict, List

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Score Distribution (Histogram)
def create_score_distribution(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Histogram showing how match scores are distributed across candidates.
    Adds a vertical line at the mean so you can see who's above/below average.

    I made this after my mentor said "I want to see where the scores cluster."
    It helps HR understand if the JD is too strict or too loose.
    """
    scores = [r["score"] for r in results if not r.get("error", False)]

    if not scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid results to display",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper",
        )
        fig.update_layout(height=350, title="Score Distribution")
        return fig

    mean_score = float(np.mean(scores))

    fig = px.histogram(
        x=scores,
        nbins=10,
        title="Score Distribution",
        labels={"x": "Match Score (%)", "y": "Candidates"},
        color_discrete_sequence=["#3366cc"],
        template="plotly_white",
    )
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_score:.1f}%",
        annotation_position="top right",
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


# Skill Gap Chart (Top Missing Skills)
def create_skill_gap_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Bar chart showing the top 10 skills missing across all candidates.

    This is more actionable than showing averages. HR can immediately see
    what skills are hard to find in the candidate pool and adjust expectations.

    I originally showed average matched vs missing, but my mentor asked:
    "Which specific skills are candidates missing?" — so I built this.
    """
    missing_counts: Dict[str, int] = {}
    for r in results:
        if not r.get("error", False):
            for skill in r.get("missing_skills", []):
                missing_counts[skill] = missing_counts.get(skill, 0) + 1

    if not missing_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No skill gap data available",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper",
        )
        fig.update_layout(height=350, title="Top Missing Skills")
        return fig

    sorted_skills = sorted(missing_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    skill_names = [s[0] for s in sorted_skills]
    skill_counts = [s[1] for s in sorted_skills]

    fig = px.bar(
        x=skill_names,
        y=skill_counts,
        title="Top Missing Skills Across Candidates",
        labels={"x": "Skill", "y": "Candidates Missing This Skill"},
        color=skill_counts,
        color_continuous_scale="Reds",
        template="plotly_white",
    )
    fig.update_layout(
        height=350,
        showlegend=False,
        xaxis_tickangle=-35,
        coloraxis_showscale=False,
    )
    return fig


# Candidate Score Comparison (Ranked Bar Chart)

def create_score_comparison(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Horizontal bar chart comparing all candidates, sorted by score.
    Color-coded: green (strong), orange (potential), red (low).

    This is the main ranking visualisation. HR can quickly see who's on top.
    """
    valid = [r for r in results if not r.get("error", False)]
    valid.sort(key=lambda x: x["score"], reverse=True)

    if not valid:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid candidates to compare",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper",
        )
        fig.update_layout(height=350, title="Candidate Rankings")
        return fig

    # Shorten long filenames so they don't break the layout
    names = [
        (r["filename"][:28] + "...") if len(r["filename"]) > 30 else r["filename"]
        for r in valid
    ]
    scores = [r["score"] for r in valid]

    colors = []
    for s in scores:
        if s >= 80:
            colors.append("#28a745")  # Green
        elif s >= 60:
            colors.append("#f0ad4e")  # Orange
        elif s >= 40:
            colors.append("#fd7e14")  # Yellow-orange
        else:
            colors.append("#dc3545")  # Red

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=names,
                orientation="h",
                marker_color=colors,
                text=[f"{s:.1f}%" for s in scores],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Candidate Rankings by Match Score",
        xaxis_title="Match Score (%)",
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(valid) * 50),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 108]),
        showlegend=False,
    )
    return fig

# Bias Detection (Pearson Correlation — Statistics Week 7)
def calculate_bias_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if longer resumes unfairly score higher (length bias).

    Method: Pearson correlation between word count and match score.
    Source: Statistics Week 7 — measures of correlation, linear relationships.

    Why this matters: A fair screening system should score based on content
    quality and relevance, not resume length. If correlation > 0.5, the tool
    may be rewarding verbose candidates, not skilled ones.

    I added this after noticing a 1900-word resume ranking #2 unfairly.
    The correlation check helps quantify whether the length penalty is working.
    """
    valid = [r for r in results if not r.get("error", False)]

    if len(valid) < 3:
        return {"error": "Need at least 3 valid resumes for bias analysis."}

    lengths = np.array([r.get("word_count", 0) for r in valid], dtype=float)
    scores = np.array([r["score"] for r in valid], dtype=float)

    # Handle edge case: no variance in lengths or scores
    if np.std(lengths) == 0 or np.std(scores) == 0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(lengths, scores)[0, 1])

    # IQR analysis (Statistics Week 7 — measures of dispersion)
    score_q1 = float(np.percentile(scores, 25))
    score_q3 = float(np.percentile(scores, 75))
    iqr = score_q3 - score_q1

    if correlation > 0.5:
        interpretation = (
            "Strong positive correlation detected. "
            "Longer resumes tend to score higher. Length bias likely — review scoring weights."
        )
    elif correlation > 0.3:
        interpretation = (
            "Moderate correlation. Some length sensitivity exists. "
            "Consider reviewing results for very long resumes."
        )
    elif correlation < -0.3:
        interpretation = (
            "Negative correlation: shorter resumes score higher. "
            "Unusual — check if very concise resumes are being over-rewarded."
        )
    else:
        interpretation = (
            "Weak correlation. Scoring appears fair with respect to resume length."
        )

    return {
        "correlation": round(correlation, 3),
        "interpretation": interpretation,
        "avg_length": round(float(np.mean(lengths)), 1),
        "median_length": round(float(np.median(lengths)), 1),
        "std_length": round(float(np.std(lengths)), 1),
        "avg_score": round(float(np.mean(scores)), 1),
        "median_score": round(float(np.median(scores)), 1),
        "std_score": round(float(np.std(scores)), 1),
        "score_iqr": round(iqr, 1),
        "score_q1": round(score_q1, 1),
        "score_q3": round(score_q3, 1),
        "min_score": round(float(np.min(scores)), 1),
        "max_score": round(float(np.max(scores)), 1),
        "n_candidates": len(valid),
    }


# Section Breakdown Radar Chart (Per Candidate)
def create_section_breakdown(section_scores: Dict[str, float]) -> go.Figure:
    """
    Radar chart showing per-section similarity scores for a single candidate.
    
    This lets HR see at a glance:
    - Strong in skills (75%+) but weak in experience (40%)
    - Well-rounded vs. lopsided candidates
    
    I built this after my mentor said "show me where they're strong and weak."
    The radar chart is perfect for that — it shows the shape of the candidate.
    """
    if not section_scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No section data available",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper",
        )
        fig.update_layout(height=300)
        return fig

    # Get scores and labels
    labels = list(section_scores.keys())
    values = list(section_scores.values())
    
    # Close the radar loop (so the line connects back to start)
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,                        # SCORES (0-100)
                theta=labels,                    # SECTION NAMES
                fill="toself",
                fillcolor="rgba(51, 102, 204, 0.2)",
                line=dict(color="#3366cc"),
                name="Section Scores",
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix="%",
            )
        ),
        showlegend=False,
        height=300,
        title="Section Score Breakdown",
        template="plotly_white",
    )
    return fig

# Skill Radar Chart (for test compatibility)
def create_skill_radar(result: Dict[str, Any]) -> go.Figure:
    """
    Dummy function to keep old unit tests passing.
    In the current app, section breakdown is shown via create_section_breakdown.
    This function is not used in production.
    """
    if result.get("error", False):
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            x=0.5, y=0.5, showarrow=False,
            xref="paper", yref="paper",
        )
        fig.update_layout(height=300)
        return fig

    # Simple radar using matched skills if available
    matched = result.get("matched_skills", [])
    categories = ["Python", "SQL", "ML", "Cloud", "Soft"]
    values = [0] * len(categories)
    for skill in matched:
        skill_lower = skill.lower()
        if "python" in skill_lower:
            values[0] += 1
        elif "sql" in skill_lower:
            values[1] += 1
        elif "machine" in skill_lower or "ml" in skill_lower:
            values[2] += 1
        elif "aws" in skill_lower or "azure" in skill_lower or "gcp" in skill_lower:
            values[3] += 1
        else:
            values[4] += 1

    # Normalize to 0-100 (arbitrary scale for demo)
    max_val = max(values) if max(values) > 0 else 1
    values = [v / max_val * 100 for v in values]

    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                fillcolor="rgba(51, 102, 204, 0.2)",
                line=dict(color="#3366cc"),
            )
        ]
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100])
        ),
        showlegend=False,
        height=300,
        title="Skill Radar (Test Compatibility)",
        template="plotly_white",
    )
    return fig