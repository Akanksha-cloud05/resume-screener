"""
analytics.py - Day 8 Analytics Module

Helper functions for:
- Score distribution charts
- Skill gap analysis charts
- Bias detection (Pearson correlation)
- Candidate comparison charts
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def create_score_distribution(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a histogram showing the distribution of match scores.
    Uses Plotly for interactive visualization.
    """
    scores = [r["score"] for r in results if not r["error"]]
    
    if not scores:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    fig = px.histogram(
        scores,
        nbins=10,
        title="Score Distribution",
        labels={"value": "Match Score (%)", "count": "Number of Candidates"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def create_skill_gap_chart(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a bar chart showing average matched vs missing skills.
    Helps identify common skill gaps across candidates.
    """
    valid_results = [r for r in results if not r["error"]]
    
    if not valid_results:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    matched_counts = [len(r.get("matched_skills", [])) for r in valid_results]
    missing_counts = [len(r.get("missing_skills", [])) for r in valid_results]
    
    avg_matched = sum(matched_counts) / len(matched_counts) if matched_counts else 0
    avg_missing = sum(missing_counts) / len(missing_counts) if missing_counts else 0
    
    fig = go.Figure(data=[
        go.Bar(name="Matched", x=["Skills"], y=[avg_matched], marker_color="#28a745"),
        go.Bar(name="Missing", x=["Skills"], y=[avg_missing], marker_color="#dc3545")
    ])
    fig.update_layout(
        title="Average Skills per Candidate",
        yaxis_title="Number of Skills",
        height=400,
        barmode="group"
    )
    return fig


def create_score_comparison(results: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a horizontal bar chart comparing all candidates.
    Color-coded by score threshold (Strong, Potential, Weak).
    """
    valid_results = [r for r in results if not r["error"]]
    valid_results.sort(key=lambda x: x["score"], reverse=True)
    
    if not valid_results:
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=400)
        return fig
    
    names = [r["filename"][:20] for r in valid_results]
    scores = [r["score"] for r in valid_results]
    
    colors = []
    for score in scores:
        if score >= 80:
            colors.append("#28a745")  # Green
        elif score >= 60:
            colors.append("#f0ad4e")  # Orange
        elif score >= 40:
            colors.append("#f39c12")  # Yellow
        else:
            colors.append("#dc3545")  # Red
    
    fig = go.Figure(data=[
        go.Bar(
            x=scores,
            y=names,
            orientation="h",
            marker_color=colors,
            text=[f"{s:.1f}%" for s in scores],
            textposition="outside"
        )
    ])
    fig.update_layout(
        title="Candidate Match Scores",
        xaxis_title="Match Score (%)",
        yaxis_title="Candidate",
        height=400,
        showlegend=False,
        xaxis=dict(range=[0, 100])
    )
    return fig


def calculate_bias_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate bias metrics using Pearson correlation.
    Checks if longer resumes score higher.
    
    Uses concepts from Statistics Week 7:
    - Pearson correlation coefficient
    - Measures linear relationship between word count and score
    """
    valid_results = [r for r in results if not r["error"]]
    
    if len(valid_results) < 3:
        return {"error": "Need at least 3 valid resumes for bias analysis."}
    
    word_counts = [r.get("word_count", 0) for r in valid_results]
    scores = [r["score"] for r in valid_results]
    
    # Calculate Pearson correlation
    correlation = np.corrcoef(word_counts, scores)[0, 1]
    
    # Interpretation
    if correlation > 0.4:
        interpretation = "Warning: Moderate positive correlation detected. Longer resumes tend to score higher. Consider reviewing."
    elif correlation > 0.2:
        interpretation = "Mild correlation detected. Some length bias may exist."
    else:
        interpretation = "No significant correlation detected. The system appears fair regarding resume length."
    
    return {
        "correlation": round(correlation, 3),
        "avg_length": round(sum(word_counts) / len(word_counts), 1),
        "std_score": round(np.std(scores), 2),
        "min_score": min(scores),
        "max_score": max(scores),
        "num_candidates": len(valid_results),
        "interpretation": interpretation
    }


def create_skill_radar(result: Dict[str, Any]) -> go.Figure:
    """
    Create a radar chart for a single candidate's skills.
    Shows distribution across skill categories.
    """
    if result.get("error", False):
        fig = go.Figure()
        fig.add_annotation(text="No data available", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Group skills by category (if available)
    categories = ["Programming", "ML/AI", "Cloud/DevOps", "Data", "Soft Skills"]
    
    # Simple categorization (basic implementation)
    matched_skills = result.get("matched_skills", [])
    
    cat_counts = [0] * len(categories)
    for skill in matched_skills:
        skill_lower = skill.lower()
        if skill_lower in ["python", "java", "c++", "javascript", "sql", "git"]:
            cat_counts[0] += 1
        elif skill_lower in ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch"]:
            cat_counts[1] += 1
        elif skill_lower in ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins"]:
            cat_counts[2] += 1
        elif skill_lower in ["pandas", "numpy", "statistics", "excel", "tableau"]:
            cat_counts[3] += 1
        else:
            cat_counts[4] += 1
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=cat_counts,
            theta=categories,
            fill="toself",
            marker_color="#1f77b4",
            name=result["filename"]
        )
    ])
    fig.update_layout(
        title="Skill Category Distribution",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(cat_counts) + 1] if max(cat_counts) > 0 else [0, 1]
            )
        ),
        height=400
    )
    return fig