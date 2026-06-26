# app.py
"""
Day 8 Elite: Streamlit Dashboard with Caching and Analytics

Features preserved from Day 7:
    - Recruiter recommendation engine
    - Edge-case handling (scanned, empty, too-short PDFs)
    - CSV export for HR teams
    - Progress bars, expanders, skill tags
    - Plotly bar chart visualization

Features added Day 8:
    - Streamlit caching (st.cache_resource for processor, st.cache_data for results)
    - Class-based backend architecture
    - Analytics dashboard (distribution, skill gaps, bias detection)
    - AI explainability (sentence-level matching)
    - Section-weighted vs simple toggle
    - Professional CSS styling and metrics cards
"""

import streamlit as st
import pandas as pd
import io
from typing import List, Dict, Any, Tuple

import engine
import analytics

# Page Config

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS

st.markdown(
    """
    <style>
    .main-header { font-size: 2.2rem; font-weight: bold; color: #1f1f1f; }
    .sub-header { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
    .score-high { color: #28a745; font-weight: bold; font-size: 1.2rem; }
    .score-mid { color: #f0ad4e; font-weight: bold; font-size: 1.2rem; }
    .score-low { color: #dc3545; font-weight: bold; font-size: 1.2rem; }
    .skill-tag { display: inline-block; padding: 2px 8px; margin: 2px; border-radius: 12px; font-size: 0.85rem; }
    .matched-tag { background-color: #d4edda; color: #155724; }
    .missing-tag { background-color: #f8d7da; color: #721c24; }
    .extra-tag { background-color: #d1ecf1; color: #0c5460; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">AI Resume Screener</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">IIT Patna Internship Project | Hybrid AI Engine (BERT + TF-IDF) | Section-Weighted Scoring</p>',
    unsafe_allow_html=True,
)

with st.expander("About this project"):
    st.markdown("""
    **AI Resume Screener** uses a hybrid AI engine to compare resumes against job descriptions.

    **Key Features:**
    - Semantic Similarity: Sentence-BERT understands synonyms and context.
    - Section-Weighted Scoring: Skills (40%), Experience (35%), Education (15%), Summary (10%).
    - Skill Gap Analysis: Synonym-aware extraction with technical and soft skills.
    - Bias Detection: Pearson correlation checks if longer resumes unfairly score higher.
    - AI Explainability: Shows which resume sentences matched the JD, building HR trust.
    - Edge-Case Hardened: Handles scanned PDFs, empty uploads, and extremely short resumes.

    **Limitations:** Works only on text-based PDFs (no OCR). Creative layouts may miss section headers.
    """)

# Sidebar: Inputs and Settings

with st.sidebar:
    st.header("Engine Status")
    if engine.BERT_LOADED:
        st.success("BERT Active")
        st.caption("Sentence-BERT | Hugging Face Transformers")
    else:
        st.info("TF-IDF Fallback")
        st.caption("BERT failed to load. Using TF-IDF.")

    st.markdown("---")
    st.header("Job Description")
    jd_text = st.text_area(
        "Paste the job description here:",
        height=280,
        placeholder="e.g., Looking for a Data Scientist with Python, SQL, and AWS experience...",
    )

    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    st.markdown("---")
    st.header("Settings")
    use_weighted = st.checkbox(
        "Section-weighted scoring",
        value=True,
        help="Skills 40%, Experience 35%, Education 15%, Summary 10%",
    )
    show_sections = st.checkbox(
        "Show section breakdown",
        value=False,
        help="Display per-section similarity scores in candidate details.",
    )
    show_explanations = st.checkbox(
        "Show AI explainability",
        value=True,
        help="See which resume sentences matched the JD (trust feature).",
    )

    st.markdown("---")
    run_btn = st.button(
        "Screen Candidates",
        type="primary",
        use_container_width=True,
    )

# Caching Layer

@st.cache_resource
def get_processor():
    """Cache the ResumeProcessor (and its lazy-loaded BERT model) across sessions."""
    return engine.ResumeProcessor()


@st.cache_data(show_spinner=False)
def _process_all(uploaded_files_data: List[Tuple[str, bytes]], jd_text: str, use_weighted: bool) -> List[Dict[str, Any]]:
    """
    Process all resumes with Streamlit caching.
    Accepts serialized (name, bytes) tuples to ensure hashability.
    """
    processor = get_processor()
    results = []
    for name, file_bytes in uploaded_files_data:
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = name
        result = processor.process(file_obj, jd_text, use_weighted)
        results.append(result)
    return results


# Main Processing

if run_btn:
    if not jd_text.strip():
        st.warning("Please paste a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        # Serialize files for caching (Streamlit needs hashable inputs)
        uploaded_files_data = [(f.name, f.getvalue()) for f in uploaded_files]

        with st.spinner("Analyzing resumes with AI..."):
            progress_bar = st.progress(0)
            results = _process_all(uploaded_files_data, jd_text.strip(), use_weighted)
            progress_bar.empty()

        valid_results = [r for r in results if not r["error"]]
        error_results = [r for r in results if r["error"]]
        valid_results.sort(key=lambda x: x["score"], reverse=True)

        # DASHBOARD METRICS
    
        st.markdown("---")
        st.subheader("Screening Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Candidates", len(uploaded_files))
        c2.metric("Successfully Parsed", len(valid_results))
        c3.metric(
            "Top Score",
            f"{valid_results[0]['score']:.1f}%" if valid_results else "N/A",
        )
        c4.metric(
            "Average Score",
            f"{sum(r['score'] for r in valid_results)/len(valid_results):.1f}%"
            if valid_results
            else "N/A",
        )

        # ANALYTICS CHARTS (Day 8)
        
        if len(valid_results) >= 2:
            st.markdown("---")
            st.subheader("Analytics Dashboard")
            v1, v2 = st.columns(2)
            with v1:
                st.plotly_chart(
                    analytics.create_score_distribution(valid_results),
                    use_container_width=True,
                    key="dist",
                )
            with v2:
                st.plotly_chart(
                    analytics.create_skill_gap_chart(valid_results),
                    use_container_width=True,
                    key="gap",
                )

            st.plotly_chart(
                analytics.create_score_comparison(valid_results),
                use_container_width=True,
                key="rank",
            )

        # BIAS ANALYSIS (Day 8 - Statistics Week 7)

        if len(valid_results) >= 3:
            st.markdown("---")
            st.subheader("Bias Analysis")
            bias = analytics.calculate_bias_metrics(valid_results)

            if "error" in bias:
                st.info(bias["error"])
            else:
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Correlation", f"{bias['correlation']}")
                b2.metric("Avg Length", f"{bias['avg_length']} words")
                b3.metric("Score Std Dev", f"{bias['std_score']}")
                b4.metric(
                    "Score Range",
                    f"{bias['min_score']:.0f}% - {bias['max_score']:.0f}%",
                )
                st.info(bias["interpretation"])
                st.caption(
                    "Pearson correlation from Statistics Week 7 - checks if longer resumes unfairly score higher"
                )


        # CANDIDATE RANKINGS (Day 7 features preserved)

        st.markdown("---")
        st.subheader("Candidate Rankings")

        for idx, result in enumerate(valid_results):
            rank = idx + 1
            score = result["score"]
            filename = result["filename"]
            recommendation = result["recommendation"]

            if score >= 80:
                match_class, match_label = "score-high", "Strong Match"
            elif score >= 60:
                match_class, match_label = "score-mid", "Potential Match"
            elif score >= 40:
                match_class, match_label = "score-low", "Weak Match"
            else:
                match_class, match_label = "score-low", "Low Match"

            with st.container(border=True):
                h1, h2, h3 = st.columns([3, 1, 1])
                with h1:
                    st.markdown(f"**#{rank}** {filename}")
                with h2:
                    st.markdown(
                        f"<span class='{match_class}'>{score:.1f}%</span>",
                        unsafe_allow_html=True,
                    )
                with h3:
                    st.caption(match_label)

                st.progress(score / 100, text=f"{match_label} ({score:.1f}%)")

                with st.expander("View Details"):
                    st.info(f"Recruiter Action: {recommendation}")
                    st.markdown("---")

                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Matched Skills**")
                        if result["matched_skills"]:
                            for skill in result["matched_skills"]:
                                st.markdown(
                                    f'<span class="skill-tag matched-tag">{skill}</span>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown("*None*")

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("**Extra Skills**")
                        if result["extra_skills"]:
                            for skill in result["extra_skills"][:8]:
                                st.markdown(
                                    f'<span class="skill-tag extra-tag">{skill}</span>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown("*None*")

                    with d2:
                        st.markdown("**Missing Skills**")
                        if result["missing_skills"]:
                            for skill in result["missing_skills"]:
                                st.markdown(
                                    f'<span class="skill-tag missing-tag">{skill}</span>',
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.markdown("*All skills matched*")

                    # Section breakdown (Day 8 toggle)
                    if show_sections and result.get("sections"):
                        st.markdown("---")
                        st.markdown("**Section Breakdown**")
                        scorer = engine.ScoringEngine()
                        for sec_name in ["skills", "experience", "education", "summary"]:
                            if sec_name in result["sections"] and result["sections"][sec_name]:
                                sec_score = scorer.calculate_semantic_score(
                                    jd_text, result["sections"][sec_name]
                                )
                                st.markdown(f"- **{sec_name.title()}:** {sec_score:.1f}%")

                    # Explainability (Day 8)
                    if show_explanations and result.get("explanations"):
                        st.markdown("---")
                        st.markdown("Why this score? (AI Explainability)")
                        for exp in result["explanations"]:
                            st.markdown(f"- {exp}")

        # CSV EXPORT (Day 7 feature preserved)
    
        if valid_results:
            st.markdown("---")
            st.subheader("Export Results")
            export_data = []
            for r in valid_results:
                export_data.append(
                    {
                        "Rank": valid_results.index(r) + 1,
                        "Candidate": r["filename"],
                        "Score (%)": r["score"],
                        "Match Level": (
                            "Strong"
                            if r["score"] >= 80
                            else "Potential"
                            if r["score"] >= 60
                            else "Weak"
                            if r["score"] >= 40
                            else "Low"
                        ),
                        "Recommendation": r["recommendation"],
                        "Matched Skills": ", ".join(r["matched_skills"]) if r["matched_skills"] else "None",
                        "Missing Skills": ", ".join(r["missing_skills"]) if r["missing_skills"] else "None",
                        "Extra Skills": ", ".join(r["extra_skills"]) if r["extra_skills"] else "None",
                        "Word Count": r.get("word_count", "N/A"),
                    }
                )

            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                csv,
                "resume_screening_results.csv",
                "text/csv",
                use_container_width=True,
            )

        # ERRORS (Day 7 feature preserved)
    
        if error_results:
            st.markdown("---")
            st.warning(f"{len(error_results)} file(s) could not be processed:")
            for err in error_results:
                st.error(f"{err['filename']} - {err['error_msg']}")