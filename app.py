"""
app.py — Streamlit Dashboard

I started this as a simple UI on Day 2. It's grown a lot since then.
The comments in this file are honest — I'm documenting what I built and why.

What I kept from earlier days:
  - Recruiter recommendation engine (from Day 5-6)
  - CSV export (HR asked for it on Day 5)
  - Progress bars and expanders (Day 5-6)
  - Plotly charts (from EDA Week 6)

What I added on Day 8:
  - Analytics dashboard with score distribution and skill gaps
  - Bias detection (Pearson correlation from Statistics Week 7)
  - Section radar chart
  - AI explainability panel
  - Streamlit caching for performance

I refactored the orchestration into engine.py on Day 8.
app.py now just calls processor.process() and displays results.
"""

import io
import os
import sys
import logging
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px

import engine
import analytics
from config import STREAMLIT_CONFIG

# Configure logging once to prevent duplicate handler attachment during Streamlit hot-reloads
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("resume_screener.log", encoding="utf-8")
        ]
    )
logger = logging.getLogger(__name__)

# Log dashboard initialization
logger.info("Initializing AI Resume Screener UI Server")
logger.info(f"Subsystems Status -> BERT Loaded: {engine.BERT_LOADED} | Skills Registry: Available")

# --- SYSTEM VALIDATION CHECKS ---
if not os.path.exists("assets/skills.json"):
    st.warning("⚠️ 'assets/skills.json' is missing — using minimal default skills.")

if not getattr(engine, 'BERT_LOADED', False):
    st.info("ℹ️ BERT models unavailable — engine is running in high-speed TF-IDF mode.")

# Page Config
st.set_page_config(
    page_title="AI Resume Screener",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS — clean, professional, no emojis
st.markdown(
    """
    <style>
    .score-high  { color: #28a745; font-weight: 600; font-size: 1.15rem; }
    .score-mid   { color: #f0ad4e; font-weight: 600; font-size: 1.15rem; }
    .score-warn  { color: #fd7e14; font-weight: 600; font-size: 1.15rem; }
    .score-low   { color: #dc3545; font-weight: 600; font-size: 1.15rem; }
    .skill-tag   { display: inline-block; padding: 2px 9px; margin: 2px 3px;
                   border-radius: 12px; font-size: 0.82rem; }
    .matched-tag { background: #d4edda; color: #155724; }
    .missing-tag { background: #f8d7da; color: #721c24; }
    .extra-tag   { background: #d1ecf1; color: #0c5460; }
    .info-box    { background: #f8f9fa; padding: 12px; border-radius: 6px; margin: 8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
st.title("AI Resume Screener")
st.caption(
    "Internship Project | "
    "Hybrid AI Engine: BERT + TF-IDF | Section-Weighted Scoring"
)

with st.expander("About this project"):
    st.markdown("""
    **AI Resume Screener** uses a hybrid AI engine to rank candidates.

    **How it works:**
    - PDF text extraction via PyMuPDF
    - Semantic similarity via Sentence-BERT (falls back to TF-IDF)
    - Section-weighted scoring: Skills 40%, Experience 35%, Education 15%, Summary 10%
    - Synonym-aware skill matching from skills.json
    - Bias detection: Pearson correlation between word count and score
    - Explainability: shows which sentences matched the JD
    - Analytics dashboard: score distribution, skill gaps, candidate comparison

    **Known limitations:**
    - No OCR — scanned PDFs return an error
    - Section extraction works on ~80% of standard templates
    - Creative layouts may miss section headers
    - BERT takes ~30s to load first time (cached afterwards)
    """)

# Caching — BERT is expensive to load
@st.cache_resource(show_spinner="Loading AI engine...")
def get_processor() -> engine.ResumeProcessor:
    """Cache the processor so BERT loads once."""
    return engine.ResumeProcessor()


@st.cache_data(show_spinner=False)
def process_all(
    file_tuples: Tuple[Tuple[str, bytes], ...],
    jd_text: str,
    use_weighted: bool,
) -> List[Dict[str, Any]]:
    """
    Process all resumes with caching.

    Accepts a tuple of (name, bytes) — hashable, so st.cache_data works.
    I tried passing a list first. It didn't work. Tuple fixed it.
    """
    processor = get_processor()
    results = []
    for name, file_bytes in file_tuples:
        file_obj = io.BytesIO(file_bytes)
        file_obj.name = name
        result = processor.process(file_obj, jd_text, use_weighted)
        results.append(result)
    return results


# Sidebar
with st.sidebar:
    st.header("Engine Status")
    if engine.BERT_LOADED:
        st.success("BERT Active")
        st.caption("Sentence-BERT | Hugging Face")
    else:
        st.info("TF-IDF Fallback")
        st.caption("Lightweight, no GPU required")

    st.markdown("---")
    st.header("Job Description")
    jd_text = st.text_area(
        "Paste the JD here:",
        height=260,
        placeholder="e.g., Looking for a Data Scientist with Python, SQL, and ML experience...",
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
        help="Display per-section similarity scores and radar chart.",
    )
    show_explanations = st.checkbox(
        "Show AI explainability",
        value=True,
        help="See which resume sentences matched the JD — builds HR trust.",
    )

    st.markdown("---")
    run_btn = st.button("Screen Candidates", type="primary", width='stretch')


# Main Processing
if run_btn:
    if not jd_text.strip():
        st.warning("Please paste a job description before screening.")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")
        st.stop()

    # Serialise to tuple for cache safety
    file_tuples: Tuple[Tuple[str, bytes], ...] = tuple(
        (f.name, f.getvalue()) for f in uploaded_files
    )

    # Initialize logger and progress
    logger.info(f"Starting pipeline for {len(uploaded_files)} resumes.")
    progress_bar = st.progress(0, text="Starting analysis...")
    
    with st.spinner("Analyzing resumes..."):
        processor = get_processor()
        results: List[Dict[str, Any]] = []
        
        for i, (name, file_bytes) in enumerate(file_tuples):
            # Update UI Progress
            progress_bar.progress(
                int((i + 1) / len(file_tuples) * 100),
                text=f"Processing {name} ({i + 1}/{len(file_tuples)})",
            )
            
            # Log individual file processing
            logger.info(f"Extracting and scoring: {name}")
            
            try:
                file_obj = io.BytesIO(file_bytes)
                file_obj.name = name
                
                # Execute core logic
                result = processor.process(file_obj, jd_text.strip(), use_weighted)
                results.append(result)
                
                logger.info(f"✅ Successfully processed {name}")
                
            except Exception as e:
                # Log error and continue to the next file
                logger.error(f"❌ Error processing {name}: {str(e)}")
                results.append({
                    "filename": name, 
                    "score": 0, 
                    "error": True, 
                    "error_msg": str(e)
                })
                
    progress_bar.empty()

    valid_results = sorted(
        [r for r in results if not r["error"]],
        key=lambda x: x["score"],
        reverse=True,
    )
    error_results = [r for r in results if r["error"]]

    # OVERVIEW METRICS
    st.markdown("---")
    st.subheader("Screening Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Uploaded", len(uploaded_files))
    c2.metric("Successfully Parsed", len(valid_results))
    c3.metric(
        "Top Score",
        f"{valid_results[0]['score']:.1f}%" if valid_results else "N/A",
    )
    c4.metric(
        "Average Score",
        f"{sum(r['score'] for r in valid_results) / len(valid_results):.1f}%"
        if valid_results
        else "N/A",
    )

    # ANALYTICS DASHBOARD (Day 8 — EDA Week 6)
    if len(valid_results) >= 2:
        st.markdown("---")
        st.subheader("Analytics Dashboard")

        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(
                analytics.create_score_distribution(valid_results),
                width='stretch',
                key="chart_dist",
            )
        with col_right:
            st.plotly_chart(
                analytics.create_skill_gap_chart(valid_results),
                width='stretch',
                key="chart_gap",
            )

        st.plotly_chart(
            analytics.create_score_comparison(valid_results),
            width='stretch',
            key="chart_rank",
        )

    # BIAS ANALYSIS (Statistics Week 7 — Pearson correlation)
    if len(valid_results) >= 3:
        st.markdown("---")
        st.subheader("Bias Analysis")
        bias = analytics.calculate_bias_metrics(valid_results)

        if "error" in bias:
            st.info(bias["error"])
        else:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Pearson Correlation", f"{bias['correlation']:.3f}")
            b2.metric("Avg Resume Length", f"{bias['avg_length']} words")
            b3.metric("Score IQR", f"{bias['score_iqr']:.1f}%")
            b4.metric(
                "Score Range",
                f"{bias['min_score']:.0f}% – {bias['max_score']:.0f}%",
            )
            st.info(bias["interpretation"])
            st.caption(
                "Pearson correlation (Statistics Week 7): checks if longer resumes "
                "unfairly score higher. Values near 0 indicate fair scoring."
            )

    # CANDIDATE RANKINGS
    st.markdown("---")
    st.subheader("Candidate Rankings")

    for rank, result in enumerate(valid_results, start=1):
        score = result["score"]
        filename = result["filename"]

        if score >= 80:
            score_class, label = "score-high", "Strong Match"
        elif score >= 60:
            score_class, label = "score-mid", "Potential Match"
        elif score >= 40:
            score_class, label = "score-warn", "Weak Match"
        else:
            score_class, label = "score-low", "Low Match"

        with st.container(border=True):
            col_name, col_score, col_label = st.columns([4, 1, 2])
            with col_name:
                st.markdown(f"**#{rank}** {filename}")
            with col_score:
                st.markdown(
                    f'<span class="{score_class}">{score:.1f}%</span>',
                    unsafe_allow_html=True,
                )
            with col_label:
                st.caption(label)

            st.progress(score / 100)

            with st.expander("View Details"):
                # Recommendation at the top
                st.info(f"**Recruiter Action:** {result['recommendation']}")
                st.markdown("---")

                # Skills
                skill_col1, skill_col2 = st.columns(2)
                with skill_col1:
                    st.markdown("**Matched Skills**")
                    if result["matched_skills"]:
                        tags = " ".join(
                            f'<span class="skill-tag matched-tag">{s}</span>'
                            for s in result["matched_skills"]
                        )
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.markdown("*None matched*")

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**Extra Skills**")
                    if result["extra_skills"]:
                        tags = " ".join(
                            f'<span class="skill-tag extra-tag">{s}</span>'
                            for s in result["extra_skills"][:8]
                        )
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.markdown("*None*")

                with skill_col2:
                    st.markdown("**Missing Skills**")
                    if result["missing_skills"]:
                        tags = " ".join(
                            f'<span class="skill-tag missing-tag">{s}</span>'
                            for s in result["missing_skills"]
                        )
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.markdown("*All required skills present*")

                # Section breakdown (Day 8 toggle)
                if show_sections and result.get("section_scores"):
                    st.markdown("---")
                    st.markdown("**Section Score Breakdown**")
                    for sec_name, sec_score in result["section_scores"].items():
                        weight = {
                            "skills": 40, "experience": 35,
                            "education": 15, "summary": 10
                        }.get(sec_name, 0)
                        st.markdown(
                            f"- **{sec_name.title()}** (weight {weight}%): {sec_score:.1f}%"
                        )

                    if len(result["section_scores"]) >= 2:
                        st.plotly_chart(
                            analytics.create_section_breakdown(result["section_scores"]),
                            width='stretch',
                            key=f"radar_{rank}",
                        )

                # Explainability (Day 8 trust feature)
                if show_explanations and result.get("explanations"):
                    st.markdown("---")
                    st.markdown("**Why this score? (AI Explainability)**")
                    st.caption("Resume sentences that semantically matched the JD:")
                    for exp in result["explanations"]:
                        st.markdown(f"- {exp}")

    # CSV EXPORT
    if valid_results:
        st.markdown("---")
        st.subheader("Export Results")

        export_rows = []
        for rank, r in enumerate(valid_results, start=1):
            export_rows.append(
                {
                    "Rank": rank,
                    "Candidate": r["filename"],
                    "Score (%)": r["score"],
                    "Match Level": (
                        "Strong" if r["score"] >= 80
                        else "Potential" if r["score"] >= 60
                        else "Weak" if r["score"] >= 40
                        else "Low"
                    ),
                    "Recruiter Action": r["recommendation"],
                    "Matched Skills": ", ".join(r["matched_skills"]) or "None",
                    "Missing Skills": ", ".join(r["missing_skills"]) or "None",
                    "Extra Skills": ", ".join(r["extra_skills"]) or "None",
                    "Word Count": r.get("word_count", "N/A"),
                }
            )

        df_export = pd.DataFrame(export_rows)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_bytes,
            file_name="resume_screening_results.csv",
            mime="text/csv",
            width='stretch',
        )

    # PARSE ERRORS
    if error_results:
        st.markdown("---")
        st.warning(f"{len(error_results)} file(s) could not be processed:")
        for err in error_results:
            st.error(f"{err['filename']} — {err['error_msg']}")