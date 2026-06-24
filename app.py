# app.py

import streamlit as st
import pandas as pd
import importlib

# NOTE: I tried matplotlib first (from Week 6) but it looked terrible in Streamlit
# Switched to Plotly after 30 minutes of frustration. Much better.

# Import backend and force reload for iterative development
# This means I don't have to restart Streamlit every time I change engine.py
# Learned this trick after restarting the app 50 times in one day...
import engine
importlib.reload(engine)  # Force reload the latest engine.py
import plotly.express as px

# print("=" * 50)
# print("APP.PY VERSION 2 LOADED")
# print("=" * 50)

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("AI Resume Screener")
st.caption("Internship Project | Text-based PDFs only")

# RECRUITER RECOMMENDATION
def generate_recommendation(score, matched_count, missing_count):
    """
    Generate a human-readable recommendation for the recruiter.
    """
    # Thresholds I settled on after manually reviewing 20 resumes:
    # First version just used score thresholds. HR mentor said "what about skill gaps?"
    # So I added missing_count logic. Second version was much better.
    if score >= 80:
        rec = "Strong Match - Interview Recommended"
    elif score >= 60 and missing_count <= 2:
        rec = "Good Match - Consider Screening"
    elif score >= 60 and missing_count > 2:
        rec = "Potential Match - Needs Review"
    elif score >= 40:
        rec = "Weak Match - Keep as Backup"
    else:
        rec = "Poor Match - Not Recommended"

    # I found that recruiters care most about what's missing
    # So I always highlight the skill gap count
    if missing_count > 0:
        rec += f" (Missing {missing_count} key skills)"
    
    return rec

# SIDEBAR: USER INPUTS
with st.sidebar:
    st.header("Engine Status")
    if engine.AI_READY:
        st.success("BERT Active")
        st.caption("Semantic similarity using Sentence-BERT")
    else:
        st.info("TF-IDF Fallback")
        st.caption("Lightweight fallback - no GPU required")

    st.markdown("---")
    st.header("Job Description")
    jd_text = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="e.g., Looking for Data Scientist with Python, SQL, and AWS..."
    )

    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    st.markdown("---")
    run_button = st.button("Screen Candidates", type="primary", use_container_width=True)

# MAIN
if run_button:
    if not jd_text:
        st.warning("Please paste a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
    else:
        with st.spinner("Analyzing resumes with AI..."):
            results = []

            for file in uploaded_files:
                # Reset file pointer before reading
                # Took me an hour to figure out why only the first PDF worked...
                # Streamlit reads files once and doesn't rewind automatically
                file.seek(0)
                resume_text = engine.extract_text_from_pdf(file)
                
                # EDGE CASE HANDLING (Learned these the hard way)

                # 1. Scanned or image-based PDFs
                if resume_text is None:
                    results.append({
                        'Filename': file.name,
                        'Score': 0.0,
                        'Error': True,
                        'ErrorMsg': 'Cannot parse PDF (scanned or image-based)',
                        'Matched': [],
                        'Missing': [],
                        'MatchedCount': 0,
                        'MissingCount': 0,
                        'WordCount': 0
                    })
                    continue

                # 2. Empty PDFs 
                if len(resume_text.strip()) == 0:
                    results.append({
                        'Filename': file.name,
                        'Score': 0.0,
                        'Error': True,
                        'ErrorMsg': 'PDF is empty',
                        'Matched': [],
                        'Missing': [],
                        'MatchedCount': 0,
                        'MissingCount': 0,
                        'WordCount': 0
                    })
                    continue

                # 3. Extremely short resumes (< 10 words)
                # Found a test resume with just "Python developer" and it was
                # scoring 60% because of vocabulary overlap...
                # So I added this check to catch those cases
                
                word_count = len(resume_text.split())
                if word_count < 10:
                    results.append({
                        'Filename': file.name,
                        'Score': 0.0,
                        'Error': True,
                        'ErrorMsg': f'Resume too short ({word_count} words)',
                        'Matched': [],
                        'Missing': [],
                        'MatchedCount': 0,
                        'MissingCount': 0,
                        'WordCount': word_count
                    })
                    continue

                # CORE SCORING

                score = engine.calculate_match_score(jd_text, resume_text)
                # Just in case the backend returns None (shouldn't happen anymore)
                if score is None or not isinstance(score, (int, float)):
                    score = 0.0

                matched, missing = engine.compare_skills(jd_text, resume_text)

                results.append({
                    'Filename': file.name,
                    'Score': score,
                    'Error': False,
                    'ErrorMsg': '',
                    'Matched': matched,
                    'Missing': missing,
                    'MatchedCount': len(matched),
                    'MissingCount': len(missing),
                    'WordCount': word_count
                })
      
            # RESULTS 
            # Sort by score, highest first (the ones the recruiter should look at)
            results_sorted = sorted(results, key=lambda x: x['Score'], reverse=True)
            df = pd.DataFrame(results_sorted)
            # Separate the ones that worked from the ones that didn't
            df_valid = df[df['Error'] == False].copy()
            df_errors = df[df['Error'] == True].copy()
             # Convert skill lists to readable strings
            df_valid['Matched'] = df_valid['Matched'].apply(lambda x: ', '.join(x) if x else 'None')
            df_valid['Missing'] = df_valid['Missing'].apply(lambda x: ', '.join(x) if x else 'None')

            # Generate recruiter recommendations
            df_valid['Recommendation'] = df_valid.apply(
                lambda row: generate_recommendation(
                    row['Score'], 
                    row.get('MatchedCount', 0), 
                    row.get('MissingCount', 0)
                ),
                axis=1
            )
            # DASHBOARD

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Candidates", len(df))
            col2.metric("Successfully Parsed", len(df_valid))
            col3.metric("Failed", len(df_errors))

            if not df_valid.empty:
                col1, col2, col3 = st.columns(3)
                col1.metric("Top Score", f"{df_valid['Score'].max():.1f}%")
                col2.metric("Average Score", f"{df_valid['Score'].mean():.1f}%")
                col3.metric("Lowest Score", f"{df_valid['Score'].min():.1f}%")
            
            # RANKINGS TABLE

            st.markdown("---")
            st.subheader("Candidate Rankings with Recruiter Action")

            st.dataframe(
                df_valid[['Filename', 'Score', 'Matched', 'Missing', 'Recommendation']],
                use_container_width=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Match Score",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                    "Recommendation": "Recruiter Action"
                },
                hide_index=True
            )
             
            # CSV Export (for HR teams)
            st.markdown("---")
            st.subheader("Export Results")

            csv_data = df_valid[['Filename', 'Score', 'Matched', 'Missing', 'Recommendation']].copy()
            csv_data.columns = ['Candidate', 'Match Score (%)', 'Matched Skills', 'Missing Skills', 'Recruiter Action']

            csv = csv_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="resume_ranking_results.csv",
                mime="text/csv"
            )

            # Plotly bar chart

            if not df_valid.empty:
                st.markdown("---")
                st.subheader("Score Visualization")
                
                fig = px.bar(
                    df_valid,
                    x='Filename',
                    y='Score',
                    color='Score',
                    color_continuous_scale='Viridis',
                    title="Match Scores by Candidate",
                    labels={'Score': 'Match Score (%)', 'Filename': 'Candidate'}
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
           
            # Skill Gap Profiles

            st.markdown("---")
            st.subheader("Skill Gap Profiles")

            for idx, row in df_valid.iterrows():
                rank = idx + 1
                with st.expander(f"#{rank} {row['Filename']} ({row['Score']:.1f}%)"):
                    st.info(f"Recruiter Action: {row['Recommendation']}")
                    st.markdown("---")
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("Matched Skills")
                        if row['Matched'] and row['Matched'] != 'None':
                            for skill in row['Matched'].split(', '):
                                st.markdown(f"{skill}")
                        else:
                            st.markdown("*No skills matched*")

                    with c2:
                        st.markdown("Missing Skills")
                        if row['Missing'] and row['Missing'] != 'None':
                            for skill in row['Missing'].split(', '):
                                st.markdown(f"{skill}")
                        else:
                            st.markdown("*All skills matched!*")
            
            # Show errors if any

            if not df_errors.empty:
                st.markdown("---")
                st.warning(f"{len(df_errors)} file(s) could not be processed:")
                for _, row in df_errors.iterrows():
                    st.error(f"{row['Filename']}: {row['ErrorMsg']}")
