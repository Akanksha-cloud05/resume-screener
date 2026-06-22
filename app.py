# app.py

import streamlit as st
import pandas as pd
import importlib
import engine
importlib.reload(engine)  # Force reload the latest engine.py

print("=" * 50)
print("APP.PY VERSION 2 LOADED")
print("=" * 50)

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("AI Resume Screener")
st.caption("Internship Project | Text-based PDFs only")

with st.sidebar:
    st.header("Engine Status")
    if engine.AI_READY:
        st.success("BERT Active")
        st.caption("Semantic similarity using Sentence-BERT")
    else:
        st.info("TF-IDF Fallback")
        st.caption("Lightweight fallback — no GPU required")

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

# Main Area: Results
if run_button:
    if not jd_text:
        st.warning("Please paste a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
    else:
        with st.spinner("Analyzing resumes with AI..."):
            results = []

            for file in uploaded_files:
                resume_text = engine.extract_text_from_pdf(file)

                if resume_text is None:
                    results.append({
                        'Filename': file.name,
                        'Score': 0.0,
                        'Error': True,
                        'Matched': [],
                        'Missing': []
                    })
                    continue

                score = engine.calculate_match_score(jd_text, resume_text)
                
                # Safety: ensure score is a number
                if score is None or not isinstance(score, (int, float)):
                    score = 0.0

                matched, missing = engine.compare_skills(jd_text, resume_text)
                print(f"DEBUG APP: {file.name} | Matched={matched} | Missing={missing}")

                results.append({
                    'Filename': file.name,
                    'Score': score,
                    'Error': False,
                    'Matched': matched,
                    'Missing': missing
                })

            # Sort by score (highest first)
            results_sorted = sorted(results, key=lambda x: x['Score'], reverse=True)

            # Create DataFrame for display
            df = pd.DataFrame(results_sorted)

            # Remove error entries from display
            df_display = df[df['Error'] == False].copy()
            df_display['Matched'] = df_display['Matched'].apply(lambda x: ', '.join(x) if x else 'None')
            df_display['Missing'] = df_display['Missing'].apply(lambda x: ', '.join(x) if x else 'None')

            # Display metrics
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Candidates", len(df_display))
            col2.metric("Top Score", f"{df_display['Score'].max():.1f}%" if not df_display.empty else "N/A")
            col3.metric("Average Score", f"{df_display['Score'].mean():.1f}%" if not df_display.empty else "N/A")

                        # Rankings table
            st.subheader("Candidate Rankings")
            
            # Use st.table instead of st.dataframe (avoids caching bug)
            st.table(df_display[['Filename', 'Score', 'Matched', 'Missing']])

            # Skill Gap Profiles
            st.markdown("---")
            st.subheader("Skill Gap Profiles")

            for idx, row in df_display.iterrows():
                rank = idx + 1
                with st.expander(f"#{rank} {row['Filename']} ({row['Score']:.1f}%)"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**✅ Matched Skills**")
                        if row['Matched'] and row['Matched'] != 'None':
                            for skill in row['Matched'].split(', '):
                                st.markdown(f"🟢 {skill}")
                        else:
                            st.markdown("*No skills matched*")

                    with c2:
                        st.markdown("**❌ Missing Skills**")
                        if row['Missing'] and row['Missing'] != 'None':
                            for skill in row['Missing'].split(', '):
                                st.markdown(f"🔴 {skill}")
                        else:
                            st.markdown("*All skills matched!*")

            # Show errors if any
            errors = df[df['Error'] == True]
            if not errors.empty:
                st.markdown("---")
                st.warning("Some files could not be parsed:")
                for _, row in errors.iterrows():
                    st.error(f"❌ {row['Filename']} (scanned or corrupted PDF)")
