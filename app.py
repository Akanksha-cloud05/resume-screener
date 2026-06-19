# Resume Screener Frontend or UI
import streamlit as st 
# import pandas as pd # TODO: Uncomment for (Ranking Tables)
# import plotly.express as px # TODO: Uncomment for (Charts)
 
#Import the backend (just the parser for now)
import engine

# Page Configuration

st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screener")
st.caption("Internship Project | Works with text-based PDFs (no OCR)")

# Sidebar: Inputs

with st.sidebar:
    st.header("Job Description")
    jd_text = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="e.g., Looking for Data Scientist with Python, SQL, and AWS..."
  )
    
    st.header(" Upload Resumes")
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True
        )
    
    st.markdown("---")
    run_button = st.button("Screen Candidates", type="primary", use_container_width=True)
    
# Main Are : Placeholder for tomorrow's logic

if run_button:
    if not jd_text:
        st.warning("Please paste a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume PDF.")
    else:
        st.info("UI Layer is ready! Day 2 will connect the BERT AI engine here.")
        st.write("**Files Uploaded:**:")
        for file in uploaded_files:
            st.write(f"- {file.name}")
