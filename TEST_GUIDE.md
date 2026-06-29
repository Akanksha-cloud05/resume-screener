# 🧪 Complete Testing Guide

Run this checklist before submission.

## Step 1: Environment Setup (2 mins)

```bash
# Activate venv
source screener_env/bin/activate  # On Windows: screener_env\Scripts\activate

# Verify Python version
python --version  # Should be 3.9+

# Verify pip
pip --version
Step 2: Dependencies Check (3 mins)
bash
# Test all required imports
python -c "
import streamlit
import pandas
import numpy
import sklearn
import fitz
import plotly
print('✅ All core imports OK')
"

# Test optional imports (won't fail if missing)
python -c "
try:
    import torch
    print('✅ PyTorch available')
except:
    print('⚠️  PyTorch not available (TF-IDF fallback will be used)')
"
Step 3: Unit Tests (5 mins)
bash
# Run all tests with verbose output
pytest tests/ -v

# Expected output:
#   collected 41 items
#   tests/test_analytics.py ................ 12 PASSED
#   tests/test_engine.py .................. 29 PASSED
#   ========================= 41 passed in X.XXs =========================

# If you see failures:
#   - Check error message
#   - Run single test: pytest tests/test_engine.py::TestWeightedScoring::test_length_penalty_very_short -v
#   - Check config.py for misconfigurations
Step 4: Config Validation (1 min)
bash
# Validates weights, thresholds, patterns on import
python -c "import config; print('✅ Config validation passed')"

# Should see:
#   ============================================================
#   ✅ ALL CONFIG VALIDATION PASSED
#   ============================================================
Step 5: Engine Smoke Test (2 mins)
bash
# Test engine functions directly (no Streamlit)
python << 'EOF'
import engine
import io
import fitz  # PyMuPDF

# Test 1: PDF parsing (creates sample PDF)
print("Testing PDF extraction...")
doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "Python developer with SQL skills")
pdf_bytes = doc.write()
doc.close()
pdf_io = io.BytesIO(pdf_bytes)
pdf_io.name = "test.pdf"
text = engine.extract_text_from_pdf(pdf_io)
assert text and "Python" in text
print("✅ PDF extraction OK")

# Test 2: Section extraction
sections = engine.extract_sections("SKILLS\nPython\n\nEXPERIENCE\nData Scientist")
assert "skills" in sections
print("✅ Section extraction OK")

# Test 3: Skill extraction
skills = engine.extract_skills("Python, SQL, AWS, Machine Learning")
assert "python" in skills
print("✅ Skill extraction OK")

# Test 4: Similarity scoring
score = engine.calculate_similarity("Python developer", "Python developer")
assert score > 90
print("✅ Similarity scoring OK")

# Test 5: Weighted scoring
jd = "Data Scientist with Python and SQL"
resume = "SKILLS\nPython, SQL\n\nEXPERIENCE\nBuilt ML models"
score = engine.calculate_weighted_score(jd, resume)
assert 0 <= score <= 100
print("✅ Weighted scoring OK")

print("\n" + "="*60)
print("✅ ALL ENGINE SMOKE TESTS PASSED")
print("="*60)
EOF
Step 6: Streamlit App Test (10 mins)
bash
# Start the app
streamlit run app.py

# In browser (http://localhost:8501):
# 1. Check sidebar loads without errors
#    - Should see "BERT Active" or "TF-IDF Fallback"
# 2. Paste JD:
#    - Copy from sample_jd.txt
# 3. Upload resume:
#    - Generate sample_resume.pdf from sample_resume.txt
#    - OR download from test data
# 4. Click "Screen Candidates"
# 5. Verify output:
#    - Score 70-85%
#    - Matched skills shown (Python, SQL, AWS, etc.)
#    - Missing skills shown (if any)
#    - Explainability appears
#    - Recommendations shown
# 6. Test CSV export:
#    - Click "Download Results as CSV"
#    - File downloads successfully
# 7. Test analytics (if 2+ resumes):
#    - Score distribution chart appears
#    - Skill gap chart appears
#    - Candidate ranking chart appears

# Stop app: Ctrl+C
Step 7: Logging Verification (2 mins)
bash
# Check that logs are being written
cat resume_screener.log

# Should show:
#   2026-06-28 13:42:16,123 - __main__ - INFO - AI Resume Screener Dashboard Started
#   2026-06-28 13:42:16,234 - __main__ - INFO - BERT Available: False
#   2026-06-28 13:42:45,567 - __main__ - INFO - Screening started: 2 PDFs, JD length: 345 chars
#   ...
Step 8: Final Checklist ✅
pytest tests/ -v — 41 tests passing

python -c "import config" — Config validation passes

Engine smoke tests pass (PDF, sections, skills, scoring)

Streamlit app runs without errors

Sample PDF uploads and scores correctly

Analytics charts render (for 2+ resumes)

CSV export works

Logging file created (resume_screener.log)

No uncommitted changes in git

Troubleshooting
Issue: PyMuPDF DLL error on Windows
text
OSError: [WinError 126] The specified module could not be found
Fix:

bash
pip uninstall torch sentence-transformers -y
pip install PyMuPDF --upgrade
BERT will auto-fallback to TF-IDF.

Issue: Tests fail with "module not found"
Fix:

bash
# Verify conftest.py exists in project root
ls conftest.py  # or dir conftest.py on Windows

# If missing, recreate it (see conftest.py section above)
Issue: Streamlit cache errors
Fix:

bash
streamlit cache clear
streamlit run app.py
Issue: Very slow first run
Expected: First BERT load takes ~30 seconds (model download + initialize)
Normal: Subsequent runs cached, instant

Success Criteria ✅
You're ready to submit when:

✅ All 41 tests pass

✅ Streamlit app runs without errors

✅ Sample resume scores 70-85%

✅ CSV export works

✅ Logging file created

✅ git status shows no uncommitted files