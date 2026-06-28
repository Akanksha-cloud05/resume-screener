import engine

# Test with the exact text from your PDFs
resume1_text = """JOHN DOE
Data Scientist
SKILLS
Python, SQL, MySQL, PostgreSQL, Machine Learning, Deep Learning, NLP, Pandas, NumPy, TensorFlow, PyTorch, AWS, Docker, Git, Statistics, Tableau"""

jd = "Looking for Data Scientist with Python, SQL, Machine Learning, Deep Learning, AWS, and Statistics skills"

print("=== JD SKILLS ===")
jd_skills = engine.extract_skills(jd)
print(jd_skills)

print("\n=== RESUME 1 SKILLS ===")
r1_skills = engine.extract_skills(resume1_text)
print(r1_skills)

print("\n=== COMPARE ===")
matched, missing = engine.compare_skills(jd, resume1_text)
print(f"Matched: {matched}")
print(f"Missing: {missing}")
