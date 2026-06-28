# test_bert.py
# Testing BERT vs TF-IDF synonym understanding

import engine

print("=" * 50)
print("BERT Semantic Similarity Tests")
print("=" * 50)

# Test 1: Exact keyword match
print("\nTest 1: Exact keyword match")
score1 = engine.calculate_match_score(
    "Python developer with machine learning",
    "Python developer with machine learning"
)
print(f"Score: {score1}%")
print("Expected: ~100% (identical text)")

# Test 2: Synonym match (BERT should score high, TF-IDF low)
print("\nTest 2: Synonym match (ML = machine learning)")
score2 = engine.calculate_match_score(
    "Looking for machine learning expert",
    "Experienced in ML and deep learning"
)
print(f"Score: {score2}%")
print("BERT: ~80-90% (understands ML = machine learning)")
print("TF-IDF: ~30-50% (no keyword overlap)")

# Test 3: Related skills, different wording
print("\nTest 3: Related skills, different wording")
score3 = engine.calculate_match_score(
    "Data scientist with Python, SQL, and statistics",
    "Analyst using Python, MySQL, and hypothesis testing"
)
print(f"Score: {score3}%")
print("BERT: ~70-80% (understands SQL ≈ MySQL, statistics ≈ hypothesis testing)")

# Test 4: Completely unrelated
print("\nTest 4: Unrelated roles")
score4 = engine.calculate_match_score(
    "Data scientist with Python and SQL",
    "Web developer with JavaScript and React"
)
print(f"Score: {score4}%")
print("Expected: ~20-30% (different domains)")

print("\n" + "=" * 50)
print("Tests complete")
print("=" * 50)    