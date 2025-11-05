import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from sklearn.utils.validation import check_is_fitted

# Create folders if not exist
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load your resume dataset
df = pd.read_csv("data/AI_Resume_Screening.csv")

# Combine text columns
df["Combined_Text"] = (
    df["Skills"].fillna("") + " " +
    df["Education"].fillna("") + " " +
    df["Certifications"].fillna("") + " " +
    df["Job Role"].fillna("")
)

# Sample job description text (to give context)
job_description = "Looking for a data analyst skilled in Python, SQL, statistics, and visualization."

# Create and fit TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words="english")
tfidf.fit([job_description] + df["Combined_Text"].tolist())

# Verify it's fitted (this will raise error if not)
check_is_fitted(tfidf)

# Save the fitted vectorizer
joblib.dump(tfidf, "model/tfidf_vectorizer.pkl")

print("✅ TF-IDF Vectorizer fitted and saved successfully!")
