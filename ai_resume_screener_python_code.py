# IntelliMatch: AI-Powered Resume Screener

# 📘 1. Load and Prepare the Data
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load Resume Data
resumes_df = pd.read_csv("data/AI_Resume_Screening.csv")

# Sample Job Description for Data Scientist
job_description = """
We are seeking a highly motivated Data Scientist to join our analytics team. The ideal candidate should have a strong background in statistics, machine learning, and data wrangling.
Responsibilities include:
- Building predictive models and machine learning algorithms.
- Analyzing large amounts of information to discover trends and patterns.
- Performing data mining and feature engineering.
- Collaborating with engineering and product teams.

Requirements:
- Proficient in Python, SQL, and machine learning libraries (scikit-learn, XGBoost, TensorFlow).
- Strong understanding of NLP, data visualization tools, and statistical techniques.
- Experience with cloud platforms like AWS or GCP is a plus.
"""

# Combine fields from resumes into a single text
resumes_df["Combined_Text"] = (
    resumes_df["Skills"].fillna("") + " " +
    resumes_df["Education"].fillna("") + " " +
    resumes_df["Certifications"].fillna("") + " " +
    resumes_df["Job Role"].fillna("")
)

# 🧠 2. TF-IDF Matching
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform([job_description] + resumes_df["Combined_Text"].tolist())

# Calculate cosine similarity with JD
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
resumes_df["TFIDF_Match_Score"] = cosine_sim * 100

# Save vectorizer for future use
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

# 🎯 3. AI Recruiter Model (Optional Prediction)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Convert labels
resumes_df['Label'] = resumes_df['Recruiter Decision'].map({"Hire": 1, "Reject": 0})

# Define features
features = resumes_df[["Experience (Years)", "Projects Count", "AI Score (0-100)", "TFIDF_Match_Score"]]
labels = resumes_df['Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "model/ai_recruiter_model.pkl")

# ✅ Output top matching resumes
top_resumes = resumes_df.sort_values(by="TFIDF_Match_Score", ascending=False).head(10)
print(top_resumes[["Name", "Skills", "TFIDF_Match_Score", "AI Score (0-100)", "Projects Count", "Recruiter Decision"]])
