import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Create necessary folders
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load your dataset
df = pd.read_csv("data/AI_Resume_Screening.csv")

# Fill missing numeric columns safely
df["Experience (Years)"] = df["Experience (Years)"].fillna(0)
df["Projects Count"] = df["Projects Count"].fillna(0)
df["AI Score (0-100)"] = df["AI Score (0-100)"].fillna(0)
df["TFIDF_Match_Score"] = df.get("TFIDF_Match_Score", pd.Series([50]*len(df)))

# Create a dummy target column (for training example)
df["Label"] = (df["AI Score (0-100)"] > 50).astype(int)

# Select features
X = df[["Experience (Years)", "Projects Count", "AI Score (0-100)", "TFIDF_Match_Score"]]
y = df["Label"]

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/ai_recruiter_model.pkl")

print("✅ Model retrained and saved successfully!")
