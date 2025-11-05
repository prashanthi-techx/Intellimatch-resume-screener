import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import TfidfTransformer
# Page Title
st.set_page_config(page_title="IntelliMatch Resume Screener", layout="wide")
st.title("🤖 IntelliMatch: AI Resume Screener")

# # Input Section
# streamlit run app/streamlit_app.py ← move this to a separate note or terminal

jd_input = st.text_area("📄 Paste the Job Description Here")
resumes_file = st.file_uploader("📤 Upload Resume CSV File", type=["csv"])

# If both inputs are provided
if jd_input and resumes_file:
    resumes_df = pd.read_csv(resumes_file)


    # Combine fields
    resumes_df["Combined_Text"] = (
        resumes_df["Skills"].fillna("") + " " +
        resumes_df["Education"].fillna("") + " " +
        resumes_df["Certifications"].fillna("") + " " +
        resumes_df["Job Role"].fillna("")
    )

    # Load vectorizer and compute score
    tfidf = joblib.load("model/tfidf_vectorizer.pkl")
    input_texts = [jd_input] + resumes_df["Combined_Text"].tolist()
    # DEBUG: Print to check values
    st.write("🔎 Job Description Text:", jd_input[:200])
    st.write("📄 First Resume Sample:", resumes_df["Combined_Text"].iloc[0][:200])
    st.write("🧠 Total texts to transform:", len(input_texts))
    #Prepare input_texts
    check_is_fitted(tfidf)  # ✅ Optional sanity check
    
    tfidf_matrix = tfidf.transform (input_texts)
    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    resumes_df["TFIDF_Match_Score"] = cosine_sim * 100

    # Load trained model and predict
    model = joblib.load("model/ai_recruiter_model.pkl")
    input_features = resumes_df[["Experience (Years)", "Projects Count", "AI Score (0-100)", "TFIDF_Match_Score"]]
    resumes_df["AI_Prediction"] = model.predict(input_features)
    resumes_df["AI_Prediction"] = resumes_df["AI_Prediction"].map({1: "Hire", 0: "Reject"})

    # Show results
    st.success("✅ Matching Completed!")
    st.subheader("🎯 Top 10 Recommended Candidates")
    st.dataframe(resumes_df.sort_values(by="TFIDF_Match_Score", ascending=False).head(10))

    # Download
    csv = resumes_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Full Results", csv, "intellimatch_results.csv", "text/csv")
