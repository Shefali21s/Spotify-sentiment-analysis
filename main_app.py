import streamlit as st
import pandas as pd
import joblib

# Load model and vectorizer
model = joblib.load("models/logistic_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# App title
st.title("🎧 Spotify Sentiment Analysis")
st.subheader("Logistic Regression ML Classifier")

# === Text Prediction ===
st.write("### 📥 Enter Text to Analyze")
user_input = st.text_area("Type something related to a song, playlist, or user experience:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        st.success(f"🧠 Predicted Sentiment: **{prediction}**")
    else:
        st.warning("Please enter some text!")

# === CSV Upload Prediction ===
st.write("---")
st.write("### 📁 Upload CSV File (with a 'text' column)")

uploaded_file = st.file_uploader("Upload a CSV file for bulk sentiment analysis", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("CSV must contain a column named 'text'.")
        else:
            X = vectorizer.transform(df['text'].astype(str))
            df['predicted_sentiment'] = model.predict(X)
            st.success("✅ Sentiment predicted for uploaded data.")
            st.dataframe(df[['text', 'predicted_sentiment']].head())

            # Optional: Download output
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Results CSV", data=csv, file_name="sentiment_results.csv", mime="text/csv")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
