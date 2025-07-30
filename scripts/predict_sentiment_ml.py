import joblib
import os

# Load model and vectorizer
model_path = "models/logistic_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model and vectorizer loaded successfully.")
else:
    print("❌ Model files not found.")
    exit()

# === Predict Sentiment on Custom Input ===
sample_texts = [
    "I love this song, it makes me feel alive!",
    "The lyrics are depressing and dark.",
    "It's just okay, nothing special."
]

# Transform text using TF-IDF vectorizer
X_sample = vectorizer.transform(sample_texts)

# Predict
predictions = model.predict(X_sample)

# Display results
for text, pred in zip(sample_texts, predictions):
    print(f"\n📝 Text: {text}\n➡️ Predicted Sentiment: {pred}")
