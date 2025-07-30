import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import os

# Load dataset (we'll use VADER-labeled data)
df = pd.read_csv("results/combined_sentiment.csv")

# Filter only Positive, Negative, Neutral
df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]

# Features and labels
X = df['text']
y = df['sentiment']

# Text Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model & vectorizer (optional)
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved in /models/")
