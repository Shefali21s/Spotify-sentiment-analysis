import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Load cleaned datasets
lyrics = pd.read_csv("processed/lyrics_clean.csv")
playlists = pd.read_csv("processed/playlists_clean.csv")
user = pd.read_csv("processed/user_clean.csv")

# Combine them
combined = pd.concat([lyrics, playlists, user], ignore_index=True)

# Initialize VADER analyzer
analyzer = SentimentIntensityAnalyzer()

# Analyze sentiment
combined['compound'] = combined['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Convert compound score to label
def get_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

combined['sentiment'] = combined['compound'].apply(get_label)

# Save results
os.makedirs("results", exist_ok=True)
combined.to_csv("results/combined_sentiment.csv", index=False)

print("✅ Sentiment analysis complete.")
print("🔄 Sample results:")
print(combined.sample(5))
