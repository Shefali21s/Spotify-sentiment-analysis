import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Load data
df = pd.read_csv("data/playlists_from_json.csv")

# Combine title + description into one 'text' column
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')
df['text'] = df['title'] + ' ' + df['description']

# Analyze sentiment
analyzer = SentimentIntensityAnalyzer()
df['compound'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Label the sentiment
def get_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['compound'].apply(get_label)

# Save results
os.makedirs("results", exist_ok=True)
df.to_csv("results/playlists_sentiment.csv", index=False)
print("[OK] Sentiment analysis completed and saved as results/playlists_sentiment.csv")
print(df[['title', 'sentiment']].sample(5))
