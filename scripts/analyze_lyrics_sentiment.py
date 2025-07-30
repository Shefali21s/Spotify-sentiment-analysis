import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Load dataset
df = pd.read_csv("data/lyrics.csv")

# Setup sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Clean text and score sentiment
df['text'] = df['text'].fillna('')
df['compound'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

# Label sentiment
def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df['sentiment'] = df['compound'].apply(label_sentiment)

# Save results
os.makedirs("results", exist_ok=True)
df.to_csv("results/lyrics_sentiment.csv", index=False)
print("✅ Saved: results/lyrics_sentiment.csv")
print(df[['song', 'artist', 'sentiment']].sample(5))
