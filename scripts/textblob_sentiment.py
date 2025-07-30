import pandas as pd
from textblob import TextBlob
import os

# Load cleaned datasets (lyrics + playlists + user)
lyrics = pd.read_csv("processed/lyrics_clean.csv")
playlists = pd.read_csv("processed/playlists_clean.csv")
user = pd.read_csv("processed/user_clean.csv")

combined = pd.concat([lyrics, playlists, user], ignore_index=True)

# Apply TextBlob polarity
combined['textblob_polarity'] = combined['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Convert polarity score to label
def get_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

combined['textblob_sentiment'] = combined['textblob_polarity'].apply(get_label)

# Save result
os.makedirs("results", exist_ok=True)
combined.to_csv("results/textblob_sentiment.csv", index=False)

print("✅ TextBlob sentiment saved to results/textblob_sentiment.csv")
print(combined[['text', 'textblob_polarity', 'textblob_sentiment']].sample(5))
