from preprocess import clean_text
from sentiment import analyze_sentiment
from visualize import plot_sentiment_distribution
import pandas as pd

# Step 1: Load dataset
df = pd.read_csv("data/reviews.csv")  # Make sure your dataset is in /data

# Step 2: Clean text
df['cleaned'] = df['text'].apply(clean_text)

# Step 3: Analyze sentiment
df['sentiment'] = df['cleaned'].apply(analyze_sentiment)

# Step 4: Visualize result
plot_sentiment_distribution(df['sentiment'])

# Step 5: Save output (optional)
df.to_csv("results/processed_reviews.csv", index=False)
