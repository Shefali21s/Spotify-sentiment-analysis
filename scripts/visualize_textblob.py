import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load TextBlob results
df = pd.read_csv("results/textblob_sentiment.csv")

# Setup
sns.set(style="whitegrid")
os.makedirs("results", exist_ok=True)

# === 1. Bar Chart: Overall Sentiment (TextBlob) ===
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="textblob_sentiment", palette="pastel", order=["Positive", "Neutral", "Negative"])
plt.title("TextBlob - Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/textblob_sentiment_distribution.png")
plt.show()

# === 2. Sentiment by Source ===
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="textblob_sentiment", hue="source", palette="Set2", order=["Positive", "Neutral", "Negative"])
plt.title("TextBlob - Sentiment by Dataset Source")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.legend(title="Source")
plt.tight_layout()
plt.savefig("results/textblob_sentiment_by_source.png")
plt.show()

# === 3. Boxplot: Sentiment Score by Source ===
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="source", y="textblob_polarity", palette="Set3")
plt.title("TextBlob - Polarity Score Distribution by Source")
plt.xlabel("Dataset Source")
plt.ylabel("Polarity Score")
plt.tight_layout()
plt.savefig("results/textblob_score_boxplot.png")
plt.show()

print("✅ All TextBlob plots saved to results/")
