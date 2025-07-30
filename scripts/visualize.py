import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("results/combined_sentiment.csv")

# Setup
sns.set(style="whitegrid")
os.makedirs("results", exist_ok=True)

# === 1. Overall Sentiment Bar Chart ===
try:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x="sentiment", palette="pastel", order=["Positive", "Neutral", "Negative"])
    plt.title("Overall Sentiment Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/sentiment_overall.png")
    print("✅ Saved: sentiment_overall.png")
    plt.close()
except Exception as e:
    print("❌ Error creating bar chart:", e)

# === 2. Sentiment by Source ===
try:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="sentiment", hue="source", palette="Set2", order=["Positive", "Neutral", "Negative"])
    plt.title("Sentiment by Dataset Source")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")
    plt.legend(title="Source")
    plt.tight_layout()
    plt.savefig("results/sentiment_by_source.png")
    print("✅ Saved: sentiment_by_source.png")
    plt.close()
except Exception as e:
    print("❌ Error creating bar chart by source:", e)

# === 3. Pie Chart ===
try:
    plt.figure(figsize=(5, 5))
    df['sentiment'].value_counts().plot.pie(
        autopct='%1.1f%%',
        startangle=140,
        colors=["#98FB98", "#FFDAB9", "#FF9999"],
        wedgeprops=dict(width=0.4)
    )
    plt.title("Sentiment Share")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("results/sentiment_pie_chart.png")
    print("✅ Saved: sentiment_pie_chart.png")
    plt.close()
except Exception as e:
    print("❌ Error creating pie chart:", e)

# === 4. Heatmap ===
try:
    pivot = df.pivot_table(index=df.index, columns='source', values='compound', aggfunc='mean')
    plt.figure(figsize=(6, 4))
    sns.heatmap(pivot.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation of Sentiment Scores Between Sources")
    plt.tight_layout()
    plt.savefig("results/sentiment_correlation_heatmap.png")
    print("✅ Saved: sentiment_correlation_heatmap.png")
    plt.close()
except Exception as e:
    print("❌ Error creating heatmap:", e)

# === 5. Boxplot ===
try:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="source", y="compound", palette="Set3")
    plt.title("Sentiment Score Distribution by Source")
    plt.xlabel("Dataset Source")
    plt.ylabel("Compound Sentiment Score")
    plt.tight_layout()
    plt.savefig("results/sentiment_score_boxplot.png")
    print("✅ Saved: sentiment_score_boxplot.png")
    plt.close()
except Exception as e:
    print("❌ Error creating boxplot:", e)
