import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv("data/user_behaviour.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Show how mood affects music listening
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="music_Influencial_mood", palette="pastel", order=df["music_Influencial_mood"].value_counts().index)
plt.title("How Mood Influences Music Listening")
plt.xticks(rotation=45)
plt.tight_layout()
os.makedirs("results", exist_ok=True)
plt.savefig("results/user_mood_influence.png")
plt.show()

# Plot preferred genres
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="fav_music_genre", palette="Set2", order=df["fav_music_genre"].value_counts().index)
plt.title("Favourite Music Genres Among Users")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/fav_genres.png")
plt.show()

print("[OK] User behaviour analysis graphs saved in results/")
