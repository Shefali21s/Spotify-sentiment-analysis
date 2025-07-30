import pandas as pd
import os

# Create output directory
os.makedirs("processed", exist_ok=True)

# Clean lyrics
lyrics = pd.read_csv("data/lyrics.csv")
lyrics = lyrics.rename(columns={"text": "text"})
lyrics["source"] = "lyrics"
lyrics = lyrics[["text", "source"]]
lyrics.to_csv("processed/lyrics_clean.csv", index=False)

# Clean playlists
playlists = pd.read_csv("data/playlists_from_json.csv")
playlists["title"] = playlists["title"].fillna("")
playlists["description"] = playlists["description"].fillna("")
playlists["text"] = playlists["title"] + " " + playlists["description"]
playlists["source"] = "playlist"
playlists = playlists[["text", "source"]]
playlists.to_csv("processed/playlists_clean.csv", index=False)

# Clean user behaviour (convert to text-friendly format)
user = pd.read_csv("data/user_behaviour.csv")
user["text"] = user["music_Influencial_mood"].astype(str) + " " + user["fav_music_genre"].astype(str)
user["source"] = "user"
user = user[["text", "source"]]
user.to_csv("processed/user_clean.csv", index=False)

print("[OK] All data preprocessed and saved to /processed/")
