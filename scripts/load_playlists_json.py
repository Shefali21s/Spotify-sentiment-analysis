import json
import pandas as pd

# Load the JSON file
with open("data/mpd.slice.0-999.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# Extract playlists
playlists = data['playlists']

# Get playlist titles & descriptions
extracted = []
for p in playlists:
    title = p['name']
    description = p.get('description', '')
    extracted.append({'title': title, 'description': description})

# Save to CSV
df = pd.DataFrame(extracted)
df.to_csv("data/playlists_from_json.csv", index=False)

print("✅ playlists_from_json.csv saved successfully!")
print(df.head())

