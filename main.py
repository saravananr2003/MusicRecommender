# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample dataset: A list of dictionaries, each representing a Tamil song.
# In a real-world scenario, you would load this data from a CSV or a database.
data = [
    {"song": "Kanave Kanave", "artist": "Anirudh Ravichander", "genre": "Romantic", "description": "Soft melody with romantic lyrics"},
    {"song": "Rowdy Baby", "artist": "Dhanush", "genre": "Folk", "description": "Energetic beats and catchy tune"},
    {"song": "Vaathi Coming", "artist": "Anirudh Ravichander", "genre": "Dance", "description": "Upbeat dance number with energetic performance"},
    {"song": "Thalli Pogathey", "artist": "Sid Sriram", "genre": "Melodious", "description": "Smooth vocals with a blend of modern and classical"},
    {"song": "Why This Kolaveri Di", "artist": "Dhanush", "genre": "Comedy", "description": "Catchy and quirky with simple lyrics"},
    {"song": "Surviva", "artist": "Anirudh Ravichander", "genre": "Energetic", "description": "Fast-paced beats and motivating lyrics"}
]

# Create a DataFrame
df = pd.DataFrame(data)

# Combine relevant text features into a single string for each song.
# You can adjust the combination as per the available features.
df['combined_features'] = df['genre'] + " " + df['artist'] + " " + df['description']

# Use TF-IDF Vectorizer to convert text to a matrix of TF-IDF features.
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Compute cosine similarity between songs
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(song_title, df, cosine_sim, top_n=3):
    """
    Given a song title, return top_n similar songs.
    """
    # Create a Series mapping song titles to indices
    indices = pd.Series(df.index, index=df['song']).drop_duplicates()

    # Get the index of the song that matches the title
    idx = indices.get(song_title)
    if idx is None:
        print(f"Song '{song_title}' not found in the dataset.")
        return []

    # Get the pairwise similarity scores of all songs with that song
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the songs based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar songs (skip the first one since it's the song itself)
    sim_scores = sim_scores[1:top_n+1]

    # Get the song indices
    song_indices = [i[0] for i in sim_scores]

    # Return the top_n most similar songs
    return df['song'].iloc[song_indices]

# Example usage:
selected_song = "Aalaporan Thamizhan"
recommended_songs = get_recommendations(selected_song, df, cosine_sim, top_n=3)
print(f"Songs similar to '{selected_song}':")
print(recommended_songs.tolist())
