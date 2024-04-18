# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from fuzzywuzzy import process
#If these package not available, run this
#%pip install pandas
#%pip install -U scikit-learn
#%pip install fuzzywuzzy
#%pip install fuzzywuzzy[speedup]


#Displaying the complete output
pd.set_option('display.max_rows', None,
              'display.width', 9000)

# Load the Spotify dataset
spotify_data = pd.read_csv("genres_v2.csv", low_memory=False)
#using low_memory=False because they contain large data

# Select relevant features for clustering
X = spotify_data[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=10, random_state=42)
# Convert DataFrame to numpy array
X_array = X.values

# Fit K-Means clustering model with numpy array
kmeans.fit(X_array)

# Get cluster labels
spotify_data['cluster'] = kmeans.labels_

# Define function to recommend songs based on user input
def recommend_songs(user_input, num_recommendations= 5, preferred_genre=None):
    user_cluster = kmeans.predict([user_input])[0]
    cluster_songs = spotify_data[spotify_data['cluster'] == user_cluster]
    if preferred_genre:
        cluster_songs = cluster_songs[cluster_songs['genre'].str.lower().str.contains(preferred_genre.lower())]
    recommendations = cluster_songs.sample(num_recommendations)
    #print(recommendations)-Dummycheck
    def merge_columns(row):
        if pd.isnull(row['song_name']):
            return row['title']
        else:
            return row['song_name']
    recommendations['merged_column']= recommendations.apply(merge_columns, axis=1)
    recommendations= recommendations.drop(columns=['song_name', 'title'])
    recommendations= recommendations.rename(columns={'merged_column':'Title of Track'})
    #print(recommendations)-dummycheck
    return recommendations[['Title of Track', 'genre', 'id']]

# Function to get features of user's favorite song
def get_favorite_song_features(song_name):
    # Filter out NaN values in the song_name column
    filtered_spotify_data = spotify_data.dropna(subset=['song_name'])
    song_name_lower = song_name.lower()
    matching_song = process.extractOne(song_name_lower, filtered_spotify_data['song_name'].str.lower())
    print("Matching song:", matching_song)
    if matching_song and matching_song[1] >= 90:  # Adjust the threshold as needed
        song_features = filtered_spotify_data[filtered_spotify_data['song_name'].str.lower() == matching_song[0]]
        return song_features.iloc[0][['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    else:
        print("Sorry, no similar song found in the dataset.")
        return None

# Function to get user input for preferred genre
def get_preferred_genre():
    while True:
        preferred_genre = input("Enter your preferred genre (or leave blank for any genre): ")
        if preferred_genre.strip() == "":
            return None
        else:
            return preferred_genre

# Main function to run the recommendation system
def main():
    print("Welcome to the Spotify Music Recommendation System!")
    print("Please provide the name of your favorite song.")
    favorite_song = input("Favorite Song: ")
    favorite_song_features = get_favorite_song_features(favorite_song)
    if favorite_song_features is not None:
        preferred_genre = get_preferred_genre()
        num_recommendations = int(input("How many recommendations do you want? "))
        recommended_songs = recommend_songs(favorite_song_features.values.tolist(), num_recommendations, preferred_genre)
        print("\nRecommended Songs:")
        print(recommended_songs)

if __name__ == "__main__":
    main()
