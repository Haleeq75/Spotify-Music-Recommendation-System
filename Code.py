# -*- coding: utf-8 -*-
"""Code (1) (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1PrydqSJ73XV1D-lY5NHcEqc336iBr6tW

Installing necessary libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install pandas
# %pip install -U scikit-learn
# %pip install fuzzywuzzy
# %pip install fuzzywuzzy[speedup]
# %pip install yellowbrick
# %pip install seaborn
# %pip install plotly.express
# %pip install matplot
# %pip install nbformat>=4.2.0
# %pip install --upgrade nbformat

"""Importing necessary libraries"""

import pandas as pd
from sklearn.cluster import KMeans
from fuzzywuzzy import process
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

"""Importing the suitable dataset"""

spotify_data = pd.read_csv("genres_v2.csv", low_memory=False)
#using low_memory=False because they contain large data

print(spotify_data.info())

"""Feature Correlation"""

from sklearn.preprocessing import LabelEncoder

# Instantiate LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'genre' column
spotify_data['genre_encoded'] = label_encoder.fit_transform(spotify_data['genre'])

# Use 'genre_encoded' as the target variable for correlation analysis
y = spotify_data['genre_encoded']

from yellowbrick.target import FeatureCorrelation

feature_names = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

y = spotify_data['genre_encoded']
x = spotify_data[feature_names]

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(x, y)     # Fit the data to the visualizer
visualizer.show()

import matplotlib.pyplot as plt

# Plot a histogram of danceability
plt.hist(spotify_data['danceability'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Danceability')
plt.ylabel('Frequency')
plt.title('Distribution of Danceability')
plt.show()

"""Analysis Of Top Genre"""

# Bar plot of top 10 genres by popularity
top_genres = spotify_data['genre'].value_counts().head(10)
plt.bar(top_genres.index, top_genres.values, color='orange')
plt.xlabel('Genre')
plt.ylabel('Number of Songs')
plt.title('Top 10 Genres by Popularity')
plt.xticks(rotation=45, ha='right')
plt.show()

"""Analysis of Danceability over Energy"""

import seaborn as sns

# Scatter plot of danceability vs. energy
sns.scatterplot(x='tempo', y='energy', data=spotify_data)
plt.xlabel('Tempo')
plt.ylabel('Energy')
plt.title('Scatter Plot of Danceability vs. Energy')
plt.show()

"""Analysis of Loudness Over Acousticness"""

import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'spotify_data' with columns 'release_date' and 'loudness'
# Convert 'release_date' column to datetime if it's not already in datetime format


# Sort the DataFrame by 'release_date' to ensure the data is plotted in chronological order
spotify_data = spotify_data.sort_values(by='acousticness')

# Plot the line graph
plt.figure(figsize=(10, 6))
plt.plot(spotify_data['acousticness'], spotify_data['loudness'], color='blue', marker='o', linestyle='-')
plt.title('Loudness Over Acousticness')
plt.xlabel('Acousticness')
plt.ylabel('Loudness')
plt.grid(True)
plt.show()

"""Defining the necessary data form dataset for the program"""

X = spotify_data[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]

"""Displaying the complete output"""

pd.set_option('display.max_rows', None,
              'display.width', 9000)

"""Fitting K-Means clustering model and geting cluster model"""

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=10, random_state=42)
# Convert DataFrame to numpy array
X_array = X.values

# Fit K-Means clustering model with numpy array
kmeans.fit(X_array)

# Get cluster labels
spotify_data['cluster'] = kmeans.labels_

"""Definign the function for recommended song from user input"""

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

"""Defining a function to get features of user's favorite song"""

# Function to get features of user's favorite song
def get_favorite_song_features(song_name):
    # Filter out NaN values in the song_name column
    filtered_spotify_data = spotify_data.dropna(subset=['song_name'])
    song_name_lower = song_name.lower()
    matching_song = process.extractOne(song_name_lower, filtered_spotify_data['song_name'].str.lower())
    #print("Matching song:", matching_song)
    if matching_song and matching_song[1] >= 90:  # Adjust the threshold as needed
        song_features = filtered_spotify_data[filtered_spotify_data['song_name'].str.lower() == matching_song[0]]
        return song_features.iloc[0][['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']]
    else:
        print("Sorry, no similar song found in the dataset.")
        return None

"""Defining a function to get user input for preferred genre"""

# Function to get user input for preferred genre
def get_preferred_genre():
    while True:
        preferred_genre = input("Enter your preferred genre (or leave blank for any genre): ")
        if preferred_genre.strip() == "":
            return None
        else:
            return preferred_genre

"""Defining wokring main function to run the recommendation system"""

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

"""To make this program run"""

if __name__ == "__main__":
    main()
