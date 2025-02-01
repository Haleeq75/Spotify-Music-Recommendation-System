# Spotify Music Recommendation System


  This Python program is designed to recommend songs based on user input using the Spotify dataset. It utilizes machine learning techniques such as K-Means clustering to group songs with similar features and recommend relevant tracks to users. The program leverages the pandas library for data manipulation, scikit-learn for machine learning tasks, and fuzzywuzzy for string matching.

## Installaltion:
Install Dependencies: Ensure that the required packages are installed by running the following commands:
```
%pip install pandas
%pip install -U scikit-learn
%pip install fuzzywuzzy
%pip install fuzzywuzzy[speedup]
```
## Process
The steps involved in the process are listed below

  **1.Import Dataset:** Import the suitable dataset into the program. The current implementation loads the dataset from a CSV file named "genres_v2.csv". Adjust the file path accordingly to load your dataset.
  
  **2.Define Necessary Data:** Specify the relevant features from the dataset required for the recommendation system. This includes attributes such as danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, and tempo.
  
  **3.Fit Clustering Model:** Fit a K-Means clustering model to the dataset to group songs into clusters based on their features. This step involves specifying the number of clusters and fitting the model to the dataset.
  
  **4.Recommend Songs:** Define a function to recommend songs based on user input. The program prompts the user to provide the name of their favorite song and optionally specify a preferred genre. It then recommends a specified number of songs similar to the user's favorite track.
  
  **5.Run the Program:** Execute the main function to run the Spotify Music Recommendation System. Follow the prompts to input your favorite song and receive personalized recommendations.
  

Note:
  i)Ensure that the dataset is in the appropriate format and contains the required features for clustering and recommendation.
  ii)Adjust the parameters such as the number of clusters and the threshold for string matching as needed to optimize the recommendation system.
