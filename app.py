from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Route to get all books
@app.route('/movies', methods=['GET'])
def get_books():
    # Extract movie name from request
    movie_name = request.args.get('movie_name')
    result = []

    # Load movie dataset
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['genres','cast','director']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    # Combine selected features into a single text representation
    combined_features = movies_data['genres'] + ' ' + \
                        movies_data['cast'] + ' '+ movies_data['director']

    # Convert text data into numerical feature vectors
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)

    # Train a K-Nearest Neighbors model with cosine similarity
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(feature_vectors)

    # Find the closest matching movie title
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]

    # Get index of the closest movie and find similar movies
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    distances, indices = knn_model.kneighbors(feature_vectors[index_of_the_movie], n_neighbors=11)

    # Print similarity matrices
    print("Distance Matrix:", distances)
    print("Indices Matrix:", indices)

    # Extract recommended movie titles along with distances
    recommendations = []
    recommended_movie_indices = indices.flatten()[1:]
    for idx, index in enumerate(recommended_movie_indices):
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        similarity_score = 1 - distances[0][idx + 1]  # Convert distance to similarity
        recommendations.append({"title": title_from_index, "similarity_score": similarity_score})

    # Return recommendations as JSON response
    return jsonify(recommendations)


if __name__ == '__main__':
    app.run(debug=True)