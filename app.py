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
    movie_name = request.args.get('movie_name')
    result = []
  
    movies_data = pd.read_csv('movies.csv')
    selected_features = ['title','genres','cast','keywords','director','original_language']
    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    combined_features = movies_data['title']+' '+movies_data['genres']+' '+movies_data['cast']+' '+movies_data['keywords']+' '+movies_data['director']+' '+movies_data['original_language']
    
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    
    knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    knn_model.fit(feature_vectors)
   
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    close_match = find_close_match[0]

    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
    distances, indices = knn_model.kneighbors(feature_vectors[index_of_the_movie], n_neighbors=11)

    print("feature_vectors", feature_vectors[index_of_the_movie])

    recommended_movie_indices = indices.flatten()[1:]

    for i, index in enumerate(recommended_movie_indices, start=1):
     title_from_index = movies_data[movies_data.index == index]['title'].values[0]
     result.append(title_from_index)

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)