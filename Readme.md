# Movie Recommendation API

## Overview

This API function, `get_books()`, recommends movies similar to a given movie title using content-based filtering with TF-IDF vectorization and K-Nearest Neighbors (KNN) for similarity measurement.

## Technologies Used

- Python
- Pandas
- Scikit-learn (TF-IDF Vectorizer, Nearest Neighbors)
- Flask (for API request handling)
- Difflib (for approximate string matching)

## How It Works

1. **Retrieve User Input**
   - The function extracts the movie name from the request arguments.
2. **Load Movie Data**
   - Reads `movies.csv` containing movie information.
3. **Feature Engineering**
   - Selected features: `title`, `genres`, `cast`, `keywords`, `director`, `original_language`.
   - Fills missing values with an empty string.
   - Combines all selected features into a single text representation.
4. **Vectorization**
   - Uses `TfidfVectorizer()` to transform the combined text into numerical vectors.
5. **Train Nearest Neighbors Model**
   - Fits a `NearestNeighbors` model with cosine similarity.
6. **Find Closest Movie Match**
   - Uses `difflib.get_close_matches()` to find the closest title match.
7. **Retrieve Similar Movies**
   - Identifies the movie’s index.
   - Finds 10 nearest neighbors based on similarity.
   - Extracts and returns the recommended movie titles as a JSON response.

## Code Breakdown

```python
# Extract movie name from request
movie_name = request.args.get('movie_name')
result = []

# Load movie dataset
movies_data = pd.read_csv('movies.csv')
selected_features = ['title','genres','cast','keywords','director','original_language']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# Combine selected features into a single text representation
combined_features = movies_data['title'] + ' ' + movies_data['genres'] + ' ' + \
                    movies_data['cast'] + ' ' + movies_data['keywords'] + ' ' + \
                    movies_data['director'] + ' ' + movies_data['original_language']

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

# Extract recommended movie titles
recommended_movie_indices = indices.flatten()[1:]
for index in recommended_movie_indices:
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    result.append(title_from_index)

# Return recommendations as JSON response
return jsonify(result)
```

## API Endpoint

### Request:

```
GET /get_books?movie_name=<movie_title>
```

### Response:

```json
[
    "Recommended Movie 1",
    "Recommended Movie 2",
    "Recommended Movie 3",
    ...
]
```

## Possible Improvements

- Improve error handling for cases where no close match is found.
- Optimize model performance with a more robust similarity algorithm.
- Expand features used in recommendation (e.g., user ratings, reviews).

## Dependencies

Ensure you have the following dependencies installed:

```sh
pip install pandas scikit-learn flask difflib
```
