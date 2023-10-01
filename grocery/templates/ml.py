import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
dataset = pd.read_csv("data.csv")

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(dataset['Name'].str.lower())

# Calculate cosine similarity
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get recommendations for a given product index
def get_recommendations(product_index, num_recommendations=5):
    similarity_scores = list(enumerate(cosine_similarities[product_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similar_products = similarity_scores[1:num_recommendations+1]
    return similar_products

# Function to recommend products based on label
def recommend_products_by_label(label, num_recommendations=5):
    label_indices = dataset[dataset['Label'] == label].index
    recommended_products = []
    
    for idx in label_indices:
        similar_products = get_recommendations(idx, num_recommendations)
        recommended_products.extend(similar_products)
    
    recommended_products = sorted(recommended_products, key=lambda x: x[1], reverse=True)
    return recommended_products[:num_recommendations]

# Example usage
product_index = 0  # Index of the product for which you want recommendations
recommended_products = recommend_products_by_label(dataset['Label'][product_index])

for idx, score in recommended_products:
    print(f"Product: {dataset['Name'][idx]}, Similarity Score: {score:.2f}")
