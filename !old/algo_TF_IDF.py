# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # Sample data
# data = {
#     'titles': [
#         "Paper A on topic X",
#         "Paper B on topic Y",
#         "Paper C on topic X and Y",
#         "Paper D on topic Z"
#     ],
#     'abstracts': [
#         "This paper discusses topic X.",
#         "This paper discusses topic Y.",
#         "This paper discusses topics X and Y.",
#         "This paper discusses topic Z."
#     ]
# }

# df = pd.DataFrame(data)

# # TF-IDF Vectorizer
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['abstracts'])

# # Compute Cosine Similarity
# cosine_sim = cosine_similarity(tfidf_matrix)

# # Convert to DataFrame
# tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# # Display the DataFrame
# print("TF-IDF Matrix:")
# print(tfidf_df)






import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example data (Replace with actual data)
data = {
    'titles': [
        "Paper A on topic X",
        "Paper B on topic Y",
        "Paper C on topic X and Y",
        "Paper D on topic Z"
    ],
    'abstracts': [
        "This paper discusses topic X.",
        "This paper discusses topic Y.",
        "This paper discusses topics X and Y.",
        "This paper discusses topic Z."
    ],
    'scores_other_method': [0.8, 0.6, 0.9, 0.4]  # Scores from the other retrieval method
}

df = pd.DataFrame(data)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['abstracts'])

# Compute Cosine Similarity for the query (e.g., Paper A)
query_index = 0  # Index of the query paper
query_vector = tfidf_matrix[query_index]
cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

# Combine the results
df['tfidf_similarity'] = cosine_similarities

# Compare scores
comparison_df = df[['titles', 'scores_other_method', 'tfidf_similarity']]

# Display the comparison
print(comparison_df)
