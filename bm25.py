import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import pandas as pd

# Ensure you have the necessary NLTK data files
nltk.download('punkt')
nltk.download('stopwords')

# Example abstracts and titles
documents = [
    {"title": "Deep Learning for NLP", "abstract": "This paper explores the use of deep learning techniques in natural language processing..."},
    {"title": "Machine Learning in Healthcare", "abstract": "The application of machine learning in healthcare has been growing rapidly..."}
]

# Preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    print("tokens: ", tokens)
    return tokens

# Apply preprocessing to abstracts and titles
for doc in documents:
    doc['title_tokens'] = preprocess_text(doc['title'])
    doc['abstract_tokens'] = preprocess_text(doc['abstract'])

# Prepare data for BM25
titles = [doc['title_tokens'] for doc in documents]
abstracts = [doc['abstract_tokens'] for doc in documents]

print("titles: ", titles)
print("abstracts: ", abstracts)
# Example of how to use BM25
from rank_bm25 import BM25Okapi

# Combine titles and abstracts
all_documents = titles + abstracts

# Initialize BM25
bm25 = BM25Okapi(all_documents)

# Example query
query = "deep learning in healthcare"
query_tokens = preprocess_text(query)

# Get BM25 scores
scores = bm25.get_scores(query_tokens)

# Print results
print(scores)


# Combine titles and abstracts
all_documents = titles + abstracts

# BM25 scores
scores = [0.14708367, 0.14708367, 0.08984029, 0.11154667]

# Map scores to documents
results = [{"document": doc, "score": score} for doc, score in zip(all_documents, scores)]

# Print results
for result in results:
    print(f"Document: {result['document']}, Score: {result['score']}")