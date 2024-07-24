# to run algorithms related to bm25 search
import nltk
import string
import pandas as pd
from rank_bm25 import BM25Okapi
from tokeniser import clean_and_tokenise


# Prepare data for BM25
def initialize_bm25(papers):
    bm25 = BM25Okapi(papers)
    return bm25

def run_bm25_query(bm25, query):
    query_tokens = clean_and_tokenise(query, "query")
    scores = bm25.get_scores(query_tokens)
    return scores

def rank_documents():
    # Rank documents by scores in descending order
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    ranked_docs = [corpus[i] for i in ranked_indices]
    ranked_scores = [scores[i] for i in ranked_indices]

    # Output ranked documents and their scores
    for doc, score in zip(ranked_docs, ranked_scores):
        print(f"Document: {doc} \nScore: {score}\n")

    # Get the highest score
    highest_score = ranked_scores[0]
    highest_ranked_doc = ranked_docs[0]
    print(f"Highest Ranked Document: {highest_ranked_doc} \nHighest Score: {highest_score}")
