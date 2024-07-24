# to run algorithms related to bm25 search
import nltk
import string
import pandas as pd
from rank_bm25 import BM25Okapi

import tokeniser as tk
from global_methods import save_to_json, load_json, save_bm25, load_bm25

from resources.objects.paper import Paper
from resources.objects.ranked_papers import RankedPapers



# ====================================================================================================
# MAIN BM25 FUNCTION
# ====================================================================================================
def run_bm25(data, query, title_weight, abstract_weight):
    # Clean and tokenize the data
    papers = tk.tokenise_papers(data)
    papers = [Paper(p['ss_id'], p['title'], p['abstract'], p['title_tokens'], p['abstract_tokens']) for p in papers]
    save_to_json([p.__dict__ for p in papers], "test", "tokenised_text")

    # Preprocess the data
    titles, abstracts = formatting_data(papers)
    bm25_titles = initialize_bm25(titles)
    bm25_abstracts = initialize_bm25(abstracts)
    # save_bm25(bm25, "bm25")
    # bm25 = load_bm25("bm25")

    # run the query
    title_scores = run_bm25_query(bm25_titles, query) # scores = [0.14708367, 0.14708367, 0.08984029, 0.11154667]
    abstract_scores = run_bm25_query(bm25_abstracts, query) # scores = [0.14708367, 0.14708367, 0.08984029, 0.11154667]

    combined_scores = [
        (title_weight * title_score + abstract_weight * abstract_score)
        for title_score, abstract_score in zip(title_scores, abstract_scores)
    ]

    ranked_papers = RankedPapers("BM25")
    for paper in papers:
        ranked_papers.add_paper(paper)

    ranked_papers.rank_papers_by_score(combined_scores)
    save_to_json([p.__dict__ for p in ranked_papers.papers], "ranked_papers_one", "results")
    print("top 5 papers: ", ranked_papers.papers[:5])
    
    return ranked_papers





# ====================================================================================================
# HELPER FUNCTIONS
# ====================================================================================================

# def formatting_data(papers):
#     titles = [paper['title_tokens'] for paper in papers]
#     abstracts = [paper['abstract_tokens'] for paper in papers]
#     # papers_formatted = titles + abstracts # papers_formatted data format: ['title token 1', 'title token 2', 'title token 3', ..., 'abstract token 1', 'abstract token 2', 'abstract token 3', ...]
    
#     return titles, abstracts

def formatting_data(papers):
    titles = [paper.title_tokens for paper in papers]
    abstracts = [paper.abstract_tokens for paper in papers]
    return titles, abstracts



# ====================================================================================================
# BM25 FUNCTIONS
# ====================================================================================================
# Prepare data for BM25
def initialize_bm25(papers):
    bm25 = BM25Okapi(papers)
    return bm25

def run_bm25_query(bm25, query):
    query_tokens = tk.clean_and_tokenise(query, "query")
    # print("query_tokens: ", query_tokens)
    scores = bm25.get_scores(query_tokens)
    return scores

def rank_papers(results):
    ranked_papers = sorted(results, key=lambda x: x["combined_score"], reverse=True)

    # for rank, item in enumerate(ranked_papers, start=1):
    #     paper = item["paper"]
    #     print(f"Rank {rank}:")
    #     print(f"  SS ID: {paper['ss_id']}")
    #     print(f"  Title: {paper['title']}")
    #     print(f"  Abstract: {paper['abstract']}")
    #     print(f"  Combined Score: {item['combined_score']:.4f}")
    #     print()

    return ranked_papers
# ranked_papers = [
#     {
#         "paper": {
#             "ss_id": <paper_id>,
#             "title": <title>,
#             "abstract": <abstract>,
#             "title_tokens": <list_of_title_tokens>,
#             "abstract_tokens": <list_of_abstract_tokens>
#         },
#         "combined_score": <combined_score>
#     },
#     ...
# ]






