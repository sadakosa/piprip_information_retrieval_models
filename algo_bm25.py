import nltk
import string
import pandas as pd
from rank_bm25 import BM25Okapi

import algo_tokeniser as tk
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list

from db import db_operations

# ====================================================================================================
# MAIN BM25 FUNCTION
# ====================================================================================================

def run_bm25(db_client, raw_papers_df, chunk_size, target_ss_ids, title_weight, abstract_weight):
    combined_scores_list = []
    bm25_titles, bm25_abstracts = None, None

    # Process data in chunks
    for start in range(0, len(raw_papers_df), chunk_size):
        chunk = raw_papers_df.iloc[start:start + chunk_size]
        papers_df = tk.tokenise_papers_df(chunk)
        titles, abstracts = formatting_data(papers_df)

        if bm25_titles is None and bm25_abstracts is None:
            bm25_titles = initialize_bm25(titles)
            bm25_abstracts = initialize_bm25(abstracts)

        target_papers = db_operations.get_papers_by_ss_ids(db_client, target_ss_ids)
        combined_scores_df = pd.DataFrame(index=papers_df['ss_id'], columns=[target[0] for target in target_papers])

        for target_paper in target_papers:
            title_title_scores = run_bm25_query(bm25_titles, target_paper[1], "target_title")
            title_abstract_scores = run_bm25_query(bm25_abstracts, target_paper[1], "target_title")
            abstract_title_scores = run_bm25_query(bm25_titles, target_paper[2], "target_abstract")
            abstract_abstract_scores = run_bm25_query(bm25_abstracts, target_paper[2], "target_abstract")

            combined_scores = [
                (title_weight * (title_title_score + abstract_title_score) + abstract_weight * (title_abstract_score + abstract_abstract_score))
                for title_title_score, title_abstract_score, abstract_title_score, abstract_abstract_score in zip(title_title_scores, title_abstract_scores, abstract_title_scores, abstract_abstract_scores)
            ]

            combined_scores_df[target_paper[0]] = combined_scores

        combined_scores_list.append(combined_scores_df)

    final_combined_scores_df = pd.concat(combined_scores_list, axis=0)
    ranked_papers_with_scores = pd.DataFrame()

    for target in final_combined_scores_df.columns:
        temp_df = pd.DataFrame({
            f"{target}_ss_id": final_combined_scores_df[target].sort_values(ascending=False).index,
            f"{target}_score": final_combined_scores_df[target].sort_values(ascending=False).values
        })
        ranked_papers_with_scores = pd.concat([ranked_papers_with_scores, temp_df], axis=1)

    save_to_csv(ranked_papers_with_scores, "ranked_papers_with_scores_bm25", "results")

    return ranked_papers_with_scores





# ====================================================================================================
# HELPER FUNCTIONS
# ====================================================================================================

def formatting_data(papers_df):
    titles = papers_df['title_tokens'].tolist()
    abstracts = papers_df['abstract_tokens'].tolist()
    return titles, abstracts

# ====================================================================================================
# BM25 FUNCTIONS
# ====================================================================================================
# Prepare data for BM25
def initialize_bm25(papers):
    bm25 = BM25Okapi(papers)
    return bm25

def run_bm25_query(bm25, text, text_type):
    if text_type == "target_title":
        tokens = tk.clean_and_tokenise(text, "title")
    else:
        tokens = tk.clean_and_tokenise(text, "abstract")

    scores = bm25.get_scores(tokens)

    return scores

def rank_papers(papers_df):
    ranked_papers_df = papers_df.sort_values(by="combined_score", ascending=False)
    return ranked_papers_df

# ====================================================================================================
# OLD MAIN FUNCTION (BEFORE CHUNKING)
# ====================================================================================================

# def run_bm25(db_client, raw_papers_df, target_ss_ids, title_weight, abstract_weight):
#     # Clean and tokenize the data
#     print("raw_papers_df: ", raw_papers_df.head(10))
#     papers_df = tk.tokenise_papers_df(raw_papers_df)
#     save_to_csv(papers_df, "tokenised_text", "tokenised_text")

#     # Preprocess the data
#     titles, abstracts = formatting_data(papers_df)
#     bm25_titles = initialize_bm25(titles)
#     bm25_abstracts = initialize_bm25(abstracts) 
#     overall_combined_scores = [0] * len(papers_df) # [0, 0, 0, ...]
#     target_papers = db_operations.get_papers_by_ss_ids(db_client, target_ss_ids) # [(ss_id, title, abstract), ...]
#     combined_scores_df = pd.DataFrame(index=papers_df['ss_id'], columns=[target[0] for target in target_papers])


#     for target_paper in target_papers:
#         # Run the query
#         title_title_scores = run_bm25_query(bm25_titles, target_paper[1], "target_title") # target title to other titles
#         title_abstract_scores = run_bm25_query(bm25_abstracts, target_paper[1], "target_title") # target title to other abstracts
#         abstract_title_scores = run_bm25_query(bm25_titles, target_paper[2], "target_abstract") # target abstract to other titles
#         abstract_abstract_scores = run_bm25_query(bm25_abstracts, target_paper[2], "target_abstract") # target abstract to other abstracts

#         # Combine scores for the current target paper
#         combined_scores = [
#             (title_weight * (title_title_score + abstract_title_score) + abstract_weight * (title_abstract_score + abstract_abstract_score))
#             for title_title_score, title_abstract_score, abstract_title_score, abstract_abstract_score in zip(title_title_scores, title_abstract_scores, abstract_title_scores, abstract_abstract_scores)
#         ]

#         combined_scores_df[target_paper[0]] = combined_scores
        
#     ranked_papers_with_scores = pd.DataFrame()

#     for target in combined_scores_df.columns:
#         temp_df = pd.DataFrame({
#             f"{target}_ss_id": combined_scores_df[target].sort_values(ascending=False).index,
#             f"{target}_score": combined_scores_df[target].sort_values(ascending=False).values
#         })
#         ranked_papers_with_scores = pd.concat([ranked_papers_with_scores, temp_df], axis=1)

#     save_to_csv(ranked_papers_with_scores, "ranked_papers_with_scores_bm2", "results")

#     return

