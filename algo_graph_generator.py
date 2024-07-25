from global_methods import save_to_csv, load_from_csv
from db import db_operations

def generate_graphs(dbclient, target_ss_ids):
    for target_ss_id in target_ss_ids:
        generate_graph(dbclient, target_ss_id)

def generate_graph(dbclient, target_ss_id):
    citation_similarity_df = load_from_csv("ranked_papers_with_scores_citation_similarity", "results")
    bm25_df = load_from_csv("ranked_papers_with_scores_bm25", "results")

    # get the top 10 papers from citation similarity and bm25 excluding the target paper
    citation_similarity_filtered = citation_similarity_df[citation_similarity_df[target_ss_id + "_ss_id"] != target_ss_id]
    citation_similarity_top_10 = citation_similarity_filtered.head(10)[target_ss_id + "_ss_id"].tolist()

    bm25_filtered = bm25_df[bm25_df[target_ss_id + "_ss_id"] != target_ss_id]
    bm25_top_10 = bm25_filtered.head(10)[target_ss_id + "_ss_id"].tolist()

    # print(f"Top 10 papers from citation similarity: {citation_similarity_top_10}")
    # print(f"Top 10 papers from bm25: {bm25_top_10}")

    # get the details of the top 10 papers from citation similarity and bm25 from the database
    citation_similarity_top_10_details = db_operations.get_papers_by_ss_ids(dbclient, citation_similarity_top_10)
    bm25_top_10_details = db_operations.get_papers_by_ss_ids(dbclient, bm25_top_10)

    # print(f"target paper details: {db_operations.get_papers_by_ss_ids(dbclient, [target_ss_id])}")
    # print(f"Top 10 papers from citation similarity details: {citation_similarity_top_10_details}")
    # print(f"Top 10 papers from bm25 details: {bm25_top_10_details}")

    # find if any of the ss_ids are in the top 10 of both citation similarity and bm25
    common_papers = set(citation_similarity_top_10).intersection(set(bm25_top_10))
    print(f"Common papers: {common_papers}")

    return 


    