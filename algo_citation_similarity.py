from db import db_operations
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
import pandas as pd
import time
import numpy as np

# ====================================================================================================
# Full Citation Similarity
# ====================================================================================================


# def get_full_citation_similarity(db_client, target_ss_ids, chunk_size):
#     # Get all co-citations and bibliographic couples
#     co_citation_data = db_operations.get_all_co_citations(db_client)
#     bibliographic_coupling_data = db_operations.get_all_bibliographic_couples(db_client)

#     # Load data into DataFrames
#     co_citation_df = load_dataframe_from_list(co_citation_data, ["paper1", "paper2", "co_citation_count"])
#     bibliographic_coupling_df = load_dataframe_from_list(bibliographic_coupling_data, ["paper1", "paper2", "coupling_count"])

#     all_related_papers_list = []

#     # Process data in chunks
#     for start in range(0, len(target_ss_ids), chunk_size):
#         chunk_ss_ids = target_ss_ids[start:start + chunk_size]

#         for ss_id in chunk_ss_ids:
#             related_papers_df = full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, ss_id)
            
#             ss_id_column = f"{ss_id}_ss_id"
#             score_column = f"{ss_id}_score"
#             related_papers_df.rename(columns={"ss_id": ss_id_column, "combined_score": score_column}, inplace=True)

#             related_papers_df.reset_index(drop=True, inplace=True)  # Reset index to ensure proper alignment during concatenation
#             all_related_papers_list.append(related_papers_df)

#     all_related_papers_df = pd.concat(all_related_papers_list, axis=1)
#     save_to_csv(all_related_papers_df, "ranked_papers_with_scores_citation_similarity", "results")

#     return all_related_papers_df



# def full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, ss_id):
#     combined_df = pd.merge(co_citation_df, bibliographic_coupling_df, on=["paper1", "paper2"], how="outer").fillna(0)
#     combined_df["combined_score"] = combined_df["co_citation_count"] + combined_df["coupling_count"]  # Calculate the combined score

#     related_papers = combined_df[(combined_df["paper1"] == ss_id) | (combined_df["paper2"] == ss_id)]  # Get the related papers
#     related_papers = related_papers.sort_values(by="combined_score", ascending=False)  # Sort the related papers by combined score

#     related_papers_df = pd.DataFrame({
#         "ss_id": related_papers.apply(lambda row: row["paper2"] if row["paper1"] == ss_id else row["paper1"], axis=1),
#         "combined_score": related_papers["combined_score"]
#     })

#     return related_papers_df






# def get_full_citation_similarity(db_client, chunk_size):

#     # Get all co-citations and bibliographic couples
#     co_citation_data = db_operations.get_all_co_citations(db_client)
#     bibliographic_coupling_data = db_operations.get_all_bibliographic_couples(db_client)

#     # Load data into DataFrames
#     co_citation_df = load_dataframe_from_list(co_citation_data, ["paper1", "paper2", "co_citation_count"])
#     bibliographic_coupling_df = load_dataframe_from_list(bibliographic_coupling_data, ["paper1", "paper2", "bibliographic_coupling_count"])

#     all_paper_ids = db_operations.get_all_paper_ids(db_client)
#     all_related_papers_list = []

#     # Process data in chunks
#     for start in range(0, len(all_paper_ids), chunk_size):
#         chunk_paper_ids = all_paper_ids[start:start + chunk_size]

#         for paper_id in chunk_paper_ids:
#             related_papers_df = full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, paper_id)
            
#             paper_id_column = f"{paper_id}_paper_id"
#             score_column = f"{paper_id}_score"
#             related_papers_df.rename(columns={"paper_id": paper_id_column, "combined_score": score_column}, inplace=True)

#             related_papers_df.reset_index(drop=True, inplace=True)  # Reset index to ensure proper alignment during concatenation
#             all_related_papers_list.append(related_papers_df)

#     all_related_papers_df = pd.concat(all_related_papers_list, axis=1)
#     save_to_csv(all_related_papers_df, "ranked_papers_with_scores_citation_similarity", "results")

#     return all_related_papers_df


# def full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, paper_id):
#     combined_df = pd.merge(co_citation_df, bibliographic_coupling_df, on=["paper1", "paper2"], how="outer").fillna(0)
#     combined_df["combined_score"] = combined_df["co_citation_count"] + combined_df["coupling_count"]  # Calculate the combined score

#     related_papers = combined_df[(combined_df["paper1"] == paper_id) | (combined_df["paper2"] == paper_id)]  # Get the related papers
#     related_papers = related_papers.sort_values(by="combined_score", ascending=False)  # Sort the related papers by combined score

#     related_papers_df = pd.DataFrame({
#         "paper_id": related_papers.apply(lambda row: row["paper2"] if row["paper1"] == paper_id else row["paper1"], axis=1),
#         "combined_score": related_papers["combined_score"]
#     })

#     return related_papers_df

















def get_full_citation_similarity(db_client, chunk_size):
    time_start = time.time()
    # Get all co-citations and bibliographic couples
    co_citation_data = db_operations.get_all_co_citations(db_client)
    bibliographic_coupling_data = db_operations.get_all_bibliographic_couples(db_client)

    co_citation_df = load_dataframe_from_list(co_citation_data, ["paper1", "paper2", "co_citation_count"])
    bibliographic_coupling_df = load_dataframe_from_list(bibliographic_coupling_data, ["paper1", "paper2", "coupling_count"])

    finding_similarities_runtime = time.time() - time_start
    print(f"Finding similarities runtime: {finding_similarities_runtime}")

    # Sort paper IDs in each row to ensure uniqueness
    co_citation_df[['paper1', 'paper2']] = pd.DataFrame(np.sort(co_citation_df[['paper1', 'paper2']], axis=1), index=co_citation_df.index)
    bibliographic_coupling_df[['paper1', 'paper2']] = pd.DataFrame(np.sort(bibliographic_coupling_df[['paper1', 'paper2']], axis=1), index=bibliographic_coupling_df.index)
    co_citation_df = co_citation_df.drop_duplicates(subset=['paper1', 'paper2'])
    bibliographic_coupling_df = bibliographic_coupling_df.drop_duplicates(subset=['paper1', 'paper2'])

    # Save to CSV
    combined_df = pd.merge(co_citation_df, bibliographic_coupling_df, on=["paper1", "paper2"], how="outer").fillna(0)
    combined_df.rename(columns={"paper1": "paper_id_one", "paper2": "paper_id_two", "co_citation_count": "co_citation_score", "coupling_count": "coupling_score"}, inplace=True)
    
    print("len(combined_df):", len(combined_df))
    save_to_csv(combined_df, "paper_paper_citation_similarity", "citation_similarity")

    save_to_csv_run_time = time.time() - time_start
    print(f"Save to CSV runtime: {save_to_csv_run_time}")

    return combined_df









# ====================================================================================================
# In Moment Citation Similarity
# ====================================================================================================
# def get_inmoment_citation_similarity(db_client, ss_ids):
#     # Initialize empty DataFrames to store results
#     all_similar_papers_by_references_df = pd.DataFrame()
#     all_similar_papers_by_citations_df = pd.DataFrame()

#     for ss_id in ss_ids:
#         # Get similar papers by references and citations
#         similar_papers_by_references_df = load_dataframe_from_list(db_operations.get_bibliographic_couples(db_client, ss_id), ['ss_id', 'title', 'abstract', 'num_similar_references'])
#         similar_papers_by_citations_df = load_dataframe_from_list(db_operations.get_co_citation(db_client, ss_id), ['ss_id', 'title', 'abstract', 'num_similar_citations'])

#         # Append the results to the main DataFrames
#         similar_papers_by_references_df['target_ss_id'] = ss_id
#         similar_papers_by_citations_df['target_ss_id'] = ss_id

#         all_similar_papers_by_references_df = pd.concat([all_similar_papers_by_references_df, similar_papers_by_references_df], ignore_index=True)
#         all_similar_papers_by_citations_df = pd.concat([all_similar_papers_by_citations_df, similar_papers_by_citations_df], ignore_index=True)

#     save_to_csv(all_similar_papers_by_references_df, "all_similar_papers_by_references", "citation_similarity")
#     save_to_csv(all_similar_papers_by_citations_df, "all_similar_papers_by_citations", "citation_similarity")

#     return all_similar_papers_by_references_df, all_similar_papers_by_citations_df

# def find_similar_papers_by_co_citation(target_paper, co_citation_data):
#     similar_papers = []
#     for paper1, paper2, count in co_citation_data:
#         if paper1 == target_paper:
#             similar_papers.append((paper2, count))
#         elif paper2 == target_paper:
#             similar_papers.append((paper1, count))
#     return similar_papers

# def find_similar_papers_by_bibliographic_coupling(target_paper, bibliographic_couples_data):
    similar_papers = []
    for paper1, paper2, count in bibliographic_couples_data:
        if paper1 == target_paper:
            similar_papers.append((paper2, count))
        elif paper2 == target_paper:
            similar_papers.append((paper1, count))
    return similar_papers


# ====================================================================================================
# OLD MAIN FUNCTION (BEFORE CHUNKING)
# =================================================================================================

# def get_full_citation_similarity(db_client, target_ss_ids):
#     # Get all co-citations and bibliographic couples
#     co_citation_df = load_dataframe_from_list(db_operations.get_all_co_citations(db_client), ["paper1", "paper2", "co_citation_count"])
#     bibliographic_coupling_df = load_dataframe_from_list(db_operations.get_all_bibliographic_couples(db_client), ["paper1", "paper2", "coupling_count"])

#     all_related_papers_list = []

#     for ss_id in target_ss_ids:
#         related_papers_df = full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, ss_id)
        
#         ss_id_column = f"{ss_id}_ss_id"
#         score_column = f"{ss_id}_score"
#         related_papers_df.rename(columns={"ss_id": ss_id_column, "combined_score": score_column}, inplace=True)

#         related_papers_df.reset_index(drop=True, inplace=True) # Reset index to ensure proper alignment during concatenation
#         all_related_papers_list.append(related_papers_df)

#     all_related_papers_df = pd.concat(all_related_papers_list, axis=1)
#     save_to_csv(all_related_papers_df, "ranked_papers_with_scores_citation_similarity", "results")

#     return
