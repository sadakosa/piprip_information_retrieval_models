from db import db_operations 
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
import pandas as pd


# ====================================================================================================
# Full Citation Similarity
# ====================================================================================================
def get_full_citation_similarity(db_client, ss_id):
    # get all co-citations and bibliographic couples and save them to csv
    # co_citation_df = load_dataframe_from_list(db_operations.get_all_co_citations(db_client), ["paper1", "paper2", "co_citation_count"])
    # bibliographic_coupling_df = load_dataframe_from_list(db_operations.get_all_bibliographic_couples(db_client, ss_id), ["paper1", "paper2", "coupling_count"])
  
    # save_to_csv(co_citation_df, "co_citation_df", "citation_similarity")
    # save_to_csv(bibliographic_coupling_df, "bibliographic_coupling_df", "citation_similarity")

    # load the co-citations and bibliographic couples from csv and calculate the full citation similarity
    co_citation_df = load_from_csv("co_citation_df", "citation_similarity")
    bibliographic_coupling_df = load_from_csv("bibliographic_coupling_df", "citation_similarity")

    # print("duplicates: ", check_for_duplicates(co_citation_df))
    # print("duplicates: ", check_for_duplicates(bibliographic_coupling_df))

    related_papers = full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, ss_id)

    return 

def full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, ss_id):
    combined_df = pd.merge(co_citation_df, bibliographic_coupling_df, on=["paper1", "paper2"], how="outer").fillna(0)
    combined_df["combined_score"] = combined_df["co_citation_count"] + combined_df["coupling_count"] # Calculate the combined score

    related_papers = combined_df[(combined_df["paper1"] == ss_id) | (combined_df["paper2"] == ss_id)] # Get the related papers
    related_papers = related_papers.sort_values(by="combined_score", ascending=False) # Sort the related papers by combined score

    # Print the related papers and their combined scores
    # print(f"Related papers to {ss_id}:")
    for index, row in related_papers.iterrows():
        related_paper = row["paper2"] if row["paper1"] == ss_id else row["paper1"]
        # print(f"Paper {related_paper} with combined score {row['combined_score']}")

    # save_to_csv(related_papers, f"related_papers_{ss_id}", "citation_similarity")
    
    return related_papers

def check_for_duplicates(df):
    pairs_set = set()
    duplicates = []

    for index, row in df.iterrows():
        paper1, paper2 = row['paper1'], row['paper2']
        pair = tuple(sorted((paper1, paper2)))  # Sort the pair to ensure consistent ordering
        if pair in pairs_set:
            duplicates.append(pair)
        else:
            pairs_set.add(pair)

    return duplicates

# ====================================================================================================
# In Moment Citation Similarity
# ====================================================================================================

def get_inmoment_citation_similarity(db_client, ss_id):
    # from db, returns [('ss_id', 'clean_title', 'abstract', similar_references=76), ...]
    similar_papers_by_references_df = load_dataframe_from_list(db_operations.get_bibliographic_couples(db_client, ss_id), ['ss_id', 'title', 'abstract', 'num_similar_references'])
    similar_papers_by_citations_df = load_dataframe_from_list(db_operations.get_co_citation(db_client, ss_id), ['ss_id', 'title', 'abstract', 'num_similar_citations'])
  
    save_to_csv(similar_papers_by_references_df, "similar_papers_by_references_df", "citation_similarity")
    save_to_csv(similar_papers_by_citations_df, "similar_papers_by_citations_df", "citation_similarity")

    # similar_papers_by_references_df = load_from_csv("similar_papers_by_references_df", "citation_similarity")
    # similar_papers_by_citations_df = load_from_csv("similar_papers_by_citations_df", "citation_similarity")

    # # citation_similarity
    # full_citation_similarity_ranking(co_citation_df, bibliographic_coupling_df, ss_id)

    return 

def find_similar_papers_by_co_citation(target_paper, co_citation_data):
    similar_papers = []
    for paper1, paper2, count in co_citation_data:
        if paper1 == target_paper:
            similar_papers.append((paper2, count))
        elif paper2 == target_paper:
            similar_papers.append((paper1, count))
    return similar_papers

def find_similar_papers_by_bibliographic_coupling(target_paper, bibliographic_couples_data):
    similar_papers = []
    for paper1, paper2, count in bibliographic_couples_data:
        if paper1 == target_paper:
            similar_papers.append((paper2, count))
        elif paper2 == target_paper:
            similar_papers.append((paper1, count))
    return similar_papers