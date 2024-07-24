from db import db_operations 
from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25

def get_citation_similarity(db_client, ss_id):
    # from db, returns [('ss_id', 'clean_title', 'abstract', similar_references=76), ...]
    # similar_papers_by_references = db_operations.get_bibliographic_couples(db_client, ss_id)
    # similar_papers_by_citations = db_operations.get_co_citation(db_client, ss_id)
    co_citation_matrix = db_operations.get_all_co_citations(db_client)
    bibliographic_coupling_matrix = db_operations.get_all_bibliographic_couples(db_client, ss_id)

    save_to_json(co_citation_matrix, "co_citation_matrix", "citation_similarity")
    save_to_json(bibliographic_coupling_matrix, "bibliographic_coupling_matrix", "citation_similarity")

    # citation_similarity

    return 


def citation_similarity_ranking(papers):
    return ranked_papers

