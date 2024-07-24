from db import db_operations 

def citation_similarity(db_client, ss_id):
    bc_result = bibliographic_coupling(db_client, ss_id)
    cc_result = co_citation(db_client, ss_id)

    return 
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

def citation_similarity_ranking(papers):
    return ranked_papers

def bibliographic_coupling():
    # get a list of papers that have the same references as this one [[ss_id, clean_title, clean_abstract], [ss_id, clean_title, clean_abstract], ...]


    pass

def co_citation():
    # get a list of papers that have the same papers citing it as this one [[ss_id, clean_title, clean_abstract], [ss_id, clean_title, clean_abstract], ...]
    pass