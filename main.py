from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
from logger.logger import Logger


from db.db_client import DBClient
from db import db_operations
import time

import algo_tokeniser as tk
# from algo_bm25 import run_bm25
from algo_citation_similarity import get_full_citation_similarity
from algo_graph_generator import GraphGenerator

from evaluator import Evaluator



def setup_db():
    config = load_yaml_config('config/config.yaml')
    rds_db = config['RDS_DB']
    
    # PostgreSQL database connection details
    psql_user = config['PSQL_USER'] if rds_db else config['LOCAL_PSQL_USER']
    psql_password = config['PSQL_PASSWORD'] if rds_db else config['LOCAL_PSQL_PASSWORD']
    psql_host = config['PSQL_HOST'] if rds_db else config['LOCAL_PSQL_HOST']
    psql_port = config['PSQL_PORT'] if rds_db else config['LOCAL_PSQL_PORT']
    psql_read_host = config['PSQL_READ_HOST'] if rds_db else config['LOCAL_PSQL_HOST']

    dbclient = DBClient("postgres", psql_user, psql_password, psql_host, psql_port)
    dbclient_read = DBClient("postgres", psql_user, psql_password, psql_read_host, psql_port)

    return dbclient, dbclient_read

   

def clean_citation_sim_results_for_eval(results):
    cleaned_results = []

    for result in results:
        cleaned_results.append([result[0], result[1], result[2]])

    return cleaned_results


def main():
    logger = Logger()
    dbclient, dbclient_read = setup_db()
    # data = db_operations.get_all_paper_ids(dbclient_read)    
    # raw_papers_df = load_dataframe_from_list(data, ["ss_id", "title", "abstract"])
    # save_to_csv(data_df, "raw_papers", "")

    # raw_papers_df = load_from_csv("raw_papers", "")
    target_ss_ids = load_json("ss_ids", "test_data")
    # target_ss_ids = load_json("ss_ids_large", "test_data")
    target_ss_id = target_ss_ids[0]

    chunk_size = 10000

    # ================== Citation Similarity ==================
    db_operations.create_citation_similarity_table(dbclient)
    get_full_citation_similarity(dbclient, chunk_size)

    # insert into db
    citation_similarities_df = load_from_csv(file_name="paper_paper_citation_similarity", folder_name="citation_similarity")
    citation_similarities = list(citation_similarities_df.itertuples(index=False, name=None))

    # time_start = time.time()

    # max_retries = 5
    # for attempt in range(max_retries):
    #     try:
    #         print(f"Attempt {attempt + 1} of batch insert")
    #         db_operations.batch_insert_citation_similarity(dbclient, logger, citation_similarities, chunk_size)
    #         break
    #     except Exception as e:
    #         print(f"Error during batch insert (attempt {attempt + 1}): {e}")
    #         logger.log_message(f"Error during batch insert (attempt {attempt + 1}): {e}")
    #         if attempt + 1 == max_retries:
    #             print("Max retries reached. Exiting.")
    #             logger.log_message("Max retries reached. Exiting.")
    #             return

    # time_end = time.time()
    # runtime = time_end - time_start
    # print(f"Runtime: {runtime}")
    # logger.log_message(f"Runtime: {runtime}")
    

    # ================== BM25 ==================
    title_weight = 0.6
    abstract_weight = 0.4

    # run_bm25(dbclient, raw_papers_df, chunk_size, target_ss_ids, title_weight, abstract_weight)

    # ================== Search ==================
    # returns a list of papers, topics, their details and scores as applicable
    # graph_generator = GraphGenerator(dbclient)

    # produce graph based on scibert (3 variations: 1. only title, 2. only abstract, 3. title + abstract)
    # top_topic_paper_edges_df, top_10_topics_df = graph_generator.generate_semantic_graph(target_ss_id)
    # semantic_results_df = top_topic_paper_edges_df[['ss_id', 'title', 'abstract']]
    # semantic_results = semantic_results_df.values.tolist() # [(ss_id, title, abstract), ...]

    # produce graph based on citation similarity
    # similar_papers_by_citations_df = graph_generator.generate_co_citation_graph(target_ss_id)
    # cs_results_df = similar_papers_by_citations_df[['ss_id', 'title', 'abstract']]
    # cs_results = cs_results_df.values.tolist() # [(ss_id, title, abstract), ...]



    # ================== Evaluation ==================
    # evaluator = Evaluator(dbclient)

    # evaluate using bm25
    # bm25_combined_scores = evaluator.run_bm25_eval(target_ss_id, semantic_results, cs_results)
    # print("target paper: ", target_ss_id)
    # print("bm25 semantic score: ", bm25_combined_scores['median_semantic_score'])
    # print("bm25 cs score: ", bm25_combined_scores['median_cs_score'])
    
    # evaluate using scibert
    # scibert_combined_scores = evaluator.run_bert_eval(target_ss_id, semantic_results, cs_results)
    # print("target paper: ", target_ss_id)
    # print("scibert semantic score: ", scibert_combined_scores['median_semantic_score'])
    # print("scibert cs score: ", scibert_combined_scores['median_cs_score'])

    # find whether there is beneficial new papers being discovered in semantic search
    # scibert_score, bm25_score = evaluator.new_papers_scoring(target_ss_id, semantic_results, cs_results)
    # print("target paper: ", target_ss_id)
    # print("median scibert score of new undiscovered papers: ", scibert_score)
    # print("median bm25 score of new undiscovered papers: ", bm25_score)

if __name__ == "__main__":
    main()