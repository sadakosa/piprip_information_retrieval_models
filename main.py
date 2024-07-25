from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
from logger.logger import Logger


from db.db_client import DBClient
from db import db_operations
import time

import algo_tokeniser as tk
from algo_bm25 import run_bm25
from algo_citation_similarity import get_full_citation_similarity
from algo_graph_generator import generate_graphs




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




def main():
    logger = Logger()
    dbclient, dbclient_read = setup_db()
    # data = db_operations.get_all_paper_ids(dbclient_read)    
    # raw_papers_df = load_dataframe_from_list(data, ["ss_id", "title", "abstract"])
    # save_to_csv(data_df, "raw_papers", "")

    # raw_papers_df = load_from_csv("raw_papers", "")
    # target_ss_ids = load_json("ss_ids", "test_data")
    # target_ss_ids = load_json("ss_ids_large", "test_data")
    # target_ss_id = target_ss_ids[0]

    chunk_size = 10000

    # ================== Citation Similarity ==================
    # db_operations.create_citation_similarity_table(dbclient)
    # get_full_citation_similarity(dbclient, chunk_size)

    citation_similarities_df = load_from_csv(file_name="paper_paper_citation_similarity", folder_name="citation_similarity")
    citation_similarities_df['co_citation_score'] = citation_similarities_df['co_citation_score'].astype(int)
    citation_similarities_df['coupling_score'] = citation_similarities_df['coupling_score'].astype(int)
    citation_similarities = list(citation_similarities_df.itertuples(index=False, name=None))
    # print(citation_similarities)
    print(type(citation_similarities))
    print(len(citation_similarities))

    # print("Sample of citation_similarities after conversion to tuples:", citation_similarities[:5])

    time_start = time.time()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1} of batch insert")
            db_operations.batch_insert_citation_similarity(dbclient, logger, citation_similarities, chunk_size)
            break
        except Exception as e:
            print(f"Error during batch insert (attempt {attempt + 1}): {e}")
            logger.log_message(f"Error during batch insert (attempt {attempt + 1}): {e}")
            if attempt + 1 == max_retries:
                print("Max retries reached. Exiting.")
                logger.log_message("Max retries reached. Exiting.")
                return

    time_end = time.time()
    runtime = time_end - time_start
    print(f"Runtime: {runtime}")
    logger.log_message(f"Runtime: {runtime}")
    # get_inmoment_citation_similarity(dbclient, ss_id)
    

    # ================== BM25 ==================
    title_weight = 0.6
    abstract_weight = 0.4

    # run_bm25(dbclient, raw_papers_df, chunk_size, target_ss_ids, title_weight, abstract_weight)

    # ================== Produce Graph ==================
    # produce graph based on citation similarity and bm25 results
    # produce graph based on citation similarity 
    # generate_graphs(dbclient, target_ss_ids)

    # ================== Evaluation ==================
    


if __name__ == "__main__":
    main()