from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
import tokeniser as tk
from resources.objects.paper import Paper

from db.db_client import DBClient
from db import db_operations

from bm25 import run_bm25
from citation_similarity import get_full_citation_similarity, get_inmoment_citation_similarity

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
    dbclient, dbclient_read = setup_db()
    data = db_operations.get_all_paper_ids(dbclient_read)    
    data_df = load_dataframe_from_list(data, ["ss_id", "title", "abstract"])
    save_to_csv(data_df, "raw_papers", "")

    data_df = load_from_csv("raw_papers", "")
    ss_ids = load_json("ss_ids", "test_data")

    # ================== Citation Similarity ==================
    # get_full_citation_similarity(dbclient, ss_id)
    # get_inmoment_citation_similarity(dbclient, ss_id)
    

    # ================== BM25 ==================
    query = "deep learning in healthcare"
    title_weight = 0.6
    abstract_weight = 0.4
    # run_bm25(raw_papers, queries, title_weight, abstract_weight)
    import bm25_test 
    # bm25_test.run_bm25(raw_papers, query, title_weight, abstract_weight)
    bm25_test.run_bm25(dbclient, data_df, ss_ids, title_weight, abstract_weight)

if __name__ == "__main__":
    main()