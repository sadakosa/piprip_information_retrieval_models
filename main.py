from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25
from bm25 import initialize_bm25, run_bm25_query # Importing all the functions from bm25.py
from tokeniser import tokenise_papers, clean_and_tokenise

from db.db_client import DBClient
from db import db_operations


# main function

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


def formatting_data(papers):
    titles = [paper['title_tokens'] for paper in papers]
    abstracts = [paper['abstract_tokens'] for paper in papers]
    papers_formatted = titles + abstracts
    return papers_formatted


def main():
    # Setup the database
    dbclient, dbclient_read = setup_db()

    # Load the data
    data = db_operations.get_all_paper_ids(dbclient_read)
    # print(data[0])

    # Clean and tokenize the data
    papers = tokenise_papers(data)
    save_to_json(papers, "test", "tokenised_text")

    # Preprocess the data
    papers_formatted = formatting_data(papers)
    bm25 = initialize_bm25(papers_formatted)
    # save_bm25(bm25, "bm25")

    # run the query
    query = "deep learning in healthcare"
    scores = run_bm25_query(bm25, query) # scores = [0.14708367, 0.14708367, 0.08984029, 0.11154667]
    results = [{"paper": paper, "score": score} for paper, score in zip(papers_formatted, scores)]

    # # Print results
    # for result in results:
    #     print(f"Paper: {result['paper']}, Score: {result['score']}")
    
    # Save the data
    save_to_json(results, "results_one", "results")

    return

if __name__ == "__main__":
    main()