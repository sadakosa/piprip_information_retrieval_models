from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25
from tokeniser import tokenise_papers, clean_and_tokenise
from resources.objects.paper import Paper

from db.db_client import DBClient
from db import db_operations

from bm25 import run_bm25
from citation_similarity import get_citation_similarity

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
    # data = [
    #     ["ss_id_1", "Deep Learning Approaches in Natural Language Processing", 
    #     "Deep learning techniques have revolutionized natural language processing (NLP) by providing powerful models capable of handling complex language tasks. This paper reviews the state-of-the-art deep learning methods applied to NLP tasks such as text classification, sentiment analysis, and machine translation. Various architectures like RNNs, CNNs, and Transformers are discussed, highlighting their advantages and limitations."],
    #     ["ss_id_2", "The Impact of Climate Change on Global Agriculture", 
    #     "Climate change poses a significant threat to global agriculture by altering weather patterns, affecting crop yields, and increasing the frequency of extreme weather events. This study examines the potential impacts of climate change on agricultural productivity and proposes adaptive strategies to mitigate these effects. The role of sustainable agricultural practices and technological innovations in enhancing resilience to climate change is also explored."],
    #     ["ss_id_3", "Quantum Computing: Advances and Applications", 
    #     "Quantum computing is an emerging field that leverages the principles of quantum mechanics to perform computations far beyond the capabilities of classical computers. This paper provides an overview of recent advances in quantum computing, including developments in quantum algorithms, error correction, and hardware implementations. The potential applications of quantum computing in cryptography, material science, and complex system simulations are also discussed."],
    #     ["ss_id_4", "The Role of Microbiome in Human Health and Disease", 
    #     "The human microbiome, comprising trillions of microorganisms living in and on our bodies, plays a crucial role in maintaining health and influencing disease states. This review highlights the latest research on the composition and function of the human microbiome, its interactions with the host, and its impact on conditions such as obesity, diabetes, and inflammatory bowel disease. Potential therapeutic interventions targeting the microbiome are also considered."]
    # ]
    raw_papers = [Paper(ss_id, title, abstract) for ss_id, title, abstract in data]


    # ss_ids = load_json("ss_ids", "test_data")
    # for ss_id in ss_ids:
    #     get_citation_similarity(dbclient, ss_id)
    ss_id = "ss_id_1"
    get_citation_similarity(dbclient, ss_id)
    
    # query = "deep learning in healthcare"
    # title_weight = 0.6
    # abstract_weight = 0.4
    # run_bm25(raw_papers, query, title_weight, abstract_weight)

if __name__ == "__main__":
    main()