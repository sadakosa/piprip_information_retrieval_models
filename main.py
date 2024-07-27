from global_methods import load_yaml_config, save_to_json, load_json, save_bm25, load_bm25
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
from logger.logger import Logger


from db.db_client import DBClient
from db import db_operations
from psycopg2.extras import execute_batch

import time

import numpy as np

import algo_tokeniser as tk
# from algo_bm25 import run_bm25
from algo_citation_similarity import get_full_citation_similarity
from algo_graph_generator import GraphGenerator

from evaluator import Evaluator
from algo_colbert import ColBERT



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


def clean_category_results(target_ss_id, bm25_results, scibert_results, category, have_cs):

    # tested_paper TEXT NOT NULL,
    # 25p_score_bm25 NUMERIC NOT NULL,
    # 50p_score_bm25 NUMERIC NOT NULL,
    # 75p_score_bm25 NUMERIC NOT NULL,,
    # max_score_bm25 NUMERIC NOT NULL,
    # 25p_score_scibert NUMERIC NOT NULL,
    # 50p_score_scibert NUMERIC NOT NULL,
    # 75p_score_scibert NUMERIC NOT NULL,
    # max_score_scibert NUMERIC NOT NULL,
    # category TEXT NOT NULL,
    if have_cs:
        cs_array = [
            bm25_results["25p_cs_score"],
            bm25_results["50p_cs_score"],
            bm25_results["75p_cs_score"],
            bm25_results["max_cs_score"],
            scibert_results["25p_cs_score"],
            scibert_results["50p_cs_score"],
            scibert_results["75p_cs_score"],
            scibert_results["max_cs_score"],
            category
        ]
    else:
        cs_array = None
    

    semantic_array = [
        target_ss_id,
        bm25_results["25p_semantic_score"],
        bm25_results["50p_semantic_score"],
        bm25_results["75p_semantic_score"],
        bm25_results["max_semantic_score"],
        scibert_results["25p_semantic_score"],
        scibert_results["50p_semantic_score"],
        scibert_results["75p_semantic_score"],
        scibert_results["max_semantic_score"],
        category
    ]

    return semantic_array, cs_array







def search_and_evaluate(dbclient, dbclient_read, logger, target_ss_id):
    results_arr = []
    # ================== BM25 ==================
    title_weight = 0.6
    abstract_weight = 0.4

    # run_bm25(dbclient, raw_papers_df, chunk_size, target_ss_ids, title_weight, abstract_weight)

    # ================== Search ==================
    # returns a list of papers, topics, their details and scores as applicable
    graph_generator = GraphGenerator(dbclient)

    # produce graph based on scibert (3 variations: 1. only title, 2. only abstract, 3. title + abstract)
    top_topic_paper_edges_df, top_10_topics_df = graph_generator.generate_semantic_graph(target_ss_id)
    semantic_results_df = top_topic_paper_edges_df[['ss_id', 'title', 'abstract']]
    semantic_results = semantic_results_df.values.tolist() # [(ss_id, title, abstract), ...]

    # produce graph based on citation similarity
    similar_papers_by_citations_df = graph_generator.generate_co_citation_graph(target_ss_id)
    cs_results_df = similar_papers_by_citations_df[['ss_id', 'title', 'abstract']]
    cs_results = cs_results_df.values.tolist() # [(ss_id, title, abstract), ...]

    # print("semantic_results: ", semantic_results)
    # print("cs_results: ", cs_results)

    # cs_ss_ids = [result[0] for result in cs_results]
    # semantic_ss_ids = [result[0] for result in semantic_results]


    # ================== Evaluation ==================
    colbert = ColBERT(logger, "scibert")
    evaluator = Evaluator(dbclient, logger, colbert)
    # evaluate using bm25
    bm25_title_scores = evaluator.run_bm25_eval(target_ss_id, semantic_results, cs_results, "title", 1)
    scibert_title_scores = evaluator.run_bert_eval(target_ss_id, semantic_results, cs_results, "title", 1)
    bm25_full_text_scores = evaluator.run_bm25_eval(target_ss_id, semantic_results, cs_results, "full_text", 2)
    scibert_full_text_scores = evaluator.run_bert_eval(target_ss_id, semantic_results, cs_results, "full_text", 2)

    common_papers = []
    novel_papers = []
    for idx, result in enumerate(semantic_results):
        is_added = False
        for result_cs in cs_results:
            if result[0] == result_cs[0]:
                common_papers.append(idx)
                is_added = True
                break
        if not is_added:
            novel_papers.append(idx)

    template_vals = {'25p_semantic_score': 0, '25p_cs_score': 0, '50p_semantic_score': 0, '50p_cs_score': 0, '75p_semantic_score': 0, '75p_cs_score': 0, 'max_semantic_score': 0, 'max_cs_score': 0, 'target_paper': ''}
    bm25_full_text_scores_novel = template_vals
    scibert_full_text_scores_novel = template_vals
    bm25_full_text_scores_common = template_vals
    scibert_full_text_scores_common = template_vals
    if len(novel_papers) != 0:
        bm25_full_text_scores_novel = gen_percentile([bm25_full_text_scores["semantic_scores"][idx] for idx in novel_papers], target_ss_id)
        scibert_full_text_scores_novel = gen_percentile([scibert_full_text_scores["semantic_scores"][idx] for idx in novel_papers], target_ss_id)
    if len(common_papers) != 0:
        bm25_full_text_scores_common = gen_percentile([bm25_full_text_scores["semantic_scores"][idx] for idx in common_papers], target_ss_id)
        scibert_full_text_scores_common = gen_percentile([scibert_full_text_scores["semantic_scores"][idx] for idx in common_papers], target_ss_id)

    # print("bm25_title_scores: ", bm25_title_scores, "\n")
    # print("scibert_title_scores: ", scibert_title_scores, "\n")
    # print("bm25_full_text_scores: ", bm25_full_text_scores, "\n")
    # print("scibert_full_text_scores: ", scibert_full_text_scores, "\n")

    # print("bm25_full_text_scores_novel: ", bm25_full_text_scores_novel, "\n")
    # print("scibert_full_text_scores_novel: ", scibert_full_text_scores_novel, "\n")
    # print("bm25_full_text_scores_common: ", bm25_full_text_scores_common, "\n")
    # print("scibert_full_text_scores_common: ", scibert_full_text_scores_common, "\n")
    return {
        'bm25_title_scores': bm25_title_scores,
        'scibert_title_scores': scibert_title_scores,
        'bm25_full_text_scores': bm25_full_text_scores,
        'scibert_full_text_scores': scibert_full_text_scores,
        'bm25_full_text_scores_novel': bm25_full_text_scores_novel,
        'scibert_full_text_scores_novel': scibert_full_text_scores_novel,
        'bm25_full_text_scores_common': bm25_full_text_scores_common,
        'scibert_full_text_scores_common': scibert_full_text_scores_common
    }

    # semantic_array, cs_array = clean_category_results(target_ss_id, bm25_title_scores, scibert_title_scores, 1, have_cs=True)
    # if cs_array is not None:
    #     results_arr.append(cs_array)
    # results_arr.append(semantic_array)

    # semantic_array, cs_array = clean_category_results(target_ss_id, bm25_title_scores, scibert_title_scores, 2, have_cs=True)
    # if cs_array is not None:
    #     results_arr.append(cs_array)
    # results_arr.append(semantic_array)

    # # find whether there is beneficial new papers being discovered in semantic search
    # semantic_ss_ids = [result[0] for result in semantic_results]
    # cs_ss_ids = [result[0] for result in cs_results]
    # novel_papers = list(set(semantic_ss_ids) - set(cs_ss_ids))
    # non_novel_papers = list(set(semantic_ss_ids) & set(cs_ss_ids))
    # p_uncommon_papers = len(novel_papers) / (len(semantic_ss_ids))
    # p_common_papers = len(novel_papers) / (len(cs_ss_ids))
    # print(f"% common papers: {p_uncommon_papers}")
    # print(f"len uncommon papers: {len(novel_papers)}")
    # print(f"len common papers: {len(non_novel_papers)}")
    # print(f"novel_papers: {novel_papers}")
    # print(f"non_novel_papers: {non_novel_papers}")
   

    # if len(novel_papers) == 0:
    #     bm25_full_text_scores_non_novel = evaluator.run_bm25_eval_novel(target_ss_id, non_novel_papers, "full_text", 4)
    #     scibert_full_text_scores_non_novel = evaluator.run_bert_eval_novel(target_ss_id, non_novel_papers, "full_text", 4)    

    #     if len(bm25_full_text_scores_non_novel) == 0 or len(scibert_full_text_scores_non_novel) == 0:
    #         print(f"No results found for paper: {target_ss_id}, skipping paper")
    #         logger.log_message(f"No results found for paper: {target_ss_id}, skipping paper")
    #         return None

    #     semantic_array, cs_array = clean_category_results(target_ss_id, bm25_full_text_scores_non_novel, scibert_full_text_scores_non_novel, 4, have_cs=False)
    #     if cs_array is not None:
    #         results_arr.append(cs_array)
    #     results_arr.append(semantic_array)

    # if len(non_novel_papers) == 0:
    #     bm25_full_text_scores_novel = evaluator.run_bm25_eval_novel(target_ss_id, novel_papers, "full_text", 3)
    #     scibert_full_text_scores_novel = evaluator.run_bert_eval_novel(target_ss_id, novel_papers, "full_text", 3)

    #     if len(bm25_full_text_scores_novel) == 0 or len(scibert_full_text_scores_novel) == 0:
    #         print(f"No results found for paper: {target_ss_id}, skipping paper")
    #         logger.log_message(f"No results found for paper: {target_ss_id}, skipping paper")
    #         return None
        
    #     semantic_array, cs_array = clean_category_results(target_ss_id, bm25_full_text_scores_novel, scibert_full_text_scores_novel, 3, have_cs=False)
    #     if cs_array is not None:
    #         results_arr.append(cs_array)
    #     results_arr.append(semantic_array)

    # bm25_full_text_scores_non_novel = evaluator.run_bm25_eval_novel(target_ss_id, non_novel_papers, "full_text", 4)
    # scibert_full_text_scores_non_novel = evaluator.run_bert_eval_novel(target_ss_id, non_novel_papers, "full_text", 4)
    # bm25_full_text_scores_novel = evaluator.run_bm25_eval_novel(target_ss_id, novel_papers, "full_text", 3)
    # scibert_full_text_scores_novel = evaluator.run_bert_eval_novel(target_ss_id, novel_papers, "full_text", 3)

    # if len(bm25_full_text_scores_novel) == 0 or len(scibert_full_text_scores_novel) == 0:
    #         print(f"No results found for paper: {target_ss_id}, skipping paper")
    #         logger.log_message(f"No results found for paper: {target_ss_id}, skipping paper")
    #         return None
    # if len(bm25_full_text_scores_non_novel) == 0 or len(scibert_full_text_scores_non_novel) == 0:
    #     print(f"No results found for paper: {target_ss_id}, skipping paper")
    #     logger.log_message(f"No results found for paper: {target_ss_id}, skipping paper")
    #     return None

    # semantic_array, cs_array = clean_category_results(target_ss_id, bm25_full_text_scores_non_novel, scibert_full_text_scores_non_novel, 4, have_cs=False)
    # if cs_array is not None:
    #     results_arr.append(cs_array)
    # results_arr.append(semantic_array)
    # semantic_array, cs_array = clean_category_results(target_ss_id, bm25_full_text_scores_novel, scibert_full_text_scores_novel, 3, have_cs=False)
    # if cs_array is not None:
    #     results_arr.append(cs_array)
    # results_arr.append(semantic_array)

    
    


    # scibert_score, bm25_score = evaluator.new_papers_scoring(target_ss_id, semantic_results, cs_results)
    # print("target paper: ", target_ss_id)
    # print("median scibert score of new undiscovered papers: ", scibert_score)
    # print("median bm25 score of new undiscovered papers: ", bm25_score)

    # find whether the non-novel papers are beneficial in semantic search
    # scibert_score, bm25_score = evaluator.new_papers_scoring(target_ss_id, semantic_results, cs_results)
    # print("target paper: ", target_ss_id)
    # print("median scibert score of new undiscovered papers: ", scibert_score)
    # print("median bm25 score of new undiscovered papers: ", bm25_score)

    # db_operations.create_tested_papers_table(dbclient)
    # db_operations.insert_tested_paper(dbclient, target_ss_id, bm25_combined_scores['average_semantic_score'], bm25_combined_scores['average_cs_score'], bm25_combined_scores['25%_semantic_score'], bm25_combined_scores['25%_cs_score'], bm25_combined_scores['50%_semantic_score'], bm25_combined_scores['50%_cs_score'], bm25_combined_scores['75%_semantic_score'], bm25_combined_scores['75%_cs_score'])   


def gen_percentile(semantic_scores, target_ss_id):
    return {
        "semantic_scores": semantic_scores,
        "25p_semantic_score": np.percentile(semantic_scores, 25),
        "25p_cs_score": 0,
        "50p_semantic_score": np.percentile(semantic_scores, 50),
        "50p_cs_score": 0,
        "75p_semantic_score": np.percentile(semantic_scores, 75),
        "75p_cs_score": 0,
        "max_semantic_score": max(semantic_scores),
        "max_cs_score": 0,
        "target_paper": target_ss_id
    }



def main():
    logger = Logger()
    dbclient, dbclient_read = setup_db()
    # data = db_operations.get_all_paper_ids(dbclient_read)    
    # raw_papers_df = load_dataframe_from_list(data, ["ss_id", "title", "abstract"])
    # save_to_csv(data_df, "raw_papers", "")

    # raw_papers_df = load_from_csv("raw_papers", "")

    while True:
        select_query = """
        SELECT ss_id, clean_title, clean_abstract FROM papers
        WHERE is_cleaned = True
        AND is_tested = False
        ORDER BY RANDOM()
        LIMIT 100;
        """

        update_query = """
        UPDATE papers
        SET is_tested = True
        WHERE ss_id = %s;
        """
        cursor = dbclient.execute(select_query)
        target_papers = dbclient.cur.fetchall()
        target_ss_ids = [target_paper[0] for target_paper in target_papers]
        # print("target_ss_ids: ", target_ss_ids)

        for paper in target_ss_ids:
            ss_id = paper[0]  # Assuming ss_id is the first column
            dbclient.cur.execute(update_query, (ss_id,))
        dbclient.conn.commit()

        if len(target_ss_ids) == 0:
            print("No papers to test. Exiting.")
            logger.log_message("No papers to test. Exiting.")
            break
        
        chunk_size = 50
        
        results = []
        latest_chunk = 0
        latest_result = 0
        for idx_ssid, target_ss_id in enumerate(target_ss_ids):
            try:
                # print("Processing target paper: ", target_ss_id)
                logger.log_message(f"Processing target paper: {target_ss_id}")
                temp_similarities = search_and_evaluate(dbclient, dbclient_read, logger, target_ss_id)
                # combine idx 0, 1
                score_keys = [
                    ['bm25_title_scores', 'scibert_title_scores', 1],
                    ['bm25_full_text_scores', 'scibert_full_text_scores', 2],
                    ['bm25_full_text_scores_novel', 'scibert_full_text_scores_novel', 3],
                    ['bm25_full_text_scores_common', 'scibert_full_text_scores_common', 4]
                ]
                sample_results = []
                is_failed = False
                for score_key in score_keys:
                    scores1 = score_key[0]
                    scores2 = score_key[1]
                    scores3 = score_key[2]
                    if (temp_similarities[scores1]['target_paper'] == '') or (temp_similarities[scores2]['target_paper'] == '') :
                        is_failed = scores3 == 1 or scores3 == 2
                        continue
                    sample_results.append((
                        target_ss_id,
                        float(temp_similarities[scores1]['25p_semantic_score']),
                        float(temp_similarities[scores1]['50p_semantic_score']),
                        float(temp_similarities[scores1]['75p_semantic_score']),
                        float(temp_similarities[scores1]['max_semantic_score']),
                        float(temp_similarities[scores2]['25p_semantic_score']),
                        float(temp_similarities[scores2]['50p_semantic_score']),
                        float(temp_similarities[scores2]['75p_semantic_score']),
                        float(temp_similarities[scores2]['max_semantic_score']),
                        scores3
                    ))
                    sample_results.append((
                        target_ss_id,
                        float(temp_similarities[scores1]['25p_cs_score']),
                        float(temp_similarities[scores1]['50p_cs_score']),
                        float(temp_similarities[scores1]['75p_cs_score']),
                        float(temp_similarities[scores1]['max_cs_score']),
                        float(temp_similarities[scores2]['25p_cs_score']),
                        float(temp_similarities[scores2]['50p_cs_score']),
                        float(temp_similarities[scores2]['75p_cs_score']),
                        float(temp_similarities[scores2]['max_cs_score']),
                        scores3
                    ))
                if not is_failed:
                    results = results + sample_results
                # print("Results: ", results)
                print(f"Processed {idx_ssid + 1} of {len(target_ss_ids)}")
                
                if latest_chunk + chunk_size < idx_ssid:
                    print(f"Inserting chunk {latest_result} to {len(results)}")
                    chunk = results[latest_result:]
                    execute_batch(dbclient.conn.cursor(), """
                    INSERT INTO tested_papers (
                        tested_paper,
                        p25p_score_bm25,
                        p50p_score_bm25,
                        p75p_score_bm25,
                        max_score_bm25,
                        p25p_score_scibert,
                        p50p_score_scibert,
                        p75p_score_scibert,
                        max_score_scibert,
                        category
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """, chunk)
                    dbclient.conn.commit()
                    latest_chunk += chunk_size
                    latest_result = len(results)
            except Exception as e:
                continue
            

        # upload to db in chunks

        # ================== Citation Similarity ==================
        # db_operations.create_citation_similarity_table(dbclient)
        # get_full_citation_similarity(dbclient, chunk_size)

        # insert into db
        # citation_similarities_df = load_from_csv(file_name="paper_paper_citation_similarity", folder_name="citation_similarity")
        # citation_similarities = list(citation_similarities_df.itertuples(index=False, name=None))

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
    

    
if __name__ == "__main__":
    main()