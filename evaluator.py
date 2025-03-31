

import nltk
import string
import pandas as pd
from rank_bm25 import BM25Okapi
import numpy as np

import algo_tokeniser as tk
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
from db import db_operations
from algo_bm25 import get_scores_for_target_paper


class Evaluator: 
    def __init__(self, db_client, logger, colbert):
        self.db_client = db_client
        self.colbert = colbert
        self.logger = logger
        pass

    def run_bm25_eval(self, target_ss_id, semantic_result, cs_result, title_check, category): # for 1 target paper
        # target_paper = (ss_id, title, abstract)
        # semantic_result = [(ss_id, title, abstract), ...]
        # co_citation_result = [(ss_id, title, abstract), ...]

        target_paper = db_operations.get_papers_by_ss_id(self.db_client, target_ss_id)
        semantic_scores = get_scores_for_target_paper(target_paper, semantic_result, title_check)
        cs_scores = get_scores_for_target_paper(target_paper, cs_result, "cs")

        combined_scores = {
            "semantic_scores": semantic_scores,
            "25p_semantic_score": np.percentile(semantic_scores, 25) if len(semantic_scores) > 0 else 0,
            "25p_cs_score": np.percentile(cs_scores, 25),
            "50p_semantic_score": np.percentile(semantic_scores, 50),
            "50p_cs_score": np.percentile(cs_scores, 50),
            "75p_semantic_score": np.percentile(semantic_scores, 75),
            "75p_cs_score": np.percentile(cs_scores, 75),
            "max_semantic_score": max(semantic_scores),
            "max_cs_score": max(cs_scores),
            "target_paper": target_paper[0]
        }

        returned_scores = [
            target_ss_id,
            combined_scores["25p_semantic_score"],
            combined_scores["25p_cs_score"],
            combined_scores["50p_semantic_score"],
            combined_scores["50p_cs_score"],
            combined_scores["75p_semantic_score"],
            combined_scores["75p_cs_score"],
            combined_scores["max_semantic_score"],
            combined_scores["max_cs_score"],
            category
        ]

        return combined_scores

    def run_bert_eval(self, target_ss_id, semantic_results, cs_results, title_check, category): # for 1 target paper
        # get the similarity scores between the target papr and the papers in the results from the database
        target_paper = db_operations.get_papers_by_ss_id(self.db_client, target_ss_id)

        semantic_scores = self.colbert.get_scores_for_target_paper(target_paper, semantic_results, title_check)
        cs_scores = self.colbert.get_scores_for_target_paper(target_paper, cs_results, title_check)

        combined_scores = {
            "semantic_scores": semantic_scores,
            "25p_semantic_score": np.percentile(semantic_scores, 25),
            "25p_cs_score": np.percentile(cs_scores, 25),
            "50p_semantic_score": np.percentile(semantic_scores, 50),
            "50p_cs_score": np.percentile(cs_scores, 50),
            "75p_semantic_score": np.percentile(semantic_scores, 75),
            "75p_cs_score": np.percentile(cs_scores, 75),
            "max_semantic_score": max(semantic_scores),
            "max_cs_score": max(cs_scores),
            "target_paper": target_paper[0]
        }

        return combined_scores


    def run_bm25_eval_novel(self, target_ss_id, results, title_check, category): # for 1 target paper
        # target_paper = (ss_id, title, abstract)
        # semantic_result = [(ss_id, title, abstract), ...]
        # co_citation_result = [(ss_id, title, abstract), ...]

        target_paper = db_operations.get_papers_by_ss_id(self.db_client, target_ss_id)
        semantic_scores = get_scores_for_target_paper(target_paper, results, title_check)
        # print(cs_scores)
        # print("YAYYY")
        # print("semantic_scores: ", semantic_scores)

        all_zero_float64 = all(isinstance(x, np.float64) and x == 0.0 for x in semantic_scores)
        if all_zero_float64:
            return []

        combined_scores = {
            "25p_semantic_score": np.percentile(semantic_scores, 25),
            "25p_cs_score": 0,
            "50p_semantic_score": np.percentile(semantic_scores, 50),
            "50p_cs_score": 0,
            "75p_semantic_score": np.percentile(semantic_scores, 75),
            "75p_cs_score": 0,
            "max_semantic_score": max(semantic_scores),
            "max_cs_score": 0,
            "target_paper": target_paper[0]
        }

        return combined_scores

    def run_bert_eval_novel(self, target_ss_id, results, title_check, category): # for 1 target paper
        target_paper = db_operations.get_papers_by_ss_id(self.db_client, target_ss_id)
        semantic_scores = self.colbert.get_scores_for_target_paper(target_paper, results, title_check)

        combined_scores = {
            "25p_semantic_score": np.percentile(semantic_scores, 25),
            "25p_cs_score": 0,
            "50p_semantic_score": np.percentile(semantic_scores, 50),
            "50p_cs_score": 0,
            "75p_semantic_score": np.percentile(semantic_scores, 75),
            "75p_cs_score": 0,
            "max_semantic_score": max(semantic_scores),
            "max_cs_score": 0,
            "target_paper": target_paper[0]
        }

        return combined_scores




    # def novel_papers_scoring(self, target_ss_id, semantic_results, cs_results):
    #     print(f"Target paper: {target_ss_id}")

    #     # get the similarity scores between the target paper and the papers in the results from the database
    #     semantic_ss_ids = [result[0] for result in semantic_results]
    #     cs_ss_ids = [result[0] for result in cs_results]
    #     print(f"cs_ss_ids length: {len(cs_ss_ids)}")
    #     print(f"Semantic length: {len(semantic_ss_ids)}")
    #     print(f"Semantic ss_ids: {semantic_ss_ids}")
    #     print(f"CS ss_ids: {cs_ss_ids}")
    #     # find if there are any uncommon papers in the results
    #     uncommon_papers = set(semantic_ss_ids) - set(cs_ss_ids)
    #     # print(f"Uncommon papers: {uncommon_papers}")
    #     print(f"% uncommon papers: {len(uncommon_papers) / (len(semantic_ss_ids) + len(cs_ss_ids))}")
    #     uncommon_papers_list = list(uncommon_papers)

    #     print(f"Uncommon papers type: {type(uncommon_papers_list)}")
    #     print(f"cs_ss_ids type: {type(cs_ss_ids)}")
    #     print(f"target_ss_id: {target_ss_id}")
    #     print(f"semantic_results length: {len(semantic_results)}")
    #     print(f"cs_results length: {len(cs_results)}")
    #     print(f"Uncommon papers length: {len(uncommon_papers_list)}")

    #     scibert_score = self.run_bert_eval(target_ss_id, semantic_results, cs_results) # cs_ss_ids are useless and are just here to fit in as a param
    #     bm25_score = self.run_bm25_eval(target_ss_id, uncommon_papers_list, cs_ss_ids) # cs_ss_ids are useless and are just here to fit in as a param

    #     # print(f"Scibert score: {scibert_score}")
    #     # print(f"BM25 score: {bm25_score}")

    #     return scibert_score['median_semantic_score'], bm25_score['median_semantic_score']
    
    

