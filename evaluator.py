

import nltk
import string
import pandas as pd
from rank_bm25 import BM25Okapi
from statistics import median

import algo_tokeniser as tk
from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
from db import db_operations
from algo_bm25 import get_scores_for_target_paper


class Evaluator: 
    def __init__(self, db_client):
        self.db_client = db_client
        pass
    
    def run_bm25_eval(self, target_ss_id, semantic_result, cs_result): # for 1 target paper
        # target_paper = (ss_id, title, abstract)
        # semantic_result = [(ss_id, title, abstract), ...]
        # co_citation_result = [(ss_id, title, abstract), ...]

        target_paper = db_operations.get_papers_by_ss_id(self.db_client, target_ss_id)
        semantic_scores = get_scores_for_target_paper(target_paper, semantic_result)
        cs_scores = get_scores_for_target_paper(target_paper, cs_result)
        print(cs_scores)

        combined_scores = {
            "semantic_scores": semantic_scores,
            "cs_scores": cs_scores,
            "median_semantic_score": median(semantic_scores),
            "median_cs_score": median(cs_scores),
            "target_paper": target_paper[1]
        }

        return combined_scores

    def run_bert_eval(self, target_ss_id, semantic_results, cs_results): # for 1 target paper
        # get the similarity scores between the target papr and the papers in the results from the database
        semantic_ss_ids = [result[0] for result in semantic_results]
        cs_ss_ids = [result[0] for result in cs_results]
        scores = db_operations.get_scibert_scores_by_ss_id(self.db_client, target_ss_id, semantic_ss_ids, cs_ss_ids)

        # print(f"Scores: {scores}")

        # get the median similarity scores for the target paper
        # semantic_scores = [score[1] for score in scores if score[0] in semantic_ss_ids]
        # cs_scores = [score[1] for score in scores if score[0] in cd_ss_ids]
        semantic_scores = []
        cs_scores = []
        # for ss_id in semantic_ss_ids:
        #     semantic_scores.append(scores[ss_id]['combined_similarity'])
        # for ss_id in cs_ss_ids:
        #     cs_scores.append(scores[ss_id]['combined_similarity'])

        for ss_id in semantic_ss_ids:
            if ss_id in scores:
                semantic_scores.append(scores[ss_id]['combined_similarity'])
            else:
                print(f"Warning: ss_id {ss_id} not found in scores")
        
        for ss_id in cs_ss_ids:
            if ss_id in scores:
                cs_scores.append(scores[ss_id]['combined_similarity'])
            else:
                print(f"Warning: ss_id {ss_id} not found in scores")
        
        
        combined_scores = {
            "semantic_scores": semantic_scores,
            "cs_scores": cs_scores,
            "median_semantic_score": median(semantic_scores),
            "median_cs_score": median(cs_scores),
            "target_paper": target_ss_id
        }

        return combined_scores

    def new_papers_scoring(self, target_ss_id, semantic_results, cs_results):
        print(f"Target paper: {target_ss_id}")

        # get the similarity scores between the target paper and the papers in the results from the database
        semantic_ss_ids = [result[0] for result in semantic_results]
        cs_ss_ids = [result[0] for result in cs_results]
        print(f"cs_ss_ids length: {len(cs_ss_ids)}")
        print(f"Semantic length: {len(semantic_ss_ids)}")
        print(f"Semantic ss_ids: {semantic_ss_ids}")
        print(f"CS ss_ids: {cs_ss_ids}")
        # find if there are any uncommon papers in the results
        uncommon_papers = set(semantic_ss_ids) - set(cs_ss_ids)
        # print(f"Uncommon papers: {uncommon_papers}")
        print(f"% uncommon papers: {len(uncommon_papers) / (len(semantic_ss_ids) + len(cs_ss_ids))}")
        uncommon_papers_list = list(uncommon_papers)

        print(f"Uncommon papers type: {type(uncommon_papers_list)}")
        print(f"cs_ss_ids type: {type(cs_ss_ids)}")
        print(f"target_ss_id: {target_ss_id}")
        print(f"semantic_results length: {len(semantic_results)}")
        print(f"cs_results length: {len(cs_results)}")
        print(f"Uncommon papers length: {len(uncommon_papers_list)}")

        scibert_score = self.run_bert_eval(target_ss_id, semantic_results, cs_results) # cs_ss_ids are useless and are just here to fit in as a param
        bm25_score = self.run_bm25_eval(target_ss_id, uncommon_papers_list, cs_ss_ids) # cs_ss_ids are useless and are just here to fit in as a param

        # print(f"Scibert score: {scibert_score}")
        # print(f"BM25 score: {bm25_score}")

        return scibert_score['median_semantic_score'], bm25_score['median_semantic_score']

