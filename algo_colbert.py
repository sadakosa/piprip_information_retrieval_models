from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import time
from global_methods import save_to_csv

class ColBERT:
    def __init__(self, logger, model_type):
        if model_type == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
        elif model_type == 'scibert':
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        self.model_type = model_type
        self.logger = logger

    # def get_topic_paper_embeddings(self, topics, topic_ids, titles, abstracts, combined_texts, ss_ids): # these are all lists of strings
    #     start_time = time.time()
    #     title_embeddings = [self.__get_embeddings(title) for title in titles]
    #     abstract_embeddings = [self.__get_embeddings(abstract) for abstract in abstracts]
    #     combined_text_embeddings = [self.__get_embeddings(combined_text) for combined_text in combined_texts]
    #     topic_embeddings = [self.__get_embeddings(topic) for topic in topics]

    #     print("Lengths of topic_paper_embeddings:")
    #     print(len(topic_embeddings), len(title_embeddings), len(abstract_embeddings), len(combined_text_embeddings))

    #     similarities = {
    #         'topic_id': [],
    #         'paper_ss_id': [],
    #         'title_similarity': [],
    #         'abstract_similarity': [],
    #         'combined_similarity': []
    #     }

    #     for i in range(len(topic_embeddings)):
    #         for j in range(i, len(title_embeddings)):
    #             cossim_title = torch.nn.functional.cosine_similarity(topic_embeddings[i], title_embeddings[j])
    #             cossim_abstract = torch.nn.functional.cosine_similarity(topic_embeddings[i], abstract_embeddings[j])
    #             cossim_combined = torch.nn.functional.cosine_similarity(topic_embeddings[i], combined_text_embeddings[j])
    #             similarities['topic_id'].append(topic_ids[i])
    #             similarities['paper_ss_id'].append(ss_ids[j])
    #             similarities['title_similarity'].append(cossim_title.item())
    #             similarities['abstract_similarity'].append(cossim_abstract.item())
    #             similarities['combined_similarity'].append(cossim_combined.item())

    #     print("Length of similarities:")
    #     print(len(similarities['topic_id']), len(similarities['paper_ss_id']), len(similarities['title_similarity']), len(similarities['abstract_similarity']), len(similarities['combined_similarity']))
    #     similarities_df = pd.DataFrame(similarities)

    #     print(similarities_df) # DataFrame with topics as columns and papers as rows

    #     # save to csv
    #     save_to_csv(similarities_df, f'topic_paper_similarities_{self.model_type}', 'similarities')
    #     self.logger.log_message("Saved topic-paper similarities to CSV")
    #     end_time = time.time()
    #     print("Time taken to get topic-paper similarities: ", end_time - start_time)

    #     # save to database
    #     # db_operations.batch_insert_similarities(dbclient, similarities_df)


    #     return similarities_df

    # def get_topic_topic_embeddings(self, topics, topic_ids): # these are all lists of strings
    #     start_time = time.time()    
    #     topic_embeddings = [self.__get_embeddings(topic) for topic in topics]

    #     similarities = {
    #         'topic_id_one': [],
    #         'topic_id_two': [],
    #         'similarity': [],
    #     }

    #     for i in range(len(topic_embeddings)):
    #         for j in range(i, len(topic_embeddings)):
    #             cossim = torch.nn.functional.cosine_similarity(topic_embeddings[i], topic_embeddings[j])
    #             similarities['topic_id_one'].append(topic_ids[i])
    #             similarities['topic_id_two'].append(topic_ids[j])
    #             similarities['similarity'].append(cossim.item())

    #     similarities_df = pd.DataFrame(similarities)

    #     save_to_csv(similarities_df, f'topic_topic_similarities_{self.model_type}', 'similarities')

    #     self.logger.log_message("Saved topic-topic similarities to CSV")
    #     end_time = time.time()
    #     print("Time taken to get topic-topic similarities: ", end_time - start_time)
    #     self.logger.log_message("Time taken to get topic-topic similarities: " + str(end_time - start_time))

    #     print(similarities_df)

    def get_scores_for_target_paper(self, target_paper, papers_to_check, title_check): 
        # target_paper = (ss_id, title, abstract)
        # results = [(ss_id, title, abstract), ...]
        # output = [score1, score2, ...]

        if title_check == "title":
            texts = [paper[1] for paper in papers_to_check]
            text_embeddings = [self.__get_embeddings(text) for text in texts]

            target_text_repeated = [target_paper[1]] * len(papers_to_check)
            target_text_embeddings = [self.__get_embeddings(target_text) for target_text in target_text_repeated]
        else:
            texts = [paper[1] + paper[2] for paper in papers_to_check]
            text_embeddings = [self.__get_embeddings(text) for text in texts]
            target_combined = target_paper[1] + " " + target_paper[2]
            target_text_repeated = [target_combined] * len(papers_to_check)
            target_text_embeddings = [self.__get_embeddings(target_text) for target_text in target_text_repeated]

        similarities = []
        for i in range(len(papers_to_check)):
            cossim = torch.nn.functional.cosine_similarity(text_embeddings[i], target_text_embeddings[i])
            similarities.append(cossim.item())

        return similarities 
    
    def __get_embeddings(self, text):
        # inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        # with torch.no_grad():
        #     outputs = self.model(**inputs)
        # return outputs.last_hidden_state.mean(dim=1)

        max_length = 512
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)