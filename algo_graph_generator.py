from global_methods import save_to_csv, load_from_csv, load_dataframe_from_list
from db import db_operations
import pandas as pd
class GraphGenerator:
    def __init__(self, dbclient):
        self.dbclient = dbclient

    def generate_graphs(self, target_ss_ids):
        for target_ss_id in target_ss_ids:
            self.generate_graph(target_ss_id)

    def generate_semantic_graph(self, target_ss_id):
        print(f"Generating graph for target paper {target_ss_id}")
        # 1. get the top 5 papers from each category (36 categories), discounted by levels
        # this ensures diversity, and specificity due to discounting
        test_results = db_operations.get_discounted_topics_by_combined_scores_only(self.dbclient, target_ss_id)
        test_results_df = load_dataframe_from_list(test_results, ["topic_id", "topic", "level", "category", "discounted_combined_similar"])   
        save_to_csv(test_results_df, "get_discounted_topics_by_combined_scores_only", "results") # [[topic_id,topic,level,category,discounted_combined_similar], ...]

        # 2. out of the 35 x 5 topics, take the top 20 topics including rewards. y = sum(weight) + x * reward, reward being the numer of categories that the top 10 topics belong to
        # this gives more weight to the topics when there is diversity of topics??

        top_175_topics_df = load_from_csv("get_discounted_topics_by_combined_scores_only", "results")
        
        # find the top 10 topics, and calculate the number of unique categories
        top_175_topics_df = top_175_topics_df.sort_values(by="discounted_combined_similar", ascending=False)
        top_10_topics_df = top_175_topics_df.head(20)
        # print(top_10_topics_df[['topic', 'level']].to_string(index=False))

        reward_columns = top_10_topics_df["category"].unique()
        reward = len(reward_columns.tolist())
        reward_weight = 1

        # in the top_175_topics_df, create a new column that calculates discounted_combined_similar + reward * 0.1, and then get the top 20 topics based on the new column
        top_175_topics_df["should_reward"] = top_175_topics_df["category"].apply(lambda x: 1 if x in reward_columns else 0)
        top_175_topics_df["score_after_reward"] = top_175_topics_df["discounted_combined_similar"] + top_175_topics_df["should_reward"] * reward * reward_weight
        top_20_topics_df = top_175_topics_df.sort_values(by="score_after_reward", ascending=False).head(20)
        # print(top_20_topics_df[['topic', 'level']].to_string(index=False))
        

        # 3. out of the 20 remaining topics, get top 10 with lowest topic-topic simialrity, y = sum(weight) - x * similarity, similarity being the similarity between the topics
        # go to db and extract out all the topic-topic similarity scores, based on the topic_ids in top_20_topics_df
        topic_topic_similarity = db_operations.get_topic_topic_similarity_by_topic_ids(self.dbclient, top_20_topics_df["topic_id"].tolist()) # [[topic1_id, topic2_id, similarity], ...]
        topic_topic_similarity_df = load_dataframe_from_list(topic_topic_similarity, ["topic1_id", "topic2_id", "topic_similarity"])
        topic_topic_similarity_discount = 1

        # add the sum of the similarity between each of the topics in the top 20 topics and all the other top 20 topics
        topic_topic_similarity_df_symmetric = pd.concat([
            topic_topic_similarity_df,
            topic_topic_similarity_df.rename(columns={"topic1_id": "topic2_id", "topic2_id": "topic1_id"})
        ])
        # Calculate the sum of similarities for each topic
        topic_similarity_sum = topic_topic_similarity_df_symmetric.groupby("topic1_id")["topic_similarity"].sum().reset_index()

        # Rename columns for clarity
        topic_similarity_sum.columns = ["topic_id", "topic_similarity"]

        # Merge with the top 20 topics DataFrame to include similarity sums
        top_20_topics_df = top_20_topics_df.merge(topic_similarity_sum, on="topic_id", how="left")

        # print(top_20_topics_df)
        top_20_topics_df['score_after_reward'] = top_20_topics_df['score_after_reward'].astype(float)
        top_20_topics_df['topic_similarity'] = top_20_topics_df['topic_similarity'].astype(float)

        # Min-Max Normalization
        top_20_topics_df['normalized_score'] = (top_20_topics_df['score_after_reward'] - top_20_topics_df['score_after_reward'].min()) / (top_20_topics_df['score_after_reward'].max() - top_20_topics_df['score_after_reward'].min())
        top_20_topics_df['normalized_similarity'] = (top_20_topics_df['topic_similarity'] - top_20_topics_df['topic_similarity'].min()) / (top_20_topics_df['topic_similarity'].max() - top_20_topics_df['topic_similarity'].min())

        topic_score_weight = 2
        topic_similarity_weight = 1
        top_20_topics_df['final_topic_score'] = topic_score_weight * top_20_topics_df['normalized_score'] - topic_similarity_weight * top_20_topics_df['normalized_similarity']
        top_10_topics_df = top_20_topics_df[['topic_id', 'topic', 'normalized_score', 'normalized_similarity', 'final_topic_score']].sort_values(by="final_topic_score", ascending=False).head(10)
        print(top_10_topics_df[['topic_id', 'topic', 'normalized_score', 'normalized_similarity', 'final_topic_score']].sort_values(by="final_topic_score", ascending=False))
        
        # 4. from top 10 topic nodes, get the paper nodes with the highest similarities to the topic nodes, 10 papers per topic node
        selected_topic_ids = top_10_topics_df['topic_id'].tolist()
        top_topic_paper_edges = db_operations.get_papers_by_topic_ids(self.dbclient, selected_topic_ids)

        if not top_topic_paper_edges:
            return None, None
        
        flattened_data = []
        # print("top_topic_paper_edges.items()")
        # print(top_topic_paper_edges.items())
        for topic_id, papers in top_topic_paper_edges.items():
            for paper in papers:
                paper_data = {
                    'topic_id': topic_id,
                    'ss_id': paper['ss_id'],
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'combined_similarity': paper['combined_similarity']
                }
                flattened_data.append(paper_data)

        top_topic_paper_edges_df = pd.DataFrame(flattened_data)
        # print("top_papers", top_topic_paper_edges_df.head(5))

        return top_topic_paper_edges_df, top_10_topics_df

    def generate_co_citation_graph(self, target_ss_id):
        citation_similarity = db_operations.get_highest_citation_similarity(self.dbclient, target_ss_id)
        citation_similarity_df = load_dataframe_from_list(citation_similarity, ["ss_id", "title", "abstract", "co_citation_count", "coupling_count", "combined_count"])
        save_to_csv(citation_similarity_df, "ranked_papers_with_scores_citation_similarity", "results")

        return citation_similarity_df



    