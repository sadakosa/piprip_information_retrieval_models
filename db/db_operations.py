
from psycopg2.extras import execute_values

# ======================== SELECTION OPERATIONS ========================

def get_all_paper_ids(db_client):
    select_query = """
    SELECT ss_id, clean_title, clean_abstract FROM papers
    WHERE is_cleaned = True;
    """
    cursor = db_client.execute(select_query)
    return cursor.fetchall()

def get_papers_by_ss_ids(db_client, ss_ids):
    select_query = """
    SELECT ss_id, clean_title, clean_abstract FROM papers
    WHERE ss_id IN %s 
    AND clean_title IS NOT NULL
    AND clean_abstract IS NOT NULL;
    """
    cursor = db_client.execute(select_query, (tuple(ss_ids),))
    return cursor.fetchall()

def get_papers_by_ss_id(db_client, ss_id):
    select_query = """
    SELECT ss_id, clean_title, clean_abstract FROM papers
    WHERE ss_id = %s
    AND is_cleaned = True;
    """
    cursor = db_client.execute(select_query, (ss_id,))
    return cursor.fetchone()

# def get_scibert_scores_by_ss_id(db_client, target_ss_id, semantic_ss_ids, cs_ss_ids):
#     """
#     Retrieve SciBERT similarity scores between the target paper and a list of other papers.

#     Parameters:
#     db_client (DatabaseClient): The database client instance.
#     target_ss_id (str): The ss_id of the target paper.
#     semantic_ss_ids (list): A list of ss_id for semantic results papers.
#     cs_ss_ids (list): A list of ss_id for CS results papers.

#     Returns:
#     dict: A dictionary with ss_id as keys and their corresponding similarity scores as values.
#     """
#     # Combine the list of ss_ids
#     combined_ss_ids = list(set(semantic_ss_ids + cs_ss_ids))
    
#     # Prepare the query
#     query = """
#     SELECT ss_id_two, title_similarity, abstract_similarity, combined_similarity
#     FROM scibert_paper_paper_edges
#     WHERE ss_id_one = %s AND ss_id_two = ANY(%s)
#     UNION
#     SELECT ss_id_one, title_similarity, abstract_similarity, combined_similarity
#     FROM scibert_paper_paper_edges
#     WHERE ss_id_two = %s AND ss_id_one = ANY(%s);
#     """
    
#     # Execute the query
#     cursor = db_client.execute(query, (target_ss_id, combined_ss_ids, target_ss_id, combined_ss_ids))
    
#     # Fetch the results
#     results = cursor.fetchall()
    
#     # Organize the results in a dictionary
#     scores = {
#         result[0]: {
#             'title_similarity': result[1],
#             'abstract_similarity': result[2],
#             'combined_similarity': result[3]
#         }
#         for result in results
#     }
    
#     return scores







def get_scibert_scores_by_ss_id(db_client, target_ss_id, semantic_ss_ids, cs_ss_ids):
    """
    Retrieve SciBERT similarity scores between the target paper and a list of other papers.

    Parameters:
    db_client (DatabaseClient): The database client instance.
    target_ss_id (str): The ss_id of the target paper.
    semantic_ss_ids (list): A list of ss_id for semantic results papers.
    cs_ss_ids (list): A list of ss_id for CS results papers.

    Returns:
    dict: A dictionary with ss_id as keys and their corresponding similarity scores as values.
    """
    # Combine the list of ss_ids
    combined_ss_ids = list(set(semantic_ss_ids + cs_ss_ids))
    
    # Print combined_ss_ids for debugging
    print(f"Combined ss_ids: {combined_ss_ids}")
    
    # Prepare the query
    query = """
    SELECT ss_id_one, ss_id_two, title_similarity, abstract_similarity, combined_similarity
    FROM scibert_paper_paper_edges
    WHERE ss_id_one = %s AND ss_id_two = ANY(%s)
    UNION
    SELECT ss_id_one, ss_id_two, title_similarity, abstract_similarity, combined_similarity
    FROM scibert_paper_paper_edges
    WHERE ss_id_two = %s AND ss_id_one = ANY(%s);
    """
    
    # Execute the query
    cursor = db_client.execute(query, (target_ss_id, combined_ss_ids, target_ss_id, combined_ss_ids))
    
    # Fetch the results
    results = cursor.fetchall()
    
    # Print results for debugging
    # print(f"Query results: {results}")

    # Organize the results in a dictionary
    scores = {}
    for result in results:
        ss_id_one = result[0]
        ss_id_two = result[1]
        title_similarity = result[2]
        abstract_similarity = result[3]
        combined_similarity = result[4]
        
        # Add scores to dictionary
        if ss_id_two != target_ss_id and ss_id_two not in scores:
            scores[ss_id_two] = {
                'title_similarity': title_similarity,
                'abstract_similarity': abstract_similarity,
                'combined_similarity': combined_similarity
            }
        
        if ss_id_one != target_ss_id and ss_id_one not in scores:
            scores[ss_id_one] = {
                'title_similarity': title_similarity,
                'abstract_similarity': abstract_similarity,
                'combined_similarity': combined_similarity
            }

    return scores






# ======================== CITATION SIMILARITY OPERATIONS ========================

# returns [['ss_id_one', 'ss_id_two', 'co_citation_count'], ...]
def get_all_co_citations(db_client):
    query = """
    SELECT LEAST(r1.reference_id, r2.reference_id) AS paper1, GREATEST(r1.reference_id, r2.reference_id) AS paper2, COUNT(*) AS co_citation_count
    FROM "references" r1
    JOIN "references" r2 ON r1.ss_id = r2.ss_id AND r1.reference_id <> r2.reference_id
    GROUP BY LEAST(r1.reference_id, r2.reference_id), GREATEST(r1.reference_id, r2.reference_id);
    """
    cursor = db_client.execute(query)
    return cursor.fetchall()

# returns [['ss_id_one', 'ss_id_two', 'bibliographic_coupling_count'], ...]
def get_all_bibliographic_couples(db_client):
    query = """
    SELECT LEAST(r1.ss_id, r2.ss_id) AS paper1, GREATEST(r1.ss_id, r2.ss_id) AS paper2, COUNT(*) AS coupling_count
    FROM "references" r1
    JOIN "references" r2 ON r1.reference_id = r2.reference_id AND r1.ss_id <> r2.ss_id
    GROUP BY LEAST(r1.ss_id, r2.ss_id), GREATEST(r1.ss_id, r2.ss_id);
    """
    cursor = db_client.execute(query)
    return cursor.fetchall()

# returns [('ss_id', 'clean_title', 'abstract', similar_references=76), ...]
def get_bibliographic_couples(db_client, target_ss_id):
    query = """
    WITH cited_papers AS (
        SELECT reference_id
        FROM references
        WHERE ss_id = %s
    ),
    papers_that_cite_same_references AS (
        SELECT DISTINCT r1.ss_id AS citing_paper
        FROM references r1
        JOIN cited_papers cp ON r1.reference_id = cp.reference_id
        WHERE r1.ss_id != %s
    )
    SELECT p.ss_id, p.clean_title, p.abstract, COUNT(*) AS similar_references
    FROM papers_that_cite_same_references ptcsr
    JOIN references r2 ON ptcsr.citing_paper = r2.ss_id
    JOIN cited_papers cp ON r2.reference_id = cp.reference_id
    JOIN papers p ON ptcsr.citing_paper = p.ss_id
    GROUP BY p.ss_id, p.clean_title, p.abstract
    ORDER BY similar_references DESC;
    """
    cursor = db_client.execute(query, (target_ss_id, target_ss_id))
    return cursor.fetchall()


# returns [('ss_id', 'clean_title', 'abstract', similar_citations=76), ...]
def get_co_citation(db_client, target_ss_id):
    query = """
    WITH citing_papers AS (
        SELECT r1.ss_id AS cited_paper, r1.reference_id AS citing_paper
        FROM "references" r1
        WHERE r1.ss_id = %s
    ),
    similar_papers AS (
        SELECT r2.ss_id AS similar_paper_id
        FROM "references" r2
        INNER JOIN citing_papers cp
        ON r2.reference_id = cp.citing_paper
    ),
    paper_details AS (
        SELECT p.ss_id, p.clean_title, p.clean_abstract
        FROM papers p
        WHERE p.ss_id IN (SELECT similar_paper_id FROM similar_papers)
    )
    SELECT pd.ss_id AS similar_paper, pd.clean_title, pd.clean_abstract, COUNT(sp.similar_paper_id) AS num_references
    FROM similar_papers sp
    JOIN paper_details pd ON sp.similar_paper_id = pd.ss_id
    GROUP BY pd.ss_id, pd.clean_title, pd.clean_abstract
    ORDER BY num_references DESC;
    """
    cursor = db_client.execute(query, (target_ss_id,))
    return cursor.fetchall()

# ======================== CITATION SIMILARITY CREATE AND INSERT ========================

def create_citation_similarity_table(db_client):
    create_query = """
    CREATE TABLE IF NOT EXISTS citation_similarity(
        ss_id TEXT NOT NULL,
        similar_paper TEXT NOT NULL,
        co_citation_count NUMERIC NOT NULL,
        bibliographic_coupling_count NUMERIC NOT NULL,
        PRIMARY KEY (ss_id, similar_paper),
        FOREIGN KEY (ss_id) REFERENCES papers(ss_id),
        FOREIGN KEY (similar_paper) REFERENCES papers(ss_id)
    );
    """
    db_client.execute(create_query)
    return

# def batch_insert_citation_similarity(db_client, citation_similarities, chunk_size): # citation_similarities = [(ss_id, similar_paper, co_citation_count, bibliographic_coupling_count), ...]
#     insert_query = """
#     INSERT INTO citation_similarity (ss_id, similar_paper, co_citation_count, bibliographic_coupling_count)
#     VALUES %s
#     ON CONFLICT (ss_id, similar_paper) DO NOTHING;
#     """
#     # print("Sample of citation_similarities:", citation_similarities[:5])

#     # Using psycopg2's execute_values to handle batch inserts
#     from psycopg2.extras import execute_values
#     for i in range(0, len(citation_similarities), chunk_size):
#         chunk = citation_similarities[i:i+chunk_size]
#         # print("Inserting chunk:", chunk[:5])
#         execute_values(db_client.cur, insert_query, chunk)
#     return


from psycopg2.extras import execute_values
def batch_insert_citation_similarity(db_client, logger, citation_similarities, chunk_size):
    insert_query = """
    INSERT INTO citation_similarity (ss_id, similar_paper, co_citation_count, bibliographic_coupling_count)
    VALUES %s
    ON CONFLICT (ss_id, similar_paper) DO NOTHING;
    """


    for i in range(0, len(citation_similarities), chunk_size):
        chunk = citation_similarities[i:i+chunk_size]
        print("len(chunk):", len(chunk))    
        
        # Debug: Check the structure of the current chunk
        print(f"Inserting chunk {i//chunk_size + 1}/{-(-len(citation_similarities)//chunk_size)}: {chunk[:5]}")
        logger.log_message(f"Inserting chunk {i//chunk_size + 1}/{-(-len(citation_similarities)//chunk_size)}: {chunk[:5]}")
        # Inserting chunk: [('e31606c0cdb2b2cf1a8c749dd71402053b8f2b12', 'e5e85b506969e276487458add9d75fe2d44b9188', 2.0, 0.0), ('e31606c0cdb2b2cf1a8c749dd71402053b8f2b12', 'e6a7d2f8818052c0ef5272446654317d44a8a825', 2.0, 0.0), ('e31606c0cdb2b2cf1a8c749dd71402053b8f2b12', 'e6e6acbd1067e448848cd48fde6de6c3b0edf82e', 2.0, 0.0), ('e31606c0cdb2b2cf1a8c749dd71402053b8f2b12', 'e8aa63ae69334bc33135b0e0dd8066fd768215e9', 2.0, 0.0), ('e31606c0cdb2b2cf1a8c749dd71402053b8f2b12', 'eb8cdb317a51e0cfa913790f966baf988baaf49e', 2.0, 0.0)]
        
        try:
            execute_values(db_client.cur, insert_query, chunk)
            db_client.commit()  # Commit after each chunk insertion
        except Exception as e:
            print(f"Error inserting chunk {i//chunk_size + 1}: {e}")
            logger.log_message(f"Error inserting chunk {i//chunk_size + 1}: {e}")
    
    print("Batch insertion complete.")
    logger.log_message("Batch insertion complete.")
    return





# ==========================================================
# GRAPH GENERATION OPERATIONS
# ==========================================================

def get_discounted_topics_by_combined_scores_only(db_client, target_ss_id): # returns all the topic-paper edges of the top 5 topics within each category
    query = """
    WITH RankedTopics AS (
        SELECT
            t.id,
            t.topic,
            t.level,
            t.category,
            e.combined_similarity,
            (e.combined_similarity * ((100 - (80.0 / t.level^3)) / 100)) AS discounted_combined_similarity,
            RANK() OVER (PARTITION BY t.category ORDER BY (e.combined_similarity * ((100 - (80.0 / t.level^3)) / 100)) DESC) AS rank
        FROM
            topics t
            JOIN scibert_topic_paper_edges e ON t.id = e.topic_id
        WHERE
            e.ss_id = %s
    ), 
    TopTopics AS (
        SELECT
            id,
            topic,
            level,
            category,
            discounted_combined_similarity
        FROM
            RankedTopics
        WHERE
            rank <= 5
    )
    SELECT
        id AS topic_id,
        topic,
        level,
        category,
        discounted_combined_similarity
    FROM
        TopTopics
    ORDER BY
        category, topic_id;
    """
    cursor = db_client.execute(query, (target_ss_id,))
    return cursor.fetchall()

def get_topic_topic_similarity_by_topic_ids(db_client, topic_ids):
    if not topic_ids:
        return []

    query = """
    SELECT
        t1.topic_id_one AS topic1_id,
        t1.topic_id_two AS topic2_id,
        t1.weight AS similarity
    FROM
        topic_topic_edges t1
    WHERE
        t1.topic_id_one IN %s
        OR t1.topic_id_two IN %s;
    """
    cursor = db_client.execute(query, (tuple(topic_ids), tuple(topic_ids)))
    return cursor.fetchall()

def get_papers_by_topic_ids(db_client, selected_topic_ids):
    if not selected_topic_ids:
        return []

    topic_ids_str = ', '.join([str(id) for id in selected_topic_ids])
    print("topic_ids_str:", topic_ids_str)

    limit_per_topic = 5
    min_unique_papers_per_topic = 5
    unique_papers_by_topic = {}

    while True:
        # SQL query to fetch papers with the highest similarity to the given topics
        query = f"""
        WITH ranked_papers AS (
            SELECT 
                tpe.topic_id, 
                tpe.ss_id, 
                p.clean_title, 
                p.clean_abstract, 
                tpe.combined_similarity,
                ROW_NUMBER() OVER (PARTITION BY tpe.topic_id ORDER BY tpe.combined_similarity DESC) as rank
            FROM 
                topic_paper_edges tpe
            JOIN 
                papers p ON tpe.ss_id = p.ss_id
            WHERE 
                tpe.topic_id IN ({topic_ids_str})
        )
        SELECT 
            topic_id, 
            ss_id, 
            clean_title, 
            clean_abstract, 
            combined_similarity
        FROM 
            ranked_papers
        WHERE 
            rank <= ({limit_per_topic})
        ORDER BY 
            topic_id, rank;
        """

        
        cursor = db_client.execute(query)
        results = cursor.fetchall()

        print("length: ",len(results))

        # Process and store the results
        papers_by_topic = {}
        for row in results:
            topic_id = row[0]
            if topic_id not in papers_by_topic:
                papers_by_topic[topic_id] = []
            papers_by_topic[topic_id].append({
                'topic_id': row[0],
                'ss_id': row[1],
                'title': row[2],
                'abstract': row[3],
                'combined_similarity': row[4]
            })

        # Check if each topic has at least 10 unique papers
        unique_papers_by_topic = {topic_id: set([paper['ss_id'] for paper in papers])
                                    for topic_id, papers in papers_by_topic.items()}
        
        all_topics_met_criteria = all(len(papers) >= min_unique_papers_per_topic for papers in unique_papers_by_topic.values())
        
        if all_topics_met_criteria:
            break

        # Increase the limit and try again
        limit_per_topic += 10
        print("Increasing limit to", limit_per_topic)

    final_papers_by_topic = {topic_id: list(papers) for topic_id, papers in unique_papers_by_topic.items()}
    return papers_by_topic

def get_highest_citation_similarity(db_client, target_ss_id):
    query = """
    SELECT
        rp.ss_id,
        rp.clean_title AS related_paper_title,
        rp.clean_abstract AS related_paper_abstract,
        cs.co_citation_count,
        cs.bibliographic_coupling_count,
        cs.co_citation_count + cs.bibliographic_coupling_count AS combined_score
    FROM
        citation_similarity cs
    JOIN 
        papers p ON cs.similar_paper = p.ss_id OR cs.ss_id = p.ss_id
    JOIN 
        papers rp ON rp.ss_id = CASE
                                WHEN cs.similar_paper = p.ss_id THEN cs.ss_id
                                ELSE cs.similar_paper
                                END
    WHERE
        p.ss_id = %s AND rp.is_cleaned = True
    ORDER BY
        combined_score DESC
    LIMIT %s;
    """
    cursor = db_client.execute(query, (target_ss_id,50))
    return cursor.fetchall()



# ==========================================================
# EVALUATOR OPERATIONS
# ==========================================================

def create_tested_papers_table():
    """
    Create a table to store the tested papers and their similarity scores.
    
    Parameters:
    target_paper (str): The ss_id of the tested_paper.
    results (list): A list of tuples containing the  of the tested papers.
    """
    create_query = """
    CREATE TABLE IF NOT EXISTS tested_papers (
        id SERIAL PRIMARY KEY,
        tested_paper TEXT NOT NULL,
        25p_score_bm25 NUMERIC NOT NULL,
        50p_score_bm25 NUMERIC NOT NULL,
        75p_score_bm25 NUMERIC NOT NULL,,
        max_score_bm25 NUMERIC NOT NULL,
        25p_score_scibert NUMERIC NOT NULL,
        50p_score_scibert NUMERIC NOT NULL,
        75p_score_scibert NUMERIC NOT NULL,
        max_score_scibert NUMERIC NOT NULL,
        category TEXT NOT NULL,
        FOREIGN KEY (tested_paper) REFERENCES papers(ss_id)
    );
    """
    db_client.execute(create_query)
    return

def batch_insert_bm25_results(target_paper, results): 
    """    
    # results = [(
        tested_paper,
        25p_semantic_score,
        50p_semantic_score,
        75p_semantic_score,
        average_semantic_score,
        25p_cs_score, 
        50p_cs_score,
        75p_cs_score,
        average_cs_score), ...]
    """
    insert_query = """
    INSERT INTO tested_papers (
        tested_paper,
        25p_semantic_score,
        50p_semantic_score,
        75p_semantic_score,
        average_semantic_score,
        25p_cs_score, 
        50p_cs_score,
        75p_cs_score,
        average_cs_score
    )
    VALUES %s
    ON CONFLICT (tested_paper) DO NOTHING;
    """
    
    data = [(target_paper, *paper) for paper in results]
    
    # Insert the data
    execute_values(db_client.cur, insert_query, data)
    db_client.commit()
    return