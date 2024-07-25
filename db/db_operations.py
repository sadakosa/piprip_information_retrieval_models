

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