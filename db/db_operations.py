

# ======================== SELECTION OPERATIONS ========================

def get_all_paper_ids(db_client):
    select_query = """
    SELECT ss_id, clean_title, clean_abstract FROM papers
    WHERE is_cleaned = True;
    """
    cursor = db_client.execute(select_query)
    return cursor.fetchall()





