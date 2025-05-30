import psycopg2
# from psycopg2.extras import execute_values


class DBClient:
    def __init__(self, db_name, user, password, host, port):
        self.conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )
        self.cur = self.conn.cursor()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    # def execute(self, query, params=None):
    #     self.cur.execute(query, params)
    #     return self.cur
    
    def execute(self, query, params=None):
        try:
            # print(f"Executing query: {query} with params: {params}")
            self.cur.execute(query, params)
            return self.cur
        except Exception as e:
            print(f"Error executing query: {e}")
            self.rollback()
            raise


    def commit(self):
        # print("Committing transaction")
        self.conn.commit()

    def begin(self):
        # Start a new transaction block
        self.cur.execute("BEGIN")

    def rollback(self):
        # Roll back the current transaction
        self.conn.rollback()