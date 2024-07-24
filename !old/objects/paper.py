class Paper:
    def __init__(self, ss_id, title, abstract, title_tokens=None, abstract_tokens=None):
        self.ss_id = ss_id
        self.title = title
        self.abstract = abstract
        self.title_tokens = title_tokens or []
        self.abstract_tokens = abstract_tokens or []
        # self.combined_score will be added later

    def __repr__(self):
        return f"Paper(ID: {self.ss_id}, Title: {self.title})"
