from resources.objects.paper import Paper


# ranked_papers_dict = {
#     "algo_used": "BM25",
#     "papers": [
#         {
#             "ss_id": "12345",
#             "title": "Sample Paper Title",
#             "abstract": "This is a sample abstract of the paper.",
#             "title_tokens": ["Sample", "Paper", "Title"],
#             "abstract_tokens": ["This", "is", "a", "sample", "abstract", "of", "the", "paper"]
#         },
#         {
#             "ss_id": "67890",
#             "title": "Another Sample Paper",
#             "abstract": "This is another sample abstract of the paper.",
#             "title_tokens": ["Another", "Sample", "Paper"],
#             "abstract_tokens": ["This", "is", "another", "sample", "abstract", "of", "the", "paper"]
#         }
#     ],
#     "map": { << this is the index of the paper in the papers list
#         "12345": 0,
#         "67890": 1
#     },
#     "bm25_ranking": ["12345", "67890"],
#     "overall_ranking": ["12345", "67890"]
# }
class RankedPapers:
    def __init__(self, algo_used):
        self.papers = []
        self.algo_used = algo_used
        self.map = {}
        self.bm25_ranking = []
        self.overall_ranking = []

    def add_paper(self, paper):
        if isinstance(paper, Paper):
            self.papers.append(paper)
            self.map[paper.ss_id] = len(self.papers) - 1
        else:
            raise TypeError("Only Paper objects can be added.")

    def rank_papers_by_score(self, scores):
        if self.algo_used == "BM25":
            for paper, score in zip(self.papers, scores):
                paper.combined_bm25_score = score
            sorted_indices = sorted(range(len(self.papers)), key=lambda i: self.papers[i].combined_bm25_score, reverse=True)
            self.bm25_ranking = [self.papers[i].ss_id for i in sorted_indices]
        else:
            for paper, score in zip(self.papers, scores):
                paper.combined_score = score
            sorted_indices = sorted(range(len(self.papers)), key=lambda i: self.papers[i].combined_score, reverse=True)
            self.overall_ranking = [self.papers[i].ss_id for i in sorted_indices]


    def __repr__(self):
        return "\n".join([str(paper) for paper in self.papers])

    def get_papers_by_bm25_rank(self):
        return [self.papers[self.map[ss_id]] for ss_id in self.bm25_ranking]

    def get_papers_by_overall_rank(self):
        return [self.papers[self.map[ss_id]] for ss_id in self.overall_ranking]