# information_retrieval_models

## Purpose
For search and evaluation
1. Search - citation similarity and semantic similarity
2. Evaluation 

## Set Up
1. config/config.yaml
2. resources/results

## Description
To parse research paper titles and abstracts for information retrieval
1. Test BM2 and other variations
2. Test BERT and other variations

**BM25 (Best Matching 25)**
BM25 is a ranking function used by search engines to estimate the relevance of documents to a given search query. It is part of the family of probabilistic information retrieval models and is an evolution of the classic TF-IDF (Term Frequency-Inverse Document Frequency) model. The key features of BM25 include:
- Term Frequency (TF): The frequency of a term in a document.
- Inverse Document Frequency (IDF): A measure of how much information the word provides, based on its rarity across documents.
- Document Length Normalization: Adjusts for the length of documents, giving a fairer comparison between short and long documents.
BM25's formula enhances TF-IDF by adding parameters that control term frequency saturation and document length normalization, making it more effective for large and diverse document collections.

**BERT (Bidirectional Encoder Representations from Transformers)**
BERT is a deep learning model developed by Google that has revolutionized natural language processing (NLP). It uses transformers, a type of neural network architecture, to understand the context of words in a sentence by considering the entire sentence rather than processing words in isolation. Key features of BERT include:
- Bidirectional Context: BERT reads text in both directions (left-to-right and right-to-left), allowing it to understand context more effectively.
- Pre-training and Fine-tuning: BERT is pre-trained on large text corpora and can be fine-tuned for specific tasks, such as question answering or sentiment analysis.
- Transfer Learning: It leverages transfer learning, where the knowledge gained from the pre-training phase is applied to various downstream tasks.

**Comparison: BM25 vs BERT**
- Complexity: BM25 is a relatively simple and efficient model, while BERT is a complex and resource-intensive model requiring significant computational power.
- Context Understanding: BM25 relies on term frequency and document statistics, making it less effective in understanding context compared to BERT, which uses deep learning to capture contextual relationships.
- Use Cases: BM25 is commonly used in search engines and information retrieval systems where efficiency is critical. BERT, on the other hand, is used in tasks requiring deep contextual understanding, such as machine translation, question answering, and named entity recognition.


## On BM25
https://pypi.org/project/rank-bm25/

- to save the tokenised papers in resources under 'tokenised_scientific.txt' or whatever
1. Load the raw paper data into a dataframe.
2. Tokenize and process the data in the dataframe.
3. Use the dataframe to initialize BM25 and run queries.
5. Store the results back in the dataframe and manipulate it as needed.

## On Citation Similarity
We have either full citation similarity or in the moment citation similarity. 

**Full citation similarity:**
1. Get and Save Data: Get non duplicated co citation from the database and save as CSV
2. Load Data: Load co-citation and bibliographic coupling data from CSV files.
3. Combine Data: Merge the co-citation and bibliographic coupling dataframes on the pairs of papers.
4. Calculate Combined Score: Compute a combined similarity score for each pair of papers.
5. Sort and Rank: Sort the pairs based on the combined score to determine the ranking.
6. Output Results: Save or return the results.

## On Bert





### On Computation Power for BERT
The time it will take to run 300,000 abstracts through BERT largely depends on several factors including:
- Hardware: The specifications of the GPU/TPU or CPU you're using.
- Batch Size: The number of abstracts processed in one batch.
- Sequence Length: The length of each abstract (number of tokens).
- Model Variant: The specific BERT model being used (e.g., BERT-base, BERT-large).
- Optimization: Any optimizations or accelerations in your code.

**Estimated cost by ChatGPT:**
- p2.xlarge Instance: Estimated Cost: $1.89 for approximately 2.1 hours.
- p3.2xlarge Instance -> Estimated Cost: Approximately $0.64 for approximately 12.5 minutes of processing time. Total Cost: 0.2083 hours * $3.06 per hour ≈ $0.64
- c5.large Instance: Estimated Cost: $4.17 for approximately 41.67 hours.


## On Database queries
- 
```
// Creating indexes on your database tables can significantly improve the performance of queries, especially those involving joins and where clauses
CREATE INDEX idx_papers_ss_id ON papers(ss_id);
CREATE INDEX idx_references_ss_id ON references(ss_id);
CREATE INDEX idx_references_reference_id ON references(reference_id);

// query that first identifies the citing papers and then finds the similar papers that those citing papers reference
WITH citing_papers AS (
    SELECT r1.ss_id AS cited_paper, r1.reference_id AS citing_paper
    FROM references r1
    WHERE r1.ss_id = 'YOUR_SS_ID_HERE'
),
similar_papers AS (
    SELECT r2.reference_id AS similar_paper
    FROM references r2
    INNER JOIN citing_papers cp
    ON r2.ss_id = cp.citing_paper
)
SELECT sp.similar_paper, COUNT(sp.similar_paper) AS num_references
FROM similar_papers sp
GROUP BY sp.similar_paper
ORDER BY num_references DESC;


```


## In Resources Folder
The save_to_json function in global_methods handles this
- bm25: saves bm25 objects
- objects: Paper and RankedPapers object definitions
- results: results
- test_data: test_data
- tokenised_text: to save tokenised texts