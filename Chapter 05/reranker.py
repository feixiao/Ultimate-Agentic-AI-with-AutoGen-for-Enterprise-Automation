"""
Reranker module implementing BM25 and CrossEncoder based re-ranking of documents.
"""

# Import required libraries
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Initialize CrossEncoder model globally with a pre-trained model
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def bm25_rerank(query, documents):
    """
    Re-rank documents using the BM25 algorithm.

    Parameters:
    * query: string query to match
    * documents: list of document strings

    Returns:
    * List of documents sorted by relevance score
    """
    # Tokenize each document into words
    tokenized_docs = [doc.split() for doc in documents]

    # Create a BM25 model with tokenized documents
    bm25 = BM25Okapi(tokenized_docs)

    # Compute BM25 scores for the tokenized query
    scores = bm25.get_scores(query.split())

    # Pair each document with its score and sort by descending score
    ranked_results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    # Return only the documents in sorted order
    return [doc for doc, _ in ranked_results]


def cross_encode_rerank(query, candidates):
    """
    Re-rank candidate documents using the CrossEncoder model.

    Parameters:
    * query: string query for evaluation
    * candidates: list of candidate document strings

    Returns:
    * List of candidate documents sorted by relevance score
    """
    # Create (query, candidate) pairs for prediction
    paired_inputs = [(query, doc) for doc in candidates]

    # Predict relevance scores for each pair
    scores = cross_encoder.predict(paired_inputs)

    # Pair each candidate with its score and sort by descending score
    ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # Return only the candidate documents in sorted order
    return [doc for doc, _ in ranked_results]


if __name__ == "__main__":
    # Example usage for BM25 Reranker
    documents = [
        "User prefers detailed answers",
        "User likes technical explanations",
        "User enjoys summaries",
    ]
    query = "Give a detailed response"
    print("BM25 Reranked Results:")
    print(bm25_rerank(query, documents))  # Print BM25 re-ranked results

    # Example usage for CrossEncoder Reranker
    query = "Explain in detail"
    candidates = ["User prefers summaries", "User likes detailed explanations"]
    print("Cross-encoded Reranked Results:")
    print(
        cross_encode_rerank(query, candidates)
    )  # Print CrossEncoder re-ranked results
