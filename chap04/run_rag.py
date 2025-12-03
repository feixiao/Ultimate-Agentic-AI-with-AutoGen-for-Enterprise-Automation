"""
Module: run_rag
Description: Demonstrates the usage of SimpleRAGTool for Retrieval Augmented Generation (RAG).
"""

from tool_rag import SimpleRAGTool


def main():
    """
    Main function to initialize the RAG tool, add documents, perform search queries, and generate a response.
    """
    # Initialize the Retrieval Augmented Generation (RAG) tool
    rag = SimpleRAGTool()

    # Example documents for indexing
    documents = [
        "OpenAI was founded in 2015 with the goal of ensuring AGI benefits humanity.",
        "Python is a programming language created by Guido van Rossum.",
        "Machine learning is a subset of artificial intelligence.",
        "FAISS is a library for efficient similarity search developed by Facebook.",
        "The transformer architecture was introduced in the paper 'Attention is All You Need'.",
    ]

    # Add documents to the RAG index
    print("Adding documents...")
    doc_ids = rag.add_documents(documents)
    print(f"Added {len(doc_ids)} documents with IDs: {doc_ids}")

    # Test single document addition
    new_doc = "Vector databases are essential for modern information retrieval."
    new_id = rag.add_document(new_doc)
    print(f"\nAdded single document with ID: {new_id}")

    # Test search functionality
    queries = ["Transformer paper"]
    print("\nTesting searches:")
    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.search(query, top_k=2)
        # Iterate through search results
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Document ID: {result['id']}")
            print(f"Similarity Score: {result['similarity']:.4f}")
            print(f"Content: {result['content']}")

    # Generate a response based on the query and retrieved context
    response = rag.generate_response(query=queries[0], context=results)
    print(f"\n ############# \nFinal Response: {response}")


if __name__ == "__main__":
    main()
