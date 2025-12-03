import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import together
import os
import sys

# Append the project root to sys.path for module discovery
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Import configuration settings from model_config module
from model_config import *


class SimpleRAGTool:
    """
    A simple Retrieval Augmented Generation (RAG) tool that indexes documents using FAISS and
    generates responses using an LLM based on retrieved context.
    """

    def __init__(self):
        """
        Initialize the FAISS index, embedding model, and generation model.

        - Initializes the SentenceTransformer for generating embeddings.
        - Sets up a FAISS index for cosine similarity search.
        - Prepares document storage and a generation model via Together.
        """
        # Initialize the embedding model using a pre-defined EMBEDDING_MODEL
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL)
        self.embedding_dim = 384  # Dimension for all-MiniLM-L6-v2

        # Initialize FAISS index for cosine similarity
        # Using IndexFlatIP with normalized vectors to compute cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Storage for document texts with unique IDs
        self.documents: Dict[int, str] = {}
        self.current_id: int = 0

        # Setup generation model using Together
        together.api_key = API_KEY
        self.generation_model = GENERATION_MODEL

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.

        Parameters:
        * vectors: np.ndarray of embeddings

        Returns:
        * Normalized vectors (in-place normalization by FAISS)
        """
        faiss.normalize_L2(vectors)  # Normalizes vectors in-place
        return vectors

    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.

        Parameters:
        * text: The input text to be embedded

        Returns:
        * Normalized embedding as a np.ndarray
        """
        embedding = self.embed_model.encode([text])[0]
        return self._normalize_vectors(embedding.reshape(1, -1))

    def add_document(self, text: str) -> int:
        """
        Add a single document to the index and store its text.

        Parameters:
        * text: The document content

        Returns:
        * Unique document ID
        """
        # Generate embedding for the document
        embedding = self.generate_embedding(text)

        # Add embedding to FAISS index
        self.index.add(embedding)

        # Store document text with a unique ID
        doc_id = self.current_id
        self.documents[doc_id] = text
        self.current_id += 1

        return doc_id

    def add_documents(self, texts: List[str]) -> List[int]:
        """
        Add multiple documents to the index and store their texts.

        Parameters:
        * texts: List of document contents

        Returns:
        * List of unique document IDs
        """
        # Generate embeddings for all documents
        embeddings = self.embed_model.encode(texts)
        embeddings = self._normalize_vectors(embeddings)

        # Add embeddings to FAISS index
        self.index.add(embeddings)

        # Store documents and assign unique IDs
        doc_ids = []
        for text in texts:
            doc_id = self.current_id
            self.documents[doc_id] = text
            doc_ids.append(doc_id)
            self.current_id += 1

        return doc_ids

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, any]]:
        """
        Search for the top_k most similar documents given a query.

        Parameters:
        * query: The input query text
        * top_k: Number of top documents to retrieve (default: 2)

        Returns:
        * List of dictionaries containing document ID, content, and similarity score
        """
        # Generate embedding for the query
        query_embedding = self.generate_embedding(query)

        # Perform search in the FAISS index
        similarities, indices = self.index.search(query_embedding, top_k)

        # Format the search results
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if idx != -1:
                doc = self.documents[int(idx)]
                results.append(
                    {"id": int(idx), "content": doc, "similarity": float(similarity)}
                )

        return results

    def generate_response(
        self, query: str, context: List[Dict], prompt_template: Optional[str] = None
    ):
        """
        Generate a response using the LLM based on the query and retrieved context.

        Parameters:
        * query: The user's question
        * context: List of retrieved documents with similarity scores
        * prompt_template: (Optional) Custom prompt template

        Returns:
        * A generated text response from the LLM
        """
        # Use default prompt template if none is provided
        if not prompt_template:
            prompt_template = """
                You are a helpful AI assistant. Use the following retrieved documents to answer the user's question.
                If the information is not in the documents, say so. Do not make up information.
                
                Retrieved Documents:
                {context}
                
                User Question: {query}
                
                Using ONLY the information from the retrieved documents, provide a clear and concise answer.
                Do not return the context as-is; formulate a smooth response for a chat interface.
            """

        # Format the retrieved documents for the prompt
        formatted_context = "\n\n".join(
            [
                f"Document {i + 1} (Similarity: {doc['similarity']:.2f}):\n{doc['content']}"
                for i, doc in enumerate(context)
            ]
        )

        # Create the final prompt using the template
        final_prompt = prompt_template.format(context=formatted_context, query=query)

        # Generate response using the Together LLM
        response = together.Complete.create(
            prompt=final_prompt,
            model=self.generation_model,
            max_tokens=512,
            temperature=0.3,
            stop=["User:", "Assistant:"],
        )

        # Debug: Print the response type
        print(type(response))

        # Return the generated text response after stripping whitespace
        return response["choices"][0]["text"].strip()

    def run(self, query: str):
        """
        Execute the RAG process:
        - Searches for similar documents based on the query
        - Generates an LLM response using the retrieved context

        Parameters:
        * query: The user's question

        Returns:
        * The final response generated by the LLM
        """
        # Retrieve relevant documents
        context = self.search(query)

        # Generate response using the query and retrieved context
        response = self.generate_response(query, context)

        return response
