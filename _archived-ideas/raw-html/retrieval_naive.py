"""
Code was generated using AI assistance

RAG Pipeline - Retrieval Module (HTML Header-Based)
Performs vector search against MongoDB to retrieve relevant KB chunks.
Supports filtering by category, subtopic, and HTML header hierarchy (header_1, header_2, header_3).
"""

import os
import certifi
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (must match web_ingestion_html.py)
DB_NAME = "rag_assignment"
COLLECTION_NAME = "kb_articles_html"
INDEX_NAME = "kb_vector_index_html"

# Retrieval configuration
DEFAULT_TOP_K = 10


def get_vector_store():
    """Connect to the MongoDB vector store."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    vector_store = MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    
    return vector_store, client


def retrieve_chunks(
    query: str, 
    top_k: int = DEFAULT_TOP_K,
    category: str = None,
    subtopic: str = None,
    header_1: str = None,
    header_2: str = None,
    header_3: str = None
) -> list:
    """
    Retrieve the top-k most relevant chunks for a given query.
    
    Args:
        query: The search query string
        top_k: Number of chunks to retrieve (default: 5)
        category: Optional filter by category (e.g., "Accounting")
        subtopic: Optional filter by subtopic (e.g., "User Guides")
        header_1: Optional filter by H1 header section
        header_2: Optional filter by H2 header section
        header_3: Optional filter by H3 header section
    
    Returns:
        List of relevant chunks with metadata
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build pre-filter if any filters specified
        pre_filter = None
        conditions = []
        
        if category:
            conditions.append({"category": {"$eq": category}})
        if subtopic:
            conditions.append({"subtopic": {"$eq": subtopic}})
        if header_1:
            conditions.append({"header_1": {"$eq": header_1}})
        if header_2:
            conditions.append({"header_2": {"$eq": header_2}})
        if header_3:
            conditions.append({"header_3": {"$eq": header_3}})
        
        if conditions:
            if len(conditions) == 1:
                pre_filter = conditions[0]
            else:
                pre_filter = {"$and": conditions}
        
        # Perform similarity search
        if pre_filter:
            results = vector_store.similarity_search(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search(
                query=query,
                k=top_k
            )
        
        return results
    finally:
        client.close()


def retrieve_chunks_with_scores(
    query: str, 
    top_k: int = DEFAULT_TOP_K,
    category: str = None,
    subtopic: str = None,
    header_1: str = None,
    header_2: str = None,
    header_3: str = None
) -> list:
    """
    Retrieve the top-k most relevant chunks with similarity scores.
    
    Args:
        query: The search query string
        top_k: Number of chunks to retrieve (default: 5)
        category: Optional filter by category
        subtopic: Optional filter by subtopic
        header_1: Optional filter by H1 header section
        header_2: Optional filter by H2 header section
        header_3: Optional filter by H3 header section
    
    Returns:
        List of tuples (chunk, score)
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build pre-filter if any filters specified
        pre_filter = None
        conditions = []
        
        if category:
            conditions.append({"category": {"$eq": category}})
        if subtopic:
            conditions.append({"subtopic": {"$eq": subtopic}})
        if header_1:
            conditions.append({"header_1": {"$eq": header_1}})
        if header_2:
            conditions.append({"header_2": {"$eq": header_2}})
        if header_3:
            conditions.append({"header_3": {"$eq": header_3}})
        
        if conditions:
            if len(conditions) == 1:
                pre_filter = conditions[0]
            else:
                pre_filter = {"$and": conditions}
        
        # Perform similarity search with scores
        if pre_filter:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k,
                pre_filter=pre_filter
            )
        else:
            results = vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
        
        return results
    finally:
        client.close()


def format_retrieved_context(chunks: list) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    Includes header hierarchy information for better context.
    
    Args:
        chunks: List of retrieved chunk objects
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, doc in enumerate(chunks, 1):
        title = doc.metadata.get("title", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        subtopic = doc.metadata.get("subtopic", "")
        source_url = doc.metadata.get("source_url", "Unknown")
        
        # Get header hierarchy
        header_1 = doc.metadata.get("header_1", "")
        header_2 = doc.metadata.get("header_2", "")
        header_3 = doc.metadata.get("header_3", "")
        
        # Build category label
        category_label = category
        if subtopic:
            category_label += f" / {subtopic}"
        
        # Build section path from headers
        section_parts = []
        if header_1:
            section_parts.append(header_1)
        if header_2:
            section_parts.append(header_2)
        if header_3:
            section_parts.append(header_3)
        section_path = " > ".join(section_parts) if section_parts else "N/A"
        
        context_parts.append(
            f"[Chunk {i}]\n"
            f"Title: {title}\n"
            f"Category: {category_label}\n"
            f"Section: {section_path}\n"
            f"Source: {source_url}\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def debug_collection():
    """Debug function to check MongoDB collection status."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print("=" * 50)
    print("DEBUG: MongoDB Collection Status (HTML-Based)")
    print("=" * 50)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Index: {INDEX_NAME}")
    
    # Check chunk count
    doc_count = collection.count_documents({})
    print(f"\nTotal chunks in collection: {doc_count}")
    
    if doc_count > 0:
        # Get a sample chunk to check structure
        sample = collection.find_one()
        print(f"\nSample chunk fields: {list(sample.keys())}")
        
        # Check if embedding field exists
        if "embedding" in sample:
            print(f"Embedding field exists with {len(sample['embedding'])} dimensions")
        else:
            print("WARNING: 'embedding' field NOT found!")
            print(f"Available fields: {list(sample.keys())}")
        
        # Show sample metadata
        print(f"\nSample metadata:")
        print(f"  Title: {sample.get('title', 'N/A')}")
        print(f"  Category: {sample.get('category', 'N/A')}")
        print(f"  Subtopic: {sample.get('subtopic', 'N/A')}")
        print(f"  Header 1: {sample.get('header_1', 'N/A')}")
        print(f"  Header 2: {sample.get('header_2', 'N/A')}")
        print(f"  Header 3: {sample.get('header_3', 'N/A')}")
        print(f"  Source URL: {sample.get('source_url', 'N/A')}")
        
        # Get unique categories
        categories = collection.distinct("category")
        print(f"\nUnique categories: {categories}")
        
        # Get unique subtopics
        subtopics = collection.distinct("subtopic")
        print(f"Unique subtopics: {subtopics}")
        
        # Get unique header_1 values (top-level sections)
        headers_1 = collection.distinct("header_1")
        print(f"Unique H1 headers: {len(headers_1)} found")
        if headers_1 and len(headers_1) <= 10:
            for h in headers_1:
                if h:
                    print(f"  - {h}")
    
    # List indexes
    print("\nIndexes on collection:")
    for index in collection.list_indexes():
        print(f"  - {index['name']}: {index.get('key', 'N/A')}")
    
    # Try to list search indexes (Atlas Vector Search)
    try:
        search_indexes = list(collection.list_search_indexes())
        print(f"\nVector Search Indexes: {len(search_indexes)}")
        for idx in search_indexes:
            print(f"  - Name: {idx.get('name')}")
            print(f"    Status: {idx.get('status', 'unknown')}")
    except Exception as e:
        print(f"\nCould not list search indexes: {e}")
    
    client.close()
    return doc_count


def main():
    """Test retrieval with a sample query."""
    print("=" * 50)
    print("RAG Pipeline - Retrieval Test (HTML Header-Based)")
    print("=" * 50)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # First, debug the collection
    doc_count = debug_collection()
    
    if doc_count == 0:
        print("\nNo chunks found! Run web_ingestion_html.py first.")
        return
    
    print("\n" + "=" * 50)
    print("Running Retrieval Test")
    print("=" * 50)
    
    # Test query - adjust this based on your knowledge base content
    test_query = "What is time clock?"
    print(f"\nQuery: {test_query}")
    print("-" * 50)
    
    # Test 1: Basic retrieval (no filters)
    print("\n--- Test 1: Basic Retrieval (no filters) ---")
    results = retrieve_chunks_with_scores(test_query, top_k=3)
    
    print(f"\nRetrieved {len(results)} chunks:\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Chunk {i} (Score: {score:.4f}) ---")
        print(f"Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Category: {doc.metadata.get('category', 'Unknown')}")
        if doc.metadata.get('subtopic'):
            print(f"Subtopic: {doc.metadata.get('subtopic')}")
        
        # Show header hierarchy
        headers = []
        if doc.metadata.get('header_1'):
            headers.append(f"H1: {doc.metadata.get('header_1')}")
        if doc.metadata.get('header_2'):
            headers.append(f"H2: {doc.metadata.get('header_2')}")
        if doc.metadata.get('header_3'):
            headers.append(f"H3: {doc.metadata.get('header_3')}")
        if headers:
            print(f"Section: {' > '.join(headers)}")
        
        print(f"Source: {doc.metadata.get('source_url', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200]}...")
        print()
    
    # Test 2: Filtered retrieval (if categories exist)
    print("\n--- Test 2: Filtered Retrieval (by category) ---")
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    categories = collection.distinct("category")
    client.close()
    
    if categories:
        test_category = "Labor" # categories[0]
        print(f"Filtering by category: {test_category}")
        
        filtered_results = retrieve_chunks_with_scores(
            test_query, 
            top_k=3, 
            category=test_category
        )
        
        print(f"\nRetrieved {len(filtered_results)} chunks:\n")
        for i, (doc, score) in enumerate(filtered_results, 1):
            print(f"--- Chunk {i} (Score: {score:.4f}) ---")
            print(f"Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"Category: {doc.metadata.get('category', 'Unknown')}")
            if doc.metadata.get('header_2'):
                print(f"Section (H2): {doc.metadata.get('header_2')}")
            print(f"Content preview: {doc.page_content[:150]}...")
            print()
    else:
        print("No categories found to test filtering.")
    
    # Show formatted context
    #print("\n" + "=" * 50)
    #print("Formatted Context for LLM:")
    #print("=" * 50)
    #chunks = [doc for doc, _ in results]
    #context = format_retrieved_context(chunks)
    #print(context[:1500] + "..." if len(context) > 1500 else context)


if __name__ == "__main__":
    main()
