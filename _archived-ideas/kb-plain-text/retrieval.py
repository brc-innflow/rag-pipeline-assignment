"""
RAG Pipeline - Retrieval Module
Performs vector search against MongoDB to retrieve relevant kb_url chunks.
Supports filtering by category and subtopic.
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

# MongoDB configuration (must match web_ingestion.py)
DB_NAME = "rag_assignment"
COLLECTION_NAME = "kb_articles"
INDEX_NAME = "kb_vector_index"

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


def retrieve_kb_urls(
    query: str, 
    top_k: int = DEFAULT_TOP_K,
    category: str = None,
    subtopic: str = None
) -> list:
    """
    Retrieve the top-k most relevant kb_urls for a given query.
    
    Args:
        query: The search query string
        top_k: Number of kb_urls to retrieve (default: 5)
        category: Optional filter by category (e.g., "Accounting")
        subtopic: Optional filter by subtopic (e.g., "User Guides")
    
    Returns:
        List of relevant kb_url chunks with metadata
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build pre-filter if category or subtopic specified
        pre_filter = None
        if category or subtopic:
            conditions = []
            if category:
                conditions.append({"category": {"$eq": category}})
            if subtopic:
                conditions.append({"subtopic": {"$eq": subtopic}})
            
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


def retrieve_kb_urls_with_scores(
    query: str, 
    top_k: int = DEFAULT_TOP_K,
    category: str = None,
    subtopic: str = None
) -> list:
    """
    Retrieve the top-k most relevant kb_urls with similarity scores.
    
    Args:
        query: The search query string
        top_k: Number of kb_urls to retrieve (default: 5)
        category: Optional filter by category
        subtopic: Optional filter by subtopic
    
    Returns:
        List of tuples (kb_url, score)
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build pre-filter if category or subtopic specified
        pre_filter = None
        if category or subtopic:
            conditions = []
            if category:
                conditions.append({"category": {"$eq": category}})
            if subtopic:
                conditions.append({"subtopic": {"$eq": subtopic}})
            
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


def format_retrieved_context(kb_urls: list) -> str:
    """
    Format retrieved kb_urls into a context string for the LLM.
    
    Args:
        kb_urls: List of retrieved kb_url objects
    
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, doc in enumerate(kb_urls, 1):
        title = doc.metadata.get("title", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        subtopic = doc.metadata.get("subtopic", "")
        source_url = doc.metadata.get("source_url", "Unknown")
        
        # Build category label
        category_label = category
        if subtopic:
            category_label += f" / {subtopic}"
        
        context_parts.append(
            f"[kb_url {i}]\n"
            f"Title: {title}\n"
            f"Category: {category_label}\n"
            f"Source: {source_url}\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def debug_collection():
    """Debug function to check MongoDB collection status."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print("=" * 50)
    print("DEBUG: MongoDB Collection Status")
    print("=" * 50)
    
    # Check kb_url count
    doc_count = collection.count_documents({})
    print(f"\nTotal kb_urls in collection: {doc_count}")
    
    if doc_count > 0:
        # Get a sample kb_url to check structure
        sample = collection.find_one()
        print(f"\nSample kb_url fields: {list(sample.keys())}")
        
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
        print(f"  Source URL: {sample.get('source_url', 'N/A')}")
        
        # Get unique categories
        categories = collection.distinct("category")
        print(f"\nUnique categories: {categories}")
        
        # Get unique subtopics
        subtopics = collection.distinct("subtopic")
        print(f"Unique subtopics: {subtopics}")
    
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
    print("RAG Pipeline - Retrieval Test")
    print("=" * 50)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # First, debug the collection
    doc_count = debug_collection()
    
    if doc_count == 0:
        print("\nNo kb_urls found! Run web_ingestion.py first.")
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
    results = retrieve_kb_urls_with_scores(test_query, top_k=3)
    
    print(f"\nRetrieved {len(results)} kb_urls:\n")
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- kb_url {i} (Score: {score:.4f}) ---")
        print(f"Title: {doc.metadata.get('title', 'Unknown')}")
        print(f"Category: {doc.metadata.get('category', 'Unknown')}")
        if doc.metadata.get('subtopic'):
            print(f"Subtopic: {doc.metadata.get('subtopic')}")
        print(f"Source: {doc.metadata.get('source_url', 'Unknown')}")
        print(f"Content preview: {doc.page_content[:200]}...")
        print()
    
    # Test 2: Filtered retrieval (if categories exist)
    print("\n--- Test 2: Filtered Retrieval (by category) ---")
    # Get first available category for testing
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    categories = collection.distinct("category")
    client.close()
    
    if categories:
        test_category = "Labor" # categories[0]
        print(f"Filtering by category: {test_category}")
        
        filtered_results = retrieve_kb_urls_with_scores(
            test_query, 
            top_k=3, 
            category=test_category
        )
        
        print(f"\nRetrieved {len(filtered_results)} kb_urls:\n")
        for i, (doc, score) in enumerate(filtered_results, 1):
            print(f"--- kb_url {i} (Score: {score:.4f}) ---")
            print(f"Title: {doc.metadata.get('title', 'Unknown')}")
            print(f"Category: {doc.metadata.get('category', 'Unknown')}")
            print(f"Content preview: {doc.page_content[:150]}...")
            print()
    else:
        print("No categories found to test filtering.")
    
    # Show formatted context
    #print("\n" + "=" * 50)
    #print("Formatted Context for LLM:")
    #print("=" * 50)
    #kb_urls = [doc for doc, _ in results]
    #context = format_retrieved_context(kb_urls)
    #print(context[:1500] + "..." if len(context) > 1500 else context)


if __name__ == "__main__":
    main()
