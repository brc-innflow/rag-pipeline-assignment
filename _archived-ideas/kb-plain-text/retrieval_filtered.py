"""
Code was generated using AI assistance

Metadata-Filtered RAG - Retrieval Module (Plain Text)
Performs metadata filtering BEFORE vector search for more targeted results.

Supported filters:
- category: Filter by category (e.g., "Accounting", "Labor", "Inventory")
- subtopic: Filter by subtopic (e.g., "User Guides", "Payroll")
- title: Filter by article title (exact or list match)
- title_contains: Filter by partial title match (case-insensitive regex)
"""

import os
import certifi
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (must match ingestion.py)
DB_NAME = "rag_assignment"
COLLECTION_NAME = "kb_articles"
INDEX_NAME = "kb_vector_index"

# Retrieval configuration
DEFAULT_TOP_K = 10


def get_mongo_client():
    """Get MongoDB client."""
    return MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())


def get_vector_store():
    """Connect to the MongoDB vector store."""
    client = get_mongo_client()
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


def build_pre_filter(
    category: Optional[str | list[str]] = None,
    subtopic: Optional[str | list[str]] = None,
    title: Optional[str | list[str]] = None,
) -> dict:
    """
    Build a MongoDB pre-filter for vector search.
    
    Args:
        category: Single category or list of categories to include
        subtopic: Single subtopic or list of subtopics to include
        title: Single title or list of titles (exact match)
    
    Returns:
        MongoDB filter dictionary
    """
    conditions = []
    
    # Category filter
    if category is not None:
        if isinstance(category, list):
            conditions.append({"category": {"$in": category}})
        else:
            conditions.append({"category": {"$eq": category}})
    
    # Subtopic filter
    if subtopic is not None:
        if isinstance(subtopic, list):
            conditions.append({"subtopic": {"$in": subtopic}})
        else:
            conditions.append({"subtopic": {"$eq": subtopic}})
    
    # Title filter (exact match)
    if title is not None:
        if isinstance(title, list):
            conditions.append({"title": {"$in": title}})
        else:
            conditions.append({"title": {"$eq": title}})
    
    # Combine all conditions with AND
    if not conditions:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}


def retrieve_with_filter(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str | list[str]] = None,
    subtopic: Optional[str | list[str]] = None,
    title: Optional[str | list[str]] = None,
    verbose: bool = False,
) -> list:
    """
    Retrieve documents with metadata pre-filtering.
    
    Args:
        query: The search query string
        top_k: Number of documents to retrieve
        category: Filter by category/categories
        subtopic: Filter by subtopic(s)
        title: Filter by title(s) - exact match
        verbose: Print filter details
    
    Returns:
        List of relevant document chunks
    """
    vector_store, client = get_vector_store()
    
    try:
        # Build pre-filter
        pre_filter = build_pre_filter(
            category=category,
            subtopic=subtopic,
            title=title,
        )
        
        if verbose and pre_filter:
            print(f"   Pre-filter: {pre_filter}")
        
        # Perform filtered similarity search
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


def retrieve_with_filter_and_scores(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    category: Optional[str | list[str]] = None,
    subtopic: Optional[str | list[str]] = None,
    title: Optional[str | list[str]] = None,
) -> list:
    """
    Retrieve documents with metadata pre-filtering and similarity scores.
    
    Returns:
        List of tuples (document, score)
    """
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(
            category=category,
            subtopic=subtopic,
            title=title,
        )
        
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


def format_retrieved_context(documents: list) -> str:
    """Format retrieved documents into a context string for the LLM."""
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        subtopic = doc.metadata.get("subtopic", "")
        source_url = doc.metadata.get("source_url", "Unknown")
        
        # Build category label
        category_label = category
        if subtopic:
            category_label += f" / {subtopic}"
        
        context_parts.append(
            f"[Document {i}]\n"
            f"Title: {title}\n"
            f"Category: {category_label}\n"
            f"Source: {source_url}\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def get_available_categories() -> list[str]:
    """Get list of categories available in the collection."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        categories = collection.distinct("category")
        return sorted([c for c in categories if c])
    finally:
        client.close()


def get_available_subtopics(category: Optional[str] = None) -> list[str]:
    """
    Get list of subtopics available in the collection.
    Optionally filter by category.
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        if category:
            subtopics = collection.distinct("subtopic", {"category": category})
        else:
            subtopics = collection.distinct("subtopic")
        return sorted([s for s in subtopics if s])
    finally:
        client.close()


def get_category_counts() -> list[tuple[str, int]]:
    """Get counts of each category in the collection."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        pipeline = [
            {"$group": {"_id": "$category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results if r["_id"]]
    finally:
        client.close()


def get_subtopic_counts(category: Optional[str] = None) -> list[tuple[str, int]]:
    """
    Get counts of each subtopic in the collection.
    Optionally filter by category.
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        match_stage = {"$match": {"subtopic": {"$ne": None, "$ne": ""}}}
        if category:
            match_stage = {"$match": {"category": category, "subtopic": {"$ne": None, "$ne": ""}}}
        
        pipeline = [
            match_stage,
            {"$group": {"_id": "$subtopic", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results if r["_id"]]
    finally:
        client.close()


def get_title_counts(category: Optional[str] = None, limit: int = 20) -> list[tuple[str, int]]:
    """
    Get counts of chunks per article title.
    Optionally filter by category.
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        match_stage = {}
        if category:
            match_stage = {"$match": {"category": category}}
        
        pipeline = []
        if match_stage:
            pipeline.append(match_stage)
        
        pipeline.extend([
            {"$group": {"_id": "$title", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ])
        results = list(collection.aggregate(pipeline))
        return [(r["_id"], r["count"]) for r in results if r["_id"]]
    finally:
        client.close()


def debug_collection():
    """Debug function to check MongoDB collection status and available metadata."""
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    print("=" * 60)
    print("DEBUG: MongoDB Collection Status (Metadata-Filtered)")
    print("=" * 60)
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Index: {INDEX_NAME}")
    
    doc_count = collection.count_documents({})
    print(f"\nTotal documents in collection: {doc_count}")
    
    if doc_count > 0:
        sample = collection.find_one()
        print(f"\nSample document fields: {list(sample.keys())}")
        print(f"\nSample metadata:")
        print(f"  title: {sample.get('title', 'N/A')}")
        print(f"  category: {sample.get('category', 'N/A')}")
        print(f"  subtopic: {sample.get('subtopic', 'N/A')}")
        print(f"  source_url: {sample.get('source_url', 'N/A')}")
        
        # Show available categories
        print("\nCategory counts:")
        for category, count in get_category_counts():
            print(f"  {category}: {count}")
        
        # Show subtopic counts
        print("\nSubtopic counts:")
        for subtopic, count in get_subtopic_counts():
            print(f"  {subtopic}: {count}")
        
        # Show top titles
        print("\nTop 10 articles by chunk count:")
        for title, count in get_title_counts(limit=10):
            print(f"  {title[:50]}...: {count}" if len(title) > 50 else f"  {title}: {count}")
    
    client.close()
    return doc_count


def main():
    """Test retrieval with various filters."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Retrieval Test (Plain Text)")
    print("=" * 60)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Debug collection first
    doc_count = debug_collection()
    
    if doc_count == 0:
        print("\nNo documents found! Run ingestion.py first.")
        return
    
    # Test queries with different filters
    print("\n" + "=" * 60)
    print("Running Filtered Retrieval Tests")
    print("=" * 60)
    
    # Test 1: No filter (baseline)
    test_query = "What is time clock?"
    print("\nTest 1: No filter (baseline)")
    print(f"   Query: {test_query}")
    results = retrieve_with_filter(test_query, top_k=3, verbose=True)
    print(f"   Retrieved: {len(results)} documents")
    for doc in results:
        print(f"   - {doc.metadata.get('title', 'Unknown')[:50]}... | Category: {doc.metadata.get('category')}")
    
    # Test 2: Filter by category
    categories = get_available_categories()
    if categories:
        test_category = categories[0]
        print(f"\nTest 2: Filter by category='{test_category}'")
        print(f"   Query: {test_query}")
        results = retrieve_with_filter(
            test_query, 
            top_k=3, 
            category=test_category,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            print(f"   - {doc.metadata.get('title', 'Unknown')[:50]}... | Category: {doc.metadata.get('category')}")
    
    # Test 3: Filter by multiple categories
    if len(categories) >= 2:
        test_categories = categories[:2]
        print(f"\nTest 3: Filter by categories={test_categories}")
        print(f"   Query: {test_query}")
        results = retrieve_with_filter(
            test_query,
            top_k=3,
            category=test_categories,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            print(f"   - {doc.metadata.get('title', 'Unknown')[:50]}... | Category: {doc.metadata.get('category')}")
    
    # Test 4: Filter by subtopic (if available)
    subtopics = get_available_subtopics()
    if subtopics:
        test_subtopic = subtopics[0]
        print(f"\nTest 4: Filter by subtopic='{test_subtopic}'")
        results = retrieve_with_filter(
            test_query,
            top_k=3,
            subtopic=test_subtopic,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            print(f"   - {doc.metadata.get('title', 'Unknown')[:50]}... | Subtopic: {doc.metadata.get('subtopic')}")
    
    # Test 5: Combined filters
    if categories and subtopics:
        print(f"\nTest 5: Combined filters (category + subtopic)")
        results = retrieve_with_filter(
            test_query,
            top_k=3,
            category=categories[0],
            subtopic=subtopics[0] if subtopics else None,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            cat = doc.metadata.get('category', 'Unknown')
            sub = doc.metadata.get('subtopic', '')
            print(f"   - {doc.metadata.get('title', 'Unknown')[:40]}... | {cat}/{sub}")


if __name__ == "__main__":
    main()
