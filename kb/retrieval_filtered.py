"""
Metadata-Filtered RAG - Retrieval Module (HTML Header-Based)
Performs metadata filtering BEFORE vector search for more targeted results.

Filter Logic: category AND subtopic AND (header_1 OR header_2 OR header_3 OR title)

Supports filtering by document metadata AND HTML header hierarchy:
- category: Filter by category (e.g., "Accounting", "Labor", "Inventory") - AND condition
- subtopic: Filter by subtopic (e.g., "User Guides", "Payroll") - AND condition
- title: Filter by article title - grouped with headers in OR condition
- header: Filter value(s) to match against header_1, header_2, header_3 - OR condition with title

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

# MongoDB configuration (must match web_ingestion_html.py)
DB_NAME = "rag_assignment"
COLLECTION_NAME = "kb_articles_html"
INDEX_NAME = "kb_vector_index_html"

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
    header: Optional[str | list[str]] = None
) -> dict:
    """
    Build a MongoDB pre-filter for vector search.
    
    Filter logic: category AND subtopic AND (header_1 OR header_2 OR header_3 OR title)
    
    Args:
        category: Single category or list of categories to include (AND condition)
        subtopic: Single subtopic or list of subtopics to include (AND condition)
        title: Single title or list of titles - grouped with headers in OR condition
        header: Filter value(s) to match against header_1, header_2, header_3, or title
    
    Returns:
        MongoDB filter dictionary
    """
    conditions = []
    
    # Category filter (AND condition)
    if category is not None:
        if isinstance(category, list):
            conditions.append({"category": {"$in": category}})
        else:
            conditions.append({"category": {"$eq": category}})
    
    # Subtopic filter (AND condition)
    if subtopic is not None:
        if isinstance(subtopic, list):
            conditions.append({"subtopic": {"$in": subtopic}})
        else:
            conditions.append({"subtopic": {"$eq": subtopic}})
    
    # Header and Title filter (OR condition: header_1 OR header_2 OR header_3 OR title)
    # Only apply if either header or title is provided
    if header is not None or title is not None:
        or_conditions = []
        
        # Add header conditions (header_1, header_2, header_3)
        if header is not None:
            if isinstance(header, list):
                or_conditions.append({"header_1": {"$in": header}})
                or_conditions.append({"header_2": {"$in": header}})
                or_conditions.append({"header_3": {"$in": header}})
            else:
                or_conditions.append({"header_1": {"$eq": header}})
                or_conditions.append({"header_2": {"$eq": header}})
                or_conditions.append({"header_3": {"$eq": header}})
        
        # Add title condition (part of the OR group)
        if title is not None:
            if isinstance(title, list):
                or_conditions.append({"title": {"$in": title}})
            else:
                or_conditions.append({"title": {"$eq": title}})
        
        if or_conditions:
            conditions.append({"$or": or_conditions})

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
    header: Optional[str | list[str]] = None,
    verbose: bool = False,
) -> list:
    """
    Retrieve documents with metadata pre-filtering.
    
    Filter logic: category AND subtopic AND (header_1 OR header_2 OR header_3 OR title)
    
    Args:
        query: The search query string
        top_k: Number of documents to retrieve
        category: Filter by category/categories (AND condition)
        subtopic: Filter by subtopic(s) (AND condition)
        title: Filter by title(s) - grouped with headers in OR condition
        header: Filter value(s) to match against header_1, header_2, header_3 (OR with title)
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
            header=header,
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
    header: Optional[str | list[str]] = None,
) -> list:
    """
    Retrieve documents with metadata pre-filtering and similarity scores.
    
    Filter logic: category AND subtopic AND (header_1 OR header_2 OR header_3 OR title)
    
    Returns:
        List of tuples (document, score)
    """
    vector_store, client = get_vector_store()
    
    try:
        pre_filter = build_pre_filter(
            category=category,
            subtopic=subtopic,
            title=title,
            header=header,
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
    """
    Format retrieved documents into a context string for the LLM.
    Includes header hierarchy information for better context.
    """
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
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
            f"[Document {i}]\n"
            f"Title: {title}\n"
            f"Category: {category_label}\n"
            f"Section: {section_path}\n"
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


def get_available_headers(
    header_level: int = 1,
    category: Optional[str] = None,
    title: Optional[str] = None
) -> list[str]:
    """
    Get list of unique headers at a specific level.
    
    Args:
        header_level: 1, 2, or 3 for H1, H2, H3
        category: Optional filter by category
        title: Optional filter by article title
    
    Returns:
        List of unique header values
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        field_name = f"header_{header_level}"
        query = {}
        if category:
            query["category"] = category
        if title:
            query["title"] = title
        
        if query:
            headers = collection.distinct(field_name, query)
        else:
            headers = collection.distinct(field_name)
        
        return sorted([h for h in headers if h])
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


def get_header_counts(
    header_level: int = 1,
    category: Optional[str] = None,
    limit: int = 20
) -> list[tuple[str, int]]:
    """
    Get counts of each header value at a specific level.
    
    Args:
        header_level: 1, 2, or 3 for H1, H2, H3
        category: Optional filter by category
        limit: Maximum number of results
    
    Returns:
        List of tuples (header_value, count)
    """
    client = get_mongo_client()
    collection = client[DB_NAME][COLLECTION_NAME]
    
    try:
        field_name = f"header_{header_level}"
        
        match_conditions = {field_name: {"$ne": None, "$ne": ""}}
        if category:
            match_conditions["category"] = category
        
        pipeline = [
            {"$match": match_conditions},
            {"$group": {"_id": f"${field_name}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
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
    print("DEBUG: MongoDB Collection Status (HTML Metadata-Filtered)")
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
        print(f"  header_1: {sample.get('header_1', 'N/A')}")
        print(f"  header_2: {sample.get('header_2', 'N/A')}")
        print(f"  header_3: {sample.get('header_3', 'N/A')}")
        print(f"  source_url: {sample.get('source_url', 'N/A')}")
        
        # Show available categories
        print("\nCategory counts:")
        for category, count in get_category_counts():
            print(f"  {category}: {count}")
        
        # Show subtopic counts
        print("\nSubtopic counts:")
        for subtopic, count in get_subtopic_counts():
            print(f"  {subtopic}: {count}")
        
        # Show H1 header counts
        print("\nTop H1 headers:")
        for header, count in get_header_counts(header_level=1, limit=10):
            display = header[:40] + "..." if len(header) > 40 else header
            print(f"  {display}: {count}")
        
        # Show H2 header counts
        print("\nTop H2 headers:")
        for header, count in get_header_counts(header_level=2, limit=10):
            display = header[:40] + "..." if len(header) > 40 else header
            print(f"  {display}: {count}")
        
        # Show top titles
        print("\nTop 10 articles by chunk count:")
        for title, count in get_title_counts(limit=10):
            print(f"  {title[:50]}...: {count}" if len(title) > 50 else f"  {title}: {count}")
    
    client.close()
    return doc_count


def main():
    """Test retrieval with various filters."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Retrieval Test (HTML Header-Based)")
    print("=" * 60)
    
    # Validate environment
    if not MONGO_DB_URL:
        raise ValueError("MONGO_DB_URL environment variable not set")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Debug collection first
    doc_count = debug_collection()
    
    if doc_count == 0:
        print("\nNo documents found! Run web_ingestion_html.py first.")
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
        title = doc.metadata.get('title', 'Unknown')[:40]
        h2 = doc.metadata.get('header_2', 'N/A')[:30] if doc.metadata.get('header_2') else 'N/A'
        print(f"   - {title}... | H2: {h2}")
    
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
            print(f"   - {doc.metadata.get('title', 'Unknown')[:40]}... | Category: {doc.metadata.get('category')}")
    
    # Test 3: Filter by header (searches all header levels with OR)
    h2_headers = get_available_headers(header_level=2)
    if h2_headers:
        test_header = h2_headers[0]
        print(f"\nTest 3: Filter by header='{test_header[:40]}...' (searches all levels)")
        print(f"   Query: {test_query}")
        results = retrieve_with_filter(
            test_query,
            top_k=3,
            header=test_header,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            h1 = doc.metadata.get('header_1', '')[:20] if doc.metadata.get('header_1') else ''
            h2 = doc.metadata.get('header_2', '')[:20] if doc.metadata.get('header_2') else ''
            print(f"   - {doc.metadata.get('title', 'Unknown')[:30]}... | H1: {h1} | H2: {h2}")
    
    # Test 4: Combined filters (category + header)
    if categories and h2_headers:
        print(f"\nTest 4: Combined filters (category + header)")
        results = retrieve_with_filter(
            test_query,
            top_k=3,
            category=categories[0],
            header=h2_headers[0] if h2_headers else None,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            cat = doc.metadata.get('category', 'Unknown')
            h2 = doc.metadata.get('header_2', 'N/A')[:20] if doc.metadata.get('header_2') else 'N/A'
            print(f"   - {doc.metadata.get('title', 'Unknown')[:30]}... | {cat} | H2: {h2}")
    
    # Test 5: Multiple header values (OR across values AND levels)
    if len(h2_headers) >= 2:
        test_header_list = h2_headers[:2]
        print(f"\nTest 5: Filter by multiple header values")
        print(f"   Headers: {[h[:30]+'...' for h in test_header_list]}")
        results = retrieve_with_filter(
            test_query,
            top_k=3,
            header=test_header_list,
            verbose=True
        )
        print(f"   Retrieved: {len(results)} documents")
        for doc in results:
            h1 = doc.metadata.get('header_1', 'N/A')[:30] if doc.metadata.get('header_1') else 'N/A'
            h2 = doc.metadata.get('header_2', 'N/A')[:30] if doc.metadata.get('header_2') else 'N/A'
            print(f"   - H1: {h1} | H2: {h2}")


if __name__ == "__main__":
    main()
