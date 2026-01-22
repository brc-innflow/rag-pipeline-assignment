"""

Web-based RAG Ingestion Pipeline
Combines raw HTML parsing for header metadata extraction with WebBaseLoader for complete content extraction.

This approach:
1. Fetches raw HTML to extract h1, h2, h3 header tags as metadata
2. Uses WebBaseLoader for reliable full-page content extraction
3. Chunks content with RecursiveCharacterTextSplitter
4. Attaches header hierarchy metadata to each chunk

Usage:
    # Single URL mode:
    python ingestion.py <parent_url> <category>
    
    # Batch mode with JSON config file:
    python ingestion.py --config <config_file.json>
    
Examples:
    python ingestion.py "https://support.example.com/folder/123" "Accounting"
    python ingestion.py --config sources.json
"""

import json
import os
import re
import sys
import time
import certifi
import requests
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration (separate collection for hybrid approach)
DB_NAME = "rag_assignment"
COLLECTION_NAME = "kb_articles_html"
INDEX_NAME = "kb_vector_index_html"

# Chunking configuration
CHUNK_SIZE = 1500      # Max chunk size
CHUNK_OVERLAP = 150    # Overlap for context continuity

# Crawling configuration
REQUEST_DELAY = 0.3  # Seconds between requests
REQUEST_TIMEOUT = 30


def log(message: str, level: str = "INFO"):
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def log_step(step_num: int, total_steps: int, description: str):
    """Print a step header."""
    print("\n" + "-" * 60)
    log(f"STEP {step_num}/{total_steps}: {description}", "STEP")
    print("-" * 60)


def log_progress(current: int, total: int, item_description: str):
    """Print progress information."""
    percentage = (current / total) * 100 if total > 0 else 0
    log(f"Progress: {current}/{total} ({percentage:.1f}%) - {item_description}")


class WebCrawler:
    """Crawls a knowledge base folder page to find article URLs."""
    
    def __init__(self, base_domain: str = None):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
        self.base_domain = base_domain
    
    def get_soup(self, url: str) -> BeautifulSoup:
        """Fetch URL and return BeautifulSoup object."""
        log(f"Fetching: {url[:80]}...", "HTTP")
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        log(f"Received {len(response.text)} bytes", "HTTP")
        return BeautifulSoup(response.text, "html.parser")
    
    def get_html(self, url: str) -> str:
        """Fetch URL and return raw HTML string."""
        log(f"Fetching HTML: {url[:80]}...", "HTTP")
        response = self.session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        log(f"Received {len(response.text)} bytes", "HTTP")
        return response.text
    
    def extract_base_domain(self, url: str) -> str:
        """Extract base domain from URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def find_article_links(self, folder_url: str, link_pattern: str = None) -> list[str]:
        """
        Find all article links on a folder page.
        
        Args:
            folder_url: The URL of the folder/category page to crawl
            link_pattern: Optional regex pattern to match article URLs (default: /articles/ or /solutions/)
        """
        log(f"Scanning for article links on: {folder_url[:60]}...", "CRAWL")
        soup = self.get_soup(folder_url)
        base_domain = self.base_domain or self.extract_base_domain(folder_url)
        
        # Default pattern matches common KB article URL patterns
        if link_pattern is None:
            link_pattern = r"/(articles?|solutions?/articles?)/"
        
        log(f"Using link pattern: {link_pattern}", "CRAWL")
        
        all_links = soup.find_all("a", href=True)
        log(f"Found {len(all_links)} total links on page", "CRAWL")
        
        links = []
        for a in all_links:
            href = a["href"]
            
            # Skip empty, anchor-only, or javascript links
            if not href or href.startswith("#") or href.startswith("javascript:"):
                continue
            
            # Check if URL matches article pattern
            if re.search(link_pattern, href):
                # Convert relative URLs to absolute
                if href.startswith("/"):
                    href = base_domain + href
                elif not href.startswith("http"):
                    href = urljoin(folder_url, href)
                
                # Remove query parameters for deduplication
                href = href.split("?")[0]
                links.append(href)
        
        # Deduplicate while preserving order
        seen = set()
        unique_links = []
        for url in links:
            if url not in seen:
                seen.add(url)
                unique_links.append(url)
        
        log(f"Found {len(unique_links)} unique article links (filtered from {len(links)} matches)", "CRAWL")
        return unique_links
    
    def find_next_page(self, folder_url: str) -> str | None:
        """Find the 'Next' page link for pagination."""
        soup = self.get_soup(folder_url)
        base_domain = self.base_domain or self.extract_base_domain(folder_url)
        
        # Look for common pagination patterns
        next_link = soup.find("a", string=re.compile(r"Next|→|>>", re.I))
        if not next_link:
            next_link = soup.find("a", {"class": re.compile(r"next", re.I)})
        if not next_link:
            next_link = soup.find("a", {"rel": "next"})
        
        if next_link and next_link.get("href"):
            href = next_link["href"]
            if href.startswith("/"):
                return base_domain + href
            elif not href.startswith("http"):
                return urljoin(folder_url, href)
            return href
        
        return None
    
    def crawl_all_pages(self, start_url: str) -> list[str]:
        """Crawl all pages starting from start_url, following pagination."""
        log("Starting paginated crawl...", "CRAWL")
        log(f"Start URL: {start_url}", "CRAWL")
        
        all_articles = []
        current_url = start_url
        page_num = 1
        
        while current_url:
            print("\n" + "." * 40)
            log(f"CRAWLING PAGE {page_num}", "PAGE")
            log(f"URL: {current_url}", "PAGE")
            
            try:
                articles = self.find_article_links(current_url)
                all_articles.extend(articles)
                log(f"Page {page_num} complete: Found {len(articles)} articles", "PAGE")
                log(f"Running total: {len(all_articles)} articles", "PAGE")
                
                # Check for next page
                log("Checking for pagination (next page)...", "CRAWL")
                current_url = self.find_next_page(current_url)
                
                if current_url:
                    log(f"Next page found: {current_url[:60]}...", "CRAWL")
                    log(f"Waiting {REQUEST_DELAY}s before next request...", "CRAWL")
                    time.sleep(REQUEST_DELAY)
                else:
                    log("No more pages found - pagination complete", "CRAWL")
                
                page_num += 1
                
            except Exception as e:
                log(f"Error crawling page: {e}", "ERROR")
                break
        
        # Final deduplication across all pages
        log("Deduplicating articles across all pages...", "CRAWL")
        seen = set()
        unique_articles = []
        for url in all_articles:
            if url not in seen:
                seen.add(url)
                unique_articles.append(url)
        
        log(f"Crawl complete: {len(unique_articles)} unique articles (from {len(all_articles)} total)", "CRAWL")
        return unique_articles


def extract_article_title(soup: BeautifulSoup) -> str:
    """Extract the article title from page content."""
    # Try common title elements
    for selector in ["title","h1", "h2", ".article-title", "#article-title"]:
        element = soup.select_one(selector)
        if element:
            title = element.get_text(strip=True)
            if title:
                return title
    return "Untitled"


def extract_header_metadata(html: str) -> dict:
    """
    Extract all h1, h2, h3 headers from HTML as metadata.
    
    Returns a dict with:
        - header_1: List of h1 text values (joined as string)
        - header_2: List of h2 text values (joined as string)
        - header_3: List of h3 text values (joined as string)
        - all_headers: Combined string of all headers for context
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements before extracting headers
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()
    
    headers = {
        "header_1": [],
        "header_2": [],
        "header_3": [],
    }
    
    # Extract h1 headers
    for h1 in soup.find_all("h1"):
        text = h1.get_text(strip=True)
        if text and text not in headers["header_1"]:
            headers["header_1"].append(text)
    
    # Extract h2 headers
    for h2 in soup.find_all("h2"):
        text = h2.get_text(strip=True)
        if text and text not in headers["header_2"]:
            headers["header_2"].append(text)
    
    # Extract h3 headers
    for h3 in soup.find_all("h3"):
        text = h3.get_text(strip=True)
        if text and text not in headers["header_3"]:
            headers["header_3"].append(text)
    
    # Convert lists to strings for metadata storage
    metadata = {}
    
    if headers["header_1"]:
        metadata["header_1"] = " | ".join(headers["header_1"])
    
    if headers["header_2"]:
        metadata["header_2"] = " | ".join(headers["header_2"])
    
    if headers["header_3"]:
        metadata["header_3"] = " | ".join(headers["header_3"])
    
    # Create combined headers string for searchability
    all_headers = []
    all_headers.extend(headers["header_1"])
    all_headers.extend(headers["header_2"])
    all_headers.extend(headers["header_3"])
    
    if all_headers:
        metadata["all_headers"] = " | ".join(all_headers)
    
    return metadata


def load_and_chunk_urls(urls: list[str], category: str, subtopic: str = None, crawler: WebCrawler = None) -> list:
    """
    Load web pages using hybrid approach:
    1. Fetch raw HTML to extract header metadata (h1, h2, h3)
    2. Use WebBaseLoader for complete content extraction
    3. Chunk with RecursiveCharacterTextSplitter
    4. Attach header metadata to all chunks
    
    Args:
        urls: List of URLs to load
        category: Category/topic to tag documents with
        subtopic: Optional subtopic for finer categorization
        crawler: WebCrawler instance to use for fetching raw HTML
    """
    log(f"Starting HYBRID chunking for {len(urls)} URLs", "CHUNK")
    log(f"Category tag: {category}", "CHUNK")
    if subtopic:
        log(f"Subtopic tag: {subtopic}", "CHUNK")
    log(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}", "CHUNK")
    log("Method: Raw HTML for headers + WebBaseLoader for content", "CHUNK")
    
    if crawler is None:
        crawler = WebCrawler()
    
    all_documents = []
    successful = 0
    failed = 0
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    start_time = time.time()
    
    for i, url in enumerate(urls, 1):
        print("\n" + "." * 40)
        log_progress(i, len(urls), f"Processing URL")
        log(f"URL: {url}", "LOAD")
        
        try:
            # =========================================================
            # Step A: Fetch raw HTML and extract header metadata
            # =========================================================
            log("STEP A: Fetching raw HTML for header extraction...", "HTML")
            raw_html = crawler.get_html(url)
            
            # Parse for title extraction
            soup = BeautifulSoup(raw_html, "html.parser")
            title = extract_article_title(soup)
            log(f"Title: {title[:60]}...", "HTML")
            
            # Extract h1, h2, h3 headers as metadata
            log("Extracting h1, h2, h3 headers from raw HTML...", "HTML")
            header_metadata = extract_header_metadata(raw_html)
            
            # Log what headers we found
            if header_metadata.get("header_1"):
                log(f"  H1: {header_metadata['header_1'][:80]}...", "HTML")
            if header_metadata.get("header_2"):
                log(f"  H2: {header_metadata['header_2'][:80]}...", "HTML")
            if header_metadata.get("header_3"):
                log(f"  H3: {header_metadata['header_3'][:80]}...", "HTML")
            
            if not header_metadata:
                log("  No headers found in document", "HTML")
            
            # =========================================================
            # Step B: Load full content with WebBaseLoader
            # =========================================================
            log("STEP B: Loading full content with WebBaseLoader...", "LOAD")
            loader = WebBaseLoader(
                web_paths=[url],
                bs_kwargs={"parse_only": None},  # Parse full page
                requests_kwargs={"timeout": REQUEST_TIMEOUT}
            )
            documents = loader.load()
            log(f"WebBaseLoader returned {len(documents)} document(s)", "LOAD")
            
            if documents:
                content_length = len(documents[0].page_content)
                log(f"Content length: {content_length} characters", "LOAD")
            
            # =========================================================
            # Step C: Split into chunks
            # =========================================================
            log("STEP C: Splitting content into chunks...", "CHUNK")
            chunks = text_splitter.split_documents(documents)
            log(f"Created {len(chunks)} chunks", "CHUNK")
            
            # =========================================================
            # Step D: Add all metadata to chunks
            # =========================================================
            log("STEP D: Adding metadata to all chunks...", "META")
            for chunk in chunks:
                # Basic metadata
                chunk.metadata["source_url"] = url
                chunk.metadata["category"] = category
                chunk.metadata["title"] = title
                if subtopic:
                    chunk.metadata["subtopic"] = subtopic
                
                # Add header metadata from raw HTML parsing
                for key, value in header_metadata.items():
                    chunk.metadata[key] = value
            
            # Show preview of first chunk with metadata
            if chunks:
                preview = chunks[0].page_content[:200].replace('\n', ' ')
                if len(chunks[0].page_content) > 200:
                    preview += "..."
                print(f"      [PREVIEW] {preview}")
                print(f"      [METADATA] {chunks[0].metadata}")
            
            all_documents.extend(chunks)
            successful += 1
            
            log(f"SUCCESS: {title[:40]}... -> {len(chunks)} chunks", "DONE")
            log(f"Running total: {len(all_documents)} chunks from {successful} articles", "DONE")
            
            # Rate limiting
            log(f"Rate limiting: waiting {REQUEST_DELAY}s...", "WAIT")
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            failed += 1
            log(f"FAILED to process URL: {e}", "ERROR")
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 40)
    log("CHUNKING SUMMARY", "SUMMARY")
    log(f"Total URLs processed: {len(urls)}", "SUMMARY")
    log(f"Successful: {successful}", "SUMMARY")
    log(f"Failed: {failed}", "SUMMARY")
    log(f"Total chunks created: {len(all_documents)}", "SUMMARY")
    log(f"Time elapsed: {elapsed:.1f} seconds", "SUMMARY")
    if len(urls) > 0:
        log(f"Average: {elapsed/len(urls):.2f}s per URL", "SUMMARY")
    print("=" * 40)
    
    return all_documents


def setup_mongodb_collection():
    """Set up MongoDB collection for vector storage."""
    log("Connecting to MongoDB Atlas...", "MONGO")
    log(f"Database: {DB_NAME}", "MONGO")
    log(f"Collection: {COLLECTION_NAME}", "MONGO")
    
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    
    log("Connection established successfully", "MONGO")
    
    # Create collection if it doesn't exist
    if COLLECTION_NAME not in db.list_collection_names():
        log(f"Collection '{COLLECTION_NAME}' not found - creating new collection...", "MONGO")
        db.create_collection(COLLECTION_NAME)
        log(f"Created collection: {COLLECTION_NAME}", "MONGO")
    else:
        # For assignment, we append rather than clear
        existing_count = db[COLLECTION_NAME].count_documents({})
        log(f"Using existing collection: {COLLECTION_NAME}", "MONGO")
        log(f"Existing documents in collection: {existing_count}", "MONGO")
    
    return client, db[COLLECTION_NAME]


def create_vector_store(collection, documents: list):
    """Create embeddings and store in MongoDB Atlas Vector Search."""
    log("Initializing OpenAI embeddings...", "EMBED")
    log("Model: text-embedding-3-small", "EMBED")
    log(f"Documents to embed: {len(documents)}", "EMBED")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    log("Embeddings model initialized", "EMBED")
    log("Starting embedding generation and storage...", "EMBED")
    log("(This may take a while depending on the number of documents)", "EMBED")
    
    start_time = time.time()
    
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )
    
    elapsed = time.time() - start_time
    
    log(f"Embedding and storage complete!", "EMBED")
    log(f"Successfully stored {len(documents)} document chunks", "EMBED")
    log(f"Time elapsed: {elapsed:.1f} seconds", "EMBED")
    if len(documents) > 0:
        log(f"Average: {elapsed/len(documents)*1000:.1f}ms per chunk", "EMBED")
    
    return vector_store


def print_vector_search_index_instructions():
    """Print instructions for creating the vector search index in MongoDB Atlas."""
    print("\n" + "=" * 70)
    print("IMPORTANT: Create Vector Search Index in MongoDB Atlas")
    print("=" * 70)
    print(f"""
To enable vector search, create an index in MongoDB Atlas:

1. Go to MongoDB Atlas -> Your Cluster -> Atlas Search -> Create Search Index
2. Select "JSON Editor" and use this configuration:

{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }},
    {{
      "type": "filter",
      "path": "category"
    }},
    {{
      "type": "filter",
      "path": "subtopic"
    }},
    {{
      "type": "filter",
      "path": "title"
    }},
    {{
      "type": "filter",
      "path": "header_1"
    }},
    {{
      "type": "filter",
      "path": "header_2"
    }},
    {{
      "type": "filter",
      "path": "header_3"
    }},
    {{
      "type": "filter",
      "path": "all_headers"
    }}
  ]
}}

3. Set the index name to: {INDEX_NAME}
4. Select database: {DB_NAME}
5. Select collection: {COLLECTION_NAME}

Filter fields available for queries:
  - category:     Filter by main topic (e.g., "Accounting", "Inventory")
  - subtopic:     Filter by sub-category (e.g., "Payroll", "Stock Management")
  - title:        Filter by article title
  - header_1:     Filter by H1 headers (main headings)
  - header_2:     Filter by H2 headers (sub-headings)
  - header_3:     Filter by H3 headers (sub-sub-headings)
  - all_headers:  Search across all headers combined

After creating the index, wait for it to become "Active" before running queries.
""")
    print("=" * 70)


def load_config_file(config_path: str) -> list[dict]:
    """
    Load URL/topic pairs from a JSON config file.
    
    Expected format:
    [
        {{"url": "https://example.com/folder/1", "topic": "Topic1"}},
        {{"url": "https://example.com/folder/2", "topic": "Topic2"}}
    ]
    """
    log(f"Loading config file: {config_path}", "CONFIG")
    
    if not os.path.exists(config_path):
        log(f"Config file not found: {config_path}", "ERROR")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Validate config structure
    if not isinstance(config, list):
        raise ValueError("Config file must contain a JSON array of {url, topic} objects")
    
    for i, item in enumerate(config):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not an object")
        if "url" not in item:
            raise ValueError(f"Item {i} missing 'url' field")
        if "topic" not in item:
            raise ValueError(f"Item {i} missing 'topic' field")
    
    log(f"Loaded {len(config)} URL/topic pairs from config", "CONFIG")
    return config


def process_single_source(crawler: WebCrawler, parent_url: str, category: str, subtopic: str = None, source_num: int = 1, total_sources: int = 1) -> list:
    """
    Process a single URL/topic pair: crawl and chunk using hybrid approach.
    Returns list of document chunks.
    
    Args:
        crawler: WebCrawler instance
        parent_url: The folder/category URL to crawl
        category: Main topic/category for the documents
        subtopic: Optional subtopic for finer categorization
        source_num: Current source number (for logging)
        total_sources: Total number of sources (for logging)
    """
    print("\n" + "#" * 70)
    log(f"PROCESSING SOURCE {source_num}/{total_sources}", "SOURCE")
    log(f"URL: {parent_url}", "SOURCE")
    log(f"Topic: {category}", "SOURCE")
    if subtopic:
        log(f"Subtopic: {subtopic}", "SOURCE")
    print("#" * 70)
    
    # Crawl for article URLs
    log("Crawling for article URLs...", "CRAWL")
    article_urls = crawler.crawl_all_pages(parent_url)
    
    if not article_urls:
        log(f"No article URLs found for source: {parent_url}", "WARN")
        return []
    
    log(f"Found {len(article_urls)} unique articles", "CRAWL")
    
    # Print discovered URLs
    topic_label = f"{category}/{subtopic}" if subtopic else category
    print("\n" + "-" * 40)
    log(f"Discovered URLs for topic '{topic_label}':", "LIST")
    for i, url in enumerate(article_urls, 1):
        print(f"   {i:3d}. {url}")
    print("-" * 40)
    
    # Load and chunk URLs using hybrid approach
    log("Loading and chunking articles (HYBRID approach)...", "CHUNK")
    documents = load_and_chunk_urls(article_urls, category, subtopic, crawler)
    
    log(f"Source complete: {len(documents)} chunks from {len(article_urls)} articles", "SOURCE")
    
    return documents


def main():
    """Main ingestion pipeline with hybrid approach."""
    pipeline_start = time.time()
    
    # Parse command line arguments
    if len(sys.argv) < 2:
        log("Missing arguments. Usage: python ingestion.py --config <config_file.json>", "ERROR")
        sys.exit(1)
    
    # Determine mode: single URL or config file
    sources = []
    
    if sys.argv[1] == "--config":
        # Batch mode with config file
        if len(sys.argv) < 3:
            log("Missing config file path. Usage: python ingestion.py --config <config_file.json>", "ERROR")
            sys.exit(1)
        
        config_path = sys.argv[2]
        sources = load_config_file(config_path)
    elif len(sys.argv) >= 3:
        # Single URL mode (backward compatible)
        sources = [{"url": sys.argv[1], "topic": sys.argv[2]}]
    else:
        log("Invalid arguments. Usage: python ingestion.py <url> <topic> OR --config <config_file.json>", "ERROR")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("  WEB-BASED RAG INGESTION PIPELINE (HYBRID APPROACH)")
    print("  Raw HTML headers + WebBaseLoader content")
    print("=" * 70)
    log("Pipeline started", "START")
    log(f"Mode: {'Batch' if len(sources) > 1 else 'Single'} ({len(sources)} source(s))", "CONFIG")
    log(f"Approach: HYBRID (HTML headers for metadata + WebBaseLoader for content)", "CONFIG")
    log(f"Max chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}", "CONFIG")
    log(f"Request Delay: {REQUEST_DELAY}s, Timeout: {REQUEST_TIMEOUT}s", "CONFIG")
    print("=" * 70)
    
    # Print all sources to be processed
    print("\n" + "-" * 40)
    log("Sources to process:", "CONFIG")
    for i, source in enumerate(sources, 1):
        topic_label = source['topic']
        if source.get('subtopic'):
            topic_label += f" / {source['subtopic']}"
        print(f"   {i}. [{topic_label}] {source['url']}")
    print("-" * 40)
    
    # Validate environment
    log("Validating environment variables...", "ENV")
    if not MONGO_DB_URL:
        log("MONGO_DB_URL environment variable not set!", "ERROR")
        raise ValueError("MONGO_DB_URL environment variable not set")
    log("MONGO_DB_URL: Found", "ENV")
    
    if not OPENAI_API_KEY:
        log("OPENAI_API_KEY environment variable not set!", "ERROR")
        raise ValueError("OPENAI_API_KEY environment variable not set")
    log("OPENAI_API_KEY: Found", "ENV")
    log("Environment validation complete", "ENV")
    
    # =========================================================================
    # Step 1: Setup MongoDB (do this first so we fail early if connection issues)
    # =========================================================================
    log_step(1, 3, "SETUP MONGODB CONNECTION")
    client, collection = setup_mongodb_collection()
    log("Step 1 complete: MongoDB ready", "DONE")
    
    try:
        # =====================================================================
        # Step 2: Crawl and chunk all sources
        # =====================================================================
        log_step(2, 3, "CRAWL AND CHUNK ALL SOURCES (HYBRID APPROACH)")
        
        crawler = WebCrawler()
        all_documents = []
        total_articles = 0
        source_stats = []
        
        for i, source in enumerate(sources, 1):
            source_docs = process_single_source(
                crawler=crawler,
                parent_url=source["url"],
                category=source["topic"],
                subtopic=source.get("subtopic"),
                source_num=i,
                total_sources=len(sources)
            )
            all_documents.extend(source_docs)
            
            # Track stats per source
            article_count = len(set(doc.metadata.get("source_url", "") for doc in source_docs))
            total_articles += article_count
            source_stats.append({
                "topic": source["topic"],
                "subtopic": source.get("subtopic"),
                "articles": article_count,
                "chunks": len(source_docs)
            })
        
        if not all_documents:
            log("No documents were successfully loaded from any source.", "ERROR")
            sys.exit(1)
        
        # Print summary of all sources
        print("\n" + "=" * 50)
        log("CRAWLING & CHUNKING SUMMARY", "SUMMARY")
        print("-" * 50)
        for stat in source_stats:
            topic_label = stat['topic']
            if stat.get('subtopic'):
                topic_label += f" / {stat['subtopic']}"
            print(f"   [{topic_label}] {stat['articles']} articles -> {stat['chunks']} chunks")
        print("-" * 50)
        log(f"Total: {total_articles} articles -> {len(all_documents)} chunks", "SUMMARY")
        print("=" * 50)
        
        log(f"Step 2 complete: {len(all_documents)} total chunks from {len(sources)} sources", "DONE")
        
        # =====================================================================
        # Step 3: Create embeddings and store
        # =====================================================================
        log_step(3, 3, "CREATE EMBEDDINGS AND STORE")
        vector_store = create_vector_store(collection, all_documents)
        log("Step 3 complete: Embeddings stored", "DONE")
        
        # =====================================================================
        # Pipeline Complete
        # =====================================================================
        pipeline_elapsed = time.time() - pipeline_start
        
        print("\n" + "=" * 70)
        print("  INGESTION PIPELINE COMPLETE! (HYBRID APPROACH)")
        print("=" * 70)
        log("FINAL SUMMARY", "COMPLETE")
        print("-" * 70)
        print(f"   Database:           {DB_NAME}")
        print(f"   Collection:         {COLLECTION_NAME}")
        print(f"   Index Name:         {INDEX_NAME}")
        print(f"   Approach:           HYBRID (HTML headers + WebBaseLoader)")
        print(f"   Sources processed:  {len(sources)}")
        print(f"   Topics:             {', '.join(s['topic'] for s in sources)}")
        print(f"   Total articles:     {total_articles}")
        print(f"   Total chunks:       {len(all_documents)}")
        print(f"   Total time:         {pipeline_elapsed:.1f} seconds ({pipeline_elapsed/60:.1f} minutes)")
        print("-" * 70)
        print("   Breakdown by topic:")
        for stat in source_stats:
            topic_label = stat['topic']
            if stat.get('subtopic'):
                topic_label += f" / {stat['subtopic']}"
            print(f"      - {topic_label}: {stat['articles']} articles, {stat['chunks']} chunks")
        print("=" * 70)
        
        # Print index creation instructions (updated with header fields)
        print_vector_search_index_instructions()
        
    finally:
        log("Closing MongoDB connection...", "CLEANUP")
        client.close()
        log("Connection closed", "CLEANUP")


if __name__ == "__main__":
    main()
