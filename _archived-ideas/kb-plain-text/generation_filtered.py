"""
Code was generated using AI assistance

Metadata-Filtered RAG - Generation Module (Plain Text)
Uses filtered retrieval to generate answers from targeted document subsets.

Supports answering questions with constraints like:
- "Based on Labor articles..."
- "Looking at Accounting content..."
- "From the Payroll subtopic..."
"""

import os
from dotenv import load_dotenv
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval_filtered import (
    retrieve_with_filter, 
    format_retrieved_context, 
    get_available_categories,
    get_available_subtopics,
    get_category_counts,
    get_subtopic_counts,
)

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template with filter context
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided knowledge base articles.

{filter_context}

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what information is missing.

Be specific and provide step-by-step instructions when applicable. Reference the source articles when helpful.

Context:
{context}

Question: {question}

Answer:"""


def create_rag_chain():
    """Create the RAG chain with prompt template and LLM."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY
    )
    
    output_parser = StrOutputParser()
    
    chain = prompt | llm | output_parser
    return chain


def build_filter_context(
    category: Optional[str | list[str]] = None,
    subtopic: Optional[str | list[str]] = None,
    title: Optional[str | list[str]] = None,
) -> str:
    """Build a human-readable description of the applied filters."""
    parts = []
    
    if category is not None:
        if isinstance(category, list):
            parts.append(f"from categories: {', '.join(category)}")
        else:
            parts.append(f"from category: {category}")
    
    if subtopic is not None:
        if isinstance(subtopic, list):
            parts.append(f"with subtopics: {', '.join(subtopic)}")
        else:
            parts.append(f"with subtopic: {subtopic}")
    
    if title is not None:
        if isinstance(title, list):
            parts.append(f"from articles: {', '.join(title[:3])}...")
        else:
            parts.append(f"from article: {title}")
    
    if parts:
        return f"You are answering based on a filtered subset of documents: {'; '.join(parts)}."
    else:
        return "You are answering based on all available knowledge base articles."


def generate_answer(
    question: str,
    top_k: int = TOP_K,
    category: Optional[str | list[str]] = None,
    subtopic: Optional[str | list[str]] = None,
    title: Optional[str | list[str]] = None,
    verbose: bool = False,
) -> dict:
    """
    Generate an answer using the metadata-filtered RAG pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        category: Filter by category/categories
        subtopic: Filter by subtopic(s)
        title: Filter by article title(s)
        verbose: Include retrieved documents in response
    
    Returns:
        Dictionary containing the answer and metadata
    """
    # Step 1: Retrieve with filters
    documents = retrieve_with_filter(
        query=question,
        top_k=top_k,
        category=category,
        subtopic=subtopic,
        title=title,
        verbose=verbose
    )
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant information matching your criteria.",
            "sources": [],
            "filters_applied": build_filter_context(category, subtopic, title)
        }
    
    # Step 2: Format context
    context = format_retrieved_context(documents)
    filter_context = build_filter_context(category, subtopic, title)
    
    # Step 3: Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question,
        "filter_context": filter_context
    })
    
    # Prepare response
    response = {
        "answer": answer,
        "sources": [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "category": doc.metadata.get("category", "Unknown"),
                "subtopic": doc.metadata.get("subtopic", ""),
                "url": doc.metadata.get("source_url", "Unknown")
            }
            for doc in documents
        ],
        "filters_applied": filter_context
    }
    
    if verbose:
        response["retrieved_documents"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
    
    return response


def interactive_mode():
    """Run an interactive Q&A session with filter support."""
    print("\n" + "=" * 60)
    print("Metadata-Filtered RAG - Interactive Q&A (Plain Text)")
    print("=" * 60)
    print("\nFilter commands:")
    print("  category:Labor       - Filter by category")
    print("  subtopic:Payroll     - Filter by subtopic")
    print("  clear                - Reset all filters")
    print("  filters              - Show active filters")
    print("  help                 - Show available categories/subtopics")
    print("  quit                 - Exit")
    print("-" * 60)
    
    # Active filters
    active_filters = {
        "category": None,
        "subtopic": None,
        "title": None
    }
    
    while True:
        print()
        user_input = input(">> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'clear':
            active_filters = {k: None for k in active_filters}
            print("All filters cleared")
            continue
        
        if user_input.lower() == 'filters':
            active = {k: v for k, v in active_filters.items() if v is not None}
            if active:
                print("Active filters:")
                for k, v in active.items():
                    print(f"   {k}: {v}")
            else:
                print("No active filters")
            continue
        
        if user_input.lower() == 'help':
            print("\nAvailable Categories:")
            for cat, count in get_category_counts():
                print(f"   {cat} ({count} chunks)")
            print("\nAvailable Subtopics:")
            for sub, count in get_subtopic_counts():
                print(f"   {sub} ({count} chunks)")
            continue
        
        # Parse filter commands
        if user_input.startswith('category:'):
            categories = user_input.split(':')[1].split(',')
            if len(categories) == 1:
                active_filters["category"] = categories[0].strip()
            else:
                active_filters["category"] = [c.strip() for c in categories]
            print(f"Category filter: {active_filters['category']}")
            continue
        
        if user_input.startswith('subtopic:'):
            subtopics = user_input.split(':')[1].split(',')
            if len(subtopics) == 1:
                active_filters["subtopic"] = subtopics[0].strip()
            else:
                active_filters["subtopic"] = [s.strip() for s in subtopics]
            print(f"Subtopic filter: {active_filters['subtopic']}")
            continue
        
        if user_input.startswith('title:'):
            active_filters["title"] = user_input.split(':', 1)[1].strip()
            print(f"Title filter: {active_filters['title']}")
            continue
        
        # Treat as question
        question = user_input
        
        # Show active filters
        active = {k: v for k, v in active_filters.items() if v is not None}
        if active:
            print(f"Searching with filters: {active}")
        
        print("Generating answer...\n")
        
        try:
            result = generate_answer(
                question,
                **active_filters,
                verbose=False
            )
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print(f"\nSources ({len(result['sources'])} documents):")
            seen_titles = set()
            for source in result["sources"]:
                title = source['title']
                if title not in seen_titles:
                    seen_titles.add(title)
                    category_label = source['category']
                    if source['subtopic']:
                        category_label += f" / {source['subtopic']}"
                    print(f"  - {title}")
                    print(f"    Category: {category_label}")
                    print(f"    URL: {source['url']}")
            
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Run example queries or interactive mode."""
    print("=" * 60)
    print("Metadata-Filtered RAG - Generation Pipeline (Plain Text)")
    print("=" * 60)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Example queries with filters
    example_queries = [
        {
            "question": "How do I clock in and out?",
            "category": "Labor",
            "description": "Labor category query"
        },
        {
            "question": "How do I set up payroll deductions?",
            "subtopic": "Payroll",
            "description": "Payroll subtopic query"
        },
        {
            "question": "How do I manage employee schedules?",
            "category": "Labor",
            "description": "Labor scheduling query"
        },
    ]
    
    print("\nExample Queries with Filters:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q['question'][:45]}... ({q['description']})")
    
    print("\n" + "-" * 50)
    choice = input("Enter 1-3 for examples, 'i' for interactive mode, or your question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        q = example_queries[int(choice) - 1]
        print(f"\nQuestion: {q['question']}")
        print(f"Filters: {q['description']}")
        print("\nRetrieving with filters...")
        print("Generating answer...\n")
        
        # Build kwargs from example
        kwargs = {k: v for k, v in q.items() if k not in ['question', 'description']}
        
        result = generate_answer(q['question'], **kwargs, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\nSources:")
        seen_titles = set()
        for source in result["sources"]:
            title = source['title']
            if title not in seen_titles:
                seen_titles.add(title)
                category_label = source['category']
                if source['subtopic']:
                    category_label += f" / {source['subtopic']}"
                print(f"  - {title}")
                print(f"    Category: {category_label}")
                print(f"    URL: {source['url']}")
        
        print(f"\n{result['filters_applied']}")
    elif choice:
        print(f"\nQuestion: {choice}")
        print("\nRetrieving (no filters)...")
        print("Generating answer...\n")
        
        result = generate_answer(choice, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print(f"\nSources:")
        seen_titles = set()
        for source in result["sources"]:
            title = source['title']
            if title not in seen_titles:
                seen_titles.add(title)
                category_label = source['category']
                if source['subtopic']:
                    category_label += f" / {source['subtopic']}"
                print(f"  - {title}")
                print(f"    Category: {category_label}")
                print(f"    URL: {source['url']}")
    else:
        print("\nNo input provided. Run again to try!")


if __name__ == "__main__":
    main()
