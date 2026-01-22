"""
Code was generated using AI assistance

RAG Pipeline - Generation Module (Plain Text)
Uses retrieved context to generate answers using OpenAI LLM.
"""

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from retrieval import retrieve_kb_urls, format_retrieved_context

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.0
TOP_K = 5

# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided knowledge base articles.

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


def generate_answer(
    question: str, 
    top_k: int = TOP_K, 
    category: str = None,
    subtopic: str = None,
    verbose: bool = False
) -> dict:
    """
    Generate an answer using the RAG pipeline.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        category: Optional filter by category
        subtopic: Optional filter by subtopic
        verbose: If True, include retrieved documents in response
    
    Returns:
        Dictionary containing the answer and optionally the sources
    """
    # Step 1: Retrieve relevant documents
    documents = retrieve_kb_urls(
        question, 
        top_k=top_k,
        category=category,
        subtopic=subtopic
    )
    
    if not documents:
        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "sources": []
        }
    
    # Step 2: Format context
    context = format_retrieved_context(documents)
    
    # Step 3: Generate answer
    chain = create_rag_chain()
    answer = chain.invoke({
        "context": context,
        "question": question
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
        ]
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
    """Run an interactive Q&A session."""
    print("\n" + "=" * 60)
    print("RAG Pipeline - Interactive Q&A (Plain Text)")
    print("Ask questions about the knowledge base")
    print("Type 'quit' or 'exit' to end the session")
    print("=" * 60)
    
    while True:
        print()
        question = input("Your question: ").strip()
        
        if not question:
            continue
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        print("\nRetrieving relevant documents...")
        print("Generating answer...\n")
        
        try:
            result = generate_answer(question, verbose=False)
            
            print("-" * 50)
            print("Answer:")
            print("-" * 50)
            print(result["answer"])
            
            print("\nSources:")
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
    print("=" * 50)
    print("RAG Pipeline - Generation (Plain Text)")
    print("=" * 50)
    
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Example queries
    example_questions = [
        "What is time clock and how do I use it?",
        "How do I set up payroll?",
        "How do I manage inventory?",
    ]
    
    print("\nExample Questions Available:")
    for i, q in enumerate(example_questions, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "-" * 50)
    choice = input("Enter question number (1-3), 'i' for interactive mode, or your own question: ").strip()
    
    if choice.lower() == 'i':
        interactive_mode()
    elif choice in ['1', '2', '3']:
        question = example_questions[int(choice) - 1]
        print(f"\nQuestion: {question}\n")
        print("Retrieving relevant documents...")
        print("Generating answer...\n")
        
        result = generate_answer(question, verbose=True)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print("\nSources:")
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
    elif choice:
        print(f"\nQuestion: {choice}\n")
        print("Retrieving relevant documents...")
        print("Generating answer...\n")
        
        result = generate_answer(choice, verbose=False)
        
        print("-" * 50)
        print("Answer:")
        print("-" * 50)
        print(result["answer"])
        
        print("\nSources:")
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
        print("\nNo question provided. Run again to try!")


if __name__ == "__main__":
    main()
