"""
Code was generated using AI assistance

RAG Pipeline - Precision Evaluation (Plain Text)
Measures retrieval precision using LLM-as-judge to assess document relevance.

Precision = (Number of relevant documents retrieved) / (Total documents retrieved, k)

Uses the retrieval module for document retrieval to ensure consistency.
"""

import os
import sys
import json
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieval import retrieve_kb_urls

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation configuration
DEFAULT_K = 5
JUDGE_MODEL = "gpt-4o-mini"

# LLM-as-Judge prompt for relevance assessment
RELEVANCE_JUDGE_PROMPT = """You are a relevance judge. Your task is to determine if the retrieved document is relevant to answering the given question about a knowledge base system.

A document is RELEVANT if it contains information that would help answer the question, even if it doesn't fully answer it.
A document is NOT RELEVANT if it contains no useful information for answering the question.

Question: {question}

Retrieved Document:
{document}

Is this document relevant to answering the question? 
Respond with ONLY "RELEVANT" or "NOT_RELEVANT" - nothing else."""


# Test cases for evaluation
# Adjust these based on your actual knowledge base content
TEST_CASES = [
    {
        "id": "time_clock_usage",
        "question": "How do I clock in and out using the time clock?",
        "description": "Basic time clock operations"
    },
    {
        "id": "payroll_setup",
        "question": "How do I set up payroll for a new employee?",
        "description": "Payroll configuration process"
    },
    {
        "id": "schedule_management",
        "question": "How do I create and manage employee schedules?",
        "description": "Scheduling functionality"
    },
    {
        "id": "time_off_requests",
        "question": "How do employees request time off?",
        "description": "PTO and leave management"
    },
    {
        "id": "overtime_rules",
        "question": "How does the system calculate overtime?",
        "description": "Overtime calculation rules"
    },
    {
        "id": "reports_generation",
        "question": "How do I generate labor reports?",
        "description": "Reporting functionality"
    },
    {
        "id": "employee_permissions",
        "question": "How do I set up user permissions and access levels?",
        "description": "Access control setup"
    },
    {
        "id": "troubleshooting",
        "question": "What should I do if the time clock is not working?",
        "description": "Common troubleshooting"
    },
]


def create_relevance_judge():
    """Create the LLM judge for assessing relevance."""
    prompt = ChatPromptTemplate.from_template(RELEVANCE_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def judge_relevance(judge_chain, question: str, document_content: str) -> bool:
    """Use LLM to judge if a document is relevant to the question."""
    response = judge_chain.invoke({
        "question": question,
        "document": document_content
    })
    
    return "RELEVANT" in response.upper() and "NOT_RELEVANT" not in response.upper()


def calculate_precision(question: str, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Calculate precision for a single query.
    
    Precision = relevant_documents / k
    
    Returns dict with precision score and details.
    """
    # Retrieve documents using the retrieval module
    documents = retrieve_kb_urls(question, top_k=k)
    
    if not documents:
        return {
            "question": question,
            "k": k,
            "retrieved": 0,
            "relevant": 0,
            "precision": 0.0,
            "judgments": []
        }
    
    # Judge each document
    judge = create_relevance_judge()
    judgments = []
    relevant_count = 0
    
    for i, doc in enumerate(documents):
        is_relevant = judge_relevance(judge, question, doc.page_content)
        
        if is_relevant:
            relevant_count += 1
        
        judgment = {
            "doc_index": i + 1,
            "relevant": is_relevant,
            "title": doc.metadata.get("title", "Unknown"),
            "category": doc.metadata.get("category", "Unknown"),
            "url": doc.metadata.get("source_url", "Unknown"),
        }
        
        if verbose:
            judgment["content_preview"] = doc.page_content[:200] + "..."
        
        judgments.append(judgment)
    
    precision = relevant_count / len(documents)
    
    return {
        "question": question,
        "k": k,
        "retrieved": len(documents),
        "relevant": relevant_count,
        "precision": precision,
        "judgments": judgments
    }


def run_evaluation(test_cases: list = None, k: int = DEFAULT_K, verbose: bool = False) -> dict:
    """
    Run precision evaluation on all test cases.
    
    Returns aggregate results and per-question breakdown.
    """
    if test_cases is None:
        test_cases = TEST_CASES
    
    print("=" * 60)
    print("RAG Pipeline - Precision Evaluation (Plain Text)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents retrieved): {k}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    total_relevant = 0
    total_retrieved = 0
    
    for i, test_case in enumerate(test_cases, 1):
        question = test_case["question"]
        print(f"\n[{i}/{len(test_cases)}] {test_case['id']}")
        print(f"    Q: {question[:60]}...")
        
        result = calculate_precision(question, k=k, verbose=verbose)
        result["test_id"] = test_case["id"]
        result["description"] = test_case.get("description", "")
        
        results.append(result)
        total_relevant += result["relevant"]
        total_retrieved += result["retrieved"]
        
        print(f"    Precision: {result['precision']:.2%} ({result['relevant']}/{result['retrieved']} relevant)")
    
    # Calculate aggregate metrics
    avg_precision = sum(r["precision"] for r in results) / len(results) if results else 0
    overall_precision = total_relevant / total_retrieved if total_retrieved > 0 else 0
    
    summary = {
        "k": k,
        "num_test_cases": len(test_cases),
        "total_documents_retrieved": total_retrieved,
        "total_relevant_documents": total_relevant,
        "average_precision": avg_precision,
        "overall_precision": overall_precision,
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nAggregate Metrics:")
    print(f"   Average Precision (per query):  {avg_precision:.2%}")
    print(f"   Overall Precision (all docs):   {overall_precision:.2%}")
    print(f"   Total Relevant / Total Retrieved: {total_relevant} / {total_retrieved}")
    
    print(f"\nPer-Query Breakdown:")
    for r in results:
        status = "PASS" if r["precision"] >= 0.6 else "WARN" if r["precision"] >= 0.4 else "FAIL"
        print(f"   [{status}] {r['test_id']}: {r['precision']:.2%}")
    
    return summary


def main():
    """Run the precision evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run precision evaluation on RAG Pipeline")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include document previews in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--question", type=str, help="Evaluate a single custom question")
    
    args = parser.parse_args()
    
    if args.question:
        # Single question evaluation
        print(f"\nEvaluating single question with k={args.k}")
        result = calculate_precision(args.question, k=args.k, verbose=True)
        
        print(f"\nQuestion: {result['question']}")
        print(f"Precision: {result['precision']:.2%} ({result['relevant']}/{result['retrieved']} relevant)")
        print("\nJudgments:")
        for j in result["judgments"]:
            status = "RELEVANT" if j["relevant"] else "NOT_RELEVANT"
            print(f"  [{status}] Doc {j['doc_index']}: {j['title']}")
            print(f"            Category: {j['category']}")
            print(f"            URL: {j['url']}")
            if args.verbose and "content_preview" in j:
                print(f"            Preview: {j['content_preview']}")
    else:
        # Full evaluation
        summary = run_evaluation(k=args.k, verbose=args.verbose)
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
