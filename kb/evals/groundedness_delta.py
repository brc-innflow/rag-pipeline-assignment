"""
Code was generated using AI assistance

Metadata-Filtered RAG - Groundedness Delta Evaluation (HTML Header-Based)
Compares answer groundedness with and without metadata filters.

Measures whether filtering improves the faithfulness of generated answers
to the retrieved context (reduces hallucinations).

Groundedness Delta = Filtered Groundedness Rate - Unfiltered Groundedness Rate

Uses the retrieval modules for document retrieval to ensure consistency.
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

from retrieval_naive import retrieve_chunks
from retrieval_filtered import retrieve_with_filter

# Load environment variables
load_dotenv()

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Evaluation configuration
DEFAULT_K = 5
GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# RAG Generation Prompt (adapted for KB content)
RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided knowledge base articles.

Use ONLY the information from the context below to answer the question. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what information is missing.

Be specific and provide step-by-step instructions when applicable.

Context:
{context}

Question: {question}

Answer:"""

# LLM-as-Judge prompt for groundedness assessment
GROUNDEDNESS_JUDGE_PROMPT = """You are a groundedness evaluator. Your task is to determine if the given answer is fully supported by the provided context.

An answer is GROUNDED if:
- Every claim and statement in the answer can be traced back to information in the context
- The answer does not include information that isn't present in the context
- The answer does not make assumptions or inferences beyond what the context supports

An answer is NOT GROUNDED if:
- It contains claims not supported by the context (hallucinations)
- It adds information or details not present in the context
- It makes unsupported generalizations or conclusions

Context:
{context}

Question: {question}

Answer to evaluate:
{answer}

Evaluate the groundedness of this answer. First, briefly explain your reasoning (2-3 sentences), then provide your verdict.

Respond in this exact format:
REASONING: <your brief explanation>
VERDICT: <GROUNDED or NOT_GROUNDED>"""


# Test cases with questions that should benefit from filtering
# Each has a question and the filter that should improve groundedness
GROUNDEDNESS_DELTA_TEST_CASES = [
    {
        "id": "labor_category",
        "question": "How do I clock in and out?",
        "filter_type": "category",
        "filters": {"category": "Labor"},
        "description": "Time clock question with Labor category filter"
    },
    {
        "id": "accounting_category",
        "question": "How do I approve invoices?",
        "filter_type": "category",
        "filters": {"category": "Accounting", "subtopic": "User Guides"},
        "description": "Invoice approval with Accounting category filter"
    },    
    {
        "id": "payroll_subtopic",
        "question": "How do I set up payroll deductions?",
        "filter_type": "subtopic",
        "filters": {"subtopic": "User Guides"},
        "description": "Payroll setup with subtopic filter"
    },
     {
        "id": "payroll_category_subtopic",
        "question": "How do I set up payroll deductions?",
        "filter_type": "subtopic",
        "filters": {"subtopic": "User Guides", "category": "Labor"},
        "description": "Payroll setup with subtopic filter"
    },
    {

        "id": "header_smile_id",
        "question": "What is Smile iD?",
        "filter_type": "header",
        "filters": {"header": "Smile iD"},
        "description": "Setup question with header filter (searches all levels)"
    },   
    {
        "id": "combined_labor_payroll",
        "question": "How do I process payroll for overtime hours?",
        "filter_type": "combined",
        "filters": {"category": "Labor", "subtopic": "User Guides"},
        "description": "Overtime payroll with category + subtopic filter"
    },
    {
        "id": "combined_category_header",
        "question": "How do I estimate occupancy ?",
        "filter_type": "combined_header",
        "filters": {"category": "Labor", "header": "Labor Forecast", "title": "Labor Forecast"},
        "description": "Setup question with category + header filter"
    },
    {
        "id": "multi_category",
        "question": "How do I export payroll data?",
        "filter_type": "multi_category",
        "filters": {"category": ["Labor", "Accounting"]},
        "description": "Cross-category question with multiple category filter"
    },
]


def format_section_path(doc) -> str:
    """Build a section path from header hierarchy."""
    parts = []
    if doc.metadata.get('header_1'):
        parts.append(doc.metadata['header_1'])
    if doc.metadata.get('header_2'):
        parts.append(doc.metadata['header_2'])
    if doc.metadata.get('header_3'):
        parts.append(doc.metadata['header_3'])
    return " > ".join(parts) if parts else "N/A"


def format_context(documents: list) -> str:
    """Format retrieved documents into context string."""
    context_parts = []
    
    for i, doc in enumerate(documents, 1):
        title = doc.metadata.get("title", "Unknown")
        category = doc.metadata.get("category", "Unknown")
        section = format_section_path(doc)
        
        context_parts.append(
            f"[Document {i}]\n"
            f"Source: {title}\n"
            f"Category: {category}\n"
            f"Section: {section}\n"
            f"Content:\n{doc.page_content}\n"
        )
    
    return "\n---\n".join(context_parts)


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using the RAG pipeline."""
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    llm = ChatOpenAI(
        model=GENERATION_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    
    return answer


def create_groundedness_judge():
    """Create the LLM judge for assessing groundedness."""
    prompt = ChatPromptTemplate.from_template(GROUNDEDNESS_JUDGE_PROMPT)
    
    llm = ChatOpenAI(
        model=JUDGE_MODEL,
        temperature=0.0,
        openai_api_key=OPENAI_API_KEY
    )
    
    chain = prompt | llm | StrOutputParser()
    return chain


def judge_groundedness(judge_chain, question: str, context: str, answer: str) -> dict:
    """Use LLM to judge if an answer is grounded in the context."""
    response = judge_chain.invoke({
        "question": question,
        "context": context,
        "answer": answer
    })
    
    # Parse the response
    reasoning = ""
    is_grounded = False
    
    if "REASONING:" in response:
        reasoning_start = response.find("REASONING:") + len("REASONING:")
        reasoning_end = response.find("VERDICT:")
        if reasoning_end > reasoning_start:
            reasoning = response[reasoning_start:reasoning_end].strip()
    
    is_grounded = "VERDICT: GROUNDED" in response.upper() and "NOT_GROUNDED" not in response.upper()
    
    return {
        "is_grounded": is_grounded,
        "reasoning": reasoning,
        "raw_response": response
    }


def evaluate_groundedness_delta(
    question: str,
    filters: dict,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Compare groundedness with and without filters for a single question.
    
    Returns:
        Dictionary with groundedness comparison results
    """
    judge = create_groundedness_judge()
    
    # === UNFILTERED (baseline) ===
    unfiltered_docs = retrieve_chunks(question, top_k=k)
    unfiltered_context = format_context(unfiltered_docs) if unfiltered_docs else ""
    unfiltered_answer = generate_answer(question, unfiltered_context) if unfiltered_context else "No documents retrieved."
    
    if unfiltered_docs:
        unfiltered_judgment = judge_groundedness(judge, question, unfiltered_context, unfiltered_answer)
        unfiltered_grounded = unfiltered_judgment["is_grounded"]
        unfiltered_reasoning = unfiltered_judgment["reasoning"]
    else:
        unfiltered_grounded = False
        unfiltered_reasoning = "No documents retrieved"
    
    # === FILTERED ===
    filtered_docs = retrieve_with_filter(question, top_k=k, **filters)
    filtered_context = format_context(filtered_docs) if filtered_docs else ""
    filtered_answer = generate_answer(question, filtered_context) if filtered_context else "No documents retrieved."
    
    if filtered_docs:
        filtered_judgment = judge_groundedness(judge, question, filtered_context, filtered_answer)
        filtered_grounded = filtered_judgment["is_grounded"]
        filtered_reasoning = filtered_judgment["reasoning"]
    else:
        filtered_grounded = False
        filtered_reasoning = "No documents retrieved with filters"
    
    # Calculate delta (1 = grounded, 0 = not grounded)
    unfiltered_score = 1 if unfiltered_grounded else 0
    filtered_score = 1 if filtered_grounded else 0
    groundedness_delta = filtered_score - unfiltered_score
    
    result = {
        "question": question,
        "filters": filters,
        "k": k,
        "unfiltered": {
            "is_grounded": unfiltered_grounded,
            "reasoning": unfiltered_reasoning,
            "answer": unfiltered_answer,
            "num_docs": len(unfiltered_docs)
        },
        "filtered": {
            "is_grounded": filtered_grounded,
            "reasoning": filtered_reasoning,
            "answer": filtered_answer,
            "num_docs": len(filtered_docs)
        },
        "groundedness_delta": groundedness_delta,
        "improvement": groundedness_delta > 0
    }
    
    if verbose:
        result["unfiltered"]["context"] = unfiltered_context
        result["unfiltered"]["sources"] = [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "category": doc.metadata.get("category", "Unknown"),
                "section": format_section_path(doc),
                "url": doc.metadata.get("source_url", "Unknown")
            }
            for doc in unfiltered_docs
        ]
        result["filtered"]["context"] = filtered_context
        result["filtered"]["sources"] = [
            {
                "title": doc.metadata.get("title", "Unknown"),
                "category": doc.metadata.get("category", "Unknown"),
                "section": format_section_path(doc),
                "url": doc.metadata.get("source_url", "Unknown")
            }
            for doc in filtered_docs
        ]
    
    return result


def run_evaluation(
    test_cases: list = None,
    k: int = DEFAULT_K,
    verbose: bool = False
) -> dict:
    """
    Run groundedness delta evaluation on all test cases.
    
    Returns:
        Dictionary with aggregate results
    """
    if test_cases is None:
        test_cases = GROUNDEDNESS_DELTA_TEST_CASES
    
    print("=" * 60)
    print("Metadata-Filtered RAG - Groundedness Delta Evaluation (HTML)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  - k (documents): {k}")
    print(f"  - Generation model: {GENERATION_MODEL}")
    print(f"  - Judge model: {JUDGE_MODEL}")
    print(f"  - Test cases: {len(test_cases)}")
    print("\n" + "-" * 60)
    
    results = []
    unfiltered_grounded_count = 0
    filtered_grounded_count = 0
    improvements = 0
    no_change = 0
    regressions = 0
    
    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        question = test_case["question"]
        filters = test_case["filters"]
        filter_type = test_case["filter_type"]
        
        print(f"\n[{i}/{len(test_cases)}] {test_id}")
        print(f"    Type: {filter_type}")
        print(f"    Q: {question[:55]}...")
        
        result = evaluate_groundedness_delta(
            question=question,
            filters=filters,
            k=k,
            verbose=verbose
        )
        result["test_id"] = test_id
        result["description"] = test_case["description"]
        result["filter_type"] = filter_type
        
        results.append(result)
        
        # Track counts
        if result["unfiltered"]["is_grounded"]:
            unfiltered_grounded_count += 1
        if result["filtered"]["is_grounded"]:
            filtered_grounded_count += 1
        
        # Track outcomes
        delta = result["groundedness_delta"]
        if delta > 0:
            improvements += 1
            status = "+1 (improved)"
        elif delta < 0:
            regressions += 1
            status = "-1 (regressed)"
        else:
            no_change += 1
            status = "0 (no change)"
        
        unf_status = "GROUNDED" if result["unfiltered"]["is_grounded"] else "NOT_GROUNDED"
        flt_status = "GROUNDED" if result["filtered"]["is_grounded"] else "NOT_GROUNDED"
        
        print(f"    Unfiltered: {unf_status}")
        print(f"    Filtered:   {flt_status}")
        print(f"    Delta:      {status}")
    
    # Calculate aggregate metrics
    num_tests = len(results)
    unfiltered_rate = unfiltered_grounded_count / num_tests if num_tests else 0
    filtered_rate = filtered_grounded_count / num_tests if num_tests else 0
    avg_delta = (filtered_rate - unfiltered_rate)
    
    # Group by filter type
    by_filter_type = {}
    for r in results:
        ft = r["filter_type"]
        if ft not in by_filter_type:
            by_filter_type[ft] = {"grounded": 0, "total": 0}
        by_filter_type[ft]["total"] += 1
        if r["filtered"]["is_grounded"]:
            by_filter_type[ft]["grounded"] += 1
    
    summary = {
        "k": k,
        "num_test_cases": num_tests,
        "unfiltered_grounded_count": unfiltered_grounded_count,
        "filtered_grounded_count": filtered_grounded_count,
        "unfiltered_groundedness_rate": unfiltered_rate,
        "filtered_groundedness_rate": filtered_rate,
        "groundedness_delta": avg_delta,
        "improvements": improvements,
        "no_change": no_change,
        "regressions": regressions,
        "by_filter_type": {
            ft: data["grounded"] / data["total"] 
            for ft, data in by_filter_type.items()
        },
        "results": results
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\nAggregate Metrics:")
    print(f"   Unfiltered Groundedness Rate: {unfiltered_rate:.2%} ({unfiltered_grounded_count}/{num_tests})")
    print(f"   Filtered Groundedness Rate:   {filtered_rate:.2%} ({filtered_grounded_count}/{num_tests})")
    print(f"   Groundedness Delta:           {avg_delta:+.2%}")
    
    print(f"\nOutcomes:")
    print(f"   Improvements: {improvements}/{num_tests} ({100*improvements/num_tests:.0f}%)")
    print(f"   No Change:    {no_change}/{num_tests} ({100*no_change/num_tests:.0f}%)")
    print(f"   Regressions:  {regressions}/{num_tests} ({100*regressions/num_tests:.0f}%)")
    
    print(f"\nFiltered Groundedness by Filter Type:")
    for ft, rate in summary["by_filter_type"].items():
        print(f"   {ft}: {rate:.2%}")
    
    print(f"\nPer-Query Breakdown:")
    for r in results:
        unf = "G" if r["unfiltered"]["is_grounded"] else "X"
        flt = "G" if r["filtered"]["is_grounded"] else "X"
        delta = r["groundedness_delta"]
        indicator = "+" if delta > 0 else "-" if delta < 0 else "="
        print(f"   [{indicator}] {r['test_id']}: {unf} -> {flt}")
    
    return summary


def main():
    """Run the groundedness delta evaluation."""
    # Validate environment
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    import argparse
    parser = argparse.ArgumentParser(description="Run groundedness delta evaluation")
    parser.add_argument("-k", type=int, default=DEFAULT_K, help=f"Number of documents to retrieve (default: {DEFAULT_K})")
    parser.add_argument("-v", "--verbose", action="store_true", help="Include full context and sources in output")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    summary = run_evaluation(k=args.k, verbose=args.verbose)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
