# Code

Main code is under the kb folder, all other folders are considered scratchpads and testing.

kb\
    ingestion.py
    retrieval_naive.py # naive rag used for eval purposes
    retrieval_filtered.py 
    generation.py # uses filtered retrival
    evals\
        precision_delta.py # compares naive vs filterede
        groundedness_delta.py # compares naive vs filtered
        groundedness.py # just filtered
 

# Documents

I used our online Knowledge Base website as the source documentation https://support.inn-flow.net/support/home.  The site is broken up into products like Accounting, Labor, Facilities.  Each product section is furthor broken down into subtopic like User Guides, FAQ, and Best Practices.  The entire KB was not ingested.


# Ingestion & Chunking

In order to injest the KB pages I wrote a web page crawler to find and pull all the urls on a given page.  I used a Product / subtopic home page as a parent page, finding all the KB article URLs within the page.  I did this for 5 product/subcategory pages. The original source with category and subtopic are stored in sources.json.

Knowing the Product and Subcategory of each KB url allowed me to use that information as meta data for each article found.  Each were tagged with a category (product) and subcategory.

To ingest the content of each url, the raw html is read in and header tags (h1,h2,h3) are found and thier values used as additional meta data.  Then using langchain WebBasedLoader the content of the entire page is parsed to plain text.  A chunk size of 1500 was used to allow for more context.  The chunks and embeddings are stored in MongoDB and a vector search index applied making use of all the meta data.

## Ingestion Stats
   Sources processed:  4
   Topics:             Accounting, Accounting, Labor, Facilities
   Total articles:     113
   Total chunks:       443
   Total time:         165.9 seconds (2.8 minutes)

   Breakdown by topic:
      - Accounting / User Guides: 53 articles, 209 chunks
      - Accounting / ePay: 7 articles, 36 chunks
      - Labor / User Guides: 43 articles, 164 chunks
      - Facilities / User Guides: 10 articles, 34 chunks

# Retrieval

Retrieval was setup to use metadata filtered RAG.  Several metadata attributes were added to each chunk in the ingestion process making this version of RAG the logical choice.  

# Eval

Two evals are set up precision_delta.py which evaluates the precision retrieval between filtered and naive.  groundedness_delta compares the groundedness between naive and filtered. The same 8 test cases are used for both evals.  The test cases exercise the various filter options.

The results show that appling bad filters, filters where there was no meta data value, significantly hurt the results.  The inverse is also true.  Narrowing the result set improves the response.

# Run

```bash

python -m venv venv
venv\Scripts\activate # windows
python kb\ingestion.py --config sources.json 
python kb\retrieval.py  # to test retrieval not required
python kb\generation.py

python kb\evals\precision_delta.py
python kb\evals\groundedness_delta.py


```