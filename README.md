## Setup

Start up a conda environment via:
````bash
conda create -n llm_env python=3.10
conda activate llm_env
pip install -r requirements
````

## Getting started
1. you probably would benefit from getting an account with OpenAI. APIs are not free, so maybe put in there 5 bucks.
2. You need some json or similar source dataset, where the data is basically a good formatted Q and A, with some keywords or categories attached. We get the `documents.json` dataset from the LLM-zoomcamp repository.
3. We initialize RAG by training a simple search engine with the FAQ dataset. The `minsearch` search engine was extracted from alexey's gregory repository.
4. We will get a q, and all the top n answers. We use that as a context to an LLM to answer the same question.
