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

## Open source alternatives to openAI
OpenAI and GPT are not the only hosted LLMs that we can use. 
There are other services that we can use

* [mistral.ai](https://mistral.ai) (5€ free credit on sign up)
* [Groq](https://console.groq.com) (can inference from open source LLMs with rate limits)
* [TogetherAI](https://api.together.ai) (can inference from variety of open source LLMs, 25$ free credit on sign up)
* [Google Gemini](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python) (2 months unlimited access)
* [OpenRouterAI](https://openrouter.ai/) (some small open-source models, such as Gemma 7B, are free)
* [HuggingFace API](https://huggingface.co/docs/api-inference/index) (over 150,000 open-source models, rate-limited and free)
* [Cohere](https://cohere.com/) (provides a developer trail key which allows upto 100 reqs/min for generating, summarizing, and classifying text. Read more [here](https://cohere.com/blog/free-developer-tier-announcement))
* [wit](https://wit.ai/) (Facebook AI Afiliate - free)
* [Anthropic API](https://www.anthropic.com/pricing#anthropic-api) (starting from $0.25 / MTok for input and $1.25 / MTok for the output for the most affordable model)
* [AI21Labs API](https://www.ai21.com/pricing#foundation-models) (Free trial including $10 credits for 3 months)
* [Replicate](https://replicate.com/) (faster inference, can host any ML model. charges 0.10$ per 1M input tokens for llama/Mistral model)

## cleanup
cleanup essentially moves important bits into defs, and the final code is one nice readable script callling the aux fcns out.

## elasticsearch
We simulate this with docker and use it to replace the toy `minsearch` search engine.