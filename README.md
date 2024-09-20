# Introduction to RAGs and LLMs

This repository is following the [llm-zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) repository, courtesy of alexey grigorev and DataZoomCamp.

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

# Working on open source LLMs
The previous RAG model involved using either a small toy search engine `minsearch` or a proper one like `elasticsearch` to index all data sources. Then we used OpenAI as the LLM of choice to get a user-friendly response.

Now, to do that we had to give OpenAI our money. And as we know, giving up our money is never fun.

So now, we will replace OpenAI with some open source versions of that.

## Saturn Cloud + Huggingface + Google FLAN
This open source version of LLM basically consists of the following:
- A background system onto which we will run everything. `Saturn cloud` enables the use of GPUs + env setup for free as long as you give them your info and request a trial.
- An open source LLM model to actually use instead of OpenAIs 4gt-o one. For that we use Google's `flan-t5-x1` which is an open-source model and it is downloadable from `Hugging face`. Huggingface basically hosts a library with a ton of tools and models for a wide array of NLP and LLM tasks.

### Other open LLM models
check out huggingface for all the open source LLM models. Some of the models explored by Alexey grigorev include:
- Google Flan
- mistral (french stuff)
- Phi 3 (microsoft developed)

## Running LLMs on CPU only
If you don't have a GPU to run LLMs on, then there is `Ollama`.

Ollama serves as a drop-in replacement to OpenAI's API. You basically redirect the OpenAI client to ollama instead of the OpenAI AI key and stuff.

If your are on mac, download Ollama from [here](https://ollama.com/download).


to connect Ollama with the OpenAI API:
````bash
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)
````

To run Ollama we do the following on the terminal:

```bash
ollama start
ollama run phi3 # you dont need to run this bit if you only want ollama on the background for the openAI API.
```

This runs phi3 mini, which is about 2GB big and has ~3B parameters.

It will start a prompt like in python, in which you can either directly ask a question, or flat out insert a prompt to give it background information.

To stop or exit the user interface, simply do `/bye` or `control + D`

If you run the `rag-intro.ipynb` file, you will notice that it takes some time now in 'thinking' about what to answer once it calls the `llm` function. This is because we are locally now running things.

Also, it is worth noting that the stochastic nature of LLMs means that the answer may change every rerun.

There are some methods to finetune parameters affecting LLM model responses, but we won't dive into that yet.

### Using Ollama via Docker

Basically:
1. Make sure you dont have the actual Ollama running on your computer
2. Run the following:
   ```bash
   docker compose up ollama
   ```
3. Make sure to download the phi3 model by doing the following:
   ```bash
   docker exec -it ollama bash  
   ollama pull phi3
   ```

NOTE: Tried a workaround by adding a dockerfile and asking it to run there pull phi3, but that dit not work. If this ever needs to be automated, then you will likely need an entrypoint.sh file.

you only need to pull phi3 once. since we are asking docker to keep a volumne, the phi3 model will persist.

Another bit to note is that this was run on a macbook laptop, so we got an error about insufficient memory.

If you get that error:
1. Click on the docker icon
2. go to settings
3. go to resources
4. Change the memory allocated to about 6gb or so.

Yeah, docker loves to eat up memory.

Fun fact, we did not have such issues just straight up running Ollama directly on the laptop instead of through a docker image.

Another fun fact:
- Locally the LLM runs in about 10s
- When using a Docker image, this runs in 17 minutes and 39s. This is because the Docker container is using purely the CPU, whereas ollama running locally is actually using the GPU available on the machine.

# Generating a UI
You can use Streamlit to generate easily through python a UI which should work anywhere as long as you can run python in it. That includes Heroku, AWS, etc.

On this course we literally asked ChatGPT to make us a basic script. All we needed to do was to add on the python script all the functions needed for the `rag` function to actually work.

Once you have a workign script, move to the folder where your streamlit script is at and run:

```bash
streamlit run streamlit_faq.py
```

# Vector search

Google uses vector dbs to make searchs also understand the context. You will get different results if you ask about apple employees, vs places to pick up apples.

Vector dbs allow for long term data storage, which is neat for LLMs to store and retrieve data.

## Vector embedings
This is basically the bit where you convert the words into vectors. So intead of "the quick fox" you get something like [1, 34, 553]. Something of the sort.

basically:
- a vector db indexes and stores vector embeddings, for fast search n optimized storage.
- provides the ability to compare multiple things (semantically) at the same time.
- helps ml models remember past data better, making them more useful for search, text generation and recommendations.

## Elastic search
Two important concepts in elasticsearch is `documents` and `indexes`.
- documents -> a collection of fields and their associated values
- index -> a collection of documents that is stored in a highly optimized format designed to perform efficient searches.

**To work with elasticsearch you have to organize the data into documents, and then add all your documents to an index.**

### Sentence transformers

Sentence transformers is a library created by UKPLab and it is maintained by Huggingface. 

It will call a pretrained model in one line of code.

It makes the whole process very simple. 

Have a look [here](https://sbert.net/docs/sentence_transformer/pretrained_models.html) for an overview of the documentation and the different types of pretrained models available.

### Mappings
- Mapping is the process of defining how a document and its fields are stored and indexed,
- each document is a collection of fields, each of those fields have their own datatype.
- We can compare mapping to db schema, describes fields and what properties they hold, datatype, and how those fields are indexed and stored.

## Evaluation metrics
Usually we get users feedback to evaluate the performance of the LLM.

There are different evaluation metrics to check if the LLM is doing well/bad

Common ranking evaluation metrics include:

- Precision at k: measures the no of relevant documents in the top k results
- recall: measures no of relevant documents retrieved out of all relevant ones available
- mean average precision (MAP): computes avg precision for each query and averages over al queries
- etc

## Ground truth
In a FAQ each question has probably 1 or 2 relevant records that you know matter. So to evaluate the ground truth, basically you can flip over the analysis and generate 5 questions for each record and evaluate how the LLM performs.

