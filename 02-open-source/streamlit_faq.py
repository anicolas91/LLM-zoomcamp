import streamlit as st
import minsearch # alexeys small and fast search engine
import requests
from openai import OpenAI

# initialize things that you need for this to run
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

# index the data
# load json data directly from the url 
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url,timeout=10)
documents_raw = docs_response.json()

# rearrange data a bit (add course type to each faq)
documents = []
for course_dict in documents_raw:
    for doc in course_dict['documents']:
        doc['course'] = course_dict['course'] #adding it to every faq
        documents.append(doc)

# initialize class, tell the search engine what is searchable and what are keywords
index = minsearch.Index(
    text_fields=['text','section','question'],
    keyword_fields=['course']
)

#actually train the search engine
index.fit(docs=documents)


# define all the aux functions
def search(query):
    '''  
    This function runs the already trained search engine and retrieves the top 5 results,
    '''
    boost = {'question': 3.0, 'section': 0.5} # give it weights
    results = index.search(
        query=query,
        filter_dict={'course': 'data-engineering-zoomcamp'},
        boost_dict=boost,
        num_results=5
    )

    return results

def build_prompt(query, search_results):
    ''' 
    This function starts with a prompt template.
    Given the query fills the template out with the results from the search engine
    '''
    # we will give the llm some context
    # Alexey mentions that this is a bit of art and science because you somewhat
    # iterate until you find something that works for you.
    prompt_template =  """

    You are a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database. 
    Use only the facts from the CONTEXT when answering the QUESTION.
    If the CONTEXT does not contain the answer, output NONE.

    QUESTION: {question}

    CONTEXT: {context}

    """.strip() #no line break

    #convert search results into proper formatted context
    context = ""
    for doc in search_results:
        context = context + \
        f"section: {doc['section']}\nquestion: {doc['question']}\nanswer:{doc['text']}\n\n"

    # we formally add the info on the prompt
    return prompt_template.format(question=query,context=context).strip()

def llm(prompt,model='phi3'):
    ''' 
    This function trains chatGPT with our prompt (with the search engine results)
    '''
    response = client.chat.completions.create(
        model = model,
        messages=[{'role':'user','content': prompt}]
    )

    return response.choices[0].message.content

def rag(query):
    '''  
    This function, given a question, finds best answers on the search engine, trains the llm with it, and returns a result
    '''
    results = search(query)
    prompt = build_prompt(query, results)
    answer = llm(prompt)

    return answer

# Streamlit application
def main():
    ''' 
    Main function that creates the UI
    '''
    st.title("Course FAQ with RAG - App")

    # Input box
    user_input = st.text_input("Enter your text")

    # Button to trigger the RAG function
    if st.button("Ask"):
        if user_input:
            with st.spinner("Processing..."):
                result = rag(user_input)
            st.success("Done!")
            st.write(result)
        else:
            st.error("Please enter some text")

if __name__ == "__main__":
    main()
