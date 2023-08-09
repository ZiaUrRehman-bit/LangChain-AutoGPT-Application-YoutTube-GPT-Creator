import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

# Application Framework
st.title('🦜️🔗 LangChain Application (Like ChatGPT)')
prompt = st.text_input('Plug in your prompt here')

# Prompt Templates
titleTemplate = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write a topic about {topic}'
)

scriptTemplate = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write a YouTube video Sscript based on this title: {title} while leverage this wikipedia research: {wikipedia_research}'
)

# Memory
titleMemory = ConversationBufferMemory(input_key= 'topic', memory_key='chat_history')
scriptMemory = ConversationBufferMemory(input_key= 'title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9)
tittleChain = LLMChain(llm=llm, prompt = titleTemplate, 
                       verbose=True, output_key='title', memory=titleMemory)
scriptChain = LLMChain(llm=llm, prompt = scriptTemplate, 
                       verbose=True, output_key='script', memory=scriptMemory)
# sequentialChain = SequentialChain(chains=[tittleChain,scriptChain],input_variables=['topic'], 
#                                   output_variables=['title', 'script'], verbose=True)
wiki = WikipediaAPIWrapper()
# Show stuff to the sreen if there's a prompt
if prompt:
    title = tittleChain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = scriptChain.run(title = title, wikipedia_research = wiki_research)
    
    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(titleMemory.buffer)

    with st.expander('Script History'):
        st.info(scriptMemory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)