import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

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
    input_variables = ['title'],
    template = 'Write a YouTube video Sscript based on this title " {title}'
)

# Llms
llm = OpenAI(temperature=0.9)
tittleChain = LLMChain(llm=llm, prompt = titleTemplate, verbose=True)
scriptChain = LLMChain(llm=llm, prompt = scriptTemplate, verbose=True)
sequentialChain = SimpleSequentialChain(chains=[tittleChain,scriptChain], verbose=True)

# Show stuff to the sreen if there's a prompt
if prompt:
    response = sequentialChain.run(prompt)
    st.write(response)