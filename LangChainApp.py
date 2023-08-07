import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI

os.environ['OPENAI_API_KEY'] = apikey

# Application Framework
st.title('🦜️🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Llms
llm = OpenAI(temperature=0.9)

# Show stuff to the sreen if there's a prompt
if prompt:
    response = llm(prompt)
    st.write(response)