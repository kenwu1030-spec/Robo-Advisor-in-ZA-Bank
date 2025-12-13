import streamlit as st
from transformers import pipeline, AutoTokenizer
import requests
from bs4 import BeautifulSoup

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = 1024
    return pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer)

def text_summarization(url, model):
    # ... rest of your function code ...
    return input_text

st.title("Robo-Advisor")
summarization_name = st.text_input("Enter a URL:")

if st.button("Summarize") and summarization_name:
    model = load_model()
    text_description = text_summarization(summarization_name, model)
    st.write(f"Generated text: {text_description}")
