import streamlit as st
from transformers import pipeline, AutoTokenizer
import requests
from bs4 import BeautifulSoup

@st.cache_resource

def text_summarization(summarization_name, model):
    """
    Function to perform text summarization
    
    Args:
        summarization_name: The input text to summarize
        model: The model to use for summarization
    
    Returns:
        The summarized text
    """
    # The parameter name should match what you're returning
    # Option 1: Return the input parameter directly (if no processing needed)
    return summarization_name

st.title("Robo-Advisor")
summarization_name = st.text_input("Enter a URL:")

if st.button("Summarize") and summarization_name:
    model = load_model()
    text_description = text_summarization(summarization_name, model)
    st.write(f"Generated text: {text_description}")
