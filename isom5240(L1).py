
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Define the load_model function
@st.cache_resource
def load_model():
    """
    Load the sentiment analysis model and tokenizer
    Returns the sentiment analysis pipeline
    """
    model_name = "kenwuhj/CustomModel_ZA_sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_pipeline

# Define the text_summarization function
@st.cache_resource
def load_summarization_model():
    """
    Load the text summarization model
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    return summarizer

def text_summarization(input_text, summarizer):
    """
    Summarize the input text
    """
    # Truncate if text is too long (max 1024 tokens for BART)
    max_length = min(len(input_text.split()), 1024)
    summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Streamlit app
st.title("Robo-Advisor")

# Load models
model = load_model()
summarizer = load_summarization_model()

# User input
url_input = st.text_input("Enter a URL:")

if st.button("Summarize"):
    if url_input:
        # Your text processing logic here
        text_description = text_summarization(url_input, summarizer)
        st.write("Summary:", text_description)
        
        # Sentiment analysis
        sentiment = model(text_description)
        st.write("Sentiment:", sentiment)
    else:
        st.warning("Please enter a URL")
