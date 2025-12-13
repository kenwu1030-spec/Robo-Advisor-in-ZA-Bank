
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import docx
import io

# Page configuration
st.set_page_config(
    page_title="Robo-Advisor",
    page_icon="📈",
    layout="wide"
)

# Load models with caching for better performance
@st.cache_resource
def load_summarization_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    return tokenizer, model

# Define label mapping
id2label = {0: "negative", 1: "neutral", 2: "positive"}

def text_summarization(file_content):
    """Summarize text from uploaded DOCX file"""
    summarizer = load_summarization_model()
    document = docx.Document(io.BytesIO(file_content))
    text_content = "
".join([paragraph.text for paragraph in document.paragraphs])
    
    # Handle long texts by chunking
    max_chunk = 1024
    if len(text_content) > max_chunk:
        text_content = text_content[:max_chunk]
    
    summary = summarizer(text_content, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

def sentiment_analysis(text):
    """Analyze sentiment of the summarized text"""
    tokenizer, model = load_sentiment_model()
    
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
