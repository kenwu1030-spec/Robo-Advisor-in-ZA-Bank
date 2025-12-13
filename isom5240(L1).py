
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup
import numpy as np
import torch
from docx import Document
import io

# Set page configuration
st.set_page_config(page_title="Robo-Advisor", page_icon="📈", layout="wide")

# Title
st.title("📈 Robo-Advisor: Financial Article Analysis")
st.markdown("Analyze financial articles to get investment recommendations based on sentiment analysis")

# Initialize models with caching
@st.cache_resource
def load_summarization_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    tokenizer.model_max_length = 1024
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer=tokenizer)
    return summarizer

@st.cache_resource
def load_sentiment_model():
    model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return sentiment_analyzer

# Function: Text Summarization from URL
def text_summarization_url(url, summarizer):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        paragraphs = soup.find_all('p')
        text_content = "".join([p.get_text() for p in paragraphs])
        
        if not text_content.strip():
            text_content = soup.get_text()
        
        summary = summarizer(
            text_content,
            max_length=150,
            min_length=50,
            truncation=True
        )[0]["summary_text"]
        
        return summary
    except Exception as e:
        st.error(f"Error fetching URL: {str(e)}")
        return None

# Function: Text Summarization from DOCX
def text_summarization_docx(file, summarizer):
    try:
        doc = Document(io.BytesIO(file.read()))
        text_content = "".join([paragraph.text for paragraph in doc.paragraphs])
        
        if not text_content.strip():
            st.error("No text found in the document")
            return None
        
        summary = summarizer(
            text_content,
            max_length=150,
            min_length=50,
            truncation=True
        )[0]["summary_text"]
        
        return summary
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        return None

# Function: Investment Advisor
def investment_advisor(summary_text, sentiment_result):
    sentiment_label = sentiment_result[0]['label'].lower()
    confidence = sentiment_result[0]['score']
    
    if sentiment_label == 'positive':
        advice = "This stock is recommended to buy."
    elif sentiment_label == 'negative':
        advice = "This stock is not recommended to buy."
    elif sentiment_label == 'neutral':
        advice = "This stock needs to adopt a wait-and-see attitude."
    else:
        advice = "Unable to determine investment recommendation."
    
    return {
        'summary': summary_text,
        'sentiment': sentiment_label,
        'confidence': confidence,
        'advice': advice
    }

# Main App
def main():
    # Load models
    with st.spinner("Loading AI models..."):
        summarizer = load_summarization_model()
        sentiment_analyzer = load_sentiment_model()
    
    # Sidebar for input selection
    st.sidebar.header("Input Options")
    input_type = st.sidebar.radio("Choose input method:", ["URL", "Upload DOCX File"])
    
    summary_text = None
    
    # Input handling
    if input_type == "URL":
        url = st.text_input("Enter the URL of the financial article:", placeholder="https://example.com/article")
        if st.button("Analyze Article from URL"):
            if url:
                with st.spinner("Fetching and summarizing article..."):
                    summary_text = text_summarization_url(url, summarizer)
            else:
                st.warning("Please enter a valid URL")
    
    else:  # Upload DOCX File
        uploaded_file = st.file_uploader("Upload a DOCX file:", type=["docx"])
        if uploaded_file and st.button("Analyze Uploaded Document"):
            with st.spinner("Processing document..."):
                summary_text = text_summarization_docx(uploaded_file, summarizer)
    
    # Analysis and Results
    if summary_text:
        st.success("Summary generated successfully!")
        
        # Display summary
        st.subheader("📄 Article Summary")
        st.write(summary_text)
        
        # Perform sentiment analysis
        with st.spinner("Analyzing sentiment..."):
            sentiment_result = sentiment_analyzer(summary_text)
        
        # Generate investment advice
        result = investment_advisor(summary_text, sentiment_result)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Investment Analysis Report")
        
        col1, col2 = st.columns(2)
