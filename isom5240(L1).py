
import streamlit as st
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from docx import Document
import io

# Set page config
st.set_page_config(
    page_title="Robo-Advisor for Financial Articles",
    page_icon="📊",
    layout="wide"
)

# Cache models to avoid reloading
@st.cache_resource
def load_summarization_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return tokenizer, model

@st.cache_resource
def load_sentiment_model():
    tokenizer = AutoTokenizer.from_pretrained('kenwuhj/CustomModel_ZA_sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('kenwuhj/CustomModel_ZA_sentiment')
    return tokenizer, model

def extract_text_from_docx(file):
    """Extract text from uploaded DOCX file"""
    doc = Document(io.BytesIO(file.read()))
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '
'.join(full_text)

def summarize_text(text, tokenizer, model, max_length=150, min_length=50):
    """Summarize the input text"""
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def analyze_sentiment(text, tokenizer, model):
    """Analyze sentiment of the text"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Map to sentiment labels
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    sentiment = sentiment_map.get(predicted_class, "Unknown")
    
    # Get confidence scores
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    confidence = probabilities[predicted_class].item()
    
    return sentiment, confidence, probabilities.tolist()

def generate_advice(sentiment, confidence):
    """Generate investment advice based on sentiment analysis"""
    if sentiment == "Positive" and confidence > 0.7:
        return "🟢 **Strong Buy Signal**: The article sentiment is strongly positive. Consider increasing your position or entering the market."
    elif sentiment == "Positive":
        return "🟢 **Moderate Buy Signal**: The article sentiment is positive but with moderate confidence. Consider a cautious entry or small position increase."
    elif sentiment == "Negative" and confidence > 0.7:
        return "🔴 **Strong Sell Signal**: The article sentiment is strongly negative. Consider reducing your position or exiting the market."
    elif sentiment == "Negative":
        return "🔴 **Moderate Sell Signal**: The article sentiment is negative but with moderate confidence. Consider reducing exposure or holding."
    else:
        return "🟡 **Hold Signal**: The article sentiment is neutral. Maintain current positions and monitor for further developments."

# Main app
def main():
    st.title("📊 Robo-Advisor for Financial Articles")
    st.markdown("Analyze financial articles and get investment recommendations based on AI-powered sentiment analysis")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This robo-advisor app:
        1. Accepts financial article uploads (DOCX)
        2. Summarizes the content
        3. Analyzes sentiment
        4. Provides investment recommendations
        """)
        
        st.header("Settings")
        max_summary_length = st.slider("Max Summary Length", 100, 300, 150)
        min_summary_length = st.slider("Min Summary Length", 30, 100, 50)
    
    # Load models with progress indication
    with st.spinner("Loading AI models..."):
        sum_tokenizer, sum_model = load_summarization_model()
        sent_tokenizer, sent_model = load_sentiment_model()
    
    st.success("✅ Models loaded successfully!")
    
    # File upload section
    st.header("1️⃣ Upload Financial Article")
    uploaded_file = st.file_uploader("Choose a DOCX file", type=['docx'])
    
    if uploaded_file is not None:
        # Extract text
        with st.spinner("Extracting text from document..."):
            article_text = extract_text_from_docx(uploaded_file)
        
        st.success(f"✅ Extracted {len(article_text)} characters")
        
        with st.expander("View Original Article"):
            st.text_area("Article Content", article_text, height=200)
        
        # Summarization
        st.header("2️⃣ Article Summary")
        with st.spinner("Generating summary..."):
            summary = summarize_text(article_text, sum_tokenizer, sum_model, 
                                    max_length=max_summary_length, 
                                    min_length=min_summary_length)
        
        st.info(summary)
        
        # Sentiment Analysis
        st.header("3️⃣ Sentiment Analysis")
        with st.spinner("Analyzing sentiment..."):
            sentiment, confidence, probabilities = analyze_sentiment(summary, sent_tokenizer, sent_model)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with col3:
            sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
            st.metric("Signal", sentiment_color.get(sentiment, "⚪"))
        
        # Probability distribution
        st.subheader("Sentiment Probability Distribution")
        prob_df = pd.DataFrame({
            'Sentiment': ['Negative', 'Neutral', 'Positive'],
            'Probability': probabilities
        })
        st.bar_chart(prob_df.set_index('Sentiment'))
