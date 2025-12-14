
import streamlit as st
from googlesearch import search
import requests
from bs4 import BeautifulSoup
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Page config
st.set_page_config(page_title="Investment Research Assistant", page_icon="📈")
st.title("📈 Investment Research Assistant: Stock Analysis via News")
st.write("Ask questions about stocks and get investment recommendations based on latest news sentiment analysis")

# Initialize models
@st.cache_resource
def load_models():
    # Summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # Sentiment analysis model
    sentiment_model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    
    return summarizer, tokenizer, model

summarizer, tokenizer, sentiment_model = load_models()

def search_google_news(query, num_results=5):
    """Search Google for news articles"""
    try:
        # Remove the 'pause' parameter - it's no longer supported
        search_results = search(query, num_results=num_results, lang="en")
        return list(search_results)
    except Exception as e:
        st.error(f"Error searching Google: {str(e)}")
        return []

def scrape_article(url):
    """Scrape article content from URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract paragraphs
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])
        
        return content[:2000]  # Limit content length
    except Exception as e:
        return None

def summarize_text(text):
    """Summarize text using BART model"""
    try:
        if len(text) < 100:
            return text
        
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return text[:200]

def analyze_sentiment(text):
    """Analyze sentiment using FinBERT with softmax and argmax approach"""
    try:
        # Tokenize with proper configuration
        inputs = tokenizer(f"Generated text: {text}", 
                          return_tensors="pt", 
                          truncation=True, 
                          max_length=512,
                          padding=True)
        
        # Get model predictions
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)[0]
        
        # Label mapping for FinBERT
        labels = ['positive', 'negative', 'neutral']
        
        # Get all sentiment scores
        sentiment_scores = {
            labels[i]: float(probabilities[i]) * 100 
            for i in range(len(labels))
        }
        
        # Get primary sentiment using argmax
        primary_idx = torch.argmax(probabilities).item()
        primary_sentiment = labels[primary_idx]
        primary_confidence = sentiment_scores[primary_sentiment]
        
        return primary_sentiment, primary_confidence, sentiment_scores
    except Exception as e:
        return "neutral", 50.0, {"positive": 33.33, "negative": 33.33, "neutral": 33.34}

# Main app
st.header("💬 Ask About a Stock")

query = st.text_input("Enter your question:", placeholder="e.g., What's the latest news on Tesla stock?")

if st.button("Analyze Stock"):
    if query:
        with st.spinner("Searching for news articles..."):
            urls = search_google_news(query, num_results=5)
        
        if not urls:
            st.warning("Could not find any news articles. Please try a different question.")
        else:
            st.success(f"Found {len(urls)} articles. Analyzing...")
            
            articles_data = []
            
            for i, url in enumerate(urls):
                with st.spinner(f"Processing article {i+1}/{len(urls)}..."):
                    content = scrape_article(url)
                    
                    if content:
                        summary = summarize_text(content)
                        sentiment, confidence, all_scores = analyze_sentiment(summary)
                        
                        articles_data.append({
                            'url': url,
                            'summary': summary,
                            'sentiment': sentiment,
                            'confidence': confidence,
                            'all_scores': all_scores
                        })
            
            if articles_data:
                st.header("📊 Analysis Results")
                
                # Overall sentiment
                avg_positive = sum([a['all_scores']['positive'] for a in articles_data]) / len(articles_data)
                avg_negative = sum([a['all_scores']['negative'] for a in articles_data]) / len(articles_data)
                avg_neutral = sum([a['all_scores']['neutral'] for a in articles_data]) / len(articles_data)
                
                st.subheader("Overall Sentiment Distribution")
                col1, col2, col3 = st.columns(3)
                col1.metric("Positive", f"{avg_positive:.2f}%")
                col2.metric("Neutral", f"{avg_neutral:.2f}%")
                col3.metric("Negative", f"{avg_negative:.2f}%")
                
                # Investment recommendation
                st.subheader("💡 Investment Recommendation")
                if avg_positive > avg_negative + 10:
                    st.success("**BULLISH**: News sentiment is predominantly positive. Consider buying opportunities.")
                elif avg_negative > avg_positive + 10:
                    st.error("**BEARISH**: News sentiment is predominantly negative. Exercise caution.")
                else:
                    st.info("**NEUTRAL**: Mixed sentiment. Further research recommended.")
                
                # Individual articles
                st.subheader("📰 Article Summaries")
                for i, article in enumerate(articles_data):
                    with st.expander(f"Article {i+1} - {article['sentiment'].upper()} ({article['confidence']:.2f}%)"):
                        st.write(f"**URL**: {article['url']}")
                        st.write(f"**Summary**: {article['summary']}")
                        st.write(f"**Sentiment Breakdown**:")
                        st.write(f"- Positive: {article['all_scores']['positive']:.2f}%")
                        st.write(f"- Neutral: {article['all_scores']['neutral']:.2f}%")
                        st.write(f"- Negative: {article['all_scores']['negative']:.2f}%")
            else:
                st.warning("Could not extract content from articles. Please try a different question.")
    else:
        st.warning("Please enter a question.")
