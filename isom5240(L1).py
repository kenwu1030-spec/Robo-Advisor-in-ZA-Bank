
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
from bs4 import BeautifulSoup

# Page configuration
st.set_page_config(
    page_title="Investment Research Assistant",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Investment Research Assistant")
st.subheader("Financial Article Analysis")
st.write("Analyze financial articles to get investment recommendations based on sentiment analysis")

# Initialize models (cached for performance)
@st.cache_resource
def load_models():
    # Sentiment analysis model
    sentiment_model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    sentiment_tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)
    
    # Summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    return sentiment_analyzer, summarizer

# Load models
try:
    sentiment_analyzer, summarizer = load_models()
    st.success("✅ Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Helper functions
def extract_text_from_url(url):
    """Extract article text from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"Error extracting text from URL: {str(e)}")
        return None

def chunk_text(text, max_length=512):
    """Split text into chunks for processing"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def analyze_article(text):
    """Main analysis pipeline"""
    # Step 1: Summarization
    st.write("### 📝 Step 1: Text Summarization")
    with st.spinner("Generating summary..."):
        chunks = chunk_text(text, max_length=1024)
        summaries = []
        
        for chunk in chunks[:3]:  # Limit to first 3 chunks for performance
            if len(chunk.split()) > 50:
                summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
        
        full_summary = ' '.join(summaries)
        st.write(full_summary)
    
    # Step 2: Sentiment Analysis (FIXED)
    st.write("### 💭 Step 2: Sentiment Analysis")
    with st.spinner("Analyzing sentiment..."):
        sentiment_chunks = chunk_text(full_summary, max_length=512)
        sentiments = []
        
        for chunk in sentiment_chunks:
            result = sentiment_analyzer(chunk)[0]
            sentiments.append(result)
        
        # Display individual chunk sentiments for debugging
        st.write("**Sentiment breakdown by chunk:**")
        for i, s in enumerate(sentiments):
            st.write(f"Chunk {i+1}: {s['label']} (confidence: {s['score']:.3f})")
        
        # Map labels to sentiment values: LABEL_0=negative(-1), LABEL_1=neutral(0), LABEL_2=positive(+1)
        label_to_value = {
            'LABEL_0': 0,  # negative
            'LABEL_1': 1,   # neutral
            'LABEL_2': 2,    # positive
        }
        
        # Calculate weighted average sentiment
        weighted_sum = sum([label_to_value[s['label']] * s['score'] for s in sentiments])
        total_weight = sum([s['score'] for s in sentiments])
        avg_sentiment = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine overall sentiment based on weighted average
        if avg_sentiment > 0.2:
            overall_sentiment = "Positive"
            sentiment_color = "🟢"
        elif avg_sentiment < -0.2:
            overall_sentiment = "Negative"
            sentiment_color = "🔴"
        else:
            overall_sentiment = "Neutral"
            sentiment_color = "🟡"
        
        st.write(f"{sentiment_color} **Overall Sentiment:** {overall_sentiment}")
        st.write(f"**Sentiment Score:** {avg_sentiment:.3f} (range: -1 to +1)")
    
    # Step 3: Investment Recommendation (FIXED)
    st.write("### 💡 Step 3: Investment Recommendation")
    
    if overall_sentiment == "Positive":
        recommendation = "**BUY** - The article sentiment is positive, suggesting favorable market conditions or company performance."
        rec_color = "success"
    elif overall_sentiment == "Negative":
        recommendation = "**SELL/AVOID** - The article sentiment is negative, indicating potential risks or unfavorable conditions."
        rec_color = "error"
    else:
        recommendation = "**HOLD** - The article sentiment is neutral. Consider additional research before making investment decisions."
        rec_color = "warning"
    
    st.markdown(f":{rec_color}[{recommendation}]")
    
    return full_summary, overall_sentiment, recommendation

# Main interface
st.write("---")

# URL input
url = st.text_input("Enter article URL:", placeholder="https://example.com/financial-article")

if st.button("Analyze Article", type="primary"):
    if url:
        with st.spinner("Fetching article..."):
            article_text = extract_text_from_url(url)
        
        if article_text:
            if len(article_text.split()) < 50:
                st.error("Article is too short for meaningful analysis. Please provide a longer article.")
            else:
                st.write("---")
                st.write("## 📊 Analysis Results")
                
                try:
                    summary, sentiment, recommendation = analyze_article(article_text)
                    
                    st.write("---")
                    st.success("✅ Analysis complete!")
                    
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
    else:
        st.warning("Please enter a URL")

# Footer
st.write("---")
st.caption("Powered by kenwuhj/CustomModel_ZA_sentiment | Built with Streamlit")


