import streamlit as st
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
import time

# Initialize models (cached to avoid reloading)
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    sentiment_analyzer = pipeline("text-classification", model="kenwuhj/CustomModel_ZA_sentiment")
    return summarizer, sentiment_analyzer

def search_web(query, num_results=5):
    """
    Simulates web search by using DuckDuckGo HTML search
    Returns list of articles with titles and snippets
    """
    try:
        # Using DuckDuckGo HTML search (no API key needed)
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(search_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')

            if title_elem and snippet_elem:
                results.append({
                    'title': title_elem.get_text(),
                    'snippet': snippet_elem.get_text(),
                    'url': title_elem.get('href', '')
                })

        return results
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def combine_search_results(results):
    """Combine search results into a single text for analysis"""
    combined_text = ""
    for i, result in enumerate(results, 1):
        combined_text += f"{result['title']}. {result['snippet']} "
    return combined_text

def analyze_investment(text, summarizer, sentiment_analyzer):
    """
    Performs text summarization and sentiment analysis
    Returns summary, sentiment, and investment advice
    """
    # Summarization
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    summary_text = summary[0]['summary_text']

    # Sentiment Analysis
    sentiment = sentiment_analyzer(text)
    label = sentiment[0]['label'].lower()
    confidence = sentiment[0]['score']

    # Investment Advice
    if label == 'positive':
        advice = "📈 **BUY RECOMMENDATION**: This stock shows positive sentiment and may be a good investment opportunity."
        color = "green"
    elif label == 'negative':
        advice = "📉 **SELL/AVOID RECOMMENDATION**: This stock shows negative sentiment. Consider avoiding or selling."
        color = "red"
    elif label == 'neutral':
        advice = "⏸️ **HOLD/WAIT**: This stock shows neutral sentiment. Adopt a wait-and-see attitude."
        color = "orange"
    else:
        advice = "❓ Unable to determine investment recommendation."
        color = "gray"

    return {
        'summary': summary_text,
        'sentiment': label,
        'confidence': confidence,
        'advice': advice,
        'color': color
    }

def main():
    st.set_page_config(
        page_title="AI Investment Advisor",
        page_icon="📊",
        layout="wide"
    )

    st.title("🤖 AI-Powered Investment Advisor")
    st.markdown("*Analyze stock sentiment using web search, text summarization, and sentiment analysis*")

    # Load models
    with st.spinner("Loading AI models..."):
        summarizer, sentiment_analyzer = load_models()

    # Part 1: Input Question
    st.header("1️⃣ Ask Your Investment Question")
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., Is it a good price now for Tesla Stock?",
        help="Ask about any stock or company"
    )

    analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

    if analyze_button and question:
        # Part 2: Web Search
        st.header("2️⃣ Searching the Web")
        with st.spinner("Searching for relevant articles..."):
            search_results = search_web(question)

        if search_results:
            st.success(f"Found {len(search_results)} relevant articles")

            with st.expander("📰 View Search Results"):
                for i, result in enumerate(search_results, 1):
                    st.markdown(f"**{i}. {result['title']}**")
                    st.markdown(f"_{result['snippet']}_")
                    if result['url']:
                        st.markdown(f"[Read more]({result['url']})")
                    st.divider()

            # Combine results for analysis
            combined_text = combine_search_results(search_results)

            # Part 3: Text Summarization and Sentiment Analysis
            st.header("3️⃣ AI Analysis")
            with st.spinner("Analyzing sentiment and generating summary..."):
                analysis = analyze_investment(combined_text, summarizer, sentiment_analyzer)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📝 Summary")
                st.info(analysis['summary'])

            with col2:
                st.subheader("💭 Sentiment Analysis")
                st.metric(
                    label="Sentiment",
                    value=analysis['sentiment'].upper(),
                    delta=f"{analysis['confidence']:.1%} confidence"
                )

            # Part 4: Investment Suggestion
            st.header("4️⃣ Investment Recommendation")
            if analysis['color'] == 'green':
                st.success(analysis['advice'])
            elif analysis['color'] == 'red':
                st.error(analysis['advice'])
            elif analysis['color'] == 'orange':
                st.warning(analysis['advice'])
            else:
                st.info(analysis['advice'])

            st.markdown("---")
            st.caption("⚠️ Disclaimer: This is an AI-generated analysis for informational purposes only. Always conduct your own research and consult with financial advisors before making investment decisions.")
        else:
            st.error("No search results found. Please try a different question.")

    elif analyze_button:
        st.warning("Please enter a question first.")

    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This app uses:
        - **Web Search**: DuckDuckGo
        - **Summarization**: BART-large-CNN
        - **Sentiment**: Custom ZA Sentiment Model

        **How it works:**
        1. Enter your investment question
        2. AI searches the web for relevant info
        """)
