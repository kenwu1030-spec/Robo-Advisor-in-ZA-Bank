git init
git add.
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

import streamlit as st
st.write("Robo-Advisor-in-ZA-Bank")


from transformers import pipeline

# Initialize the two pipelines
text_summarization_model1 = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_pipeline = pipeline("text-classification", model="kenwuhj/CustomModel_ZA_sentiment")

def investment_advisor(text):
    """
    Analyzes stock-related text and provides investment advice.

    Args:
        text: Input text about a stock or company

    Returns:
        dict: Contains summary and investment advice
    """
    # Step 1: Summarize the text
    summary = text_summarization_model1(text, max_length=130, min_length=30, do_sample=False)
    summary_text = summary[0]['summary_text']

    # Step 2: Perform sentiment analysis on the original text
    sentiment = sentiment_pipeline(text)
    label = sentiment[0]['label'].lower()
    confidence = sentiment[0]['score']

    # Step 3: Generate investment advice based on sentiment
    if label == 'positive':
        advice = "This stock is recommended to buy."
    elif label == 'negative':
        advice = "This stock is not recommended to buy."
    elif label == 'neutral':
        advice = "This stock needs to adopt a wait-and-see attitude."
    else:
        advice = "Unable to determine investment recommendation."

    # Return results
    return {
        'summary': summary_text,
        'sentiment': label,
        'confidence': confidence,
        'advice': advice
    }

# Example usage
if __name__ == "__main__":
    # Sample stock news text
    sample_text = """
    Microstrategies price dropped 10% because of the Bitcoin price drop.
    """

    result = investment_advisor(sample_text)

    print("=" * 60)
    print("INVESTMENT ANALYSIS REPORT")
    print("=" * 60)
    print(f"""
SUMMARY:
{result['summary']}""")
    print(f"""
SENTIMENT: {result['sentiment'].upper()} (Confidence: {result['confidence']:.2%})""")
    print(f"""
INVESTMENT ADVICE:
{result['advice']}""")
    print("=" * 60)
