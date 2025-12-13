
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def main():
    tokenizer = AutoTokenizer.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("kenwuhj/CustomModel_ZA_sentiment")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    st.title("Sentiment Analysis with HuggingFace Spaces")
    st.write("Enter a sentence to analyze its sentiment:")

    user_input = st.text_input("Input text for sentiment analysis:", label_visibility="hidden")
    if user_input:
        result = sentiment_pipeline(user_input)
        sentiment = result[0]["label"]
        confidence = result[0]["score"]

        st.write(f"Sentiment: {sentiment}")
        st.write(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
import numpy as np

def investment_advisor(summary_text, sentiment_label, sentiment_predictions):
    # Calculate confidence from the sentiment_predictions array
    confidence = sentiment_predictions[0][np.argmax(sentiment_predictions[0])]

    # Step 3: Generate investment advice based on sentiment
    if sentiment_label == 'positive':
        advice = "This stock is recommended to buy."
    elif sentiment_label == 'negative':
        advice = "This stock is not recommended to buy."
    elif sentiment_label == 'neutral':
        advice = "This stock needs to adopt a wait-and-see attitude."
    else:
        advice = "Unable to determine investment recommendation."

    # Return results
    return {
        'summary': summary_text,
        'sentiment': sentiment_label,
        'confidence': confidence,
        'advice': advice
    }

# Example usage
if __name__ == "__main__":
    # Get values from kernel state (assuming these variables are defined in previous cells)
    current_summary_text = text_description
    current_predicted_label = predicted_label
    current_predictions = predictions

    result = investment_advisor(current_summary_text, current_predicted_label, current_predictions)

    print("=" * 60)
    print("INVESTMENT ANALYSIS REPORT")
    print("=" * 60)
    print(f"\nSUMMARY:\n{result['summary']}")
    print(f"\nSENTIMENT: {result['sentiment'].upper()} (Confidence: {result['confidence']:.2%})")
    print(f"\nINVESTMENT ADVICE:\n{result['advice']}")
    print("=" * 60)
