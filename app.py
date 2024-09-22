# app.py
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# Load the VADER model (optional since VADER doesn't require saving, but for demonstration)
def load_model(model_path):
    return joblib.load(model_path)

# Sentiment analysis function
def classify_sentiment(text, analyzer):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Streamlit app
def main():
    st.title("Sentiment Analysis of English-Punjabi mixed Social Media Post")
    st.write("This app classifies the sentiment of English-Punjabi mixed Social Media posts.")

    # Text input from the user
    user_input = st.text_area("Enter text for sentiment analysis:")

    # Load the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()  # Optional: load_model('vader_sentiment_analyzer.pkl')

    if st.button("Analyze"):
        if user_input:
            sentiment = classify_sentiment(user_input, analyzer)
            st.write(f"The sentiment of the entered text is: **{sentiment}**")
        else:
            st.write("Please enter text for analysis.")

if __name__ == '__main__':
    main()
