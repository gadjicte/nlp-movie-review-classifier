import streamlit as st
import spacy
import re
import os
from joblib import load
import warnings
warnings.filterwarnings('ignore')

# Initialize session state for model and NLP
if 'nlp' not in st.session_state:
    try:
        st.session_state.nlp = spacy.load("en_core_web_lg")
    except OSError:
        st.error("Downloading spaCy model... Please wait.")
        os.system("python -m spacy download en_core_web_lg")
        st.session_state.nlp = spacy.load("en_core_web_lg")

if 'model' not in st.session_state:
    try:
        import numpy as np
        # Ensure numpy version compatibility
        if np.__version__ >= '2.0.0':
            st.warning("Warning: Your NumPy version might be incompatible with the saved model. If you encounter errors, please run: pip install numpy==1.24.3")
        
        # Look for the model in the current directory and parent directory
        model_paths = [
            'sentiment_model.joblib',
            os.path.join(os.path.dirname(__file__), 'sentiment_model.joblib'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sentiment_model.joblib')
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                st.session_state.model = load(path)
                st.session_state.model_path = path
                break
        else:
            raise FileNotFoundError("Model file not found")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocessing(text):
    if not text:
        return ""
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.strip().lower()  # Normalize whitespace and lowercase
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    doc = st.session_state.nlp(text)
    preprocessed_tokens = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        preprocessed_tokens.append(token.lemma_)
    return " ".join(preprocessed_tokens)

# Create the Streamlit app
st.title("Sentiment Analysis of Movie Reviews")
st.write("This is a simple web app to analyze the sentiment of movie reviews.")



user_input = st.text_area("Enter a movie review:", height=100)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review first.")
    else:
        try:
            # Preprocess the input
            with st.spinner("Processing your review..."):
                processed_input = preprocessing(user_input)
                
                # Make prediction
                prediction = st.session_state.model.predict([processed_input])
                
                # Show result with appropriate styling
                if prediction[0] == 1:
                    st.success("Sentiment: Positive 😊")
                else:
                    st.error("Sentiment: Negative ☹️")
                
                # Show the processed text in an expander
                with st.expander("See processed text"):
                    st.write(processed_input)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try again with a different review.")