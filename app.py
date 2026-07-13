from pathlib import Path
import streamlit as st
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------------------------------------------------
# 1. Secure Artifact Loading (Fixes the FileNotFoundError)
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Dynamically find the absolute path of the current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construct exact absolute paths for cloud deployment
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pkl")
    
    # Change "gru_model.h5" to match your exact model filename if it is different
    model_path = os.path.join(BASE_DIR, "gru_sentiment_model.h5") 
    
    # Load the Tokenizer
    if not os.path.exists(tokenizer_path):
        st.error(f"Missing file: '{tokenizer_path}' not found. Ensure it is committed to GitHub.")
        st.stop()
        
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
        
    # Load the GRU Model
    if not os.path.exists(model_path):
        st.error(f"Missing file: '{model_path}' not found. Ensure it is committed to GitHub.")
        st.stop()
        
    model = load_model(model_path)
    
    return model, tokenizer

# Initialize the model and tokenizer
try:
    model, tokenizer = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.stop()

# ---------------------------------------------------------
# 2. User Interface
# ---------------------------------------------------------
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Using a Gated Recurrent Unit (GRU) Neural Network to predict if a review is Positive or Negative.")

# Text input for user review
user_review = st.text_area("Paste your movie review here:", height=150, placeholder="Type something like: 'The cinematography was brilliant and the acting was top notch!'")

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter some text before clicking analyze.")
    else:
        with st.spinner("Analyzing text..."):
            # 1. Tokenize the input text
            sequences = tokenizer.texts_to_sequences([user_review])
            
            # 2. Pad the sequences (adjust maxlen to match how you trained your GRU model, usually 100-250)
            MAX_LEN = 200 
            padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
            
            # 3. Predict sentiment
            prediction = model.predict(padded_sequences)[0][0]
            
            st.write("---")
            # 4. Display Results
            if prediction >= 0.5:
                st.success(f"### 🤩 Positive Sentiment (Score: {prediction:.2%})")
                st.balloons()
            else:
                st.error(f"### 😔 Negative Sentiment (Score: {prediction:.2%})")