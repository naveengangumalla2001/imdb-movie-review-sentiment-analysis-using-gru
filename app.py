import re
import pickle
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="IMDB Sentiment Classifier", page_icon="🎬", layout="centered")

MAX_LEN = 200


@st.cache_resource
def load_artifacts():
    model = load_model("gru_sentiment_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_sentiment(review_text):
    cleaned = clean_text(review_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob = model.predict(padded, verbose=0)[0][0]
    label = "Positive" if prob > 0.5 else "Negative"
    return label, float(prob)

# ---------------- Streamlit UI ----------------
st.title("🎬 IMDB Movie Review Sentiment Classifier")
st.write("Powered by a GRU neural network (87% test accuracy)")

review = st.text_area("Enter a movie review:", height=150, placeholder="Type or paste a review here...")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        label, prob = predict_sentiment(review)
        confidence = prob if label == "Positive" else 1 - prob

        if label == "Positive":
            st.success(f"**{label}** (confidence: {confidence:.2%})")
        else:
            st.error(f"**{label}** (confidence: {confidence:.2%})")

        st.progress(float(prob))
        st.caption(f"Raw model output (probability of positive): {prob:.4f}")