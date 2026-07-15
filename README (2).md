# 🎬 IMDb Movie Review Sentiment Analysis using GRU

## 📌 Project Overview

This project performs **Sentiment Analysis** on IMDb movie reviews using
a **Gated Recurrent Unit (GRU)** Deep Learning model. The application
predicts whether a movie review expresses a **Positive** or **Negative**
sentiment.

The project covers the complete NLP pipeline, including data
preprocessing, tokenization, sequence padding, model training,
evaluation, model saving, and deployment using Streamlit.

------------------------------------------------------------------------

## 🚀 Features

-   IMDb Movie Review Sentiment Classification
-   Text Cleaning and Preprocessing
-   Tokenization using Keras Tokenizer
-   Sequence Padding
-   Deep Learning Model using GRU
-   Binary Sentiment Prediction
-   Streamlit Web Application
-   Easy-to-use User Interface

------------------------------------------------------------------------

## 🛠️ Technologies Used

-   Python
-   TensorFlow / Keras
-   GRU (Gated Recurrent Unit)
-   NumPy
-   Pandas
-   Matplotlib
-   Scikit-learn
-   NLTK
-   Streamlit

------------------------------------------------------------------------

## 📂 Project Structure

``` text
IMDB-Movie-Review-Sentiment-Analysis-Using-GRU/
│
├── app.py
├── model.keras
├── tokenizer.pkl
├── requirements.txt
├── README.md
├── train.ipynb
├── dataset/
└── images/
```

------------------------------------------------------------------------

## 📊 Dataset

**IMDb Large Movie Review Dataset**

-   50,000 Movie Reviews
-   25,000 Training Reviews
-   25,000 Testing Reviews
-   Binary Classification:
    -   Positive
    -   Negative

------------------------------------------------------------------------

## 🔄 Project Workflow

1.  Import Dataset
2.  Data Cleaning
3.  Text Preprocessing
4.  Tokenization
5.  Sequence Padding
6.  Train-Test Split
7.  Build GRU Model
8.  Model Training
9.  Model Evaluation
10. Save Model
11. Load Saved Model
12. Streamlit Deployment

------------------------------------------------------------------------

## 🧠 GRU Architecture

``` text
Input Review
      │
      ▼
Text Cleaning
      │
      ▼
Tokenizer
      │
      ▼
Padding Sequences
      │
      ▼
Embedding Layer
      │
      ▼
GRU Layer
      │
      ▼
Dropout
      │
      ▼
Dense Layer
      │
      ▼
Sigmoid Activation
      │
      ▼
Positive / Negative
```

------------------------------------------------------------------------

## 📈 Model Performance

  Metric          Score
  --------------- ------------------------------------------
  Accuracy        89%+ *(replace with your actual result)*
  Optimizer       Adam
  Activation      Sigmoid
  Loss Function   Binary Crossentropy

------------------------------------------------------------------------

## 💻 Installation

Clone the repository:

``` bash
git clone https://github.com/naveengangumalla2001/imdb-movie-review-sentiment-analysis-using-gru.git
cd imdb-movie-review-sentiment-analysis-using-gru
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 🖥️ Example Predictions

**Review:** \> This movie was absolutely amazing. The acting was
brilliant.

**Prediction:** Positive 😊

**Review:** \> Worst movie I have ever watched.

**Prediction:** Negative 😞

------------------------------------------------------------------------

## 🎯 Future Improvements

-   Bidirectional GRU
-   Attention Mechanism
-   BERT Integration
-   Hugging Face Deployment
-   Docker Deployment

------------------------------------------------------------------------

## 👨‍💻 Author

**Naveen Kumar Gangumalla**

-   Data Science Enthusiast
-   Machine Learning
-   Deep Learning
-   Natural Language Processing

GitHub: https://github.com/naveengangumalla2001

------------------------------------------------------------------------

## ⭐ Support

If you found this project useful, please give it a **⭐ Star** on
GitHub.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License.
