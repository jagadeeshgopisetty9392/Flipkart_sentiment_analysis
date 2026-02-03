import streamlit as st
import pickle
import re
import nltk

# auto download for cloud/server
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load model + vectorizer

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()


# Text cleaning function

def clean_text(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemma.lemmatize(w) for w in words if w not in stop]
    return " ".join(words)



# Streamlit UI

st.title("🛒 Flipkart Review Sentiment Analyzer")

review = st.text_area("Enter your review here")

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = clean_text(review)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😡 Negative Review")
