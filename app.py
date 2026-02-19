import streamlit as st
import pickle
import nltk
from preprocessing import clean

# ---------- NLTK Setup ----------
def download_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

download_nltk()

# ---------- Load Model ----------
with open('Sentiment_model.pkl','rb') as file:
    model = pickle.load(file)

# ---------- Load Vectorizer ----------
with open('vectorizer.pkl', 'rb') as f:
    vect = pickle.load(f)

# ---------- UI ----------
st.title('Review Analysis')
st.write('Enter the review below to check whether its a **Positive** or **Negative**.')

text = st.text_area('Review', height=150)

if st.button('Predict'):
    if text.strip() == "":
        st.warning('Please enter some text')
    else:
        cleaned = clean(text)
        sample_vectorized = vect.transform([cleaned])
        prediction = model.predict(sample_vectorized)[0]

        if prediction == 1:
            st.success('Its a positive review 😃')
        else:
            st.error('Its a negative review 😑')
