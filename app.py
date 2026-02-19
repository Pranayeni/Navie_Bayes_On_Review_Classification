import streamlit as st
import pickle
import nltk
from preprocessing import clean
from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer()

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# load the trained model
with open('sentiment_model.pkl','rb') as file:
    model=pickle.load(file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vect = pickle.load(f)
    

# App title
st.title('Review Analysis')

st.write('Enter the review below to check whether its a **Positive** or **Negative**.')
# text input
text=st.text_area('Review',height=150)

# Now you can transform new text
sample_vectorized = vect.transform([text])



# prediction button
if st.button('Predict'):
    if text.strip()=="":
        st.warning('Please enter some text')
    else:
        prediction=model.predict(sample_vectorized)[0]

        if prediction==1:
            st.success('its a positive review😃')
        else:
            st.error('its a negative review😑')
            
