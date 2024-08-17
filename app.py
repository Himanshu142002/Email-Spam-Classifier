import pickle

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

#load 
tfidf=pickle.load(open('vectorizer.pkl','rb'))
model=pickle.load(open('model.pkl','rb'))

import string

#preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

ps=PorterStemmer()

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    text=[word for word in text if word.isalnum()]
    
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    text=[ps.stem(word) for word in text]
    return " ".join(text)

st.title("Email Spam Classifier ")
input_sms=st.text_area("Enter your message")

if st.button('Predict'):
    transformed_sms=transform_text(input_sms)
    
    vector_input=tfidf.transform([transformed_sms])
    
    result=model.predict(vector_input)[0]
    
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")  # To add more functionality, you can add more buttons and