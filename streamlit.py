import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the TfidfVectorizer used during training
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
   tfidf_vectorizer = pickle.load(vectorizer_file)

def predict(answer):
    # Transform the input message using the pre-trained TfidfVectorizer
    input_features =  tfidf_vectorizer.transform([answer])

    # Make a prediction using the pre-trained model
    prediction = model.predict(input_features)

    # Convert the numeric prediction to a human-readable label
    result = 'Spam' if prediction[0] == 0 else 'Ham'

    return result

def main():
    st.title('Spam Classifier App')

    answer = st.text_input('Enter your message:')
    if st.button('Predict'):
        result = predict(answer)
        st.write(f'The message is a {result}')

if __name__ == '__main__':
    main()
