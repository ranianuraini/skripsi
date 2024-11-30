import streamlit as st
from joblib import load
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data if not already downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('indonesian'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text
def preprocess_text(text):
    # Cleaning: Remove non-alphabet characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Case folding: Convert all letters to lowercase
    text = text.lower()
    
    # Tokenizing: Split text into words
    tokens = word_tokenize(text)
    
    # Filtering: Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization: Convert words to their base form
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return ' '.join(lemmatized_tokens)


# Define Streamlit layout
st.title("Aspect-Based Sentiment Analysis for Bromo Reviews")
st.write("Enter your review to analyze sentiment for each aspect (attraction, amenity, accessibility, image, price, and human resource).")

# Input for user text
input_text = st.text_area("Enter review text:", "")

if st.button("Analyze"):
    # Only proceed if input_text is not empty
    if input_text.strip() != "":
        # Preprocess input text
        processed_text = preprocess_text(input_text)
        
        # Load the TF-IDF vectorizer and transform the input text
        tfidf_vectorizer = load('model_tfidf/tfidf-model.joblib')
        tfidf_features = tfidf_vectorizer.transform([processed_text]).toarray()
        
        # List of aspects and sentiment labels
        list_aspek = ['attraction', 'amenity', 'accessibility', 'image', 'price', 'human resource']
        aspek_pred = {}
        label_sentimen = {-1: 'Negatif', 1: 'Positif'}
        
        # Aspect-based sentiment prediction
        for aspek in list_aspek:
            svm_model = load(f"model_tfidf/tfidf-svm-aspek-{aspek}.joblib")
            pred = svm_model.predict(tfidf_features)
            
            if pred[0] == 1:
                model_sentimen = load(f'model_tfidf/tfidf-svm-sentimen-{aspek}.joblib')
                pred_sentimen = model_sentimen.predict(tfidf_features)
                aspek_pred[aspek] = label_sentimen[pred_sentimen[0]]
        
        # Display the results
        if aspek_pred:
            st.write("Sentiment Analysis Results:")
            for aspek, sentiment in aspek_pred.items():
                st.write(f"- {aspek.capitalize()}: {sentiment}")
        else:
            st.write("No relevant aspects detected in the review.")
    else:
        st.warning("Please enter a review before analyzing.")
