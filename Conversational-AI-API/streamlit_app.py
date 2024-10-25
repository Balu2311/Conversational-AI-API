import streamlit as st
import openai
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import constant

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key from environment variable
openai.api_key = constant.open_ai_key
if not openai.api_key:
    logging.error("OpenAI API key not set in environment variables.")
    st.error("OpenAI API key not set.")

def load_model(filename='model.pkl'):
    try:
        model = joblib.load(filename)
        return model['embeddings'], model['texts'], model['filenames']
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return [], [], []

def generate_query_embedding(query):
    response = openai.Embedding.create(
        model="text-embedding-ada-002", 
        input=query
    )
    return response['data'][0]['embedding']

def find_most_similar(query_embedding, stored_embeddings):
    similarities = cosine_similarity([query_embedding], stored_embeddings)
    return np.argmax(similarities), np.max(similarities)

def retrieve_relevant_document(query, embeddings, texts, filenames):
    query_embedding = generate_query_embedding(query)
    index, similarity = find_most_similar(query_embedding, embeddings)
    return texts[index], filenames[index], similarity

# Streamlit app layout
st.title("Document Query System")

# User input for the query
user_query = st.text_input("Enter your query:")

if st.button("Submit"):
    if not user_query or not isinstance(user_query, str) or len(user_query.strip()) == 0:
        st.error("Query must be a non-empty string")
    else:
        embeddings, texts, filenames = load_model('model.pkl')
        if embeddings:
            relevant_section, filename, similarity = retrieve_relevant_document(user_query, embeddings, texts, filenames)
            st.write("### Relevant Document Section:")
            st.write(relevant_section)
            st.write("### Document Filename:")
            st.write(filename)
            st.write("### Similarity Score:")
            st.write(similarity)
        else:
            st.error("Failed to load the model.")

