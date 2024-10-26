import streamlit as st
import openai
import os
import numpy as np
from docx import Document
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
#openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_key = st.secrets["open_ai_key"]

# Function to load resources
def load_resources():
    resources = {}
    for root, dirs, files in os.walk('Sample Training Documents'):
        for file in files:
            if file.endswith('.docx'):
                doc = Document(os.path.join(root, file))
                description = ''
                referral_links = []
                link_flag = False
                for para in doc.paragraphs:
                    if para.text.strip():
                        description += para.text + '\n'
                        if "http" in para.text or "https://" in para.text:
                            referral_links.append(para.text)
                            link_flag = True
                if not link_flag:
                    referral_links.append(f"https://vbnreddy/resources/{file}")
                resources[file] = {
                    "description": description.strip(),
                    "links": referral_links
                }
    return resources

# Load resources once at the start
resources = load_resources()

# Function to embed resources
def embed_resources(resources):
    descriptions = [resource['description'] for resource in resources.values()]
    embeddings_response = openai.Embedding.create(
        input=descriptions,
        model="text-embedding-ada-002"
    )
    return [embedding['embedding'] for embedding in embeddings_response['data']]

# Create embeddings
resource_embeddings = embed_resources(resources)

# Function to analyze user input
def analyze_issue(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}]
    )
    return response['choices'][0]['message']['content']

# Function to embed user input
def embed_user_input(user_input):
    response = openai.Embedding.create(
        input=user_input,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

# Function to get similar resources
def get_similar_resources(user_input, resource_embeddings, resources):
    user_embedding = embed_user_input(user_input)
    user_embedding = np.array(user_embedding).reshape(1, -1)
    similarities = cosine_similarity(user_embedding, resource_embeddings)
    scored_resources = [(file_name, embedding, resources[file_name]['links']) for file_name, embedding in zip(resources.keys(), similarities.flatten())]
    ranked_resources = sorted(scored_resources, key=lambda x: x[1], reverse=True)
    return ranked_resources[:5]

# Streamlit UI
st.title("Conversational AI Chat-2")
user_input = st.text_input("Please describe your issue:")

if st.button("Submit"):
    if user_input:
        analysis = analyze_issue(user_input)
        suggestions = get_similar_resources(user_input, resource_embeddings, resources)
        
        st.subheader("Analysis:")
        st.write(analysis)
        
        st.subheader("Suggested Resources:")
        if suggestions:
            suggested_file = suggestions[0][0]  # Get the file name of the top suggestion
            referral_link = suggestions[0][2]    # Get the links of the top suggestion
        else:
            suggested_file = None
            referral_link = None
        st.write(f"File: {suggested_file}")
        st.write(f"Links: {referral_link}")
        # for suggestion in suggestions:
        #     # st.write(f"File: {suggestion[0]}")
        #     # st.write("Links:")
        #     # for link in suggestion[2]:
        #     #     st.write(link)
        #     st.write(f"File: {suggestion[0][0]}")
        #     st.write("Links:")
        #     for link in suggestion[0][2]:
        #         st.write(link)
    else:
        st.warning("Please provide your input.")
