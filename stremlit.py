import streamlit as st
import requests

st.title("Streamlit and Flask Integration")

if st.button("Fetch Data"):
    response = requests.get('http://localhost:5000/api/data')
    data = response.json()
    st.write(data)