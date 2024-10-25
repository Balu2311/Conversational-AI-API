import streamlit as st
import requests

st.title("Streamlit and Flask Integration")
if st.button("Fetch Data"):
    response = requests.get('http://127.0.0.1:5001')
    data = response.json()
    st.write(data)
