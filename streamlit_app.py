import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import os
import time
# from dotenv import load_dotenv

# load_dotenv()

import requests


headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post("https://api-inference.huggingface.co/models/deepset/roberta-base-squad2", headers=headers, json=payload)
	return response.json()
	

st.title('PDF Qns & Ans Chatbot')

st.write('This is a simple chatbot that can answer questions about the pdf uploaded by the user')

search_method=st.radio("Choose the search method",["Entire Document","Related Chunks"])

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Chunking text
def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return chunks

def find_related_chunks(prompt, chunks, threshold=0.1):
    # Create embeddings using CountVectorizer
    vectorizer = CountVectorizer().fit_transform([prompt] + chunks)
    
    # Compute cosine similarity between the prompt and each chunk
    similarity_matrix = cosine_similarity(vectorizer[0:1], vectorizer[1:])
    
    # Get chunks with similarity greater than the threshold
    related_chunks = [chunks[i] for i in range(len(chunks)) if similarity_matrix[0][i] >= threshold]
    
    return related_chunks

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)




if uploaded_file is not None:
    text=read_pdf(uploaded_file)
    chunks = chunk_text(text)
    
    # now clear the previous title and file uploader
    st.empty()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if search_method=="Entire Document":
            context=text
        else:
            related_chunks = find_related_chunks(prompt, chunks)
            # Combine related chunks into one context for the model
            context = " ".join(related_chunks)
        with st.chat_message("assistant"):
            output = query({
            "inputs": {
            "question": prompt,
            "context":text}})
            response = st.write_stream(response_generator(output['answer']))
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
 
else:
    st.write('Please upload a file')