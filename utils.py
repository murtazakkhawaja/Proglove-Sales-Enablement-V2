import os
import PyPDF2
import openai
from dotenv import load_dotenv  
import streamlit as st
load_dotenv()

openai.api_key = st.secrets["openai_api_key"]

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyPDF2"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def get_openai_embedding(text):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
