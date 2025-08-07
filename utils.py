import os
import PyPDF2
import openai
from dotenv import load_dotenv  

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyPDF2 and save it to a .txt file"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        # Save extracted text to txt file
        pdf_name = os.path.basename(file_path).replace(".pdf", "")
        os.makedirs("extracted_text", exist_ok=True)
        txt_path = os.path.join("extracted_text", f"{pdf_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text.strip())

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
