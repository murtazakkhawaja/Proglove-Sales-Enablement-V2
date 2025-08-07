import os
import PyPDF2
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv

# Load .env before using the API key
load_dotenv()

# Initialize OpenAI client
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file_path):
    """
    Extract text from a PDF and return as list of (page_num, text).
    Also saves extracted text to 'extracted_text' folder.
    """
    pages_text = []
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    pages_text.append((page_num, page_text.strip()))

        # Save extracted text to TXT for reference
        pdf_name = os.path.basename(file_path).replace(".pdf", "")
        os.makedirs("extracted_text", exist_ok=True)
        txt_path = os.path.join("extracted_text", f"{pdf_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            for p_num, p_text in pages_text:
                f.write(f"[Page {p_num}]\n{p_text}\n\n")

        return pages_text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return []

def chunk_text(text, max_tokens=500):
    """
    Splits text into smaller chunks based on token count.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())

    return chunks

def get_openai_embedding(text):
    """
    Generates an embedding vector for given text.
    """
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding
