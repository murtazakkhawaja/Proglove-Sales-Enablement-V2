import os
import sys
import json
from utils import extract_text_from_pdf, chunk_text, get_openai_embedding
from database import EmbeddingDatabase
import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setting up logger again
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  
        logging.FileHandler("app.log", mode="a", encoding="utf-8") 
    ]
)
logger = logging.getLogger(__name__)


def process_pdf(file_path, save_json=True, db_path="embeddings_db"):
    """Process a PDF and store embeddings in database"""
    pdf_name = os.path.basename(file_path).replace(".pdf", "")
    logger.info(f"\n--- Processing PDF: {pdf_name} ---")

    # Extract text
    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        logger.error(f"No text extracted from {file_path}")
        return None
    logger.info(f"Text extracted from {pdf_name} — {len(raw_text)} characters")
    logger.info(f"Preview of extracted text:\n{raw_text[:500]}")

    # Chunk text
    chunks = chunk_text(raw_text)
    logger.info(f"Split into {len(chunks)} chunks (max 200 words each)")

    embeddings = []
    for i, chunk in enumerate(chunks):
        try:
            emb = get_openai_embedding(chunk)
            embeddings.append({
                "text": chunk,
                "embedding": emb
            })
            logger.info(f"Embedded chunk {i+1}/{len(chunks)} — {len(chunk.split())} words")
        except Exception as e:
            logger.error(f"Error embedding chunk {i+1}: {e}")
            continue

    logger.info(f"Total embeddings generated: {len(embeddings)}")

    if not embeddings:
        logger.warning(f"No valid embeddings generated for {pdf_name}")
        return None

    # Saving to database
    db = EmbeddingDatabase(db_path)
    db.add_pdf_embeddings(pdf_name, embeddings)
    logger.info(f"Stored embeddings in local DB: {db_path}")

    # Json save just for my understanding
    if save_json:
        os.makedirs("embeddings_json", exist_ok=True)
        json_path = os.path.join("embeddings_json", f"{pdf_name}_embeddings.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved embeddings to json: {json_path}")

    logger.info(f"Finished processing {pdf_name}")
    return db.get_database_stats()

def clear_all_embeddings(db_path="embeddings_db"):
    """Clear all stored embeddings"""
    db = EmbeddingDatabase(db_path)
    db.clear_database()
    logger.info("All embeddings cleared from database")
    return "All embeddings cleared successfully"

def get_database_stats(db_path="embeddings_db"):
    """Get database statistics"""
    db = EmbeddingDatabase(db_path)
    stats = db.get_database_stats()
    logger.info(f"Database stats: {stats}")
    return stats
