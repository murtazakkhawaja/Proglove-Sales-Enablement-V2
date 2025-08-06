import os
import sys
import json
import shutil
import logging
from utils import extract_text_from_pdf, chunk_text, get_openai_embedding
from database import EmbeddingDatabase

# Clear old log handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up logger
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
    """Process a single PDF and store its embeddings"""
    pdf_name = os.path.basename(file_path).replace(".pdf", "")
    logger.info(f"\n--- Processing PDF: {pdf_name} ---")

    raw_text = extract_text_from_pdf(file_path)
    if not raw_text:
        logger.error(f"No text extracted from {file_path}")
        return None
    logger.info(f"Text extracted from {pdf_name} — {len(raw_text)} characters")
    logger.info(f"Preview of extracted text:\n{raw_text[:500]}")

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

    db = EmbeddingDatabase(db_path)
    db.add_pdf_embeddings(pdf_name, embeddings)
    logger.info(f"Stored embeddings in local DB: {db_path}")

    if save_json:
        os.makedirs("embeddings_json", exist_ok=True)
        json_path = os.path.join("embeddings_json", f"{pdf_name}_embeddings.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(embeddings, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved embeddings to JSON: {json_path}")

    logger.info(f"Finished processing {pdf_name}")
    return db.get_database_stats()


def clear_all_embeddings(db_path="embeddings_db"):
    """Clear the embedding database and related folders"""
    db = EmbeddingDatabase(db_path)
    db.clear_database()
    logger.info("All embeddings cleared from database")

    # Delete entire folders recursively
    folders_to_clear = ["embeddings_json", "chroma_db"]
    for folder in folders_to_clear:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                logger.info(f"Deleted folder and all contents: {folder}")
            except Exception as e:
                logger.warning(f"Could not delete {folder}: {e}")
        os.makedirs(folder, exist_ok=True)
    return "All embeddings cleared successfully"


def get_database_stats(db_path="embeddings_db"):
    """Get current database stats"""
    db = EmbeddingDatabase(db_path)
    stats = db.get_database_stats()
    logger.info(f"Database stats: {stats}")
    return stats


if __name__ == "__main__":
    # Step 1: Clear previous embeddings
    clear_all_embeddings()

    # Step 2: Process all PDFs in data/
    data_folder = "data"
    if not os.path.exists(data_folder):
        logger.error(f"Folder {data_folder} not found.")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".pdf")]

    if not pdf_files:
        logger.warning("No PDF files found in the data/ folder.")
    else:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(data_folder, pdf_file)
            process_pdf(pdf_path)

    logger.info("Embedding generation complete.")