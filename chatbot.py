import logging
import os
from datetime import datetime
from typing import Dict
from openai import OpenAI
from database import EmbeddingDatabase
from utils import get_openai_embedding
from dotenv import load_dotenv

load_dotenv()

# Setup logger
logging.basicConfig(
    filename="chatbot_log.log",
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CompanyChatbot:
    def __init__(self, db_path="embeddings_db", model="gpt-3.5-turbo", top_k=6):
        self.db = EmbeddingDatabase(db_path)
        self.model = model
        self.top_k = top_k
        self.log_dir = "chat_logs"
        os.makedirs(self.log_dir, exist_ok=True)

    def ask_question(self, user_query: str) -> Dict:
        return self._ask(user_query)

    def _ask(self, user_query: str) -> Dict:
        # Step 0: Create embedding for query
        try:
            query_embedding = get_openai_embedding(user_query)
        except Exception as e:
            return {"answer": f"Error embedding question: {e}", "sources": []}

        # Step 1: Search for relevant chunks
        chunks = self.db.search_similar_chunks(query_embedding, top_k=self.top_k, threshold=0.25)
        if not chunks:
            return {
                "answer": "Sorry, I couldn't find any relevant information in the documents.",
                "sources": []
            }

        pdf_sources = list({f"{c['metadata']['pdf_name']} (page {c['metadata'].get('page_num', '?')})" for c in chunks})

        # Step 2: Format retrieved context
        context_text = "[START DOCUMENT EXCERPTS]\n"
        for i, c in enumerate(chunks):
            pdf = c['metadata']['pdf_name']
            page = c['metadata'].get('page_num', 'Unknown')
            context_text += f"[Chunk {i+1} — {pdf}, page {page}]\n{c['text']}\n\n"
        context_text += "[END DOCUMENT EXCERPTS]"

        # Step 3: Extract ONLY relevant facts
        extraction_prompt = (
            "You are an information extraction assistant. "
            "From the provided context, extract ONLY the facts directly relevant to the user's question. "
            "Do not guess or use outside knowledge. "
            "List facts in bullet points, each with its PDF name and page number."
        )
        try:
            extraction_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{user_query}"}
                ],
                temperature=0
            )
            extracted_facts = extraction_response.choices[0].message.content.strip()
        except Exception as e:
            extracted_facts = f"Error extracting facts: {e}"

        # Step 4: Rewrite into natural, user-friendly answer
        final_prompt = (
            "You are a helpful company assistant. "
            "Using ONLY the extracted facts below, write a concise, natural-sounding answer. "
            "If no relevant facts exist, respond with: 'I couldn't find that in the documents.' "
            "Always include PDF name(s) and page number(s) in parentheses for each fact."
        )
        try:
            final_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": final_prompt},
                    {"role": "user", "content": f"Extracted facts:\n{extracted_facts}"}
                ],
                temperature=0.3
            )
            final_answer = final_response.choices[0].message.content.strip()
        except Exception as e:
            final_answer = f"OpenAI API error: {e}"

        # Step 5: Log interaction
        self._log_interaction(user_query, context_text, final_answer)

        return {
            "answer": final_answer,
            "sources": pdf_sources
        }

    def _log_interaction(self, query: str, context: str, response: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"chat_{timestamp}.log"
        log_path = os.path.join(self.log_dir, log_filename)

        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"QUESTION:\n{query}\n\n")
                f.write(f"RETRIEVED CONTEXT:\n{context}\n\n")
                f.write(f"RESPONSE:\n{response}\n")
            logging.info(f"Chat interaction logged to {log_path}")
        except Exception as e:
            logging.error(f"Failed to write chat log: {e}")

    def get_database_info(self) -> Dict:
        pdf_names = self.db.get_all_pdf_names()
        return {
            "total_pdfs": len(pdf_names),
            "total_chunks": self.db.get_total_chunks(),
            "pdf_names": pdf_names
        }

    def clear_database(self):
        self.db.clear()

    def clear_conversation_history(self):
        pass
