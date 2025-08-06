import logging
import os
from datetime import datetime
from typing import List, Dict
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

    def ask_question(self, user_query: str) -> str:
        return self._ask(user_query)

    def _ask(self, user_query: str) -> str:
        try:
            query_embedding = get_openai_embedding(user_query)
        except Exception as e:
            return f"Error embedding question: {e}"

        chunks = self.db.search_similar_chunks(query_embedding, top_k=self.top_k, threshold=0.25)
        if not chunks:
            return "Sorry, I couldn't find any relevant information in the documents."
        if not chunks:
            return "Sorry, I couldn't find any relevant information in the documents.\n\n(Debug: No chunks above similarity threshold.)"

        # TEMP DEBUG: return top chunks to verify what was retrieved
        debug_chunk_text = "\n\n".join([f"[{c['similarity']:.2f}] {c['text']}" for c in chunks])
        # return f" Found top chunks:\n\n{debug_chunk_text}"  # for testing can uncomment

        context_text = "\n\n".join(
            [f"[Chunk {i+1} from {c['metadata']['pdf_name']}]\n{c['text']}" for i, c in enumerate(chunks)]
        )

        system_prompt = (
            "You are a helpful assistant that only answers questions based on the provided context. "
            "Do not use external knowledge. If the answer is not in the context, say 'I couldn't find that in the documents.'"
        )

        user_prompt = (
            f"Context:\n{context_text}\n\n"
            f"Question:\n{user_query}\n\n"
            f"Answer in a professional tone:"
        )

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            final_answer = response.choices[0].message.content.strip()
        except Exception as e:
            final_answer = f"OpenAI API error: {e}"

        self._log_interaction(user_query, context_text, final_answer)
        return final_answer

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
        # for optional future use to Clear in-memory conversation context
        pass
