import logging
import os
import json
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
    def __init__(self, db_path="embeddings_db", model="gpt-4-turbo", top_k=6):
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
        chunks = self.db.search_similar_chunks(
            query_embedding, top_k=self.top_k, threshold=0.25
        )
        if not chunks:
            return {
                "answer": "Sorry, I couldn't find any relevant information in the documents.",
                "sources": []
            }

        pdf_sources = list({
            f"{c['metadata']['pdf_name']} (page {c['metadata'].get('page_num', '?')})"
            for c in chunks
        })

        # Step 2: Build retrieval context
        context_text = "[START DOCUMENT EXCERPTS]\n"
        for i, c in enumerate(chunks):
            pdf = c['metadata']['pdf_name']
            page = c['metadata'].get('page_num', 'Unknown')
            context_text += f"[Chunk {i+1} — {pdf}, page {page}]\n{c['text']}\n\n"
        context_text += "[END DOCUMENT EXCERPTS]"

        # Step 3: JSON Extraction Prompt
        extraction_prompt = (
            "You are an information extraction assistant for the company ProGlove.\n\n"
            "From the provided context, EXTRACT information and OUTPUT VALID JSON ONLY. "
            "The JSON MUST EXACTLY follow the schema below and must be the only content in the response.\n\n"
            "JSON schema (exact keys):\n"
            "{\n"
            '  "proglove_facts": [ { "text": "...", "source": "PDF name (pN)" }, ... ],\n'
            '  "other_companies": [ { "name": "...", "role": "customer|partner|collaborator|case study subject|relationship unknown", "source": "PDF name (pN)" }, ... ],\n'
            '  "people": [ { "name": "...", "role": "...|role unknown", "company": "...|company unknown", "source": "PDF name (pN)" }, ... ]\n'
            "}\n\n"
            "Rules:\n"
            "- Output valid JSON only — no commentary.\n"
            "- Always extract all facts directly stated in the context. Do not skip partial or indirect mentions.\n"
            "- Always include the exact PDF filename and page number in the 'source' field.\n"
            "- If a company's role is not clear, use 'relationship unknown'.\n"
            "- If a person’s role or company is not stated, use 'role unknown' / 'company unknown'.\n"
            "- Always list companies and people even if ProGlove is not mentioned in the same sentence.\n"
            "- Never merge separate facts; keep each fact as a separate object.\n"
            "- Do not infer beyond what is explicitly stated, but do capture all relevant entities mentioned.\n"
        )


        try:
            extraction_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": extraction_prompt},
                    {
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion:\n{user_query}"
                    }
                ],
                temperature=0
            )
            extracted_facts_raw = extraction_response.choices[0].message.content.strip()

            # Ensure valid JSON
            try:
                extracted_facts = json.loads(extracted_facts_raw)
            except json.JSONDecodeError:
                return {
                    "answer": "Extraction step failed: invalid JSON format.",
                    "sources": pdf_sources
                }

        except Exception as e:
            return {"answer": f"Error extracting facts: {e}", "sources": pdf_sources}

        # Step 4: Final user-facing rewrite
        final_prompt = (
            "You are a helpful assistant for the company ProGlove. "
            "You will be given a JSON object (from the extractor) with three sections: proglove_facts, other_companies, and people. "
            "DO NOT output or repeat the JSON. Use the JSON only as your source of truth to write a single concise, natural-language answer to the user's question.\n\n"
            "Rules:\n"
            "- Always address every part of the user's question explicitly.\n"
            "- While answering the user, Focus on the most relevant facts to answer the question.\n"
            "- Focus on ProGlove first. If other companies are relevant, state their relationship using exactly the value from 'role'.\n"
            "- When mentioning people, include their role and company from the JSON; if unknown, say 'role not specified' or 'company not specified'.\n"
            "- If a company is not the manufacturer, clearly say 'No, [Company] is not the manufacturer; they are a [role].'\n"
            "- Always include the PDF filename and page number in parentheses after each fact.\n"
            "- Be concise but cover all unique facts; avoid repeating the same fact in different sentences.\n"
            "- If the JSON is completely empty, say 'Sorry I dont have an answer to that rightnow, my memory and learning capabilities are limited yet.' \n"
        )


        try:
            final_response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": final_prompt},
                    {"role": "user", "content": json.dumps(extracted_facts)}
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
