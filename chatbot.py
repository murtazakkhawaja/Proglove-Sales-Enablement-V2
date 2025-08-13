import logging
import os
import json
from datetime import datetime
import streamlit as st
from typing import Dict
from openai import OpenAI
from database import EmbeddingDatabase
from utils import get_openai_embedding
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build


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

        
        # --- GOOGLE SHEETS SETUP ---
        service_account_info = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_INFO"])
        
        SPREADSHEET_ID = st.secrets["GSHEET_SPREADSHEET_ID"]


        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = service_account.Credentials.from_service_account_info(
           service_account_info, scopes=scopes
        )
        self.sheets_service = build('sheets', 'v4', credentials=creds)
        self.sheet_id = SPREADSHEET_ID
        self.sheet_name = "Logs"  # Sheet tab name to append logs

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
        # --- extraction prompt ---
        extraction_prompt = (
            "You are an information extraction assistant for the company ProGlove.\n\n"
            "From the provided context, extract ALL facts that could plausibly help answer the user's question, "
            "even if some details are unclear. OUTPUT VALID JSON ONLY in the format below.\n\n"
            "JSON schema (exact keys):\n"
            "{\n"
            '  "proglove_facts": [ { "text": "...", "source": "PDF name (pN)" }, ... ],\n'
            '  "other_companies": [ { "name": "...", "role": "customer|partner|collaborator|case study subject|relationship unknown", "source": "PDF name (pN)" }, ... ],\n'
            '  "people": [ { "name": "...", "role": "role specified|role unknown", "company": "company specified|company unknown", "source": "PDF name (pN)" }, ... ]\n'
            "}\n\n"
            "Rules:\n"
            "- Always include any fact that may be relevant — even if incomplete.\n"
            "- If the role or company is unclear, use 'role unknown' or 'company unknown' instead of leaving it out.\n"
            "- Keep each fact separate — do not merge multiple into one object.\n"
            "- Always include the exact PDF filename and page number in the 'source'.\n"
            "- Do not guess beyond the provided context.\n"
            "- If truly nothing in the context relates to the user's question, return exactly:\n"
            "{\n"
            '  "proglove_facts": [],\n'
            '  "other_companies": [],\n'
            '  "people": []\n'
            "}\n"
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
        # --- final prompt ---
        final_prompt = (
            "You are a helpful assistant for the company ProGlove.\n"
            "You will be given a JSON object (from the extractor) with three sections: proglove_facts, other_companies, and people.\n\n"
            "Write a clear, concise, natural-sounding answer to the user's question using ONLY the facts in the JSON.\n"
            "Do not show or mention the JSON itself.\n\n"
            "Rules:\n"
            "- Address the user's question directly, using any relevant facts from the JSON.\n"
            "- Prioritize ProGlove facts first.\n"
            "- Mention other companies only if relevant to the question, stating their role exactly as given (even if 'relationship unknown').\n"
            "- Mention people only if relevant, including role/company if given; otherwise say 'role not specified' or 'company not specified'.\n"
            "- Include the PDF filename and page number in parentheses after each fact.\n"
            "- If all three sections are empty, respond exactly:\n"
            "  'I couldn't find that in the documents.'\n"
            "- Write in a friendly, human tone — avoid robotic phrasing.\n"
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

        # Remove references if fallback answer is given
        if final_answer == "Sorry I don't have an answer to that right now, my memory and learning capabilities are limited yet.":
            pdf_sources = []

        return {
            "answer": final_answer,
            "sources": pdf_sources
        }


    def _log_interaction(self, query: str, context: str, response: str):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f"chat_{timestamp}.log"
        log_path = os.path.join(self.log_dir, log_filename)

        # --- Local log file ---
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"QUESTION:\n{query}\n\n")
                f.write(f"RETRIEVED CONTEXT:\n{context}\n\n")
                f.write(f"RESPONSE:\n{response}\n")
            logging.info(f"Chat interaction logged to {log_path}")
        except Exception as e:
            logging.error(f"Failed to write chat log: {e}")

        # --- Google Sheets log ---
        try:
            values = [[timestamp, query, response]]
            body = {'values': values}
            request = self.sheets_service.spreadsheets().values().append(
                spreadsheetId=self.sheet_id,
                range=f"{self.sheet_name}!A:C",
                valueInputOption="USER_ENTERED",
                insertDataOption="INSERT_ROWS",
                body=body
            )

            # Execute request and capture raw API response
            api_response = request.execute()

            # Print to Streamlit logs 
            print("Google Sheets API Response:", json.dumps(api_response, indent=2))

            # Also log to the file
            logging.info(f"Google Sheets API Response: {json.dumps(api_response, indent=2)}")

            logging.info("Logged interaction to Google Sheets")

        except Exception as e:
            logging.error(f"Failed to log interaction to Google Sheets: {e}")
            print(f"Google Sheets logging error: {e}")

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
