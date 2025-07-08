import os
import openai
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

#logging for debugging
import logging
logging.basicConfig(level=logging.INFO)

# Split text into chunks
def split_text(text, max_tokens=500):
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk + para) < max_tokens:
            current_chunk += para + "\n\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Main embed and insert function
def embed_and_store(file_path, page_name):
        # Skip unsupported file types
    if not file_path.endswith((".txt", ".md", ".pdf")):
        print(f"Skipping unsupported file type: {file_path}")
        return
    try:
        # Read text from PDF using PyPDF2
        if file_path.endswith(".pdf"):
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                # Extract text from all pages that contain text
                text = "\n".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
        else:
            # Read text from .txt or .md files
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
    except Exception as e:
        # Handle file reading errors
        print(f"Error reading {file_path}: {e}")
        return
    # Split the full text into smaller chunks
    chunks = split_text(text)
    # Embed each chunk and store in Supabase
    for chunk in chunks:
        if chunk.strip():
            # Get embedding from OpenAI
            response = openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunk
            )
            vector = response.data[0].embedding
            # Insert the chunk and its embedding into the Supabase table
            supabase.table("juno_embeddings").insert({
                "page": page_name,
                "chunk": chunk,
                "embedding": vector
            }).execute()
            logging.info(f"Inserted chunk to Supabase from page: {page_name}")

#Searches all notion pages for files, embeds them, and stores in Supabase
from notion_util import get_page_content
import re

def batch_embed_all_notion_files():
    # Pull a list of all pages using a Notion search
    from notion_client import Client
    logging.info("Connecting to Notion...")

    notion = Client(auth=os.getenv("NOTION_TOKEN"))
    search_results = notion.search()["results"]
    logging.info(f"Found {len(search_results)} pages from Notion search")

    for page in search_results:
        # Log entire page ID/title section for debugging
        logging.info(f"Raw page properties: {page.get('properties', {})}")
        
        title_props = page.get("properties", {}).get("title", {}).get("title", [])
        if not title_props:
            continue
        title = title_props[0]["plain_text"]
        logging.info(f"Found page: {title}")

        content = get_page_content(title)
        file_paths = re.findall(r"\[FILE_DOWNLOADED\]:(.+)", content)
        for file_path in file_paths:
            logging.info(f"Embedding file: {file_path}")
            embed_and_store(file_path.strip(), title)

if __name__ == "__main__":
    batch_embed_all_notion_files()