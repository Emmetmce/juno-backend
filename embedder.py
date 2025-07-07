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

# Make sure the juno_embeddings table exists

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
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = split_text(text)
    for chunk in chunks:
        print(f"Embedding chunk: {chunk[:40]}...")
        response = openai.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        vector = response.data[0].embedding

        supabase.table("juno_embeddings").insert({
            "page_name": page_name,
            "chunk": chunk,
            "embedding": vector,
            "metadata": {}
        }).execute()

    print(f"Uploaded {len(chunks)} chunks from '{file_path}'.")

#usage
embed_and_store("junk_kouture_privacy_policy.txt", "Privacy Policy")
embed_and_store("03_Competition_Handbook_2024-25.txt", "Competition Handbook")