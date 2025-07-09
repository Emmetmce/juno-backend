from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import openai
import numpy as np
from supabase.client import create_client, Client
from notion_util import get_page_content  
from fastapi.responses import FileResponse

#logging for debugging
import logging
logging.basicConfig(level=logging.INFO)

openai.api_key = os.environ.get("OPENAI_API_KEY")
def embed_text(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding
app = FastAPI()

@app.get("/ping")
async def ping():
    return {"status": "ok"}

#env setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Auth
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai.api_key = OPENAI_API_KEY
#serve static files
@app.get("/openapi.yaml")
def get_openapi_yaml():
    return FileResponse("openapi.yaml", media_type="text/yaml")

# Test site alive
@app.get("/")
def root():
    return {"status": "Juno backend is live!"}

# Notion endpoint
@app.get("/get-content")
def get_content(page_name: str = Query(..., description="Title of Notion page")):
    try:
        content = get_page_content(page_name)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query embedding class
class QueryRequest(BaseModel):
    query: str

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

@app.post("/query")
async def query_knowledge_base(query: dict):
    logging.info("🚀 /query route hit")
    try:
        print("✅ ROUTE HIT SUCCESSFULLY")
        return {"results": ["This is a test"]}
    except Exception as e:
        print("❌ ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))