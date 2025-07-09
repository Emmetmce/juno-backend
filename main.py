from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import openai
import numpy as np
from supabase.client import create_client, Client
from notion_util import get_page_content  
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

#logging for debugging
import logging
logging.basicConfig(level=logging.INFO)
app = FastAPI()

#allowing CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # can add ["https://chat.openai.com"] for future security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.environ.get("OPENAI_API_KEY")
def embed_text(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

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
async def query_knowledge_base(query: QueryRequest):
    logging.info("/query route hit")
    try:
        logging.info(f"Received query: {query.query}")
        embedded_query = embed_text(query.query)
        logging.info("Query embedded successfully")

        response = supabase.rpc("match_juno_embeddings", {
            "query_embedding": embedded_query,
            "match_threshold": 0.8,
            "match_count": 5
        }).execute()

        if response.error:
            logging.error(f"Supabase RPC error: {response.error}")
            raise HTTPException(status_code=500, detail=response.error.message)

        logging.info(f"Supabase returned {len(response.data)} matches")
        return {"results": response.data}

    except Exception as e:
        logging.exception("Exception in /query")
        raise HTTPException(status_code=500, detail=str(e))