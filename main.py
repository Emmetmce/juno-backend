from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import os
import openai
import numpy as np
from supabase.client import create_client, Client
from notion_util import get_page_content  
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json


#logging for debugging
import logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Juno Memory API",
    description="API for querying Junk Kouture knowledge base using vector embeddings",
    version="1.0.0"
)

#allowing CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Allow all, just for now
        "https://chat.openai.com",
        "https://chatgpt.com",
        "https://openai.com"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

#env setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create Supabase client and set OpenAI API key
if not OPENAI_API_KEY:
    logging.info("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY is required")

openai.api_key = OPENAI_API_KEY

# Supabase client
if not SUPABASE_URL or not SUPABASE_KEY:
    logging.info("Supabase credentials are not properly configured")
    raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def embed_text(text: str):
    """Generate embedding for text using OpenAI"""
    try:
        response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logging.info(f"Error generating embedding: {e}")
        raise

# Test site alive
@app.get("/")
def root():
    return {
        "status": "Juno backend is live!",
        "version": "1.0.0",
        "endpoints": {
            "health": "/ping",
            "query": "/query",
            "content": "/get-content",
            "openapi": "/openapi.yaml"
        }
    }

@app.get("/ping")
async def pingHealthCheck():
    try:
        # Test database connection
        result = supabase.table("juno_embeddings").select("id").limit(1).execute()
        db_status = "connected" if result else "disconnected"
        
        return {
            "status": "ok",
            "database": db_status,
            "timestamp": os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost")
        }
    except Exception as e:
        logging.info(f"Health check failed: {e}")
        return {
            "status": "error",
            "database": "disconnected",
            "error": str(e)
        }

#serve static files
@app.get("/openapi.yaml")
def get_openapi_yaml():
    return FileResponse("openapi.yaml", media_type="text/yaml")


# Notion endpoint
@app.get("/get-content")
def getNotionContent(page_name: str = Query(..., description="Title of Notion page")):
    try:
        content = get_page_content(page_name)
        return {"content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Query embedding class
class QueryRequest(BaseModel):
    query: str

def calc_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def get_relevant_chunks(query: str, k: int = 5):
    """
    Get top k matching chunks from Supabase using vector similarity
    """
    try:
        # Get embedding for the query
        embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        query_embedding = np.array(embedding_response.data[0].embedding, dtype=np.float32)
        
        # Query Supabase using the match_documents function (if you have it)
        # or fetch all and compute similarity
        response = supabase.table("juno_embeddings").select(
            "id, page_name, chunk, embedding, chunk_index, source_file, file_hash, chunk_hash"
        ).execute()
        
        if not response.data:
            return []
        
        chunks = response.data
        
        # Compute similarity for each chunk
        for chunk in chunks:
            try:
                # Convert embedding to json
                chunk_embedding = chunk["embedding"]
                if isinstance(chunk_embedding, str):
                    chunk_embedding = np.array(json.loads(chunk_embedding), dtype=np.float32)
                else:
                    chunk_embedding = np.array(chunk_embedding, dtype=np.float32)
                if len(chunk_embedding) != len(query_embedding):
                    logging.warning(f"Chunk embedding length mismatch: {len(chunk_embedding)} vs {len(query_embedding)}")
                    continue
                chunk["similarity"] = calc_cosine_similarity(query_embedding, chunk_embedding)
                chunk["content"] = chunk["chunk"]
                chunk["source"] = chunk.get("page_name") or chunk.get("source_file", "Unknown")
            except Exception as e:
                logging.error(f"Error processing chunk {chunk['id']}: {e}")
                continue
        # Filter out chunks with no similarity score
        valid_chunks = [chunk for chunk in chunks if "similarity" in chunk]
        logging.info(f"Valid chunks with similarity: {len(valid_chunks)}")
        
        # Sort by similarity and return top k
        top_chunks = sorted(valid_chunks, key=lambda x: x["similarity"], reverse=True)[:k]
        # Log similarity scores for debugging
        for i, chunk in enumerate(top_chunks):
            logging.info(f"Chunk {i+1}: similarity={chunk['similarity']:.3f}, source={chunk['source']}")
        
        return top_chunks
        
    except Exception as e:
        logging.error("❌ Error getting matching chunks:", exc_info=True)
        return []

@app.get("/debug/chunks")
async def debug_chunks():
   """Debug endpoint to see what chunks are in the database"""
   try:
        # Check total count
        count_response = supabase.table("juno_embeddings").select("id", count="exact").execute()
        total_count = count_response.count if hasattr(count_response, 'count') else 0
        
        # Get sample data
        response = supabase.table("juno_embeddings").select(
            "id, page_name, source_file, chunk_index, chunk"
        ).limit(5).execute()
        
        # Get first few characters of each chunk for preview
        sample_chunks = []
        for chunk in response.data:
            sample_chunks.append({
                "id": chunk.get("id"),
                "page_name": chunk.get("page_name"),
                "source_file": chunk.get("source_file"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_preview": chunk.get("chunk", "")[:200] + "..." if len(chunk.get("chunk", "")) > 200 else chunk.get("chunk", "")
            })
        
        return {
            "total_chunks": total_count,
            "sample_chunks": sample_chunks,
            "table_structure": "juno_embeddings table accessed successfully",
            "database_status": "connected"
        }
   except Exception as e:
        logging.error(f"Debug endpoint error: {e}")
        return {
            "error": str(e),
            "database_status": "error"
        }

@app.get("/debug/search-test")
async def debug_search_test():
    """Test search functionality with a simple query"""
    try:
        # Test with a simple query
        test_query = "podcast"
        
        # Get embedding
        embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",  # Make sure this matches your embedder
            input=test_query
        )
        query_embedding = embedding_response.data[0].embedding
        
        # Get all chunks with embeddings
        response = supabase.table("juno_embeddings").select(
            "id, page_name, chunk, embedding"
        ).limit(10).execute()
        
        if not response.data:
            return {"error": "No data in database"}
        
        # Calculate similarities
        similarities = []
        for chunk in response.data:
            try:
                embedding_raw = chunk["embedding"]
        
                if isinstance(embedding_raw, str):
                    embedding_vector = np.array(json.loads(embedding_raw), dtype=np.float32)
                else:
                    embedding_vector = np.array(embedding_raw, dtype=np.float32)

                similarity = calc_cosine_similarity(query_embedding, embedding_vector)

                similarities.append({
                    "id": chunk["id"],
                    "page_name": chunk["page_name"],
                    "similarity": round(similarity, 4),
                    "chunk_preview": chunk["chunk"][:100] + "..."
                })
            except Exception as e:
                similarities.append({
                    "id": chunk["id"],
                    "error": str(e)
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        
        return {
            "test_query": test_query,
            "embedding_dimension": len(query_embedding),
            "chunks_tested": len(response.data),
            "top_similarities": similarities[:5]
        }
        
    except Exception as e:
        logging.error(f"Search test error: {e}")
        return {"error": str(e)}

@app.post("/query")
async def queryKnowledgeBase(query: QueryRequest):
    logging.info(f"Received query: {query.query}")

    try: 
        user_question = query.query

        #get relevant chunks from supabase
        results = await get_relevant_chunks(user_question, k=5)
        
        if not results:
            return {
                "answer": "I couldn't find that information in the embedded knowledge base.",
                "sources": [],
                "confidence": "low"
            }
        
        # 2. Build context from chunk content
        context_parts = []
        sources = []
        
        for i, result in enumerate(results, 1):
            content = result["content"]
            source = result.get("source") or result.get("page_name") or result.get("file_name", "Unknown")
            similarity = result.get("similarity", 0.0)
            
            context_parts.append(f"[Source {i}: {source}]\n{content}")
            sources.append({
                "name": source,
                "similarity": round(float(result.get("similarity")) if isinstance(similarity, (int, float, np.floating)) else 0.0, 3),
                "rank": i
            })
        
        context = "\n\n".join(context_parts)
        
        # 3. Enhanced system prompt for better responses
        system_prompt = """You are Juno, a Digital Intelligence assistant for Junk Kouture.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions
- If the answer is not in the context, say: "I couldn't find that information in the embedded knowledge base."
- When referencing information, mention the source (e.g., "According to [Source 1]...")
- Be specific and detailed when the context supports it
- If multiple sources have conflicting information, mention both perspectives
- Do NOT make assumptions or add information not in the context

Context quality: Based on the similarity scores, prioritize information from higher-ranked sources."""

        # Make API call with client syntax
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_question}"}
            ],
            temperature=0.1,  # Lower temperature for more consistent responses
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Determine confidence based on similarity scores
        avg_similarity = sum((r.get("similarity", 0)) for r in results) / len(results)
        confidence = "high" if avg_similarity > 0.7 else "medium" if avg_similarity > 0.5 else "low"
        
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "context_used": len(results),
            "debug_info": f"avg similarity={avg_similarity:.3f}"
        }
        
    except Exception as e:
        logging.error("❌ Error in query endpoint:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

