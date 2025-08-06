from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel
import os
import openai
import numpy as np
from supabase.client import create_client, Client
from notion_util import get_page_content, upload_file_to_existing_page, add_file_to_notion_page
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List, Literal
from upload_ui import router as upload_ui_router
from fastapi.staticfiles import StaticFiles
import requests
import tempfile
import hashlib
import re
import time


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
#mount the upload UI router
app.include_router(upload_ui_router)
app.mount("/static", StaticFiles(directory="static"), name="static") 

#env setup
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create Supabase client and set OpenAI API key
if not OPENAI_API_KEY:
    logging.info("OPENAI_API_KEY environment variable is not set")
    raise ValueError("OPENAI_API_KEY is required")

openai.api_key = OPENAI_API_KEY

# Supabase client
if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    logging.info("Supabase credentials are not properly configured")
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

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
    
# TODO: make 'save chat' endpoint to save chat history with Juno to Notion
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class SaveChatRequest(BaseModel):
    chat_id: str
    messages: List[ChatMessage]
    destination: Literal["notion", "supabase", "both"] = "notion"  # Default to Notion
    notion_page_name: str = None  # Optional Notion page ID if saving to Notion

@app.post("/save_chat")
async def saveChat(req: SaveChatRequest):
    logging.info(f"💬 /save_chat route hit for {req.chat_id}")

    try:
        #add timestamp to filename to avoid conflicts
        import time
        timestamp = int(time.time())

        # Format chat as markdown
        transcript_md = f"# Chat ID: {req.chat_id}\n\n"
        transcript_md += f"**Saved at:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        for msg in req.messages:
            prefix = "You" if msg.role == "user" else "Juno" #can change you to user name later
            transcript_md += f"**{prefix}:** {msg.content.strip()}\n\n"

        filename = f"{req.chat_id}_{timestamp}.md"
        file_bytes = transcript_md.encode("utf-8")

        #save to Supabase storage
        if req.destination in ["supabase", "both"]:
            supabase.storage.from_("chats").upload(path=filename, file=file_bytes, file_options={"content-type": "text/markdown"})
            logging.info(f"Saved to Supabase: {filename}")
        #save to notion storage
        if req.destination in ["notion", "both"]:
            if not req.notion_page_name:
                raise HTTPException(status_code=400, detail="Missing `notion_page_name` to upload file")
            #upload to supabase first to get public URL
            supabase.storage.from_("user.uploaded.notion.files").upload(
                path=filename,
                file=file_bytes,
                file_options={"content-type": "text/markdown"}
            )

            public_url = f"{SUPABASE_URL}/storage/v1/object/public/user.uploaded.notion.files/{filename}"
            add_file_to_notion_page(req.notion_page_name, filename, public_url)
            logging.info(f"Saved to Notion page: {req.notion_page_name} with file {filename}")

        return {"status": "success", "saved_as": filename, "chat_id":req.chat_id, "timestamp": timestamp}

    except Exception as e:
        logging.error(f"❌ Error saving chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving chat: {str(e)}")
    
# TODO: make 'save file' endpoint to save files to Notion
#Multipart/form-data upload endpoint -- this does not work through the OpenAI chat interface, will work through a custom frontend
@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...), notion_page_name: str = Form(...)):
    logging.info(f"/upload_file route hit for page: {notion_page_name}")

    try:
        file_bytes = await file.read()
        filename = file.filename
        #wipe special characters from filename
        import re, time
        safe_filename = re.sub(r"[^a-zA-Z0-9_.-]", "_", filename)

        # Add timestamp
        timestamp = int(time.time())
        parts = safe_filename.rsplit('.', 1)

        if len(parts) == 2:
            unique_filename = f"{parts[0]}_{timestamp}.{parts[1]}"
        else:
            unique_filename = f"{safe_filename}_{timestamp}"

        # Upload to Supabase public bucket
        try:
            # Use service role client
            service_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
            
            supabase_response = service_client.storage.from_("user.uploaded.notion.files").upload(
                path=unique_filename,
                file=file_bytes,
                file_options={
                    "content-type": file.content_type or "application/octet-stream",
                    "x-upsert": "true"
                }
            )
            
            logging.info(f"📁 Supabase upload response: {supabase_response}")
            
        except Exception as e:
            logging.error(f"❌ Upload method failed: {e}")

        if 'public_url' not in locals():
            public_url = f"{SUPABASE_URL}/storage/v1/object/public/user.uploaded.notion.files/{unique_filename}"

        logging.info(f"📁 Public URL: {public_url}")

        # Upload to Notion
        logging.info(f"About to call add_file_to_notion_page with: {notion_page_name}")
        add_file_to_notion_page(notion_page_name, unique_filename, public_url)

        try:
            result = {}
            result["status"] = "success"
            result["filename"] = unique_filename
            result["original_filename"] = filename
            result["url"] = public_url
            result["notion_page"] = notion_page_name
            
            logging.info(f"✅ Return object built successfully: {result}")
            return result
        except Exception as return_error:
            logging.error(f"❌ Error building return object: {str(return_error)}")
            return {"status": "success", "message": "file uploaded successfully to notion and supabase"}
    except Exception as e:
        logging.error(f"❌ Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

class FileUploadRequest(BaseModel):
    filename: str
    file_content: str  # supports txt, md, logs, yaml, json
    notion_page_name: str

@app.post("/upload_file_from_gpt")
# Upload file from GPT chat interface to Notion and Supabase
async def upload_file_from_gpt(payload: FileUploadRequest):
    import time
    try:
        timestamp = int(time.time())
        name_parts = payload.filename.rsplit(".", 1)
        if len(name_parts) == 2:
            unique_filename = f"{name_parts[0]}_{timestamp}.{name_parts[1]}"
        else:
            unique_filename = f"{payload.filename}_{timestamp}"

        # Convert text to bytes
        file_bytes = payload.file_content.encode("utf-8")

        # Upload only to Notion (Supabase will be handled by embedder)
        from notion_util import upload_file_to_existing_page
        upload_file_to_existing_page(payload.notion_page_name, unique_filename, file_bytes)

        logging.info(f"✅ Uploaded file '{unique_filename}' to Notion page '{payload.notion_page_name}'")

        return {
            "status": "success",
            "filename": unique_filename,
            "notion_page": payload.notion_page_name
        }

    except Exception as e:
        logging.error(f"❌ Error uploading file via JSON: {str(e)}")
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
        
        # Query Supabase for both text and image results
        response = supabase.table("juno_embeddings").select(
            "id, page_name, chunk, embedding, chunk_index, source_file, file_hash, chunk_hash, file_type, image_url, description, original_filename"
        ).execute()

        if not response.data:
            logging.info("No chunks found in the database")
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

                #add image fields if available
                if chunk.get("file_type") == "image":
                    chunk["is_image"] = True
                else:
                    chunk["is_image"] = False

            except Exception as e:
                logging.error(f"Error processing chunk {chunk['id']}: {e}")
                continue
        # Filter out chunks with no similarity score
        valid_chunks = [chunk for chunk in chunks if "similarity" in chunk]
        logging.info(f"Valid chunks with similarity: {len(valid_chunks)}")
        
        # Sort by similarity and return top k
        top_chunks = sorted(valid_chunks, key=lambda x: x["similarity"], reverse=True)[:max(k, 20)]
        # Log similarity scores for debugging
        for i, chunk in enumerate(top_chunks):
            chunk_type = "image" if chunk.get("is_image") else "text"
            logging.info(f"Chunk {i+1} ({chunk_type}): similarity={chunk['similarity']:.3f}, source={chunk['source']}")
            # log if image
            if chunk_type == "image":
                logging.info(f"🖼️ Image URL: {chunk.get('image_url')}, description: {chunk.get('description', '')}")
        
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
        results = await get_relevant_chunks(user_question, k=20)
        #seperate text and image results
        text_chunks = [r for r in results if not r.get("is_image")]
        image_chunks = [r for r in results if r.get("is_image")]

        # Use top 5 text and top 3 image chunks
        final_results = text_chunks[:5] + image_chunks[:3]
        
        if not results:
            return {
                "answer": "I couldn't find that information in the embedded knowledge base.",
                "sources": [],
                "confidence": "low"
            }
        
        # 2. Build context from chunk content. seperating images and text
        context_parts = []
        image_sources = []
        sources = []
        
        for i, result in enumerate(final_results, 1):
            content = result["content"]
            source = result.get("source") or result.get("page_name") or result.get("file_name", "Unknown")
            similarity = result.get("similarity", 0.0)
            image_url = result.get("image_url", None)
            
            if image_url:
                try:
                    sim_val = float(similarity)
                except (ValueError, TypeError):
                    sim_val = 0.0
                image_sources.append({
                    "url": image_url,
                    "description": content,
                    "source": source,
                    "similarity": round(sim_val, 3),
                    "rank": i
                })
            else:
                # Add text content to context
                context_parts.append(f"[Source {i}: {source}]\n{content}")
                sim_val = result.get("similarity", 0.0)
                try:
                    sim_val = float(similarity)
                except (ValueError, TypeError):
                    sim_val = 0.0
                sources.append({
                    "name": source,
                    "similarity": round(sim_val, 3),
                    "rank": i,
                    "type": "text"
                })
        
        # 3. Enhanced system prompt for better responses
        messages =  [
            {
                "role": "system", 
                "content": """You are Juno, a Digital Intelligence assistant for Junk Kouture.

IMPORTANT INSTRUCTIONS:
- Use ONLY the provided context to answer questions
- When you see images, analyze them directly and describe what you observe
- For creative tasks like "create a poster using this image", describe in detail what you see in the reference images and how you would adapt those visual elements
- For image editing requests like "add an exclamation point to the JK dublin graphic", you can use the /edit-image endpoint to modify existing images with AI
- If the answer is not in the context, say: "I couldn't find that information in the embedded knowledge base."
- When referencing information, mention the source (e.g., "According to [Source 1]...")
- Be specific and detailed when the context supports it
- If multiple sources have conflicting information, mention both perspectives
- Do NOT make assumptions or add information not in the context

When analyzing images, focus on:
- Visual elements (colors, typography, layout, design style)
- Branding elements (logos, text, styling)
- Composition and artistic choices
- Any text or messaging visible in the image

IMAGE EDITING CAPABILITIES:
You can edit existing images using the /edit-image endpoint. When users ask to modify images:

1. First identify the image they want to edit using descriptive terms
2. Use the edit_instruction to describe the modification in natural language
3. The system will find the best matching image and apply the edit using AI

Example: "add an exclamation point to the top right corner of the JK dublin graphic"
- search_query: "Junk Kouture dublin graphic" 
- edit_instruction: "add an exclamation point to the top right corner"

Available through your existing search and creative endpoints:
- Search images: Use context from vector search results  
- Analyze images: Use vision API analysis in responses
- Create with images: Use images as creative references


Context quality: Based on the similarity scores, prioritize information from higher-ranked sources."""
            }
        ]

        # Add context and user question to the messages
        if context_parts:
            text_context = "\n\n".join(context_parts)
            user_content = [{"type": "text", "text": f"Text Context:\n{text_context}\n\nQuestion: {user_question}"}]
        else:
            user_content = [{"type": "text", "text": f"Question: {user_question}"}]
        
        # Add images to the message
        for img_source in image_sources:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": img_source["url"],
                    "detail": "high"  # high detail for better analysis
                }
            })
            # Add context about the image
            user_content.append({
                "type": "text", 
                "text": f"[Image from {img_source['source']}]: {img_source['description']}"
            })
        
        messages.append({"role": "user", "content": user_content})

        # Make API call with client syntax
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,  # Lower temperature for more consistent responses
            max_tokens=1500
        )
        
        answer = response.choices[0].message.content
        
        #combine sources and images
        all_sources = sources + [
            {
                "name": img["source"],
                "similarity": img["similarity"],
                "rank": img["rank"],
                "type": "image",
                "image_url": img["url"],
            } for img in image_sources
        ]
        # Determine confidence based on similarity scores
        if results:
            avg_similarity = sum((r.get("similarity", 0)) for r in results) / len(results)
            confidence = "high" if avg_similarity > 0.7 else "medium" if avg_similarity > 0.5 else "low"
        else:
            confidence = "low"
        
        return {
            "answer": answer,
            "sources": all_sources,
            "confidence": confidence,
            "context_used": len(results),
            "images_analyzed": len(image_sources),
            "debug_info": f"avg similarity={avg_similarity:.3f}, images={len(image_sources)}, text={len(sources)}"
        }
        
    except Exception as e:
        logging.error("❌ Error in query endpoint:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@app.get("/upload-link")
async def get_upload_link(
    page: str = Query(None, description="Suggested Notion page name"),
    file_type: str = Query(None, description="Expected file type hint")
):
    """Generate an upload link for the GPT to share with users"""
    
    base_url = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8000")
    upload_url = f"{base_url}/upload"
    
    # Add page suggestion if provided
    if page:
        upload_url += f"?page={page.replace(' ', '%20')}"
    
    return {
        "upload_url": upload_url,
        "instructions": f"Click this link to upload your file to Juno: {upload_url}",
        "suggested_page": page,
        "message": "This will open a secure upload form where you can select your file and choose the destination page."
    }



#Image operation endpoints
class CreativeRequest(BaseModel):
    task: str  # e.g., "create a poster design"
    reference_images: List[str] = []  # URLs of reference images
    additional_context: str = ""  # Any additional text context

class ImageEditRequest(BaseModel):
    search_query: str  
    edit_instruction: str  
    save_to_notion_page: str = None  # Optional

async def get_matching_images(search_query: str, k: int = 10):
    """
    Search for images using vector embeddings on the description field
    Optimized for Supabase structure where images have descriptions
    """
    try:
        # Get embedding for the search query
        embedding_response = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=search_query
        )
        query_embedding = np.array(embedding_response.data[0].embedding, dtype=np.float32)
        
        # Query Supabase for image results only
        response = supabase.table("juno_embeddings").select(
            "id, page_name, chunk, embedding, description, image_url, original_filename, file_type"
        ).eq("file_type", "image").is_("image_url", "not.null").execute()

        if not response.data:
            logging.info("No images found in the database")
            return []
        
        images = response.data
        
        # Compute similarity for each image based on its description
        for image in images:
            try:
                # Convert embedding to numpy array
                image_embedding = image["embedding"]
                if isinstance(image_embedding, str):
                    image_embedding = np.array(json.loads(image_embedding), dtype=np.float32)
                else:
                    image_embedding = np.array(image_embedding, dtype=np.float32)
                
                if len(image_embedding) != len(query_embedding):
                    logging.warning(f"Image embedding length mismatch: {len(image_embedding)} vs {len(query_embedding)}")
                    continue
                    
                image["similarity"] = calc_cosine_similarity(query_embedding, image_embedding)
                
                # Also do text matching on description for boost
                description = image.get("description", "").lower()
                filename = image.get("original_filename", "").lower()
                search_lower = search_query.lower()
                
                # Boost similarity if there are direct text matches
                text_boost = 0
                if search_lower in description:
                    text_boost += 0.1
                if search_lower in filename:
                    text_boost += 0.05
                    
                # Check for individual words
                search_words = search_lower.split()
                for word in search_words:
                    if word in description:
                        text_boost += 0.02
                
                image["similarity"] = min(1.0, image["similarity"] + text_boost)
                
            except Exception as e:
                logging.error(f"Error processing image {image['id']}: {e}")
                continue
        
        # Filter out images with no similarity score and sort
        valid_images = [img for img in images if "similarity" in img]
        logging.info(f"Valid images with similarity: {len(valid_images)}")
        
        # Sort by similarity and return top k
        top_images = sorted(valid_images, key=lambda x: x["similarity"], reverse=True)[:k]
        
        # Log similarity scores for debugging
        for i, img in enumerate(top_images[:5]):  # Log top 5
            logging.info(f"Image {i+1}: {img['original_filename']} - similarity={img['similarity']:.3f}")
            logging.info(f"  Description: {img.get('description', '')[:100]}...")
        
        return top_images
        
    except Exception as e:
        logging.error("❌ Error getting matching images:", exc_info=True)
        return []

@app.get("/search-images")
async def search_images(
    query: str = Query(..., description="Search query for images"),
    limit: int = Query(5, description="Number of images to return")
):
    """Search for images in the knowledge base using vector embeddings"""
    try:
        # Use the better vector-based search function
        matching_images = await get_matching_images(query, k=limit)
        
        if not matching_images:
            # Fallback: show what images are available
            all_images = supabase.table("juno_embeddings").select(
                "page_name, image_url, original_filename, description"
            ).eq("file_type", "image").is_("image_url", "not.null").limit(10).execute()
            
            available_samples = []
            for img in all_images.data[:5]:  # Show first 5 as samples
                available_samples.append({
                    "filename": img["original_filename"],
                    "description": img.get("description", "")[:100] + "..." if len(img.get("description", "")) > 100 else img.get("description", ""),
                    "source": img["page_name"]
                })
            
            return {
                "images": [],
                "message": f"No images found matching '{query}'",
                "suggestions": "Try terms like: dublin, junk kouture, logo, promotional, graphic",
                "available_samples": available_samples,
                "total_images_in_db": len(all_images.data)
            }
        
        # Format results for the response
        formatted_images = []
        for img in matching_images:
            formatted_images.append({
                "source": img.get("page_name", "Unknown"),
                "url": img["image_url"],
                "filename": img["original_filename"],
                "description": img.get("description", ""),
                "similarity": round(img.get("similarity", 0), 3),
                "searchable_text": img.get("chunk", "")  # Keep for compatibility
            })
        
        return {
            "images": formatted_images,
            "total_found": len(matching_images),
            "query": query,
            "search_method": "vector_embeddings",
            "debug": f"Found {len(matching_images)} images using vector search"
        }
        
    except Exception as e:
        logging.error(f"Error searching images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-images")
async def analyze_images_with_prompt(request: dict):
    """Analyze specific images with a custom prompt using GPT-4 Vision"""
    try:
        image_urls = request.get("image_urls", [])
        prompt = request.get("prompt", "Analyze these images in detail")
        
        if not image_urls:
            raise HTTPException(status_code=400, detail="No image URLs provided")
        
        # Build messages for vision API
        messages = [
            {
                "role": "system",
                "content": "You are Juno, analyzing images from the Junk Kouture knowledge base. Provide detailed, accurate descriptions of what you see."
            }
        ]
        
        # Add user prompt and images
        user_content = [{"type": "text", "text": prompt}]
        
        for url in image_urls:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": url,
                    "detail": "high"
                }
            })
        
        messages.append({"role": "user", "content": user_content})
        
        # Call vision API
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.1,
            max_tokens=1500
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "analysis": analysis,
            "images_analyzed": len(image_urls),
            "prompt_used": prompt
        }
        
    except Exception as e:
        logging.error(f"Error analyzing images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/creative-with-images")
async def creative_task_with_images(request: CreativeRequest):
    """Use images from knowledge base for creative tasks like poster design"""
    try:
        if not request.reference_images:
            # Auto-search for relevant images based on task
            search_terms = ["poster", "design", "logo", "branding", "creative", "graphic", "visual"]
            found_images = []
            
            for term in search_terms:
                image_results = supabase.table("juno_embeddings").select(
                    "image_url, page_name, description"
                ).eq("file_type", "image").ilike("chunk", f"%{term}%").limit(2).execute()
                
                for img in image_results.data:
                    if img["image_url"] not in [x["url"] for x in found_images]:
                        found_images.append({
                            "url": img["image_url"],
                            "source": img["page_name"],
                            "description": img.get("description", "")
                        })
            
            reference_images = [img["url"] for img in found_images[:3]]
        else:
            reference_images = request.reference_images
        
        if not reference_images:
            return {
                "error": "No reference images found in knowledge base",
                "suggestion": "Try uploading some design references first"
            }
        
        # Build creative prompt
        creative_prompt = f"""
Task: {request.task}

Additional Context: {request.additional_context}

Please analyze the reference images I'm showing you and create a detailed design concept based on what you see. Include:

1. Visual Analysis: What design elements, colors, typography, and styling do you observe?
2. Brand Elements: What Junk Kouture branding elements should be incorporated?
3. Creative Concept: How would you adapt these visual elements for the requested task?
4. Specific Recommendations: Colors, fonts, layout, imagery, and messaging suggestions
5. Technical Details: Dimensions, file formats, or other specifications if relevant

Be specific about what you see in each reference image and how it influences your creative recommendations.
"""
        
        # Analyze with Vision API
        messages = [
            {"role": "system", "content": "You are Juno, a creative assistant for Junk Kouture. Use the reference images to create detailed, actionable design recommendations."},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": creative_prompt}
                ] + [
                    {
                        "type": "image_url",
                        "image_url": {"url": url, "detail": "high"}
                    } for url in reference_images
                ]
            }
        ]
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,  # Slightly higher for creativity
            max_tokens=1500
        )
        
        return {
            "creative_concept": response.choices[0].message.content,
            "reference_images_used": reference_images,
            "task": request.task,
            "images_analyzed": len(reference_images)
        }
        
    except Exception as e:
        logging.error(f"Error in creative task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/edit-image")
async def edit_image_with_ai(request: ImageEditRequest):
    """Find an image and edit it using OpenAI's image editing capabilities"""
    try:
        # Step 1: Use specialized image search
        matching_images = await get_matching_images(request.search_query, k=10)
        
        if not matching_images:
            # Fallback: Get all images for user to see what's available
            all_images = supabase.table("juno_embeddings").select(
                "original_filename, description, page_name"
            ).eq("file_type", "image").is_("image_url", "not.null").limit(10).execute()
            
            available_images = [
                f"{img['original_filename']} ({img.get('description', '')[:50]}...)" 
                for img in all_images.data
            ]
            
            return {
                "error": f"No images found matching '{request.search_query}'",
                "suggestion": "Try these terms based on available images:",
                "available_images": available_images
            }
        
        # Use the best match
        best_match = matching_images[0]
        image_url = best_match["image_url"]
        
        if not image_url:
            return {"error": "Selected image has no URL"}
        
        logging.info(f"🎯 Selected image: {best_match['original_filename']} (similarity: {best_match['similarity']:.3f})")
        logging.info(f"📝 Image description: {best_match.get('description', '')}")
        
        # Step 2: Download the image
        response = requests.get(image_url)
        if response.status_code != 200:
            return {"error": f"Failed to download image from {image_url}"}
        
        # Step 3: Create temporary file for the image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(response.content)
            temp_image_path = temp_file.name
        
        try:
            # Step 4: Use OpenAI's image editing API
            client = openai.OpenAI(api_key=OPENAI_API_KEY)
            
            with open(temp_image_path, "rb") as image_file:
                edit_response = client.images.edit(
                    image=image_file,
                    prompt=request.edit_instruction,
                    n=1,
                    size="1024x1024"
                )
            
            edited_image_url = edit_response.data[0].url
            
            # Step 5: Download the edited image
            edited_response = requests.get(edited_image_url)
            if edited_response.status_code != 200:
                return {"error": "Failed to download edited image from OpenAI"}
            
            # Step 6: Store the edited image in Supabase
            timestamp = int(time.time())
            safe_instruction = re.sub(r'[^a-zA-Z0-9_.-]', '_', request.edit_instruction.replace(' ', '_'))[:50]
            edited_filename = f"edited_{safe_instruction}_{timestamp}_{best_match['original_filename']}"
            
            # Upload to Supabase storage
            supabase.storage.from_("juno_images").upload(
                path=edited_filename,
                file=edited_response.content,
                file_options={"content-type": "image/png", "x-upsert": "true"}
            )
            
            # Get public URL for the edited image
            edited_public_url = f"{SUPABASE_URL}/storage/v1/object/public/juno_images/{edited_filename}"
            
            # Step 7: Generate description for the edited image
            original_desc = best_match.get('description', '')
            edited_description = f"Edited version of '{best_match['original_filename']}'. Original: {original_desc}. Edit applied: {request.edit_instruction}"
            
            # Create embedding for the edited image description
            embedding_response = openai.embeddings.create(
                input=edited_description,
                model="text-embedding-ada-002"
            )
            
            # Step 8: Store metadata in database
            file_hash = hashlib.md5(edited_response.content).hexdigest()
            chunk_hash = hashlib.md5(edited_description.encode('utf-8')).hexdigest()
            
            supabase.table("juno_embeddings").insert({
                "page_name": "AI Edited Images",
                "chunk": f"AI edited image: {edited_description}",
                "embedding": json.dumps(embedding_response.data[0].embedding),
                "source_file": f"ai_edited_{edited_filename}",
                "chunk_index": 0,
                "file_hash": file_hash,
                "chunk_hash": chunk_hash,
                "file_type": "image",
                "image_url": edited_public_url,
                "original_filename": edited_filename,
                "description": edited_description  # This is key for future searches!
            }).execute()
            
            # Step 9: Optionally save to Notion page
            notion_saved = False
            if request.save_to_notion_page:
                try:
                    add_file_to_notion_page(
                        request.save_to_notion_page, 
                        edited_filename, 
                        edited_public_url
                    )
                    notion_saved = True
                    logging.info(f"✅ Saved edited image to Notion page: {request.save_to_notion_page}")
                except Exception as e:
                    logging.warning(f"⚠️ Failed to save to Notion: {e}")
            
            return {
                "status": "success",
                "original_image": {
                    "filename": best_match["original_filename"],
                    "url": image_url,
                    "description": best_match.get("description", ""),
                    "source_page": best_match.get("page_name"),
                    "similarity": round(best_match["similarity"], 3)
                },
                "edited_image": {
                    "filename": edited_filename,
                    "url": edited_public_url,
                    "description": edited_description
                },
                "edit_instruction": request.edit_instruction,
                "saved_to_notion": notion_saved,
                "notion_page": request.save_to_notion_page if notion_saved else None,
                "debug_info": f"Found {len(matching_images)} matching images, selected best with similarity {best_match['similarity']:.3f}"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_image_path)
            
    except Exception as e:
        logging.error(f"❌ Error editing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Update your existing queryKnowledgeBase system prompt to include image editing context
# Add this to your system prompt in the queryKnowledgeBase function:

#def get_image_editing_system_context():
    """Add this content to your system prompt in queryKnowledgeBase"""
    return """
IMAGE EDITING CAPABILITIES:
You can edit existing images using the /edit-image endpoint. When users ask to modify images:

1. First identify the image they want to edit using descriptive terms
2. Use the edit_instruction to describe the modification in natural language
3. The system will find the best matching image and apply the edit using AI

Example: "add an exclamation point to the top right corner of the JK dublin graphic"
- search_query: "JK dublin graphic" 
- edit_instruction: "add a red exclamation point to the top right corner"

Available through your existing search and creative endpoints:
- Search images: Use context from vector search results  
- Analyze images: Use vision API analysis in responses
- Create with images: Use images as creative references
"""
