import os
import openai
from supabase import create_client
from dotenv import load_dotenv
import logging
import re
from notion_client import Client
from notion_util import extract_blocks_from_page 
import hashlib
import tiktoken
import json
# This script extracts content from Notion pages and attached files, embeds them using OpenAI embeddings,
# and stores the embeddings in a Supabase PostgreSQL vector database for semantic retrieval by Juno.

#imports for enhanced file processing
import fitz  # PyMuPDF for PDF parsing
try:
    import docx  # python-docx for DOCX files
except ImportError:
    docx = None
    logging.warning("python-docx not installed. DOCX support disabled.")

try:
    from PIL import Image
    import pytesseract
    import io
except ImportError:
    Image = None
    pytesseract = None
    logging.warning("PIL/pytesseract not installed. OCR support disabled.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize clients
openai.api_key = OPENAI_API_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
notion = Client(auth=os.getenv("NOTION_TOKEN"))

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file, including image OCR if available."""
    extracted_text = []
    try:
        with fitz.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc[page_num]

                #get regular text
                text = page.get_text()
                if text and text.strip(): 
                    extracted_text.append(f"Page {page_num + 1}:\n{text.strip()}\n")
                #get images and perform OCR if available
                if Image and pytesseract:
                    for img_index, img in enumerate(page.get_images(full=True)):
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            image = Image.open(io.BytesIO(image_bytes))
                            image.convert("RGB")  # Convert to RGB for OCR compatibility
                            ocr_text = pytesseract.image_to_string(image)
                            if ocr_text.strip():
                                extracted_text.append(f"[Page {page_num + 1} - Image {img_index + 1}]\n{ocr_text.strip()}")
                        except Exception as e:
                            logger.warning(f"Error processing image on page {page_num + 1}: {e}")
    except Exception as e:
        logger.error(f"Error extracting text from PDF file {file_path}: {e}")
        return "\n\n".join(extracted_text)

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file."""
    if not docx:
        logger.warning("python-docx is not installed. Cannot process DOCX files.")
        return ""
    
    text_parts = []
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text.strip())
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text_parts.append(" | ".join(row_text))

    except Exception as e:
        logger.error(f"Error extracting text from DOCX file {file_path}: {e}")
    return "\n".join(text_parts)

def extract_text_from_image(file_path: str) -> str:
    """Extracts text from an image file using OCR."""
    if not Image or not pytesseract:
        logger.warning("PIL/pytesseract is not installed. Cannot process image files.")
        return ""
    
    try:
        image = Image.open(file_path)
        image.convert("RGB")  # Convert to RGB for better OCR compatibility
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from image file {file_path}: {e}")
        return ""

def enhanced_extract_text(file_path: str) -> str:
    """Enhanced text extractor that handles multiple file types."""
    file_type = file_path.lower()
    
    try:
        if file_type.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        elif file_type.endswith((".docx", ".doc")):
            text = extract_text_from_docx(file_path)
        elif file_type.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            text = extract_text_from_image(file_path)
        elif file_type.endswith((".txt", ".md")):
            # Use existing logic for text files
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            # Try to read as text file (fallback)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        
        # Ensure we always return a string, even if empty
        return text if text is not None else ""
        
    except Exception as e:
        logger.error(f"Cannot process file type for {file_path}: {e}")
        return ""
        
def get_file_hash(file_path):
    """Generate a hash for the file to check if it has been processed"""
    try:
        with open(file_path, 'rb') as f:
            file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        return None

def hash_chunk(text):
    """Generate a hash for a text chunk to check if it has been processed"""
    try:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Error generating hash for chunk: {e}")
        return None

#wont be used as much, going to be using is_chunk_already_embedded instead
# but keeping this for future reference
def is_file_already_embedded(file_path, page_name):
    """Check if a file has already been embedded by checking the database"""
    try:
        # Check by exact file path and page name
        result = supabase.table("juno_embeddings").select("id").eq("source_file", file_path).eq("page_name", page_name).limit(1).execute()
        
        if result.data:
            logger.info(f"File already embedded (exact match): {file_path}")
            return True
        
        # Check by file hash (for when files are re-downloaded with different paths)
        file_hash = get_file_hash(file_path)
        if file_hash:
            # Check if we have this hash in our database
            hash_result = supabase.table("juno_embeddings").select("id").eq("file_hash", file_hash).limit(1).execute()
            
            if hash_result.data:
                logger.info(f"File already embedded (hash match): {file_path}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"Error checking if file is embedded: {e}")
        return False  # If we can't check, assume it's not embedded to be safe


def split_text(text, max_tokens=2000):
    """Split text into token-limited chunks for embedding"""
    enc = tiktoken.get_encoding("cl100k_base") # Using the same encoding as OpenAI
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para_tokens = enc.encode(para)
        if len(para_tokens) > max_tokens:
            # if paragraph is too long, split it into smaller chunks
            start = 0
            while start < len(para_tokens):
                end = start + max_tokens
                chunk_tokens = para_tokens[start:end]
                chunk_text = enc.decode(chunk_tokens)
                chunks.append(chunk_text.strip())
                start = end
        else:
            # if paragraph fits, add it to the current chunk
            test_chunk = current_chunk + para + "\n\n"
            if len(enc.encode(test_chunk)) <= max_tokens:
                current_chunk += test_chunk
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def embed_and_store(file_path, page_name):
    """Embed file content and store in Supabase, enhanced to handle various file types"""
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return
    
    #gen file hash for duplicate file checking
    file_hash = get_file_hash(file_path)

    try:
        text = enhanced_extract_text(file_path)
        
        if not text.strip():
            logger.warning(f"No text content found in {file_path}")
            return
            
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return
    
    # Split the full text into smaller chunks
    chunks = split_text(text)
    logger.info(f"Split {file_path} into {len(chunks)} chunks")
    
    # Embed each chunk and store in Supabase
    for i, chunk in enumerate(chunks):

        chunk_hash = hash_chunk(chunk) # Check if this chunk has already been embedded
        result = supabase.table("juno_embeddings").select("id").eq("chunk_hash", chunk_hash).eq("page_name", page_name).limit(1).execute()
        if result.data:
            logger.info(f"Chunk {i+1}/{len(chunks)} already embedded. Skipping.")
            continue
        if chunk.strip():
            try:
                # Get embedding from OpenAI
                response = openai.embeddings.create(
                    input=chunk,
                    model="text-embedding-ada-002"
                ).data[0].embedding
                
                # Insert the chunk and its embedding into the Supabase table
                supabase.table("juno_embeddings").insert({
                    "page_name": page_name,
                    "chunk": chunk,
                    "embedding": json.dumps(response),  # Store as JSON string
                    "source_file": file_path,
                    "chunk_index": i,
                    "file_hash": file_hash,
                    "chunk_hash": chunk_hash 
                }).execute()
                
                logger.info(f"Inserted chunk {i+1}/{len(chunks)} from {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing chunk {i} from {file_path}: {e}")

#should not be used in production, only for debugging because pages can have files added/removed
def is_page_already_processed(page_name):
    """Check if a Notion page has already been processed"""
    try:
        result = supabase.table("juno_embeddings").select("id").eq("page_name", page_name).limit(1).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking if page is processed: {e}")
        return False

def get_page_title(page):
    """Extract title from Notion page object"""
    properties = page.get("properties", {})
    
    # Try different possible title property names
    for title_key in ["title", "Page", "Name"]:
        if title_key in properties:
            title_prop = properties[title_key]
            if title_prop.get("type") == "title" and title_prop.get("title"):
                return title_prop["title"][0]["plain_text"]
    
    # Fallback: try to get title from the page object itself
    if "title" in page:
        title_list = page["title"]
        if title_list and len(title_list) > 0:
            return title_list[0]["plain_text"]
    
    # Last resort: use page ID
    return f"Page_{page['id']}"

def batch_embed_all_notion_files(force_reprocess=False):#force_reprocess=False incase of overwriting existing data
    """Search all Notion pages for files, embed them, and store in Supabase, enhanced with better file support"""
    logger.info("Connecting to Notion...")
    
    try:
        # Get all pages from Notion
        search_results = notion.search()["results"]
        logger.info(f"Found {len(search_results)} pages from Notion search")
        
        total_files_processed = 0
        
        for page in search_results:
            page_id = page["id"]
            page_title = get_page_title(page)
            
            logger.info(f"Processing page: '{page_title}' (ID: {page_id})")
            
            try:
                # Extract content directly using page ID
                content = extract_blocks_from_page(page_id)
                
                if not content:
                    logger.warning(f"No content found for page: {page_title}")
                    continue
                
                # Look for downloaded files in the content
                file_paths = re.findall(r"\[FILE_DOWNLOADED\]:(.+)", content)
                
                if file_paths:
                    logger.info(f"Found {len(file_paths)} files in page '{page_title}'")
                    
                    for file_path in file_paths:
                        file_path = file_path.strip()
                        logger.info(f"Processing file: {file_path}")
                        
                        embed_and_store(file_path, page_title)
                        total_files_processed += 1
                else:
                    logger.info(f"No files found in page '{page_title}'")
                    
                    # Also embed the page content itself if it has text
                    text_content = re.sub(r"\[FILE_DOWNLOADED\]:[^\n]*", "", content).strip()
                    if text_content:
                        # Create a temporary text file for the page content
                        temp_file = f"temp_page_{page_id}.txt"
                        try:
                            with open(temp_file, "w", encoding="utf-8") as f:
                                f.write(text_content)
                            logger.info(f"Created temporary file for page content: {temp_file}")
                            embed_and_store(temp_file, page_title)
                            total_files_processed += 1
                            os.remove(temp_file)  # Clean up
                        except Exception as e:
                            logger.error(f"Error processing page content for {page_title}: {e}")
                
            except Exception as e:
                logger.error(f"Error processing page '{page_title}': {e}")
                continue
        
        logger.info(f"Batch processing complete. Total files processed: {total_files_processed}")
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")

def test_single_page(page_name):
    """Test function to debug a single page"""
    logger.info(f"Testing single page: {page_name}")
    
    # Search for the specific page
    search_results = notion.search(query=page_name)["results"]
    
    if not search_results:
        logger.error(f"Page '{page_name}' not found")
        return
    
    page = search_results[0]
    page_id = page["id"]
    page_title = get_page_title(page)
    
    logger.info(f"Found page: '{page_title}' (ID: {page_id})")
    
    # Extract content
    content = extract_blocks_from_page(page_id)
    logger.info(f"Page content preview: {content[:200]}...")
    
    # Look for files
    file_paths = re.findall(r"\[FILE_DOWNLOADED\]:(.+)", content)
    logger.info(f"Found {len(file_paths)} files: {file_paths}")
    
    return content

if __name__ == "__main__":
    # Uncomment the next line to test a specific page first
    # test_single_page("Competition Handbook & Rules")
    logger.info("Starting enhanced embedder with PDF and multipart file support...")
    
    # Check for required dependencies
    missing_deps = []
    if not fitz:
        missing_deps.append("PyMuPDF (pip install PyMuPDF)")
    if not docx:
        missing_deps.append("python-docx (pip install python-docx)")
    if not Image or not pytesseract:
        missing_deps.append("PIL and pytesseract (pip install Pillow pytesseract)")
    
    if missing_deps:
        logger.warning(f"Missing optional dependencies: {', '.join(missing_deps)}")
        logger.warning("Some file types may not be processed correctly.")
    
    batch_embed_all_notion_files(force_reprocess=False)  # Set to True to reprocess all pages
    # Note: This will process all Notion pages and embed their content/files into Supabase