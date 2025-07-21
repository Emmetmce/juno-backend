import os
import requests
from notion_client import Client
from dotenv import load_dotenv
from urllib.parse import urlparse
import time
import logging
from supabase import create_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
notion = Client(auth=os.getenv("NOTION_TOKEN"))

def get_page_content(page_name):
    """Search for page by title and extract content"""
    try:
        search = notion.search(query=page_name, filter={"value": "page", "property": "object"})
        for result in search["results"]:
            if result["object"] == "page":
                page_id = result["id"]
                return extract_blocks_from_page(page_id)
        return "Page not found."
    except Exception as e:
        logger.error(f"Error searching for page '{page_name}': {e}")
        return f"Error retrieving page: {e}"

def extract_blocks_from_page(page_id):
    """Extract content and files from a Notion page"""
    
    def get_file_extension(url, content_type=None):
        """Determine file extension from URL or content type"""
        # Try to get extension from URL
        parsed = urlparse(url)
        path = parsed.path
        if '.' in path:
            return path.split('.')[-1]
        
        # Fallback to content type
        if content_type:
            type_map = {
                'application/pdf': 'pdf',
                'image/png': 'png',
                'image/jpeg': 'jpg',
                'image/gif': 'gif',
                'text/plain': 'txt',
                'application/msword': 'doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx'
            }
            return type_map.get(content_type, 'bin')
        
        return 'bin'  # Default extension
    
    def download_file(url, filename):
        """Download file with proper error handling"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Get content type for extension detection
            content_type = response.headers.get('content-type', '')
            extension = get_file_extension(url, content_type)
            
            # Create proper filename
            if not filename.endswith(f'.{extension}'):
                filename = f"{filename}.{extension}"
            
            os.makedirs("downloads", exist_ok=True)
            filepath = os.path.join("downloads", filename)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            logger.info(f"Downloaded file: {filepath}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download file from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            return None
    
    def extract_text_from_block(block):
        """Extract text content from various block types"""

        #ensure block is a dict
        if not isinstance(block, dict):
            logger.warning(f"Expected dict block, got {type(block)}: {block}")
            return []
        
        block_type = block.get("type")
        if not block_type:
            return []
        
        text_content = []
        block_data = block.get(block_type, {})
        
        # Handle rich text blocks
        if "rich_text" in block_data:
            for text_obj in block_data["rich_text"]:
                text_content.append(text_obj.get("plain_text", ""))
        
        # Handle text property (for some block types)
        elif "text" in block_data:
            for text_obj in block_data["text"]:
                text_content.append(text_obj.get("plain_text", ""))
        
        # Handle title blocks
        elif "title" in block_data:
            for text_obj in block_data["title"]:
                text_content.append(text_obj.get("plain_text", ""))
        
        # Handle code blocks
        elif block_type == "code":
            code_text = ""
            for text_obj in block_data.get("rich_text", []):
                code_text += text_obj.get("plain_text", "")
            if code_text:
                language = block_data.get("language", "")
                text_content.append(f"```{language}\n{code_text}\n```")
        
        return text_content
    
    def extract_blocks(blocks):
        """Recursively extract content from blocks"""
        content = []
        
        for block in blocks:
            # Ensure block is a dict
            if not isinstance(block, dict):
                logger.warning(f"Skipping non-dict block: {type(block)} - {block}")
                continue
            
            block_type = block.get("type")
            if not block_type:
                continue
            
            # Extract text content
            text_content = extract_text_from_block(block)
            content.extend(text_content)
            
            # Handle file blocks
            if block_type in ["file", "pdf", "image", "video", "audio"]:
                block_data = block.get(block_type, {})
                
                # Handle both uploaded files and external links
                file_obj = block_data.get("file") or block_data.get("external")
                
                if file_obj and file_obj.get("url"):
                    file_url = file_obj.get("url")
                    
                    # Try to get filename from caption or use block ID
                    filename = None
                    if "caption" in block_data and block_data["caption"]:
                        filename = block_data["caption"][0].get("plain_text", "")
                    
                    if not filename:
                        filename = f"{block_type}_{block['id']}"
                    
                    # Clean filename
                    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
                    
                    # Download file
                    downloaded_path = download_file(file_url, filename)
                    if downloaded_path:
                        content.append(f"[FILE_DOWNLOADED]:{downloaded_path}")
                        logger.info(f"Successfully downloaded: {downloaded_path}")
                    else:
                        content.append(f"[FILE_FAILED]:{file_url}")
                        logger.warning(f"Failed to download: {file_url}")
            
            # Handle child blocks recursively
            if block.get("has_children"):
                try:
                    child_response = notion.blocks.children.list(block["id"])
                    child_blocks = child_response.get("results", [])

                    # Filter out non-dict items before recursion
                    valid_child_blocks = [b for b in child_blocks if isinstance(b, dict)]
                    
                    if len(valid_child_blocks) != len(child_blocks):
                        logger.warning(f"Filtered out {len(child_blocks) - len(valid_child_blocks)} invalid child blocks")
                    
                    content.extend(extract_blocks(valid_child_blocks))
                    # Add delay to avoid rate limiting
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error fetching child blocks for {block['id']}: {e}")
                    logger.debug(f"Block that caused error: {block}")
        
        return content
    
    try:
        # Get top-level blocks
        response = notion.blocks.children.list(page_id)
        blocks = response.get("results", [])

        # Filter out non-dict items
        valid_blocks = [b for b in blocks if isinstance(b, dict)]
        
        if len(valid_blocks) != len(blocks):
            logger.warning(f"Filtered out {len(blocks) - len(valid_blocks)} invalid top-level blocks")
        
        all_content = extract_blocks(valid_blocks)
        return "\n".join(all_content)
        
    except Exception as e:
        logger.error(f"Error extracting blocks from page {page_id}: {e}")
        return f"Error extracting page content: {e}"
    
def upload_file_to_existing_page(page_title: str, filename: str, file_bytes: bytes):
    """Upload a file to Supabase and add it to an existing Notion page."""
    try:
        # Upload to Supabase
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
        supabase.storage.from_("user.uploaded.notion.files").upload(
            path=filename, 
            file=file_bytes, 
            file_options={"x-upsert": "true"}  # Use x-upsert for Supabase
        )
        
        # Create public URL
        SUPABASE_URL = os.getenv("SUPABASE_URL")  
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/user.uploaded.notion.files/{filename}"
        
        # Add to Notion page
        add_file_to_notion_page(page_title, filename, public_url)

    except Exception as e:
        logger.error(f"❌ Failed to upload file to Notion page '{page_title}': {str(e)}")
        raise

def debug_page_blocks(page_id):
    """Debug function to see what blocks exist on a page"""
    try:
        response = notion.blocks.children.list(page_id)
        blocks = response.get("results", [])
        logger.info(f"Found {len(blocks)} blocks on page {page_id}")
        
        for i, block in enumerate(blocks):
            # Add type checking for debugging
            if not isinstance(block, dict):
                logger.info(f"Block {i}: NON-DICT TYPE - {type(block)} - {block}")
                continue
                
            block_type = block.get("type")
            logger.info(f"Block {i}: type={block_type}, has_children={block.get('has_children', False)}")
            
            if block_type in ["file", "pdf", "image"]:
                logger.info(f"  File block details: {block.get(block_type, {})}")
        
        return blocks
    except Exception as e:
        logger.error(f"Error debugging page blocks: {e}")
        return []
    
# Function to add a file to an existing Notion page using a public URL
def add_file_to_notion_page(page_title: str, filename: str, public_url: str):
    """Add a file to a full-page Notion page by title (no databases)."""
    try:
        def slugify(text):
            import re
            import unicodedata
            text = unicodedata.normalize("NFKD", text)
            text = text.encode("ascii", "ignore").decode("utf-8")  # remove accents
            text = text.lower()
            text = re.sub(r"[’'\"“”]", "", text)  # remove smart quotes
            text = re.sub(r"[^\w\s-]", "", text)  # remove non-word characters
            text = re.sub(r"[-\s]+", "-", text).strip("-_")  # convert spaces to dashes
            return text

        normalized_title = slugify(page_title)
        logger.info(f"🔎 Normalized slug to search for: {normalized_title}")

        response = notion.search(query=page_title, filter={"value": "page", "property": "object"})
        results = response.get("results", [])

        for page in results:
            url = page.get("url", "")
            if slugify(url) and normalized_title in slugify(url):
                page_id = page["id"]
                break
        else:
            urls = [page.get("url", "") for page in results]
            logger.warning(f"No match found for '{page_title}'. URLs returned: {urls}")
            raise ValueError(f"No matching page titled '{page_title}' found in Notion.")

        # Add the file
        notion.blocks.children.append(
            block_id=page_id,
            children=[
                {
                    "object": "block",
                    "type": "file",
                    "file": {
                        "type": "external",
                        "external": {"url": public_url},
                    }
                }
            ]
        )

        logger.info(f"✅ Added file {filename} to Notion page '{page_title}'")

    except Exception as e:
        logger.error(f"❌ Failed to add file to Notion page '{page_title}': {str(e)}")
        raise
# Helper function to test the utility
#def test_page_extraction(page_name):
    ###"""Test function to debug page extraction"""
    #logger.info(f"Testing extraction for page: {notion_page_name}")
    #content = get_page_content(page_name)
    #logger.info(f"Extracted content length: {len(content)}")
    ###return content