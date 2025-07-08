import os
import requests
from notion_client import Client
from dotenv import load_dotenv
from urllib.parse import urlparse
import time
import logging

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
                    child_blocks = notion.blocks.children.list(block["id"])["results"]
                    content.extend(extract_blocks(child_blocks))
                    # Add small delay to avoid rate limiting
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error fetching child blocks for {block['id']}: {e}")
        
        return content
    
    try:
        # Get top-level blocks
        blocks = notion.blocks.children.list(page_id)["results"]
        all_content = extract_blocks(blocks)
        return "\n".join(all_content)
        
    except Exception as e:
        logger.error(f"Error extracting blocks from page {page_id}: {e}")
        return f"Error extracting page content: {e}"

# Helper function to test the utility
#def test_page_extraction(page_name):
    """Test function to debug page extraction"""
    logger.info(f"Testing extraction for page: {page_name}")
    content = get_page_content(page_name)
    logger.info(f"Extracted content length: {len(content)}")
    return content