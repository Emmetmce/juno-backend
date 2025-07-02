import os
from notion_client import Client
from dotenv import load_dotenv

load_dotenv()
notion = Client(auth=os.getenv("NOTION_TOKEN"))

def get_page_content(page_name):
    #searching for page by title
    search = notion.search(query=page_name, filter={"value": "page", "property": "object"})
    for result in search["results"]:
        if result["object"] == "page":
            page_id = result["id"]
            return extract_blocks_from_page(page_id)
    return "Page not found."

def extract_blocks_from_page(page_id):
    #get files inside pages
    blocks = notion.blocks.children.list(page_id)["results"]
    content = []
    
    for block in blocks:
        block_type = block["type"]
        #text blocks
        if block_type in block and "text" in block[block_type]:
            for text_obj in block[block_type]["text"]:
                content.append(text_obj.get("plain_text", ""))
        #file blocks
        if block_type in ["file", "image", "pdf"]:
            file_data = block.get(block_type, {})
            file_url = file_data.get("file", {}).get("url")
            caption = file_data.get("caption", [])
            caption_text = caption[0]["plain_text"] if caption else "Unnamed File"
            content.append(f"[{caption_text}]({file_url})")

    return "\n".join(content) if content else "No readable content found."