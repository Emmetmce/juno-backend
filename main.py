from fastapi import FastAPI, Query
from notion_util import get_page_content

app = FastAPI()

@app.get("/get-content")
def read_notion_page(page_name: str = Query(..., description="Title of the Notion page")):
    content = get_page_content(page_name)
    return {"content": content}