# 📘 Juno Backend AI System

Juno is the centralized Digital Intelligence (DI) behind Junk Kouture's internal knowledge systems. This backend enables semantic search, transcript storage, and Notion file automation using OpenAI embeddings, FastAPI, and a PostgreSQL vector database hosted on Supabase.

---

## 🔧 Features

- 🔎 Semantic query over embedded Notion content (text + files)
- 📄 Save and attach interviews or chats to Notion and Supabase
- 📂 Upload text-based files (Markdown, JSON, YAML, etc.) directly to Notion
- 🧠 Embedding pipeline optimized for large chunks (up to 2000 tokens)
- 🚦 Health checks and robust error handling

---

## 🧱 Tech Stack

- **Python** + **FastAPI**
- **OpenAI** (text-embedding-3-small)
- **Supabase** with pgvector extension
- **Notion API**
- **Render** for deployment

---

## 🔌 API Endpoints

### `POST /query`

Query the knowledge base with natural language.

```json
{
  "query": "What are the rules for Junk Kouture competition?"
}
```

**Response includes:**

- AI-generated `answer`
- Source files and `similarity` scores
- Confidence level and `context_used`

---

### `GET /get-content`

Retrieve raw Notion page content by title.

```http
GET /get-content?page_name=Competition%20Rules
```

---

### `POST /save_chat`

Save and attach a chat or interview as Markdown.

```json
{
  "chat_id": "interview_troy_armour",
  "messages": [
    { "role": "user", "content": "What was the hardest part of your life?" },
    { "role": "assistant", "content": "Being 15 was very hard for me..." }
  ],
  "destination": "both",
  "notion_page_name": "Team Interviews"
}
```

---

### `POST /upload_file_from_gpt`

Upload a supported file to a Notion page.

```json
{
  "filename": "vision.md",
  "file_content": "Junk Kouture is on a mission to...",
  "notion_page_name": "Brand Strategy"
}
```

---

### `GET /ping`

Simple health check.

```json
{ "status": "ok" }
```

---

## 🧪 Example CLI Usage

### Querying

```bash
curl -X POST https://juno-backend-7lj5.onrender.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What did the CEO say about his toughest period?"}'
```

### Uploading File

```bash
curl -X POST https://juno-backend-7lj5.onrender.com/upload_file_from_gpt \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "notes.md",
    "file_content": "Meeting notes from team sync...",
    "notion_page_name": "Weekly Sync"
  }'
```

### Saving Transcript

```bash
curl -X POST https://juno-backend-7lj5.onrender.com/save_chat \
  -H "Content-Type: application/json" \
  -d '{
    "chat_id": "troy_interview",
    "messages": [
      { "role": "user", "content": "What was the hardest moment in your journey?" },
      { "role": "assistant", "content": "It was when I was 15 and..." }
    ],
    "destination": "both",
    "notion_page_name": "Interviews"
  }'
```

---

## ✅ Status

Juno backend is live and running at:\
[**https://juno-backend-7lj5.onrender.com**](https://juno-backend-7lj5.onrender.com)

For help or contributions, contact the Juno Builder team.

