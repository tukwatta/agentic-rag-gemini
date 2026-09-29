# Mini Agentic RAG (Gemini)

A small web app: upload documents, ask questions. A Gemini-powered agent decides
whether to search your documents (RAG), do maths, or check the time.

Stack: Streamlit (UI) + ChromaDB (vector store) + Google Gemini (LLM) + Docker.

## Files

| File | Purpose |
|---|---|
| `app.py` | Whole application: ingestion, tools, agent loop, web UI |
| `requirements.txt` | Python packages |
| `Dockerfile` | Recipe to build the container image |
| `docker-compose.yml` | Runs the container: port, secrets, persistent storage, auto-restart |
| `.env.example` | Template for your secrets (copy to `.env`) |
| `.dockerignore` | Keeps secrets and junk out of the image |
| `sample-docs/acme_policy.txt` | Small document to test with |

## Quick start (on any machine with Docker)

```bash
cp .env.example .env      # then edit .env: add GEMINI_API_KEY and APP_PASSWORD
docker compose up -d --build
# open http://localhost:8501  (or http://<server-ip>:8501)
```

Useful commands:

```bash
docker compose logs -f                 # watch logs
docker compose restart                 # restart
docker compose up -d --build           # apply code changes
docker compose up -d --force-recreate  # apply .env changes
docker compose down                    # stop (keeps your indexed data)
docker compose down -v                 # stop AND delete indexed data
```
