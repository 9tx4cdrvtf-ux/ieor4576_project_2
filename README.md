# Disney Parks Queue Oracle
Disney Parks Queue Oracle is an AI-powered assistant that helps you make smarter decisions at Disney theme parks worldwide. Built with Google ADK and Vertex AI (Gemini), it combines live wait-time data, historical analytics, and a semantic ride database to answer questions in natural language — no app switching, no guesswork.
What it can do:

🎢 Live wait times — get current queue times for any ride across all Disney parks worldwide        
📊 Historical insights — find the best time of day or month to ride, and see how today's waits compare to historical averages        
🗺️ Ride recommendations — describe what you're looking for ("thrilling roller coasters", "gentle rides for a 5-year-old") and get personalized suggestions with live waits        
🌍 Park comparison — find out which park is least crowded right now        

Powered by:

Queue-Times.com for live wait data       
Google Vertex AI (Gemini 2.0 Flash) for natural language understanding        
ChromaDB for semantic search across 400+ Disney ride descriptions       
BigQuery for historical wait-time analytics     

## Live URL
**live url**: https://disney-queue-agent-new-505448524679.us-central1.run.app/

## Data Source

Real-time queue times data: https://queue-times.com/en-US

information in RAG: The intro of each Ride in each disney park

## Run (local)

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Google Cloud project with Vertex AI. Copy `.env.example` to `.env` and set `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`.

```bash
uv run index_rides.py
uv run app.py
```

Open http://127.0.0.1:8000 . The UI includes a link to [Powered by Queue-Times.com](https://queue-times.com/) as required by the Queue Times API.

ADK web UI (optional):

```bash
cp .env disney_agent/.env   # if adk expects env next to the agent package
uv run adk web
```

Select `disney_agent` / `disney_orchestrator` in the UI (exact label depends on ADK version).

## Layout

- `disney_agent/agent.py` — `root_agent` (orchestrator) + `history_insights` + `planner` sub-agents
- `disney_agent/tools.py` — Queue Times HTTP tools and pandas EDA helpers
- `disney_agent/rag_placeholder.py` — stub until the vector store is wired
- `app.py` — FastAPI + ADK `Runner` (same pattern as `example/app.py`)
