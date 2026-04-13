Disney Parks around the World

Real-time queue times data: https://queue-times.com/en-US

information in RAG: The intro of each Ride in each disney park

## Run (local)

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Google Cloud project with Vertex AI. Copy `.env.example` to `.env` and set `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`.

```bash
uv sync
uv run python index_rides.py
uv run python app.py
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
