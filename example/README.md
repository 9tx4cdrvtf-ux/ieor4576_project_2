# Google ADK Agent

Same weather + contact agent as [LiteLLM Tool Calling](../litellm-tool-calling), rebuilt with **Google ADK**. Compare the two to see what the framework handles for you: tool wrapping, the event-driven agent loop, and session management.

## Prerequisites

You need Google Cloud set up with Vertex AI. See the **Google Cloud & Vertex AI Setup Guide**.

You also need an [OpenWeatherMap API key](https://openweathermap.org/api) with access to the One Call API 3.0.

Create a `.env` file in this directory:

```
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1
OWM_API_KEY=your-openweathermap-key
```

## Running

### Option 1: Custom FastAPI frontend

```bash
uv run python app.py
```

Open http://localhost:8000 in your browser.

### Option 2: ADK built-in web UI

Copy your `.env` into the agent directory, then run `adk web`:

```bash
cp .env weather_agent/.env
uv run adk web
```

Open http://localhost:8000 and select `weather_agent` from the dropdown.

## Try it

- "What's the weather in New York?"
- "Look up Alice's contact info"
- "What's the weather in Tokyo and look up Bob's email"

## Project Structure

```
google-adk-agent/
├── weather_agent/        # Agent package (required for `adk web`)
│   ├── __init__.py       # from . import agent
│   └── agent.py          # Defines root_agent
├── app.py                # Custom FastAPI wrapper
├── index.html            # Chat frontend
└── .env                  # Your API keys
```

## API

- `GET /` - Serves the chat UI
- `POST /chat` - Send a message, returns response (ADK runner handles tools)
- `POST /clear` - Clear session history
