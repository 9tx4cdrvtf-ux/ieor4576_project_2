import contextlib
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part
from pydantic import BaseModel

load_dotenv()

from disney_agent.agent import root_agent  # noqa: E402

# --- ADK Runner ---

session_service = InMemorySessionService()
runner = Runner(
    agent=root_agent,
    app_name="disney_queue_app",
    session_service=session_service,
)

# --- FastAPI ---

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/")
async def index():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    with contextlib.suppress(Exception):
        await session_service.create_session(
            app_name="disney_queue_app",
            user_id=session_id,
            session_id=session_id,
        )

    content = Content(role="user", parts=[Part.from_text(text=request.message)])

    try:
        response_text = ""
        async for event in runner.run_async(
            user_id=session_id,
            session_id=session_id,
            new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        response_text = part.text
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Agent or model call failed ({type(e).__name__}). "
                "Check Vertex: set VERTEXAI_PROJECT to your real GCP project id, "
                "enable Vertex AI API, and run `gcloud auth application-default login`. "
                f"Original error: {e!s}"
            ),
        ) from e

    return ChatResponse(response=response_text, session_id=session_id)


@app.post("/clear")
async def clear(session_id: str | None = None):
    if session_id:
        with contextlib.suppress(Exception):
            await session_service.delete_session(
                app_name="disney_queue_app",
                user_id=session_id,
                session_id=session_id,
            )
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
