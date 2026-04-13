import json
import os

import requests
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm


def _get_lat_lon(location: str) -> tuple[float, float]:
    api_key = os.environ["OWM_API_KEY"]
    geo = requests.get(
        "http://api.openweathermap.org/geo/1.0/direct",
        params={"q": location, "limit": 1, "appid": api_key},
    ).json()
    if not geo:
        raise ValueError(f"Location not found: {location}")
    return geo[0]["lat"], geo[0]["lon"]


def get_weather(location: str) -> str:
    """Get current weather for a location."""
    api_key = os.environ["OWM_API_KEY"]
    lat, lon = _get_lat_lon(location)
    weather = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "imperial"},
    ).json()
    return json.dumps({
        "location": location,
        "temp_f": weather["main"]["temp"],
        "humidity": weather["main"]["humidity"],
        "description": weather["weather"][0]["description"],
    })


def lookup_contact(name: str) -> str:
    """Look up a contact's info by name."""
    return json.dumps({"name": name, "email": f"{name.lower()}@example.com"})


MODEL = LiteLlm(model="vertex_ai/gemini-2.0-flash-lite")

root_agent = LlmAgent(
    name="weather_agent",
    model=MODEL,
    description="A helpful assistant that can check weather and look up contacts.",
    instruction="You are a helpful assistant. Be concise and friendly.",
    tools=[get_weather, lookup_contact],
)
