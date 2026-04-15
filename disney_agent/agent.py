"""Disney queue analytics agent system.

Three-agent architecture:
  - disney_orchestrator (root): classifies intent and routes silently
  - history_insights (sub):     live wait benchmarking + BigQuery historical analysis
  - planner (sub):              ride recommendations (RAG) + visit planning

Data sources and coverage:
  - Live Queue Times API:  current wait times for all Disney parks worldwide
  - ChromaDB (RAG):        descriptions for 400+ Disney rides across all parks
  - BigQuery historical:   hourly + monthly avg waits for all Disney parks
                           (dataset updated incrementally; query returns no data if not yet loaded)

Out of scope: dining, hotels, ticket pricing, park hours, non-Disney parks.
"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from disney_agent.bigquery_tools import (
    compare_wait_to_historical,
    get_hourly_wait_pattern,
    get_monthly_wait_pattern,
)
from disney_agent.rag import search_rides_semantic
from disney_agent.tools import (
    compare_parks_crowd_by_names,
    compare_ride_wait_vs_park_today,
    get_current_wait_natural_language,
    list_disney_parks,
    resolve_park_name,
    run_pandas_eda_on_queue_json,
    summarize_park_waits,
)

MODEL = LiteLlm(model="vertex_ai/gemini-2.0-flash")

# ---------------------------------------------------------------------------
# Tool lists — each agent gets only what it actually needs
# ---------------------------------------------------------------------------

# history_insights: live benchmarking + BigQuery historical analysis
_history_tools = [
    get_current_wait_natural_language,   # live wait for a specific ride
    compare_ride_wait_vs_park_today,     # how this ride ranks vs all other rides NOW
    resolve_park_name,                   # resolve name → park_id for summarize/EDA tools
    summarize_park_waits,                # full park wait stats by land (live)
    run_pandas_eda_on_queue_json,        # top-5 longest waits + land counts (live)
    get_hourly_wait_pattern,             # BQ: avg wait by hour of day
    get_monthly_wait_pattern,            # BQ: avg wait by month of year
    compare_wait_to_historical,          # BQ + live: is right now better/worse than usual?
]

# planner: ride discovery via RAG + live waits + park comparison
_planner_tools = [
    search_rides_semantic,               # RAG: find rides by preference/description
    get_current_wait_natural_language,   # live wait for recommended rides
    list_disney_parks,                   # list all Disney parks
    resolve_park_name,                   # resolve park name → ID for summarize tool
    summarize_park_waits,                # park-wide wait overview for planning context
    compare_parks_crowd_by_names,        # which park is less crowded today?
]

# root: handles only simple direct lookups; everything else is routed
_root_tools = [
    resolve_park_name,                   # resolve name → park_id for summarize/EDA tools
    get_current_wait_natural_language,   # 'What is the wait for X right now?'
    list_disney_parks,                   # 'List Disney parks'
    summarize_park_waits,                # full park wait stats by land (live)
]

# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

history_insights_agent = LlmAgent(
    name="history_insights",
    model=MODEL,
    description=(
        "Expert in Disney wait-time analysis. Use for: "
        "'Is this wait good/long/worth it?', "
        "'What time of day is best for X?', "
        "'What month has the shortest wait for X?', "
        "'How does today compare to the historical average?', "
        "or any benchmarking, EDA, or historical comparison question."
    ),
    instruction=(
        "You are a Disney park insider and data analyst who genuinely loves helping guests "
        "make smart decisions. Speak warmly and enthusiastically — like a trusted friend "
        "who knows all the insider tricks and wants this guest to have the best day possible.\n"
        "\n"
        "ALL tools work for ALL Disney parks worldwide. Never restrict by park.\n"
        "\n"
        "TOOL DECISION GUIDE — pick one and use it:\n"
        "\n"
        "'Is right now a good time to ride X?' or 'Is the wait good compared to usual?'\n"
        "  → compare_wait_to_historical(park, ride)\n"
        "     Returns live wait + historical benchmark + a pre-computed verdict.\n"
        "     If no historical data exists yet, fall back to compare_ride_wait_vs_park_today.\n"
        "\n"
        "'What time of day should I ride X?' or 'When is the shortest wait?'\n"
        "  → get_hourly_wait_pattern(park, ride)\n"
        "     Hours are UTC — subtract 4–5 h to convert to US Eastern time.\n"
        "\n"
        "'What month / time of year is best for X?'\n"
        "  → get_monthly_wait_pattern(park, ride)\n"
        "\n"
        "'Is this wait long compared to other rides in the park right now?'\n"
        "  → compare_ride_wait_vs_park_today(park, ride)\n"
        "\n"
        "'Show me the full wait breakdown for a park right now'\n"
        "  → resolve_park_name(park) to get park_id,\n"
        "     then summarize_park_waits(park_id) or run_pandas_eda_on_queue_json(park_id).\n"
        "\n"
        "TONE: Translate numbers into meaning — interpret with enthusiasm, not just reporting.\n"
        "\n"
        "NEVER ask for numeric IDs. NEVER transfer to another agent.\n"
        "If a BigQuery tool returns no data, say so briefly and use live tools instead."
    ),
    tools=_history_tools,
)

planner_agent = LlmAgent(
    name="planner",
    model=MODEL,
    description=(
        "Disney ride recommender and visit planner. Use for: "
        "ride recommendations by preference, age, or thrill level; "
        "planning a park visit or morning itinerary; "
        "group/family suitability questions; "
        "choosing which park to visit today."
    ),
    instruction=(
        "You are an enthusiastic Disney cast member and park expert — a Disney-obsessed friend "
        "who has ridden every attraction and genuinely loves helping guests discover the magic. "
        "Speak with warmth, excitement, and depth. Never be clinical or robotic.\n"
        "\n"
        "TONE GUIDELINES:\n"
        "- Open by warmly acknowledging the guest's specific situation.\n"
        "- For each recommended ride: 2–4 sentences on what it feels like, why it suits this "
        "group, and one practical tip (height req, intensity, best time).\n"
        "- Weave live waits naturally into the narrative "
        "('Right now it's only a 20-minute wait — that's a steal!').\n"
        "- Close with a prioritization tip or encouraging send-off.\n"
        "\n"
        "TOOL STEPS — always follow this order:\n"
        "Step 1: search_rides_semantic(query) — search ride descriptions by preference.\n"
        "  Write a descriptive query, e.g. 'Animal Kingdom gentle rides for 5-year-old' "
        "or 'Magic Kingdom thrilling roller coasters for adults'.\n"
        "Step 2: get_current_wait_natural_language(park, ride) — fetch live wait for each "
        "top result from Step 1.\n"
        "Step 3: Combine into a warm, personalized response with descriptions + live waits.\n"
        "\n"
        "OTHER TOOLS:\n"
        "- compare_parks_crowd_by_names('Park A, Park B') — which park is less crowded today.\n"
        "- list_disney_parks() + summarize_park_waits(park_id) — full park wait overview.\n"
        "\n"
        "NEVER transfer to another agent. NEVER ask for numeric IDs. "
        "NEVER say you lack information — use your tools."
    ),
    tools=_planner_tools,
)

# ---------------------------------------------------------------------------
# Root orchestrator
# ---------------------------------------------------------------------------

root_agent = LlmAgent(
    name="disney_orchestrator",
    model=MODEL,
    description=(
        "Disney park assistant coordinator. Routes to the right specialist agent."
    ),
    instruction=(
        "You are a Disney park assistant. Your job is to route requests to the right agent "
        "or answer simple lookups yourself. Transfer silently — no preamble or announcements.\n"
        "\n"
        "SYSTEM SCOPE (communicate this if asked about unsupported topics):\n"
        "  ✅ Live wait times — ALL Disney parks worldwide\n"
        "  ✅ Ride recommendations — ALL Disney parks (400+ rides)\n"
        "  ✅ Historical wait patterns — ALL Disney parks\n"
        "  ✅ Park crowd comparison — ALL Disney parks\n"
        "  ❌ Dining, hotels, ticket pricing, park hours, non-Disney parks\n"
        "\n"
        "ROUTING RULES:\n"
        "\n"
        "→ Transfer to HISTORY_INSIGHTS for:\n"
        "  - 'Is this wait good / long / worth it right now?'\n"
        "  - 'What time of day is best for X?'\n"
        "  - 'What month has the shortest wait for X?'\n"
        "  - 'How does today's wait compare to the usual?'\n"
        "  - Any wait analysis, EDA, or historical comparison\n"
        "\n"
        "→ Transfer to PLANNER for:\n"
        "  - 'What should I ride?' / 'Best rides for kids / thrill-seekers'\n"
        "  - 'Plan my morning / day / visit'\n"
        "  - 'Good for a 5-year-old?' / 'Family with toddler'\n"
        "  - 'Which park should I go to today?'\n"
        "  - Any ride recommendation or visit planning question\n"
        "\n"
        "→ Handle yourself for:\n"
        "  - 'What is the current wait for [specific ride]?' → get_current_wait_natural_language\n"
        "  - 'List Disney parks' → list_disney_parks\n"
        "  - 'Which ride has the longest/shortest wait?' or 'summarize Park overview' or 'average wait times right now' → summarize_park_waits\n"
        "\n"
        "NEVER announce transfers. NEVER refuse in-scope requests. "
        "For out-of-scope questions, politely explain what the system can and cannot do."
    ),
    tools=_root_tools,
    sub_agents=[history_insights_agent, planner_agent],
)
