"""BigQuery tools for historical Disney wait-time analysis.

Tables (project: wide-gamma-486722-m8, dataset: disney_historical_data):
  - ride_queue_hourly_stats:  park_name, ride_name, year, hour(0-23),
                               overall_avg_queue_mins, avg_max_queue_mins, created_at
  - ride_queue_monthly_stats: park_name, ride_name, year, month(1-12),
                               overall_avg_queue_mins, avg_max_queue_mins, created_at

Coverage: Disney Magic Kingdom only.
Rides: Jungle Cruise, Pirates of the Caribbean, Swiss Family Treehouse, TRON Lightcycle / Run.

All tools use keyword-based fuzzy matching — no exact names required.
'Magic Kingdom' matches 'Disney Magic Kingdom'; 'TRON' matches 'TRON Lightcycle / Run'.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT = os.environ.get("VERTEXAI_PROJECT", "wide-gamma-486722-m8")
DATASET = "disney_historical_data"
HOURLY_TABLE  = f"`{PROJECT}.{DATASET}.ride_queue_hourly_stats`"
MONTHLY_TABLE = f"`{PROJECT}.{DATASET}.ride_queue_monthly_stats`"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_client: bigquery.Client | None = None


def _bq() -> bigquery.Client:
    """Lazy-init BigQuery client (reuses connection across calls)."""
    global _client
    if _client is None:
        _client = bigquery.Client(project=PROJECT)
    return _client


def _keywords(query: str) -> list[str]:
    """Split a natural-language query into lowercase keywords, dropping stop words.

    'Magic Kingdom' → ['magic', 'kingdom']
    'TRON Lightcycle / Run' → ['tron', 'lightcycle', 'run']
    """
    stop = {"the", "of", "and", "at", "in", "for", "to", "a", "an"}
    tokens = re.split(r"[\s\-/]+", query.strip().lower())
    return [t for t in tokens if len(t) > 1 and t not in stop]


def _fuzzy_where(
    park_query: str,
    ride_query: str,
) -> tuple[str, list[bigquery.ScalarQueryParameter]]:
    """Build a parameterized fuzzy WHERE clause matching park + ride name by keywords.

    Each keyword becomes its own LIKE condition (AND within a field), so
    'magic kingdom' matches 'Disney Magic Kingdom' and 'tron' matches 'TRON Lightcycle / Run'.
    """
    params: list[bigquery.ScalarQueryParameter] = []
    conditions: list[str] = []

    def add_field(field: str, query: str, prefix: str) -> None:
        kws = _keywords(query)
        if not kws:
            return
        parts = []
        for i, kw in enumerate(kws):
            pname = f"{prefix}_{i}"
            parts.append(f"LOWER({field}) LIKE @{pname}")
            params.append(bigquery.ScalarQueryParameter(pname, "STRING", f"%{kw}%"))
        conditions.append("(" + " AND ".join(parts) + ")")

    if park_query.strip():
        add_field("park_name", park_query, "park")
    if ride_query.strip():
        add_field("ride_name", ride_query, "ride")

    where = " AND ".join(conditions) if conditions else "TRUE"
    return where, params


def _no_data_response(park_query: str, ride_query: str) -> str:
    return json.dumps({
        "ok": False,
        "error": "no_data_found",
        "park_query": park_query,
        "ride_query": ride_query,
        "hint": (
            "No historical data found for this park/ride combination. "
            "Try broader keywords, e.g. use a shorter ride name or partial park name."
        ),
    })


def _run_query(sql: str, params: list[bigquery.ScalarQueryParameter]) -> list[Any]:
    """Execute a BigQuery query and return rows, raising a clear error on failure."""
    try:
        job_config = bigquery.QueryJobConfig(query_parameters=params)
        return list(_bq().query(sql, job_config=job_config).result())
    except Exception as e:
        raise RuntimeError(f"BigQuery query failed: {e}") from e


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def get_hourly_wait_pattern(park_name_query: str, ride_name_query: str) -> str:
    """Return the historical average wait by hour of day (0–23) for a ride.

    Use this to answer:
      - 'What time of day has the shortest wait for X?'
      - 'When should I ride Pirates to avoid the crowd?'
      - 'Is morning or evening better for TRON?'

    Args:
        park_name_query: Natural language park name, e.g. 'Magic Kingdom'.
        ride_name_query: Natural language ride name, e.g. 'TRON' or 'Pirates'.

    Returns:
        JSON with per-hour avg/peak waits plus best_hour and worst_hour summary.
        Hours are UTC — subtract 4–5 hours to convert to US Eastern time.
    """
    where, params = _fuzzy_where(park_name_query, ride_name_query)
    sql = f"""
        SELECT park_name, ride_name, hour,
               ROUND(overall_avg_queue_mins, 1) AS avg_wait_mins,
               ROUND(avg_max_queue_mins, 1)     AS avg_peak_wait_mins
        FROM {HOURLY_TABLE}
        WHERE {where}
        ORDER BY ride_name, hour
    """
    try:
        rows = _run_query(sql, params)
    except RuntimeError as e:
        return json.dumps({"ok": False, "error": str(e)})

    if not rows:
        return _no_data_response(park_name_query, ride_name_query)

    rides: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.park_name}||{row.ride_name}"
        if key not in rides:
            rides[key] = {"park": row.park_name, "ride": row.ride_name, "hourly_pattern": []}
        rides[key]["hourly_pattern"].append({
            "hour_utc": row.hour,
            "label_utc": f"{row.hour:02d}:00 UTC",
            "avg_wait_mins": row.avg_wait_mins,
            "avg_peak_wait_mins": row.avg_peak_wait_mins,
        })

    result: list[dict[str, Any]] = []
    for data in rides.values():
        valid = [h for h in data["hourly_pattern"] if h["avg_wait_mins"] is not None]
        if valid:
            best  = min(valid, key=lambda x: x["avg_wait_mins"])
            worst = max(valid, key=lambda x: x["avg_wait_mins"])
            data["best_hour_utc"]        = best["label_utc"]
            data["best_avg_wait_mins"]   = best["avg_wait_mins"]
            data["worst_hour_utc"]       = worst["label_utc"]
            data["worst_avg_wait_mins"]  = worst["avg_wait_mins"]
        result.append(data)

    return json.dumps({"ok": True, "rides": result})


def get_monthly_wait_pattern(park_name_query: str, ride_name_query: str) -> str:
    """Return the historical average wait by month (1–12) for a ride.

    Use this to answer:
      - 'What month of the year has the shortest wait for X?'
      - 'Is July or December busier for TRON?'
      - 'When is the best time of year to visit for Pirates?'

    Args:
        park_name_query: Natural language park name, e.g. 'Magic Kingdom'.
        ride_name_query: Natural language ride name, e.g. 'Jungle Cruise'.

    Returns:
        JSON with per-month avg/peak waits plus best_month and worst_month summary.
    """
    where, params = _fuzzy_where(park_name_query, ride_name_query)
    sql = f"""
        SELECT park_name, ride_name, month,
               ROUND(overall_avg_queue_mins, 1) AS avg_wait_mins,
               ROUND(avg_max_queue_mins, 1)     AS avg_peak_wait_mins
        FROM {MONTHLY_TABLE}
        WHERE {where}
        ORDER BY ride_name, month
    """
    try:
        rows = _run_query(sql, params)
    except RuntimeError as e:
        return json.dumps({"ok": False, "error": str(e)})

    if not rows:
        return _no_data_response(park_name_query, ride_name_query)

    rides: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = f"{row.park_name}||{row.ride_name}"
        if key not in rides:
            rides[key] = {"park": row.park_name, "ride": row.ride_name, "monthly_pattern": []}
        rides[key]["monthly_pattern"].append({
            "month": row.month,
            "month_name": MONTH_NAMES[row.month - 1],
            "avg_wait_mins": row.avg_wait_mins,
            "avg_peak_wait_mins": row.avg_peak_wait_mins,
        })

    result: list[dict[str, Any]] = []
    for data in rides.values():
        valid = [m for m in data["monthly_pattern"] if m["avg_wait_mins"] is not None]
        if valid:
            best  = min(valid, key=lambda x: x["avg_wait_mins"])
            worst = max(valid, key=lambda x: x["avg_wait_mins"])
            data["best_month"]          = best["month_name"]
            data["best_avg_wait_mins"]  = best["avg_wait_mins"]
            data["worst_month"]         = worst["month_name"]
            data["worst_avg_wait_mins"] = worst["avg_wait_mins"]
        result.append(data)

    return json.dumps({"ok": True, "rides": result})


def compare_wait_to_historical(park_name_query: str, ride_name_query: str) -> str:
    """Compare a ride's current live wait to its historical average for the nearest recorded hour.

    Use this to answer:
      - 'Is the wait for TRON good or bad right now compared to usual?'
      - 'Is today's wait for Pirates above or below the historical average?'
      - 'Is right now a good time to ride Jungle Cruise?'

    Internally fetches ALL hourly history for the ride and picks the closest recorded hour
    (so it never fails when the exact hour has no record). Also fetches the live wait,
    returning a pre-computed verdict in one call — no need to call any other tool.

    Args:
        park_name_query: Natural language park name, e.g. 'Magic Kingdom'.
        ride_name_query: Natural language ride name, e.g. 'Pirates' or 'Jungle Cruise'.

    Returns:
        JSON with live_wait_mins, historical_avg_wait_mins, and a human-readable verdict.
    """
    from disney_agent.tools import get_current_wait_natural_language

    current_utc_hour = datetime.now(timezone.utc).hour

    where, params = _fuzzy_where(park_name_query, ride_name_query)
    sql = f"""
        SELECT park_name, ride_name, hour,
               ROUND(overall_avg_queue_mins, 1) AS avg_wait_mins,
               ROUND(avg_max_queue_mins, 1)     AS avg_peak_wait_mins
        FROM {HOURLY_TABLE}
        WHERE {where}
        ORDER BY ride_name, hour
    """
    try:
        rows = _run_query(sql, params)
    except RuntimeError as e:
        return json.dumps({"ok": False, "error": str(e)})

    if not rows:
        return _no_data_response(park_name_query, ride_name_query)

    # Group by ride
    rides_grouped: dict[str, list[dict[str, Any]]] = {}
    ride_meta: dict[str, tuple[str, str]] = {}
    for row in rows:
        key = f"{row.park_name}||{row.ride_name}"
        rides_grouped.setdefault(key, []).append({
            "hour": row.hour,
            "avg_wait_mins": row.avg_wait_mins,
            "avg_peak_wait_mins": row.avg_peak_wait_mins,
        })
        ride_meta[key] = (row.park_name, row.ride_name)

    # Fetch live wait using the exact BQ ride name for best match accuracy
    first_park, first_ride = next(iter(ride_meta.values()))
    live_wait: float | None = None
    live_is_open: bool = False
    try:
        live_raw = json.loads(get_current_wait_natural_language(first_park, first_ride))
        if live_raw.get("ok") and live_raw.get("matches"):
            m = live_raw["matches"][0]
            live_is_open = m.get("is_open", False)
            wt = m.get("wait_time")
            if wt is not None:
                live_wait = float(wt)
    except (TypeError, ValueError, KeyError):
        pass

    def _hour_distance(h: int) -> int:
        d = abs(h - current_utc_hour)
        return min(d, 24 - d)

    benchmarks: list[dict[str, Any]] = []
    for key, hourly in rides_grouped.items():
        park_name, ride_name = ride_meta[key]
        valid = [h for h in hourly if h["avg_wait_mins"] is not None]
        if not valid:
            continue
        closest = min(valid, key=lambda h: _hour_distance(h["hour"]))
        hist_avg = closest["avg_wait_mins"]

        if live_wait is not None and hist_avg is not None:
            diff = round(live_wait - hist_avg, 1)
            if diff <= -5:
                verdict = f"better than usual (live {live_wait} min vs historical avg {hist_avg} min — {abs(diff)} min shorter than normal)"
            elif diff >= 5:
                verdict = f"worse than usual (live {live_wait} min vs historical avg {hist_avg} min — {diff} min longer than normal)"
            else:
                verdict = f"about average (live {live_wait} min vs historical avg {hist_avg} min)"
        elif not live_is_open:
            verdict = "ride appears closed right now"
        else:
            verdict = "live wait unavailable — historical avg shown for context"

        benchmarks.append({
            "park": park_name,
            "ride": ride_name,
            "current_utc_hour": current_utc_hour,
            "matched_historical_hour": closest["hour"],
            "historical_avg_wait_mins": hist_avg,
            "historical_peak_wait_mins": closest["avg_peak_wait_mins"],
            "live_wait_mins": live_wait,
            "live_is_open": live_is_open,
            "verdict": verdict,
        })

    return json.dumps({"ok": True, "historical_benchmark": benchmarks})
