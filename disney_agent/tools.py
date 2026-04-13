"""Queue Times API helpers and lightweight EDA summaries."""

from __future__ import annotations

import json
import statistics
from typing import Any

import pandas as pd
import requests

PARKS_JSON = "https://queue-times.com/parks.json"
QUEUE_TIMES_TMPL = "https://queue-times.com/parks/{park_id}/queue_times.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 30.0) -> Any:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _iter_disney_parks() -> list[dict[str, Any]]:
    """Flatten Disney-related parks from live parks.json."""
    data = _get_json(PARKS_JSON)
    out: list[dict[str, Any]] = []
    for group in data:
        gname = (group.get("name") or "").lower()
        for park in group.get("parks") or []:
            pname = (park.get("name") or "").lower()
            if "disney" in gname or "disney" in pname:
                out.append({
                    "park_id": park.get("id"),
                    "name": park.get("name"),
                    "country": park.get("country"),
                    "timezone": park.get("timezone"),
                    "group": group.get("name"),
                })
    return out


def _collect_ride_rows(raw: dict[str, Any]) -> list[tuple[str | None, dict[str, Any]]]:
    """Flatten lands → rides into (land_name, ride_dict) pairs."""
    rows: list[tuple[str | None, dict[str, Any]]] = []
    for land in raw.get("lands") or []:
        lname = land.get("name")
        for ride in land.get("rides") or []:
            rows.append((lname, ride))
    return rows


def _extract_open_waits(raw: dict[str, Any]) -> list[float]:
    """Extract wait times (minutes) for all open rides from a queue_times payload."""
    waits: list[float] = []
    for land in raw.get("lands") or []:
        for ride in land.get("rides") or []:
            if not ride.get("is_open"):
                continue
            wt = ride.get("wait_time")
            if wt is None:
                continue
            try:
                waits.append(float(wt))
            except (TypeError, ValueError):
                continue
    return waits


def _compare_parks_crowd(park_ids_csv: str) -> str:
    """Compare crowd pressure across parks by numeric IDs (internal use only)."""
    park_ids = [int(x.strip()) for x in park_ids_csv.split(",") if x.strip()]
    results = []
    for pid in park_ids:
        raw = _get_json(QUEUE_TIMES_TMPL.format(park_id=pid))
        waits = _extract_open_waits(raw)
        results.append({
            "park_id": pid,
            "n_open_waits": len(waits),
            "mean_wait": round(statistics.mean(waits), 2) if waits else None,
            "median_wait": round(statistics.median(waits), 2) if waits else None,
            "max_wait": max(waits) if waits else None,
        })
    return json.dumps({"parks": results})


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------

def list_disney_parks() -> str:
    """List all Disney theme parks worldwide with their IDs and timezones."""
    matches = _iter_disney_parks()
    return json.dumps({
        "parks": [{"id": m["park_id"], **{k: v for k, v in m.items() if k != "park_id"}} for m in matches],
        "count": len(matches),
    })


def resolve_park_name(query: str) -> str:
    """Resolve a natural language park name (e.g. 'Magic Kingdom') to a numeric park_id.

    Never ask the user for a numeric ID — call this with their words instead.
    """
    q = (query or "").strip().lower()
    if not q:
        return json.dumps({"matches": [], "best_park_id": None, "note": "empty query"})

    candidates: list[dict[str, Any]] = []
    for park in _iter_disney_parks():
        pname = (park.get("name") or "").lower()
        pid = park.get("park_id")
        score = 0
        if q == pname:
            score = 100
        elif q in pname or pname in q:
            score = 85
        else:
            q_words = [w for w in q.split() if len(w) > 1]
            if q_words and all(w in pname for w in q_words):
                score = 70
            elif q_words:
                hit = sum(1 for w in q_words if w in pname)
                if hit:
                    score = 40 + 10 * hit
        # Prevent "Animal Kingdom" from matching when user says "Magic Kingdom"
        if pname == "animal kingdom" and "magic" in q and "animal" not in q:
            score = 0
        if score > 0:
            candidates.append({
                "park_id": pid,
                "name": park.get("name"),
                "score": score,
                "timezone": park.get("timezone"),
                "group": park.get("group"),
            })

    candidates.sort(key=lambda x: (-x["score"], x["name"] or ""))
    seen: set[Any] = set()
    uniq: list[dict[str, Any]] = []
    for c in candidates:
        if c["park_id"] not in seen:
            seen.add(c["park_id"])
            uniq.append(c)

    best = uniq[0]["park_id"] if uniq else None
    return json.dumps({
        "query": query.strip(),
        "matches": uniq[:10],
        "best_park_id": best,
        "note": "Use best_park_id for the clearest match; ask the user to clarify if scores are tied.",
    })


def get_current_wait_natural_language(park_name_query: str, ride_name_query: str) -> str:
    """Get the current posted wait for a ride using natural language names only.

    Example: park_name_query='Hollywood Studios', ride_name_query='Tower of Terror'.
    Never requires numeric IDs.
    """
    res = json.loads(resolve_park_name(park_name_query))
    pid = res.get("best_park_id")
    if pid is None:
        return json.dumps({
            "ok": False,
            "error": "could_not_resolve_park",
            "query": park_name_query,
            "candidates": res.get("matches", []),
        })
    raw = _get_json(QUEUE_TIMES_TMPL.format(park_id=pid))
    rq = (ride_name_query or "").strip().lower()
    rows = _collect_ride_rows(raw)

    matches: list[dict[str, Any]] = []
    for lname, ride in rows:
        rname = (ride.get("name") or "").lower()
        ok = rq in rname or rname in rq
        if not ok:
            words = [w for w in rq.split() if len(w) > 2]
            if words and all(w in rname for w in words):
                ok = True
        if ok:
            rid = ride.get("id")
            if rid not in {m["ride_id"] for m in matches}:
                matches.append({
                    "ride_id": rid,
                    "name": ride.get("name"),
                    "land": lname,
                    "wait_time": ride.get("wait_time"),
                    "is_open": ride.get("is_open"),
                    "last_updated": ride.get("last_updated"),
                })

    # Fallback: single-keyword match (e.g. "Twilight" → "Twilight Zone Tower of Terror")
    if not matches:
        words = [w for w in rq.replace("-", " ").split() if len(w) > 3]
        seen_ids: set[Any] = set()
        for word in words:
            for lname, ride in rows:
                rname = (ride.get("name") or "").lower()
                if word in rname and ride.get("id") not in seen_ids:
                    seen_ids.add(ride.get("id"))
                    matches.append({
                        "ride_id": ride.get("id"),
                        "name": ride.get("name"),
                        "land": lname,
                        "wait_time": ride.get("wait_time"),
                        "is_open": ride.get("is_open"),
                        "last_updated": ride.get("last_updated"),
                    })

    return json.dumps({
        "ok": True,
        "park_name_query": park_name_query.strip(),
        "ride_name_query": ride_name_query.strip(),
        "park_id": int(pid),
        "park_match": res.get("matches", [None])[0],
        "matches": matches,
        "summary": (
            f"Found {len(matches)} ride(s)."
            if matches
            else "No ride name match; try a different keyword from the full attraction name."
        ),
    })


def compare_ride_wait_vs_park_today(park_name_query: str, ride_name_query: str) -> str:
    """Compare a specific ride's current wait to all other open rides in the same park right now.

    Use this to answer 'Is this wait long compared to other rides in the park today?'
    This is a live cross-sectional benchmark, not a historical comparison.
    """
    cur = json.loads(get_current_wait_natural_language(park_name_query, ride_name_query))
    if not cur.get("ok"):
        return json.dumps(cur)
    matches = cur.get("matches") or []
    if not matches:
        return json.dumps({"ok": False, "error": "no_ride_match", "detail": cur})

    target = matches[0]
    your_wait = target.get("wait_time")
    if your_wait is None or target.get("is_open") is not True:
        return json.dumps({
            "ok": True,
            "ride": target,
            "note": "Ride closed or no wait posted; cannot compare.",
        })
    try:
        your_wait_f = float(your_wait)
    except (TypeError, ValueError):
        return json.dumps({"ok": False, "error": "invalid_wait", "ride": target})

    park_id = int(cur["park_id"])
    raw = _get_json(QUEUE_TIMES_TMPL.format(park_id=park_id))
    all_waits = _extract_open_waits(raw)

    if not all_waits:
        return json.dumps({"ok": False, "error": "no_waits_in_park"})

    n = len(all_waits)
    pct_shorter = round(100.0 * sum(1 for w in all_waits if w < your_wait_f) / n, 1)
    pct_longer  = round(100.0 * sum(1 for w in all_waits if w > your_wait_f) / n, 1)

    return json.dumps({
        "ok": True,
        "ride": {"name": target.get("name"), "your_wait_minutes": your_wait_f},
        "park_snapshot": {
            "park_id": park_id,
            "open_rides_with_posted_wait": n,
            "park_median_wait": round(float(statistics.median(all_waits)), 1),
            "park_mean_wait": round(float(statistics.mean(all_waits)), 1),
            "park_max_wait": max(all_waits),
        },
        "verdict": {
            "pct_rides_shorter_wait": pct_shorter,
            "pct_rides_longer_wait": pct_longer,
            "interpretation": (
                "A high pct_rides_shorter_wait means this ride's line is longer than most others right now."
            ),
        },
    })


def compare_parks_crowd_by_names(park_name_queries_csv: str) -> str:
    """Compare live crowd levels across multiple Disney parks by natural language names.

    Args:
        park_name_queries_csv: Comma-separated park names, e.g. 'Magic Kingdom, Hollywood Studios'.
    """
    parts = [x.strip() for x in park_name_queries_csv.split(",") if x.strip()]
    ids: list[int] = []
    unresolved: list[str] = []
    for p in parts:
        blob = json.loads(resolve_park_name(p))
        bid = blob.get("best_park_id")
        if bid is not None:
            ids.append(int(bid))
        else:
            unresolved.append(p)
    if not ids:
        return json.dumps({"error": "could not resolve any park name", "unresolved": unresolved})
    inner = json.loads(_compare_parks_crowd(",".join(str(i) for i in ids)))
    return json.dumps({**inner, "unresolved": unresolved})


def summarize_park_waits(park_id: int) -> str:
    """Summarize live wait times for a park: overall stats and breakdown by land.

    Use resolve_park_name first to get the park_id from a natural language name.
    """
    raw = _get_json(QUEUE_TIMES_TMPL.format(park_id=park_id))
    rows: list[dict[str, Any]] = []
    for land in raw.get("lands") or []:
        lname = land.get("name") or "Unknown"
        for ride in land.get("rides") or []:
            rows.append({
                "land": lname,
                "ride": ride.get("name"),
                "wait_time": ride.get("wait_time"),
                "is_open": ride.get("is_open"),
            })
    if not rows:
        return json.dumps({"park_id": park_id, "error": "no rides in payload"})

    df = pd.DataFrame(rows)
    open_df = df[df["is_open"] == True].copy()  # noqa: E712
    waits = pd.to_numeric(open_df["wait_time"], errors="coerce").dropna()

    by_land = (
        open_df.groupby("land")["wait_time"]
        .agg(["mean", "median", "max", "count"])
        .round(1)
        .reset_index()
        .to_dict(orient="records")
    )

    return json.dumps({
        "park_id": park_id,
        "rides_total": len(df),
        "rides_open": int(open_df.shape[0]),
        "wait_mean": round(float(waits.mean()), 1) if len(waits) else None,
        "wait_median": round(float(waits.median()), 1) if len(waits) else None,
        "wait_max": int(waits.max()) if len(waits) else None,
        "by_land": by_land,
    })


def run_pandas_eda_on_queue_json(park_id: int) -> str:
    """Return the top-5 longest waits and ride counts by land for a park right now.

    Use resolve_park_name first to get the park_id from a natural language name.
    """
    raw = _get_json(QUEUE_TIMES_TMPL.format(park_id=park_id))
    rows = []
    for land in raw.get("lands") or []:
        for ride in land.get("rides") or []:
            rows.append({
                "land": land.get("name"),
                "ride": ride.get("name"),
                "wait_time": ride.get("wait_time"),
                "is_open": ride.get("is_open"),
            })
    if not rows:
        return json.dumps({"park_id": park_id, "error": "empty"})

    df = pd.DataFrame(rows)
    open_df = df[df["is_open"] == True].copy()  # noqa: E712
    open_df["wait_time"] = pd.to_numeric(open_df["wait_time"], errors="coerce")
    open_df = open_df.dropna(subset=["wait_time"])
    top = open_df.nlargest(5, "wait_time")[["land", "ride", "wait_time"]].to_dict(orient="records")
    return json.dumps({
        "park_id": park_id,
        "top_5_longest_waits": top,
        "open_rides_by_land": open_df.groupby("land").size().to_dict(),
    })
