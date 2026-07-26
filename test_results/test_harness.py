"""
Charlotte AI Load + Quality Test Harness
30 concurrent personas, 10 questions each, 5 topic areas.
Usage: python3 test_harness.py [--mode fast|real]
"""

import asyncio
import httpx
import json
import time
import os
import sys
import uuid
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR  # Use the script's directory as output dir
CONFIG = {
    "NUM_PERSONAS": 30,
    "QUESTIONS_PER_PERSONA": 10,
    "INTERVAL_SECONDS": 300,
    "TIME_MODE": "fast",
    "OUTPUT_DIR": OUTPUT_DIR,
    "TOPICS": ["sales", "linkedin", "content", "coaching", "business"],
    "ENDPOINT": "https://charlotte-second-brain.onrender.com/chat",
}

# Override mode
if "--mode" in sys.argv:
    idx = sys.argv.index("--mode")
    CONFIG["TIME_MODE"] = sys.argv[idx + 1]
if CONFIG["TIME_MODE"] == "fast":
    CONFIG["INTERVAL_SECONDS"] = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Maximum concurrent HTTP connections
SEMAPHORE = asyncio.Semaphore(30)


async def send_question(client, question, persona_id, q_idx):
    """Send a single question to the chatbot and return response data."""
    payload = {"message": question["question"]}
    timestamp_sent = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    error = None
    http_status = None
    response_text = None
    try:
        async with SEMAPHORE:
            resp = await client.post(
                CONFIG["ENDPOINT"],
                json=payload,
                timeout=90,
            )
            http_status = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                response_text = data.get("answer", "")
            else:
                try:
                    err_data = resp.json()
                    error = err_data.get("error", err_data.get("detail", str(resp.status_code)))
                except Exception:
                    error = f"HTTP {resp.status_code}"
    except httpx.TimeoutException:
        error = "timeout"
        http_status = 0
    except httpx.ConnectError:
        error = "connection_error"
        http_status = 0
    except Exception as e:
        error = str(e)
        http_status = 0

    timestamp_received = datetime.now(timezone.utc).isoformat()
    latency_ms = int((time.monotonic() - start) * 1000)
    return {
        "index": q_idx + 1,
        "topic": question.get("topic", "unknown"),
        "question": question["question"],
        "timestamp_sent": timestamp_sent,
        "timestamp_received": timestamp_received,
        "latency_ms": latency_ms,
        "http_status": http_status or 0,
        "response_text": response_text or "",
        "error": error,
    }


async def run_persona(persona):
    """Run all questions for a single persona sequentially with intervals."""
    persona_id = persona["id"]
    questions = persona["questions"]
    profile = {k: v for k, v in persona.items() if k != "questions"}

    print(f"  [{persona_id}] {persona['name']} — starting ({len(questions)} questions)")

    results = []
    test_started = datetime.now(timezone.utc).isoformat()

    async with httpx.AsyncClient(timeout=90.0) as client:
        for idx, q_text in enumerate(questions):
            topic = await classify_topic(q_text, persona)
            question = {"question": q_text, "topic": topic}
            result = await send_question(client, question, persona_id, idx)
            results.append(result)

            status = "✓" if result["error"] is None else f"✗ {result['error']}"
            print(f"    [{persona_id}] Q{idx+1}/{len(questions)} → {result['latency_ms']}ms {status}")

            if idx < len(questions) - 1:
                await asyncio.sleep(CONFIG["INTERVAL_SECONDS"])

    test_ended = datetime.now(timezone.utc).isoformat()
    latencies = [r["latency_ms"] for r in results]
    successful = sum(1 for r in results if r["error"] is None)
    report = {
        "persona_id": persona_id,
        "persona_profile": profile,
        "test_started_at": test_started,
        "test_ended_at": test_ended,
        "questions": results,
        "summary": {
            "total_questions": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else 0,
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
        },
    }

    out_path = os.path.join(OUTPUT_DIR, f"{persona_id}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  [{persona_id}] DONE — {successful}/{len(results)} ok, avg {report['summary']['avg_latency_ms']}ms")
    return report


async def classify_topic(q_text, persona):
    """Heuristic topic classification based on keywords and persona focus."""
    topic_scores = {t: 0 for t in CONFIG["TOPICS"]}
    keywords = {
        "sales": ["close", "deal", "pipeline", "objection", "pricing", "dm", "dms", "conversation", "call", "meeting", "prospect", "buyer", "sell", "sales", "revenue", "commission", "contract"],
        "linkedin": ["linkedin", "profile", "post", "connection", "follower", "content", "engagement", "dm", "dms", "impression", "network", "headline", "reach"],
        "content": ["post", "content", "write", "video", "carousel", "engagement", "audience", "story", "storytelling", "article", "newsletter", "copy"],
        "coaching": ["coach", "coaching", "client", "program", "workshop", "group", "1:1", "session", "certification", "testimonial", "offer"],
        "business": ["business", "revenue", "pipeline", "growth", "scale", "founder", "startup", "board", "investor", "partner", "retainer", "enterprise"],
    }
    for topic, words in keywords.items():
        for w in words:
            if w in q_text.lower():
                topic_scores[topic] += 1

    # Boost by persona's topic_focus
    focus = persona.get("topic_focus", {})
    for topic in CONFIG["TOPICS"]:
        topic_scores[topic] += focus.get(topic, 0) * 3

    best = max(topic_scores, key=topic_scores.get)
    if topic_scores[best] == 0:
        return "sales"
    return best


async def run_all():
    test_run_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*70}")
    print(f"  Charlotte AI Load Test — {CONFIG['TIME_MODE'].upper()} mode")
    print(f"  Run ID: {test_run_id}")
    print(f"  {CONFIG['NUM_PERSONAS']} personas × {CONFIG['QUESTIONS_PER_PERSONA']} questions = {CONFIG['NUM_PERSONAS'] * CONFIG['QUESTIONS_PER_PERSONA']} total")
    print(f"  Interval: {CONFIG['INTERVAL_SECONDS']}s")
    print(f"  Endpoint: {CONFIG['ENDPOINT']}")
    print(f"{'='*70}\n")

    with open(os.path.join(OUTPUT_DIR, "personas.json")) as f:
        all_personas = json.load(f)

    personas = all_personas[:CONFIG["NUM_PERSONAS"]]

    first_persona_start = None
    last_persona_end = None
    start_time = time.monotonic()

    # Run all personas concurrently
    tasks = [run_persona(p) for p in personas]
    reports = await asyncio.gather(*tasks)

    total_elapsed = time.monotonic() - start_time

    # Collect concurrency evidence (overlapping timestamps)
    all_questions = []
    for r in reports:
        for q in r.get("questions", []):
            all_questions.append({
                "persona": r["persona_id"],
                "sent": q["timestamp_sent"],
                "index": q["index"],
            })
            if first_persona_start is None or q["timestamp_sent"] < first_persona_start:
                first_persona_start = q["timestamp_sent"]
            if last_persona_end is None or q["timestamp_received"] > last_persona_end:
                last_persona_end = q["timestamp_received"]

    # Build consolidated report
    all_latencies = [r["summary"]["avg_latency_ms"] for r in reports]
    all_individual_latencies = []
    errors_by_type = {"timeout": 0, "5xx": 0, "rate_limited": 0, "other": 0}
    latency_by_topic = {t: [] for t in CONFIG["TOPICS"]}
    total_sent = 0
    total_success = 0
    failed_personas = []

    for r in reports:
        for q in r["questions"]:
            total_sent += 1
            all_individual_latencies.append(q["latency_ms"])
            if q["error"]:
                if q["error"] == "timeout":
                    errors_by_type["timeout"] += 1
                elif str(q["http_status"]).startswith("5"):
                    errors_by_type["5xx"] += 1
                elif q["http_status"] == 429:
                    errors_by_type["rate_limited"] += 1
                else:
                    errors_by_type["other"] += 1
            else:
                total_success += 1
                # Topic latency
                t = q.get("topic", "sales")
                if t in latency_by_topic:
                    latency_by_topic[t].append(q["latency_ms"])

        if r["summary"]["failed"] > 0:
            failed_personas.append({
                "persona_id": r["persona_id"],
                "failed": r["summary"]["failed"],
                "total": r["summary"]["total_questions"],
            })

    # Find slowest persona
    slowest = max(reports, key=lambda r: r["summary"]["avg_latency_ms"])
    fastest = min(reports, key=lambda r: r["summary"]["avg_latency_ms"])

    # Compute p50/p95
    sorted_lats = sorted(all_individual_latencies)
    p50 = sorted_lats[len(sorted_lats) // 2] if sorted_lats else 0
    p95_idx = int(len(sorted_lats) * 0.95)
    p95 = sorted_lats[p95_idx] if sorted_lats else 0

    # Determine if concurrency was observed
    concurrency_desc = (
        f"All {CONFIG['NUM_PERSONAS']} personas started within a shared window. "
        f"First request: {first_persona_start}, last: {last_persona_end}. "
        f"Total wall-clock: {total_elapsed:.1f}s."
    )

    consolidated = {
        "test_run_id": test_run_id,
        "config": CONFIG,
        "total_personas": CONFIG["NUM_PERSONAS"],
        "total_questions_sent": total_sent,
        "overall_success_rate": round(total_success / total_sent, 4) if total_sent else 0,
        "overall_avg_latency_ms": int(sum(all_individual_latencies) / len(all_individual_latencies)) if all_individual_latencies else 0,
        "overall_p50_latency_ms": p50,
        "overall_p95_latency_ms": p95,
        "overall_max_latency_ms": max(all_individual_latencies) if all_individual_latencies else 0,
        "overall_min_latency_ms": min(all_individual_latencies) if all_individual_latencies else 0,
        "errors_by_type": errors_by_type,
        "latency_by_topic": {
            t: {
                "avg_ms": int(sum(v) / len(v)) if v else 0,
                "count": len(v),
            }
            for t, v in latency_by_topic.items()
        },
        "concurrency_observed": concurrency_desc,
        "total_wall_clock_seconds": round(total_elapsed, 1),
        "fastest_persona": {
            "persona_id": fastest["persona_id"],
            "avg_latency_ms": fastest["summary"]["avg_latency_ms"],
            "name": next(p["name"] for p in personas if p["id"] == fastest["persona_id"]),
        },
        "slowest_persona": {
            "persona_id": slowest["persona_id"],
            "avg_latency_ms": slowest["summary"]["avg_latency_ms"],
            "name": next(p["name"] for p in personas if p["id"] == slowest["persona_id"]),
        },
        "any_failed_personas": failed_personas,
        "raw_file_index": sorted([f"{p['id']}.json" for p in personas]),
    }

    out_path = os.path.join(OUTPUT_DIR, "consolidated_report.json")
    with open(out_path, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  CONSOLIDATED REPORT — {test_run_id}")
    print(f"{'='*70}")
    print(f"  Success rate:    {consolidated['overall_success_rate']*100:.1f}% ({total_success}/{total_sent})")
    print(f"  Avg latency:     {consolidated['overall_avg_latency_ms']}ms")
    print(f"  P50 / P95:       {p50}ms / {p95}ms")
    print(f"  Wall clock:      {total_elapsed:.1f}s")
    print(f"  Errors:          {errors_by_type}")
    print(f"  Failed personas: {len(failed_personas)}")
    for fp in failed_personas:
        print(f"    {fp['persona_id']}: {fp['failed']}/{fp['total']} failed")
    print(f"  Fastest:         {consolidated['fastest_persona']['persona_id']} ({consolidated['fastest_persona']['avg_latency_ms']}ms)")
    print(f"  Slowest:         {consolidated['slowest_persona']['persona_id']} ({consolidated['slowest_persona']['avg_latency_ms']}ms)")
    print(f"  Latency by topic:")
    for t, v in consolidated["latency_by_topic"].items():
        print(f"    {t}: {v['avg_ms']}ms avg ({v['count']} queries)")
    print(f"  Report:          {out_path}")
    print(f"{'='*70}\n")

    return consolidated


if __name__ == "__main__":
    asyncio.run(run_all())
