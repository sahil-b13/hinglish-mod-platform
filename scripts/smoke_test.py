"""
End-to-end smoke test that exercises the full pipeline (classify -> campaign
cluster -> trust/rate-limit) directly against data/sample/sample_messages.jsonl,
without needing Kafka running. Useful for quickly sanity-checking the pipeline
logic, and as a fallback if you hit the Kafka/Zookeeper setup friction the plan
(§7) warns about — this confirms the pipeline stages themselves are correct
independent of the message broker.

Requires only Redis running locally (docker compose up -d redis, or a local
redis-server).

Run:
    python scripts/smoke_test.py
"""

import json
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))

import redis  # noqa: E402
from classifier import get_classifier  # noqa: E402
from trust import TrustStore  # noqa: E402
from simhash_cluster import CampaignDetector  # noqa: E402

SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_messages.jsonl"


def main():
    try:
        r = redis.Redis(host="localhost", port=6380, decode_responses=True)
        r.ping()
    except redis.exceptions.ConnectionError:
        print("Could not connect to Redis on localhost:6380.")
        print("Start it with: docker compose up -d redis   (or a local redis-server)")
        sys.exit(1)

    r.flushdb()
    print("Redis connected and flushed for a clean run.\n")

    classifier = get_classifier()
    trust_store = TrustStore(r)
    campaign_detector = CampaignDetector()

    counts = {"published": 0, "flagged": 0, "held_for_review": 0}
    campaigns_seen = set()

    with open(SAMPLE_PATH) as f:
        for line in f:
            event = json.loads(line)
            text = event["message"]

            verdict = classifier.predict(text)
            message_id = str(uuid.uuid4())
            campaign_id = campaign_detector.add_message(
                message_id, event["user_id"], event["channel_id"], text
            )
            trust_result = trust_store.process_message(event["user_id"], verdict["label"])

            if verdict["label"] == "severe":
                status = "held_for_review"
            elif verdict["label"] == "abusive" or trust_result.rate_limited:
                status = "flagged"
            else:
                status = "published"

            counts[status] += 1
            if campaign_id:
                campaigns_seen.add(campaign_id)

            tag = f" [{campaign_id}]" if campaign_id else ""
            print(f"{status.upper():16s} | {verdict['label']:8s} "
                  f"| trust={trust_result.trust:5.2f}{tag} | {text[:55]}")

    print("\n--- Summary ---")
    print(f"Published: {counts['published']}  Flagged: {counts['flagged']}  "
          f"Held for review: {counts['held_for_review']}")
    print(f"Campaigns detected: {len(campaigns_seen)}")

    assert len(campaigns_seen) >= 1, "expected at least one campaign to be detected in sample data"
    assert counts["held_for_review"] >= 1, "expected at least one severe message to be held"
    print("\nSmoke test passed: pipeline stages are wired correctly.")


if __name__ == "__main__":
    main()
