"""
Simulated chat-stream producer.

Reads messages from a JSONL file (default: synthetic sample data) and publishes
them to a Kafka/Redpanda topic, one at a time, with a small delay to mimic a live
stream. Once you have real Bohra/PRISM data downloaded, point --source at that file
instead — no other code changes needed as long as it has the same JSON shape:

    {"user_id": ..., "channel_id": ..., "message": ...}

Timestamps are stamped at publish time so downstream velocity/rate-limit logic
sees realistic gaps.
"""

import argparse
import json
import time
import random
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaProducer

TOPIC = "chat-events"


def load_messages(path: Path):
    messages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            messages.append(json.loads(line))
    return messages


def make_producer(bootstrap_servers: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


def run(source: Path, bootstrap_servers: str, min_delay: float, max_delay: float, loop: bool):
    messages = load_messages(source)
    if not messages:
        raise SystemExit(f"No messages found in {source}")

    producer = make_producer(bootstrap_servers)
    print(f"Loaded {len(messages)} messages from {source}. Publishing to '{TOPIC}'...")

    sent = 0
    try:
        while True:
            for msg in messages:
                event = {
                    "user_id": msg["user_id"],
                    "channel_id": msg["channel_id"],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "message": msg["message"],
                }
                # partition by channel so per-channel ordering is preserved
                producer.send(TOPIC, key=event["channel_id"], value=event)
                sent += 1
                print(f"[{sent}] -> {event['channel_id']} / {event['user_id']}: {event['message'][:60]}")
                time.sleep(random.uniform(min_delay, max_delay))
            if not loop:
                break
            print("Replaying dataset from the start (--loop enabled)...")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        producer.flush()
        producer.close()
        print(f"Done. Sent {sent} messages total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated chat producer")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_messages.jsonl",
        help="JSONL file of messages to replay",
    )
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--min-delay", type=float, default=0.3, help="min seconds between messages")
    parser.add_argument("--max-delay", type=float, default=1.2, help="max seconds between messages")
    parser.add_argument("--loop", action="store_true", help="replay dataset forever")
    args = parser.parse_args()

    run(args.source, args.bootstrap_servers, args.min_delay, args.max_delay, args.loop)
