"""
Main pipeline consumer. Reads chat events from Kafka, runs them through:

    Stage 1: normalize text
    Stage 2: classifier -> abuse probability (RuleBasedClassifier for now)
    Stage 3: SimHash fingerprint -> campaign clustering
    Stage 4: Redis trust score read/update + rate-limit check

...then writes a combined verdict record to Redis (for the dashboard to read) and
prints it to the console.

Run: python consumer/consumer.py
"""

import argparse
import json
import re
import time
import uuid
from datetime import datetime

import redis
from kafka import KafkaConsumer

from classifier import get_classifier
from trust import TrustStore
from simhash_cluster import CampaignDetector

TOPIC = "chat-events"
RESULTS_KEY = "moderation:recent_verdicts"  # Redis list, capped, read by dashboard
CAMPAIGNS_KEY = "moderation:campaigns"      # Redis hash: campaign_id -> summary JSON
MAX_RECENT_VERDICTS = 500


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def decide_final_status(label: str, rate_limited: bool) -> str:
    if label == "severe":
        return "held_for_review"
    if label == "abusive" or rate_limited:
        return "flagged"
    return "published"


def run(bootstrap_servers: str, redis_host: str, redis_port: int):
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=True,
        group_id="moderation-pipeline",
    )
    r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    classifier = get_classifier()
    trust_store = TrustStore(r)
    campaign_detector = CampaignDetector()

    print(f"Consumer listening on topic '{TOPIC}'... (Ctrl+C to stop)")

    for msg in consumer:
        event = msg.value
        message_id = str(uuid.uuid4())

        # Stage 1: normalize
        text = normalize_text(event["message"])

        # Stage 2: classify
        verdict = classifier.predict(text)

        # Stage 3: campaign clustering
        campaign_id = campaign_detector.add_message(
            message_id, event["user_id"], event["channel_id"], text
        )

        # Stage 4: trust + rate limit
        trust_result = trust_store.process_message(event["user_id"], verdict["label"])

        final_status = decide_final_status(verdict["label"], trust_result.rate_limited)

        record = {
            "message_id": message_id,
            "user_id": event["user_id"],
            "channel_id": event["channel_id"],
            "timestamp": event.get("timestamp", datetime.utcnow().isoformat()),
            "text": text,
            "label": verdict["label"],
            "confidence": verdict["confidence"],
            "trust": trust_result.trust,
            "velocity": trust_result.velocity,
            "rate_limited": trust_result.rate_limited,
            "campaign_id": campaign_id,
            "final_status": final_status,
        }

        r.lpush(RESULTS_KEY, json.dumps(record))
        r.ltrim(RESULTS_KEY, 0, MAX_RECENT_VERDICTS - 1)

        if campaign_id:
            summary = campaign_detector.cluster_summary(campaign_id)
            r.hset(CAMPAIGNS_KEY, campaign_id, json.dumps(summary))

        tag = f" [{campaign_id}]" if campaign_id else ""
        print(f"{final_status.upper():16s} | {verdict['label']:8s} ({verdict['confidence']}) "
              f"| trust={trust_result.trust} vel={trust_result.velocity}{tag} | {text[:60]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moderation pipeline consumer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6380)
    args = parser.parse_args()

    run(args.bootstrap_servers, args.redis_host, args.redis_port)
