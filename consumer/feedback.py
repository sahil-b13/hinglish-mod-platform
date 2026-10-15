"""
Moderator feedback store — backs the active learning loop (plan §9): moderator
confirm/overturn decisions get recorded here, and scripts/export_feedback_queue.py
turns accumulated feedback into labeled rows you can append to your training data
for the next fine-tuning cycle.

Redis key used:
    moderation:feedback  -> hash, message_id -> JSON {original_label, corrected_label,
                                                        decision, text, timestamp}
"""

import json
import time
from dataclasses import dataclass, asdict

FEEDBACK_KEY = "moderation:feedback"


@dataclass
class FeedbackRecord:
    message_id: str
    text: str
    original_label: str
    decision: str          # "confirm" or "overturn"
    corrected_label: str   # same as original_label if confirmed, new label if overturned
    timestamp: float


class FeedbackStore:
    def __init__(self, redis_client):
        self.r = redis_client

    def record(self, message_id: str, text: str, original_label: str,
               decision: str, corrected_label: str = None):
        if decision not in ("confirm", "overturn"):
            raise ValueError("decision must be 'confirm' or 'overturn'")
        final_label = corrected_label if decision == "overturn" else original_label
        record = FeedbackRecord(
            message_id=message_id,
            text=text,
            original_label=original_label,
            decision=decision,
            corrected_label=final_label,
            timestamp=time.time(),
        )
        self.r.hset(FEEDBACK_KEY, message_id, json.dumps(asdict(record)))
        return record

    def already_reviewed(self, message_id: str) -> bool:
        return self.r.hexists(FEEDBACK_KEY, message_id)

    def all_feedback(self):
        raw = self.r.hgetall(FEEDBACK_KEY)
        return [json.loads(v) for v in raw.values()]

    def stats(self):
        records = self.all_feedback()
        total = len(records)
        overturns = sum(1 for r in records if r["decision"] == "overturn")
        return {"total_reviewed": total, "overturns": overturns,
                "overturn_rate": round(overturns / total, 3) if total else 0.0}
