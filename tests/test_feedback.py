"""
Tests for consumer/feedback.py. Requires Redis running locally (same as
scripts/smoke_test.py) — run with:
    redis-server --daemonize yes  (or docker compose up -d redis)
    python tests/test_feedback.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))

import redis
from feedback import FeedbackStore


def get_test_store():
    r = redis.Redis(host="localhost", port=6380, decode_responses=True)
    r.flushdb()
    return FeedbackStore(r)


def test_confirm_keeps_original_label():
    store = get_test_store()
    rec = store.record("m1", "some text", "abusive", "confirm")
    assert rec.corrected_label == "abusive"
    assert rec.decision == "confirm"


def test_overturn_uses_corrected_label():
    store = get_test_store()
    rec = store.record("m2", "some text", "abusive", "overturn", corrected_label="safe")
    assert rec.corrected_label == "safe"
    assert rec.decision == "overturn"


def test_already_reviewed():
    store = get_test_store()
    assert store.already_reviewed("m3") is False
    store.record("m3", "text", "safe", "confirm")
    assert store.already_reviewed("m3") is True


def test_stats_overturn_rate():
    store = get_test_store()
    store.record("m1", "t1", "abusive", "confirm")
    store.record("m2", "t2", "abusive", "overturn", "safe")
    store.record("m3", "t3", "severe", "confirm")
    store.record("m4", "t4", "abusive", "overturn", "safe")
    stats = store.stats()
    assert stats["total_reviewed"] == 4
    assert stats["overturns"] == 2
    assert stats["overturn_rate"] == 0.5


def test_invalid_decision_raises():
    store = get_test_store()
    try:
        store.record("m1", "text", "safe", "maybe")
        assert False, "expected ValueError for invalid decision"
    except ValueError:
        pass


if __name__ == "__main__":
    test_confirm_keeps_original_label()
    test_overturn_uses_corrected_label()
    test_already_reviewed()
    test_stats_overturn_rate()
    test_invalid_decision_raises()
    print("All feedback tests passed.")
