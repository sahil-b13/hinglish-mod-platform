"""
Per-user trust score + sliding-window rate limiting, backed by Redis.

Redis keys used (matches the plan's design in §4):
    user:{id}:trust      -> float, 0.0 (untrusted) to 1.0 (trusted)
    user:{id}:velocity   -> sorted set of message timestamps in the current window (TTL'd)

Trust score update rule (simple, tunable):
    - safe message      -> trust nudged up slightly
    - abusive message    -> trust nudged down
    - severe message      -> trust dropped hard
    - rate-limit trip     -> trust nudged down (spammy behavior)

These nudge amounts and the velocity threshold are exactly the kind of numbers the
plan flags as needing empirical tuning against your own injected spam (§6/§7) —
the defaults below are reasonable starting points, not final values.
"""

import time
from dataclasses import dataclass

import redis

DEFAULT_TRUST = 0.5
TRUST_MIN, TRUST_MAX = 0.0, 1.0

TRUST_DELTA = {
    "safe": +0.02,
    "abusive": -0.08,
    "severe": -0.25,
}
RATE_LIMIT_TRUST_PENALTY = -0.15

# Sliding window rate limit: more than N messages in WINDOW_SECONDS trips the limiter
WINDOW_SECONDS = 60
# Calibrated via scripts/tune_rate_limits.py against simulated behavior patterns:
# a fast-but-legitimate active user peaked at 16 msgs/60s in that simulation, so a
# naive value like 12 produces false positives. 20 is the lower end of the range
# (20-25) with zero false positives against casual/active legit users AND zero
# misses against the simulated burst-spammer pattern. Re-run that script against
# YOUR actual injected spam and real user behavior once you have it (Week 6) —
# these are illustrative simulated patterns, not measured real traffic.
DEFAULT_VELOCITY_LIMIT = 20  # msgs/min before rate-limiting kicks in
TIGHTENED_VELOCITY_LIMIT = 8  # msgs/min limit applied to low-trust users


@dataclass
class TrustResult:
    trust: float
    velocity: int
    rate_limited: bool


class TrustStore:
    def __init__(self, redis_client: "redis.Redis"):
        self.r = redis_client

    def _trust_key(self, user_id: str) -> str:
        return f"user:{user_id}:trust"

    def _velocity_key(self, user_id: str) -> str:
        return f"user:{user_id}:velocity"

    def get_trust(self, user_id: str) -> float:
        val = self.r.get(self._trust_key(user_id))
        return float(val) if val is not None else DEFAULT_TRUST

    def _set_trust(self, user_id: str, trust: float):
        trust = max(TRUST_MIN, min(TRUST_MAX, trust))
        self.r.set(self._trust_key(user_id), trust)

    def _record_and_get_velocity(self, user_id: str) -> int:
        key = self._velocity_key(user_id)
        now = time.time()
        window_start = now - WINDOW_SECONDS

        pipe = self.r.pipeline()
        pipe.zadd(key, {str(now): now})
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.expire(key, WINDOW_SECONDS * 2)
        _, _, count, _ = pipe.execute()
        return int(count)

    def process_message(self, user_id: str, verdict_label: str) -> TrustResult:
        """
        Called once per incoming message. Updates velocity, checks rate limit,
        updates trust score based on the classifier verdict, and returns the
        current state for logging/dashboard display.
        """
        trust = self.get_trust(user_id)
        velocity = self._record_and_get_velocity(user_id)

        limit = TIGHTENED_VELOCITY_LIMIT if trust < 0.3 else DEFAULT_VELOCITY_LIMIT
        rate_limited = velocity > limit

        delta = TRUST_DELTA.get(verdict_label, 0.0)
        if rate_limited:
            delta += RATE_LIMIT_TRUST_PENALTY

        new_trust = trust + delta
        self._set_trust(user_id, new_trust)

        return TrustResult(trust=round(max(TRUST_MIN, min(TRUST_MAX, new_trust)), 3),
                            velocity=velocity, rate_limited=rate_limited)
