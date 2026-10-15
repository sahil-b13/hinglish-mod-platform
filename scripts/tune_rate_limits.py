"""
Rate-limit / velocity threshold tuning helper.

The plan (§6/§7) flags rate-limit numbers as needing empirical tuning against
real behavior, same as the SimHash threshold. This script simulates a few
realistic user behavior patterns (normal chatter, active-but-legitimate user,
burst spammer, slow-drip spammer) and computes, for a range of candidate
velocity limits, whether each pattern would trip the rate limiter — so
DEFAULT_VELOCITY_LIMIT in consumer/trust.py is backed by numbers, not a guess.

This works offline against simulated timestamps (no real sleeping, no Redis
needed) — it directly re-implements the sliding-window counting logic from
consumer/trust.py against synthetic message-arrival times, which is equivalent
to what Redis's ZADD/ZREMRANGEBYSCORE/ZCARD sequence does in production but
runs instantly instead of taking real wall-clock time.

Run:
    python scripts/tune_rate_limits.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))
from trust import WINDOW_SECONDS  # noqa: E402


# Simulated behavior patterns: list of inter-message delays in seconds.
# e.g. [8, 8, 8, 8, 8] = 5 messages, each 8 seconds after the previous one.
PATTERNS = {
    "casual_chatter": [8] * 8,               # ~1 msg every 8s -- relaxed conversation
    "active_legit_user": [3] * 15,           # engaged user in a fast-moving discussion
    "burst_spammer": [0.3] * 25,             # rapid-fire spam burst
    "slow_drip_spammer": [5] * 20,           # spammer pacing just under naive per-message limits
    "flagged_then_quiet": [2, 2, 2] + [15] * 5,  # a few rapid msgs then settles down
}

SHOULD_TRIP = {"burst_spammer"}
SHOULD_NOT_TRIP = {"casual_chatter", "active_legit_user"}


def max_sliding_window_count(delays: list, window_seconds: int) -> int:
    """
    Given inter-message delays, compute the maximum number of messages that ever
    fall within any `window_seconds`-wide sliding window — this is exactly what
    consumer/trust.py's velocity counter measures via Redis ZCARD after
    ZREMRANGEBYSCORE trims anything older than the window.
    """
    timestamps = [0.0]
    for d in delays:
        timestamps.append(timestamps[-1] + d)

    max_count = 0
    for t in timestamps:
        window_start = t - window_seconds
        count = sum(1 for ts in timestamps if window_start < ts <= t)
        max_count = max(max_count, count)
    return max_count


def main():
    candidate_limits = [5, 8, 10, 12, 15, 20, 25]

    peak_velocities = {name: max_sliding_window_count(delays, WINDOW_SECONDS)
                        for name, delays in PATTERNS.items()}

    print(f"Sliding window: {WINDOW_SECONDS}s\n")
    print("Peak velocity (max messages in any window) per pattern:")
    for name, peak in peak_velocities.items():
        print(f"  {name:22s}: {peak} msgs/{WINDOW_SECONDS}s")
    print()

    print(f"{'limit':>6} | " + " | ".join(f"{name[:16]:>16s}" for name in PATTERNS))
    print("-" * (9 + 19 * len(PATTERNS)))

    results_by_limit = {}
    for limit in candidate_limits:
        cells = []
        any_fp = False
        any_miss = False
        for name, peak in peak_velocities.items():
            tripped = peak > limit
            marker = ""
            if tripped and name in SHOULD_NOT_TRIP:
                marker, any_fp = " FP!", True
            elif not tripped and name in SHOULD_TRIP:
                marker, any_miss = " MISS!", True
            cells.append(f"{'TRIP' if tripped else 'ok'}{marker}")
        results_by_limit[limit] = (any_fp, any_miss)
        print(f"{limit:>6} | " + " | ".join(f"{c:>16s}" for c in cells))

    print()
    clean = [l for l, (fp, miss) in results_by_limit.items() if not fp and not miss]
    if clean:
        suggested = clean[len(clean) // 2]
        print(f"Limits with no false positives and no misses on this pattern set: {clean}")
        print(f"Suggested DEFAULT_VELOCITY_LIMIT: {suggested} "
              f"(current value in consumer/trust.py: check DEFAULT_VELOCITY_LIMIT)")
    else:
        print("No candidate limit avoided both false positives and misses on this pattern "
              "set — casual_chatter/active_legit_user and burst_spammer peak velocities "
              "are too close together. This is a real signal: pure velocity may not be "
              "enough to separate your actual users from spammers; consider combining "
              "with the SimHash campaign signal (a burst that's also textually repetitive "
              "is a much stronger spam signal than velocity alone).")

    print("\nslow_drip_spammer and flagged_then_quiet are shown for reference but not")
    print("scored against SHOULD_TRIP/SHOULD_NOT_TRIP — a slow-drip spammer staying under")
    print("velocity limits is a known gap in rate-limiting alone (see plan §7, adversarial")
    print("evasion) and is exactly what the SimHash campaign layer is meant to catch instead.")
    print("\nRe-run this with delay patterns from YOUR actual injected spam once you have")
    print("it (Week 6) rather than trusting these illustrative numbers.")


if __name__ == "__main__":
    main()
