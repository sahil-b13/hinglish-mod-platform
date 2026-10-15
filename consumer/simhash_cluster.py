"""
Coordination layer: detects when many "different" messages are near-duplicates of
each other (a coordinated spam/hate campaign), using SimHash fingerprints + a
Union-Find (disjoint-set) structure to group messages into campaign clusters.

How it works:
    1. Each incoming message is fingerprinted with SimHash (64-bit).
    2. We compare its fingerprint against recent fingerprints (kept in Redis,
       windowed) using Hamming distance.
    3. If the Hamming distance to any recent message is <= HAMMING_THRESHOLD,
       the two messages are unioned into the same campaign cluster.
    4. Clusters with >= MIN_CLUSTER_SIZE messages are surfaced as "campaigns."

HAMMING_THRESHOLD is exactly the kind of number the plan (§6/§7) says needs
empirical tuning against your own injected spam variants — 3 is a reasonable
starting point for 64-bit SimHash (commonly cited range is 3-10 depending on
string length), not a final answer. Tune it by watching what does/doesn't get
grouped in your synthetic spam-injection tests (README/plan Tier 3 testing).
"""

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from simhash import Simhash

# Calibrated empirically against the synthetic spam-variant set in
# data/sample/sample_messages.jsonl: near-duplicate spam pairs land <=13 bits apart
# on 64-bit fingerprints, unrelated messages land >=18 bits apart. Run
# `python scripts/tune_simhash_threshold.py` to see the full sweep and re-derive
# this number — it prints recall/false-merge counts per threshold. 15 sits in the
# middle of the zero-false-merge zone (13-17) with margin on both sides.
# This is exactly the kind of number the plan (§6/§7) flags as needing empirical
# tuning against YOUR actual injected spam once you have more/real data — re-run
# scripts/tune_simhash_threshold.py with real labeled pairs before trusting this
# beyond the synthetic sample set.
HAMMING_THRESHOLD = 15
MIN_CLUSTER_SIZE = 3
RECENT_WINDOW_SECONDS = 600  # only compare against fingerprints from the last 10 min


def normalize_for_fingerprint(text: str) -> str:
    """Strip URLs/numbers/punctuation noise so near-duplicate spam with varying
    links or trailing digits still fingerprints as similar."""
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"[^a-z\s]", " ", t)  # keep roman-script alpha only for fingerprinting
    t = re.sub(r"\s+", " ", t).strip()
    return t


def fingerprint(text: str) -> int:
    """
    Word-level (unigram) SimHash. Using whole words as features — rather than the
    `simhash` library's default character 4-grams — makes this robust to spam
    templates that reorder a couple of words to evade filters (e.g. "FREE PAISA
    jeeto abhi" vs "FREE PAISA abhi jeeto"), while character n-grams over the
    concatenated string are highly position-sensitive and treat reordered text as
    almost completely different.
    """
    normalized = normalize_for_fingerprint(text)
    words = normalized.split()
    if not words:
        return 0
    return Simhash(words).value


class UnionFind:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


@dataclass
class ClusterMember:
    message_id: str
    user_id: str
    channel_id: str
    text: str
    fp: int
    timestamp: float


@dataclass
class CampaignDetector:
    """
    In-memory campaign detector. For the MVP this keeps recent fingerprints in a
    local list (fine for the data volumes in this project's scope, per plan §7's
    "not load-tested at real platform scale" honesty note). A production version
    would store fingerprints in Redis (recent_hashes:{channel}, per plan §4) with
    an LSH index instead of a linear scan.
    """
    uf: UnionFind = field(default_factory=UnionFind)
    members: Dict[str, ClusterMember] = field(default_factory=dict)

    def _prune_old(self):
        cutoff = time.time() - RECENT_WINDOW_SECONDS
        stale = [mid for mid, m in self.members.items() if m.timestamp < cutoff]
        for mid in stale:
            del self.members[mid]

    def add_message(self, message_id: str, user_id: str, channel_id: str, text: str) -> Optional[str]:
        """
        Fingerprints and clusters a new message. Returns a campaign_id if the
        message is part of a cluster of size >= MIN_CLUSTER_SIZE, else None.
        """
        self._prune_old()
        fp = fingerprint(text)
        now = time.time()

        member = ClusterMember(message_id, user_id, channel_id, text, fp, now)
        self.members[message_id] = member

        for other_id, other in list(self.members.items()):
            if other_id == message_id:
                continue
            distance = bin(fp ^ other.fp).count("1")
            if distance <= HAMMING_THRESHOLD:
                self.uf.union(message_id, other_id)

        root = self.uf.find(message_id)
        cluster_ids = [mid for mid in self.members if self.uf.find(mid) == root]

        if len(cluster_ids) >= MIN_CLUSTER_SIZE:
            return f"C-{root[:8]}"
        return None

    def cluster_summary(self, campaign_id: str) -> Dict:
        root_suffix = campaign_id.replace("C-", "")
        cluster_ids = [mid for mid in self.members
                       if self.uf.find(mid).startswith(root_suffix)]
        cluster_members = [self.members[mid] for mid in cluster_ids]
        users = {m.user_id for m in cluster_members}
        channels = {m.channel_id for m in cluster_members}
        timestamps = [m.timestamp for m in cluster_members]
        return {
            "campaign_id": campaign_id,
            "message_count": len(cluster_members),
            "user_count": len(users),
            "channel_count": len(channels),
            "window_seconds": round(max(timestamps) - min(timestamps), 1) if timestamps else 0,
            "sample_text": cluster_members[0].text if cluster_members else "",
        }
