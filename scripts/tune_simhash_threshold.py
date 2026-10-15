"""
Threshold-tuning helper for the SimHash campaign detector.

The plan is explicit that HAMMING_THRESHOLD (in consumer/simhash_cluster.py) needs
empirical tuning against real injected spam, not a value copied from a tutorial.
This script does that tuning: it takes labeled pairs (same-campaign vs
different-campaign) and reports, for each candidate threshold, how many
same-campaign pairs it would correctly merge and how many different-campaign
pairs it would incorrectly merge (false merges) — so you can pick a threshold with
real precision/recall numbers instead of a guess.

Comes pre-loaded with the labeled pairs implied by data/sample/sample_messages.jsonl
(the 6-message spam block vs. the 19 unrelated messages). Once you have real
injected spam variants (Week 6, per plan), replace LABELED_PAIRS / regenerate it
from your own labeled data — see build_pairs_from_groups() below for the helper.

Run:
    python scripts/tune_simhash_threshold.py
"""

import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))
from simhash_cluster import fingerprint  # noqa: E402


# Ground truth: messages that SHOULD end up in the same campaign cluster.
SPAM_CAMPAIGN = [
    "FREE PAISA jeeto abhi, click karo yahan http://bit.ly/xyz1",
    "FREE PAISA jeeto abhi, click karo yahan http://bit.ly/xyz2",
    "FREE PAISA jeeto turant, click karo yahan http://bit.ly/xyz3",
    "FREE PAISA abhi jeeto, click yahan karo http://bit.ly/xyz4",
    "muft paisa jeeto abhi, yahan click karo http://bit.ly/xyz5",
    "FREE PAISA jeeto abhi, yahan click karo http://bit.ly/xyz6",
]

# Ground truth: messages that should NOT be clustered with the spam campaign or
# with each other (pulled from the non-spam lines in data/sample/sample_messages.jsonl).
UNRELATED = [
    "yaar aaj match kitna accha tha, India ne jeet liya",
    "bhai yeh product bilkul sahi hai, maine try kiya",
    "kal weather kaisa rahega, koi update hai kya",
    "yeh movie dekhne layak hai, bahut acting achhi thi",
    "tu bilkul bewakoof hai, kuch samajh nahi aata tujhe",
    "chal ab bakwas band kar, kaam ki baat kar",
    "aaj office mein bahut kaam tha, thak gaya hoon",
    "tera dimag kharab hai kya, itni ghatiya baat mat kar",
    "yeh restaurant ka khana bahut tasty tha, must try",
    "kal exam hai mera, thoda tension ho raha hai",
    "chutiya banaya sabko is scheme ne, sab paisa doob gaya",
    "salo tum sab gaddar ho, tumhe barbaad kar denge dekh lena",
]


def build_pairs_from_groups(groups: list[list[str]], unrelated: list[str]):
    """
    Helper for building labeled pairs from your own data once you have real
    injected spam campaigns. `groups` is a list of campaigns, each a list of
    message texts that belong together; `unrelated` is a flat pool of messages
    that shouldn't match anything or each other.
    """
    same_pairs = []
    for group in groups:
        same_pairs.extend(combinations(group, 2))

    all_msgs = [m for g in groups for m in g] + unrelated
    diff_pairs = []
    for a, b in combinations(all_msgs, 2):
        is_same_group = any(a in g and b in g for g in groups)
        if not is_same_group:
            diff_pairs.append((a, b))
    return same_pairs, diff_pairs


def hamming(a: str, b: str) -> int:
    return bin(fingerprint(a) ^ fingerprint(b)).count("1")


def main():
    same_pairs, diff_pairs = build_pairs_from_groups([SPAM_CAMPAIGN], UNRELATED)

    same_distances = [hamming(a, b) for a, b in same_pairs]
    diff_distances = [hamming(a, b) for a, b in diff_pairs]

    print(f"Same-campaign pairs: {len(same_pairs)} | "
          f"distance range {min(same_distances)}-{max(same_distances)}")
    print(f"Different-campaign pairs: {len(diff_pairs)} | "
          f"distance range {min(diff_distances)}-{max(diff_distances)}")
    print()

    print(f"{'threshold':>9} | {'recall (same merged)':>22} | {'false merges (diff)':>20}")
    print("-" * 60)
    best_threshold, best_score = None, -1
    for t in range(0, 33):
        recall = sum(1 for d in same_distances if d <= t) / len(same_distances)
        false_merges = sum(1 for d in diff_distances if d <= t)
        score = recall - (false_merges / len(diff_distances))
        marker = ""
        if score > best_score:
            best_score, best_threshold = score, t
            marker = "  <-- best so far"
        print(f"{t:>9} | {recall:>21.0%} | {false_merges:>20d}{marker}")

    print()
    print(f"Suggested threshold: {best_threshold} "
          f"(current value in consumer/simhash_cluster.py: check HAMMING_THRESHOLD)")
    print("Re-run this with real injected spam data before trusting it for anything "
          "beyond this synthetic sample set.")


if __name__ == "__main__":
    main()
