import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))

from simhash_cluster import CampaignDetector, UnionFind, fingerprint


def test_union_find_basic():
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")
    assert uf.find("a") == uf.find("c")
    assert uf.find("a") != uf.find("d")


def test_near_duplicate_messages_cluster_together():
    detector = CampaignDetector()
    variants = [
        "FREE PAISA jeeto abhi, click karo yahan http://bit.ly/xyz1",
        "FREE PAISA jeeto abhi, click karo yahan http://bit.ly/xyz2",
        "FREE PAISA jeeto turant, click karo yahan http://bit.ly/xyz3",
        "FREE PAISA abhi jeeto, click yahan karo http://bit.ly/xyz4",
    ]
    campaign_ids = []
    for i, text in enumerate(variants):
        cid = detector.add_message(f"m{i}", f"u{i}", "c1", text)
        campaign_ids.append(cid)

    # last message should trigger a campaign (>= MIN_CLUSTER_SIZE=3 near-dupes)
    assert campaign_ids[-1] is not None
    summary = detector.cluster_summary(campaign_ids[-1])
    assert summary["message_count"] >= 3


def test_unrelated_messages_do_not_cluster():
    detector = CampaignDetector()
    unrelated = [
        "aaj match kitna accha tha",
        "kal exam hai mera tension ho raha hai",
        "yeh restaurant ka khana bahut tasty tha",
    ]
    campaign_ids = [
        detector.add_message(f"m{i}", f"u{i}", "c1", text)
        for i, text in enumerate(unrelated)
    ]
    assert all(cid is None for cid in campaign_ids)


def test_fingerprint_is_deterministic():
    text = "yeh product bilkul sahi hai"
    assert fingerprint(text) == fingerprint(text)


def test_campaign_spans_multiple_channels_and_users():
    """Mirrors the full spam block in data/sample/sample_messages.jsonl, including
    the 'muft paisa' variant (different keyword, same template) across 4 channels."""
    detector = CampaignDetector()
    variants = [
        ("u_2001", "c_101", "FREE PAISA jeeto abhi, click karo yahan http://bit.ly/xyz1"),
        ("u_2002", "c_102", "FREE PAISA jeeto abhi, click karo yahan http://bit.ly/xyz2"),
        ("u_2003", "c_103", "FREE PAISA jeeto turant, click karo yahan http://bit.ly/xyz3"),
        ("u_2004", "c_101", "FREE PAISA abhi jeeto, click yahan karo http://bit.ly/xyz4"),
        ("u_2005", "c_104", "muft paisa jeeto abhi, yahan click karo http://bit.ly/xyz5"),
        ("u_2006", "c_102", "FREE PAISA jeeto abhi, yahan click karo http://bit.ly/xyz6"),
    ]
    campaign_id = None
    for i, (user, channel, text) in enumerate(variants):
        campaign_id = detector.add_message(f"cm{i}", user, channel, text) or campaign_id

    assert campaign_id is not None
    summary = detector.cluster_summary(campaign_id)
    assert summary["user_count"] >= 4
    assert summary["channel_count"] >= 3


if __name__ == "__main__":
    test_union_find_basic()
    test_near_duplicate_messages_cluster_together()
    test_unrelated_messages_do_not_cluster()
    test_fingerprint_is_deterministic()
    test_campaign_spans_multiple_channels_and_users()
    print("All simhash_cluster tests passed.")
