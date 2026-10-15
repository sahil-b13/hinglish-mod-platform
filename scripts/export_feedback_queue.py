"""
Exports accumulated moderator feedback (confirm/overturn decisions from the
dashboard) into a text,label CSV ready to append to your training data — the
active learning loop the plan (§9) calls out as a future-scope extension:
"feed moderator decisions back into a retraining queue, so the system improves
from real usage over time."

Run:
    python scripts/export_feedback_queue.py --output data/bohra/feedback_batch_1.csv

Then, once you're ready to retrain:
    cat data/bohra/train.csv data/bohra/feedback_batch_1.csv > data/bohra/train_v2.csv
    python scripts/train.py --data_dir data/bohra --model_name ... (point at train_v2.csv)

Requires Redis running locally (the same instance the dashboard writes feedback to).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))
import redis  # noqa: E402
from feedback import FeedbackStore  # noqa: E402


LABEL_TO_ID = {"safe": 0, "abusive": 1, "severe": 2}


def main(args):
    try:
        r = redis.Redis(host="localhost", port=6380, decode_responses=True)
        r.ping()
    except redis.exceptions.ConnectionError:
        print("Could not connect to Redis on localhost:6380.")
        print("Start it with: docker compose up -d redis   (or a local redis-server)")
        sys.exit(1)

    store = FeedbackStore(r)
    records = store.all_feedback()

    if not records:
        print("No moderator feedback recorded yet. Review some messages in the "
              "dashboard's 'Moderator review' panel first.")
        return

    stats = store.stats()
    print(f"Total reviewed: {stats['total_reviewed']} | "
          f"Overturns: {stats['overturns']} ({stats['overturn_rate']:.1%})")

    if stats["overturn_rate"] > 0.3:
        print("\n[!] High overturn rate (>30%) — this suggests the current classifier's "
              "verdicts disagree with moderators often enough that it's worth looking at "
              "*why* before blindly retraining on this batch (e.g. is the placeholder "
              "scorer just bad at a specific pattern? check scripts/run_tricky_tests.py-style "
              "category breakdown before assuming the fix is more data).")

    rows = []
    for rec in records:
        label = rec["corrected_label"]
        if label not in LABEL_TO_ID:
            print(f"[!] Skipping message_id={rec['message_id']}: unrecognized label '{label}'")
            continue
        rows.append({"text": rec["text"], "label": LABEL_TO_ID[label]})

    if not rows:
        print("No valid rows to export after filtering.")
        return

    df = pd.DataFrame(rows).drop_duplicates(subset=["text"])
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\nExported {len(df)} labeled rows to {output_path}")
    print("Review this file before appending to training data — moderator overturns")
    print("are valuable signal but a small batch can skew a retrain if not balanced")
    print("against your existing class distribution (see scripts/prepare_data.py's "
          "quality report for that check).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export moderator feedback for retraining")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args)
