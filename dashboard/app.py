"""
Live moderation dashboard. Reads from the same Redis instance the consumer writes
to (moderation:recent_verdicts, moderation:campaigns) and auto-refreshes.

Includes a moderator review panel (confirm/overturn) that feeds the active
learning loop — see consumer/feedback.py and scripts/export_feedback_queue.py.

Run: streamlit run dashboard/app.py
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd
import redis
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))
from feedback import FeedbackStore  # noqa: E402

RESULTS_KEY = "moderation:recent_verdicts"
CAMPAIGNS_KEY = "moderation:campaigns"
LABELS = ["safe", "abusive", "severe"]
REVIEW_PANEL_LIMIT = 10

st.set_page_config(page_title="Hinglish Moderation Dashboard", layout="wide")


@st.cache_resource
def get_redis():
    return redis.Redis(host="localhost", port=6380, decode_responses=True)


def load_verdicts(r, limit=200):
    raw = r.lrange(RESULTS_KEY, 0, limit - 1)
    return [json.loads(x) for x in raw]


def load_campaigns(r):
    raw = r.hgetall(CAMPAIGNS_KEY)
    return [json.loads(v) for v in raw.values()]


def main():
    r = get_redis()

    st.title("🛡️ Hinglish Moderation Platform — Live Dashboard")
    st.caption(
        "Classifier is currently the placeholder rule-based scorer "
        "(consumer/classifier.py) until a fine-tuned model is plugged in — see README."
    )

    verdicts = load_verdicts(r)
    campaigns = load_campaigns(r)

    if not verdicts:
        st.info("No messages seen yet. Start the producer and consumer, then refresh.")
        st.stop()

    df = pd.DataFrame(verdicts)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Messages seen", len(df))
    col2.metric("Held for review", int((df["final_status"] == "held_for_review").sum()))
    col3.metric("Flagged", int((df["final_status"] == "flagged").sum()))
    col4.metric("Active campaigns", len(campaigns))

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Live feed")
        status_filter = st.multiselect(
            "Filter by status",
            options=sorted(df["final_status"].unique()),
            default=list(df["final_status"].unique()),
        )
        filtered = df[df["final_status"].isin(status_filter)]

        def highlight(row):
            color = {
                "held_for_review": "background-color: #ffcccc",
                "flagged": "background-color: #fff3cd",
                "published": "",
            }.get(row["final_status"], "")
            return [color] * len(row)

        display_cols = ["timestamp", "user_id", "channel_id", "text", "label",
                         "confidence", "trust", "final_status", "campaign_id"]
        st.dataframe(
            filtered[display_cols].style.apply(highlight, axis=1),
            use_container_width=True,
            height=500,
        )

    with right:
        st.subheader("User trust scores")
        trust_view = (
            df.sort_values("timestamp", ascending=False)
            .drop_duplicates("user_id")[["user_id", "trust", "velocity", "rate_limited"]]
            .sort_values("trust")
        )
        st.dataframe(trust_view, use_container_width=True, height=250)

        st.subheader("Campaign clusters")
        if campaigns:
            for c in sorted(campaigns, key=lambda x: -x["message_count"]):
                st.markdown(
                    f"**{c['campaign_id']}** — {c['message_count']} messages, "
                    f"{c['user_count']} users, {c['channel_count']} channels, "
                    f"{c['window_seconds']}s window\n\n"
                    f"> {c['sample_text'][:80]}"
                )
        else:
            st.write("No campaigns detected yet.")

    st.divider()

    st.subheader("🧑‍⚖️ Moderator review (active learning loop)")
    st.caption(
        "Confirm or overturn verdicts on flagged/held messages. Overturns feed "
        "scripts/export_feedback_queue.py for the next retraining cycle (plan §9)."
    )
    feedback_store = FeedbackStore(r)
    review_candidates = df[df["final_status"].isin(["flagged", "held_for_review"])]
    unreviewed = review_candidates[
        ~review_candidates["message_id"].apply(feedback_store.already_reviewed)
    ]

    if unreviewed.empty:
        st.write("No unreviewed flagged/held messages right now.")
    else:
        stats = feedback_store.stats()
        st.caption(f"Reviewed so far: {stats['total_reviewed']} "
                   f"(overturn rate: {stats['overturn_rate']:.0%})")
        for _, row in unreviewed.head(REVIEW_PANEL_LIMIT).iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.write(f"**{row['label']}** ({row['confidence']}) — {row['text']}")
                    st.caption(f"user={row['user_id']} channel={row['channel_id']} "
                               f"status={row['final_status']}")
                with c2:
                    if st.button("✅ Confirm", key=f"confirm_{row['message_id']}"):
                        feedback_store.record(row["message_id"], row["text"], row["label"], "confirm")
                        st.rerun()
                    corrected = st.selectbox(
                        "Overturn to:", LABELS,
                        index=LABELS.index(row["label"]) if row["label"] in LABELS else 0,
                        key=f"select_{row['message_id']}",
                        label_visibility="collapsed",
                    )
                    if st.button("↩️ Overturn", key=f"overturn_{row['message_id']}"):
                        feedback_store.record(row["message_id"], row["text"], row["label"],
                                               "overturn", corrected)
                        st.rerun()

    st.divider()
    if st.button("🔄 Refresh"):
        st.rerun()


if __name__ == "__main__":
    main()
