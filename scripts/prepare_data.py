"""
Prepares a raw labeled dataset (e.g. hydrated Bohra tweets) into the train.csv /
test.csv format scripts/train.py expects, and — critically — prints the data
quality report the plan (§6) says you need to look at yourself: class balance,
duplicate rate, text length distribution, missing/empty rows.

The plan is explicit: "Label quality and class balance issues are easy to miss if
you only see aggregate stats." This script surfaces those stats up front so you
see them before you spend a training run on data with a hidden problem.

Expects a raw CSV with (at minimum) text and label columns — column names are
configurable since raw dataset exports vary. Run this AFTER you've hydrated the
Bohra tweet IDs (or downloaded PRISM) and BEFORE running scripts/train.py.

Run:
    python scripts/prepare_data.py \\
        --raw_csv data/bohra/raw_hydrated.csv \\
        --text_col tweet_text --label_col hate_label \\
        --output_dir data/bohra \\
        --test_size 0.15
"""

import argparse
from pathlib import Path

import pandas as pd


def quality_report(df: pd.DataFrame, text_col: str, label_col: str):
    print("=== Data quality report ===")
    print(f"Total rows: {len(df)}")

    n_missing_text = df[text_col].isna().sum()
    n_empty_text = (df[text_col].astype(str).str.strip() == "").sum()
    print(f"Missing text: {n_missing_text} | Empty-string text: {n_empty_text}")

    n_dupes = df[text_col].duplicated().sum()
    print(f"Duplicate text rows: {n_dupes} ({n_dupes / len(df):.1%})")

    print("\nLabel distribution:")
    dist = df[label_col].value_counts(dropna=False)
    print(dist)
    dist_pct = df[label_col].value_counts(normalize=True, dropna=False)
    max_class_share = dist_pct.max()
    if max_class_share > 0.85:
        print(f"\n[!] WARNING: majority class is {max_class_share:.0%} of the data. "
              f"Severe class imbalance — macro-F1 will likely be dragged down by the "
              f"minority class(es). Consider class weighting in scripts/train.py "
              f"(Trainer's compute_loss, or a WeightedRandomSampler) rather than "
              f"training on this as-is.")

    lengths = df[text_col].dropna().astype(str).str.split().apply(len)
    print(f"\nText length (words): min={lengths.min()}, median={lengths.median():.0f}, "
          f"max={lengths.max()}, mean={lengths.mean():.1f}")
    very_short = (lengths <= 2).sum()
    if very_short > 0:
        print(f"[!] {very_short} rows have <=2 words — check these aren't junk/empty "
              f"after cleaning (e.g. tweets that were just a URL or @mention).")

    print()


def main(args):
    df = pd.read_csv(args.raw_csv)
    if args.text_col not in df.columns or args.label_col not in df.columns:
        raise SystemExit(
            f"Expected columns '{args.text_col}' and '{args.label_col}' in {args.raw_csv}. "
            f"Found: {list(df.columns)}. Use --text_col/--label_col to match your file."
        )

    df = df[[args.text_col, args.label_col]].rename(
        columns={args.text_col: "text", args.label_col: "label"}
    )

    quality_report(df, "text", "label")

    before = len(df)
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.drop_duplicates(subset=["text"])
    after = len(df)
    print(f"Dropped {before - after} rows (missing/empty/duplicate). {after} rows remain.")

    if args.label_map:
        # e.g. --label_map "hateful:2,offensive:1,none:0"
        mapping = dict(pair.split(":") for pair in args.label_map.split(","))
        mapping = {k.strip(): int(v) for k, v in mapping.items()}
        unmapped = set(df["label"].astype(str).unique()) - set(mapping.keys())
        if unmapped:
            raise SystemExit(f"--label_map doesn't cover these label values: {unmapped}")
        df["label"] = df["label"].astype(str).map(mapping)

    # Stratified split so rare classes end up in both train and test
    from sklearn.model_selection import train_test_split
    try:
        train_df, test_df = train_test_split(
            df, test_size=args.test_size, random_state=42, stratify=df["label"]
        )
    except ValueError as e:
        print(f"\n[!] Stratified split failed ({e}). This usually means a class has "
              f"too few examples after deduplication — check the label distribution "
              f"above closely, this is a real data problem worth investigating, not "
              f"just a script bug. Falling back to a non-stratified split so you can "
              f"at least inspect the data; do not treat metrics from this split as "
              f"reliable until you've addressed the class imbalance.")
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print(f"\nWrote {len(train_df)} train rows and {len(test_df)} test rows to {output_dir}/")
    print("Next: python scripts/model_bakeoff.py --data_dir", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and quality-check raw labeled data")
    parser.add_argument("--raw_csv", required=True)
    parser.add_argument("--text_col", default="text")
    parser.add_argument("--label_col", default="label")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--label_map", default=None,
                         help='e.g. "hateful:2,offensive:1,none:0" if labels are strings')
    args = parser.parse_args()
    main(args)
