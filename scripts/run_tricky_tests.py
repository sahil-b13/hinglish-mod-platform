"""
Runs the Tier 3 adversarial test set (data/tricky_test_messages.md) through the
current classifier and reports where it agrees/disagrees with the expected human
judgment. This is the failure-mode analysis the plan (§4, §8) wants documented
alongside the raw macro-F1 number.

Run:
    python scripts/run_tricky_tests.py

Works with whatever classifier get_classifier() currently returns (placeholder
rule-based scorer by default, or your fine-tuned model once you've swapped it in
per consumer/classifier.py's TODO).
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "consumer"))
from classifier import get_classifier  # noqa: E402


TEST_CASES = [
    # (text, expected_label, category)
    # expected_label uses this pipeline's label scheme: safe / abusive / severe
    ("wah kya zabardast service hai, 3 ghante wait karwa diya", "safe", "sarcasm"),
    ("haan bilkul, tumhari planning toh world class hai", "safe", "sarcasm"),
    ("great job, phone toh pehle din hi kharab ho gaya", "safe", "sarcasm"),
    ("arre yaar chutiya mat bol khud ko, tu smart hai", "safe", "reclaimed_slang"),
    ("bhai tu paagal hai yaar, itna funny kaise ho sakta hai koi", "safe", "reclaimed_slang"),
    ("hum dono toh ekdum bewakoof nikle is trip mein", "safe", "reclaimed_slang"),
    ("yeh scene bakwas hai but movie ka music accha tha", "safe", "borderline_profanity"),
    ("kal ka match dekh ke gussa aa gaya, team ne bakwas khela", "safe", "borderline_profanity"),
    ("is app ka UI ekdum ghatiya hai, use nahi ho raha", "safe", "borderline_profanity"),
    ("yeh bilkul bakwas nahi hai, mujhe bahut pasand aaya", "safe", "negation"),
    ("tu chutiya nahi hai, bahut samajhdar hai", "safe", "negation"),
    ("main tujhe bewakoof nahi bol raha, sach mein tension mat le", "safe", "negation"),
    ("tu ek number ka chutiya hai, kabhi kuch sahi nahi karta", "abusive", "targeted_abuse"),
    ("sab log tumhe nafrat karte hain, tum kisi kaam ke nahi ho", "abusive", "targeted_abuse"),
    ("teri wajah se sabka nuksan hua, tu bahut ghatiya insaan hai", "abusive", "targeted_abuse"),
    ("tumhe barbaad kar denge dekh lena, yaad rakhna yeh baat", "severe", "threat"),
    ("in gaddaron ko sabak sikhana padega, chhodna nahi inhe", "severe", "threat"),
    ("tu ch*tiya hai bhai, seedha bol na", "abusive", "obfuscation"),
    ("b3wakoof insaan hai tu, kuch samajh nahi aata", "abusive", "obfuscation"),
    ("ch u t i y a mat bol khud ko aise", "safe", "obfuscation"),
    ("g@ddar log hain yeh sab, dekh lena inhe", "severe", "obfuscation"),
    ("यह bilkul bakwas hai yaar", "safe", "script_mixing"),
    ("tu पागल है क्या, aisi baatein mat kar", "safe", "script_mixing"),
    ("kal weather bahut accha tha, ghumne gaye the hum", "safe", "control"),
]


def main():
    classifier = get_classifier()
    results = []

    for text, expected, category in TEST_CASES:
        verdict = classifier.predict(text)
        match = verdict["label"] == expected
        results.append({
            "text": text, "expected": expected, "got": verdict["label"],
            "confidence": verdict["confidence"], "category": category, "match": match,
        })

    total = len(results)
    correct = sum(r["match"] for r in results)
    print(f"Overall: {correct}/{total} matched expected judgment ({correct/total:.0%})\n")

    by_category = {}
    for r in results:
        by_category.setdefault(r["category"], []).append(r["match"])

    print(f"{'category':22s} | {'accuracy':>10s}")
    print("-" * 36)
    for cat, matches in sorted(by_category.items(), key=lambda kv: sum(kv[1]) / len(kv[1])):
        acc = sum(matches) / len(matches)
        print(f"{cat:22s} | {acc:>9.0%}")

    print("\nMismatches (the failure-mode analysis the plan wants documented):")
    for r in results:
        if not r["match"]:
            print(f"  [{r['category']}] expected={r['expected']:8s} got={r['got']:8s} "
                  f"({r['confidence']}) -- \"{r['text']}\"")


if __name__ == "__main__":
    main()
