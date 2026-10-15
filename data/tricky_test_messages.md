# Tier 3 adversarial / tricky-message test set

Per the plan §8 (Tier 3 — Synthetic load/adversarial testing), these are hand-written
"tricky" Hinglish messages meant to probe where the classifier gets it wrong:
sarcasm, reclaimed slang used non-hatefully, borderline profanity, and basic
obfuscation. Document what the classifier does with each — including the failures
— rather than only reporting the cases it gets right. This is a starting set of 24;
the plan asks for 20-30, add your own as you find more edge cases during testing.

Columns: text | expected_human_judgment | category | notes

---

## Sarcasm / irony

1. "wah kya zabardast service hai, 3 ghante wait karwa diya" — expected: mildly negative/complaint (sarcastic praise), not hateful — category: sarcasm
2. "haan bilkul, tumhari planning toh world class hai" — expected: sarcastic criticism, not abuse — category: sarcasm
3. "great job, phone toh pehle din hi kharab ho gaya" — expected: sarcastic complaint, not abuse — category: sarcasm

## Reclaimed/in-group slang used non-hatefully

4. "arre yaar chutiya mat bol khud ko, tu smart hai" — expected: friendly banter, self-deprecating, not hateful — category: reclaimed_slang
5. "bhai tu paagal hai yaar, itna funny kaise ho sakta hai koi" — expected: affectionate teasing, not abuse — category: reclaimed_slang
6. "hum dono toh ekdum bewakoof nikle is trip mein" — expected: self-deprecating humor, not abuse — category: reclaimed_slang

## Borderline profanity (context-dependent)

7. "yeh scene bakwas hai but movie ka music accha tha" — expected: mild negative opinion about content, not targeted abuse — category: borderline_profanity
8. "kal ka match dekh ke gussa aa gaya, team ne bakwas khela" — expected: frustration about performance, not hate speech — category: borderline_profanity
9. "is app ka UI ekdum ghatiya hai, use nahi ho raha" — expected: product complaint, not abuse — category: borderline_profanity

## Negation (flips polarity)

10. "yeh bilkul bakwas nahi hai, mujhe bahut pasand aaya" — expected: safe (positive, negation flips the profane word's polarity) — category: negation
11. "tu chutiya nahi hai, bahut samajhdar hai" — expected: safe (negation) — category: negation
12. "main tujhe bewakoof nahi bol raha, sach mein tension mat le" — expected: safe (negation, reassurance) — category: negation

## Targeted abuse (should be flagged)

13. "tu ek number ka chutiya hai, kabhi kuch sahi nahi karta" — expected: abusive, targeted — category: targeted_abuse
14. "sab log tumhe nafrat karte hain, tum kisi kaam ke nahi ho" — expected: abusive/hateful, targeted — category: targeted_abuse
15. "teri wajah se sabka nuksan hua, tu bahut ghatiya insaan hai" — expected: abusive, targeted — category: targeted_abuse

## Severe / threatening

16. "tumhe barbaad kar denge dekh lena, yaad rakhna yeh baat" — expected: severe, threatening language — category: threat
17. "in gaddaron ko sabak sikhana padega, chhodna nahi inhe" — expected: severe, incitement-adjacent — category: threat

## Obfuscation / leetspeak (adversarial evasion attempts)

18. "tu ch*tiya hai bhai, seedha bol na" — expected: abusive (asterisk obfuscation) — category: obfuscation
19. "b3wakoof insaan hai tu, kuch samajh nahi aata" — expected: abusive (leetspeak) — category: obfuscation
20. "ch u t i y a mat bol khud ko aise" — expected: ambiguous — spaced-out slur, likely self-deprecating per context — category: obfuscation
21. "g@ddar log hain yeh sab, dekh lena inhe" — expected: severe/threatening (leetspeak) — category: obfuscation

## Code-switching / Devanagari-Roman mixing

22. "यह bilkul bakwas hai yaar" — expected: mild negative (mixed script) — category: script_mixing
23. "tu पागल है क्या, aisi baatein mat kar" — expected: mild/ambiguous (mixed script, affectionate framing) — category: script_mixing

## Clean / control (should never be flagged)

24. "kal weather bahut accha tha, ghumne gaye the hum" — expected: safe — category: control

---

## How to use this set

Run each message through the classifier (`consumer/classifier.py`'s `predict()`)
and compare its output to the expected judgment above. Document mismatches — these
are exactly the "here's exactly where it breaks" failure-mode analysis the plan
(§4) wants alongside the raw macro-F1 number. The rule-based placeholder
classifier will fail most of the negation and sarcasm cases by design (it's pure
keyword matching) — that's expected and worth showing explicitly as the reason a
real fine-tuned model is needed, not a flaw to hide.
