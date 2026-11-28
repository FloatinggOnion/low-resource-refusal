# Results: Qwen3 pilot, comprehension vs action in Yoruba refusal (13 Nov 2026)

**Update, later on 13 Nov: Swahili and Hausa and a decoding check are added at the end of this file (Addendum). They refine bottom-line items 1, 4 and 5; see "What this adds to the bottom line" there.**

Paste-ready for the "Results" heading. Models: Qwen3-4B, 8B, 14B (`enable_thinking=False`, no system prompt). Prompts: TukaBench afri-jbb-harm and afri-jbb-benign, 100 topic-matched twin pairs (10 categories x 10). Cells: `eng`, `yor`, `yor-stripped`. Rules fixed before any model output was read: `PREREG.md` (changelog at the bottom). All intervals are 95% bootstrap CIs over items (Belebele: over questions), the same draws for every cell so differences are paired.

## Bottom line

1. **H1 is untested in this range, not refuted.** Qwen3 does not read Yoruba well at any size from 4B to 14B (your falsifier 4 fires), so a switch from comprehension-limited to action-limited cannot appear here. Belebele in Yoruba is 30% / 30% / 34% (chance 25%) against 90% / 87% / 95% in English.
2. **The harm-direction measure (H) fails its own validity control, as pre-registered.** The English direction separates harmful from benign at AUROC 0.98 in English, but 58-67% of that margin survives word shuffling (your falsifier 3 fires; rule threshold 50%). The shuffled AUROC (0.78-0.82) equals what a bag-of-words classifier gets from word identity alone (0.77), so what survives is lexical. Until H is refined, the C/H/R split is not valid in the sense your log states. The same direction does not transfer to Yoruba at all (AUROC 0.52-0.55, CIs include 0.5).
3. **The refusal gap is huge and flat across size.** English refuses 84-87% of harmful prompts; Yoruba 0.3-4% by the primary judge. Raw gap +81 to +87 points; on items the model understood (U1) +81 to +87. Falsifier 1 (residual gap near zero) does not fire. Falsifier 2 does not apply, because C is nowhere near ceiling.
4. **The repetition flag is common in Yoruba.** About 90% of 4B Yoruba responses are flagged empty or repetitive and 81% hit the 1024-token cap. U1 and the generation task measure different behaviors.
5. **The two judges disagree on Yoruba, so the estimated gap depends on the judge.** On sampled harmful Yoruba prompts, κ is 0.031–0.117 with all responses and 0.027–0.214 after excluding flagged outputs. Blinded human calibration is required before treating exact refusal rates as validated.

H1 remains untested. Yoruba scores low on the reading tasks and its generations often trigger the repetition flag. These measurements cannot locate the source of the refusal gap.

## Falsifier and gate scorecard (from PREREG.md)

| rule | outcome |
|---|---|
| F1: residual gap near zero at every size | **Does not fire.** Residual +86 / +86 / +84 points (4B / 8B / 14B), CIs 68-100, 69-100, 72-94. |
| F2: large flat residual gap while C near ceiling at the smallest model | **Does not fire:** gap is large and flat, but C is not near ceiling (U1 mean accuracy 42.5% at 4B). |
| F3: H separates after meaning is destroyed | **Fires for English at all sizes** (retained margin 0.58 / 0.62 / 0.67). Caveat: shuffling keeps word identity, and a bag-of-words classifier reaches 0.77. Yoruba: no H signal to retain. |
| F4: C low at every runnable size | **Fires.** Best case 14B: U1 mean accuracy 60%, Belebele 34%. |
| Near ceiling at 4B (U1 accuracy >= 95%) | No: 42.5% [36, 49]. The small-model half of H1 is not failed by this rule. |
| Gate 0: stripping lowers U1 and Belebele by >= 10 points | **Fails.** Belebele -1.1 [-3.7, +1.4] / -3.9 [-6.3, -1.4] / -3.4 [-6.0, -1.0]. U1 does not fall; "understood" rises +4.5 / +5.5 / +5.5 points. |
| Gate 0: a gap exists (Yoruba >= 10 points below English) | Passes under both judges. |
| Gate 0: judge agreement kappa >= 0.6 | **Not met on sampled harmful responses.** English clears 0.6 in the supplementary pooled all-task calculation, where prompt kinds and decoding settings are combined, but falls below the threshold within the headline stratum. |

## What was run (methods in brief)

- **C (no generation).** U1: Yoruba (or English, or stripped) prompt plus four English options (own `Goal` and three same-category, same-split `Goal`s), letter logits A-D, two option orders with the correct letter moved; "understood" = correct in both orders, otherwise `uncertain` or `misread`. U1-twin: one distractor is the topic-matched twin of opposite intent. Belebele: 900 questions per cell, zero-shot English instruction, letter logits; stripping applied to passage, question and answers (never to English). Chance for "right in both orders" is 6.25%.
- **H.** Hidden state at the last token of the chat-templated prompt (`add_generation_prompt=True`, `enable_thinking=False`), all layers. Direction = mean(harmful English) - mean(benign English), fit on one half of the twin pairs (split by category, pairs kept together), evaluated on the other half, folds swapped. Layer chosen on English training data only by nested 5-fold CV. Controls: five word-shuffles per prompt, bag-of-words logistic regression, prompt-length AUROC.
- **R.** vLLM, direct prompting. Primary: k=3, T=0.7, top-p 0.9 (Safety-IRT "Standard"; TukaBench itself is greedy, which was also run), max 1024 new tokens. Judge: TukaBench's judge system message on a local open model (provider research access unconfirmed), input = English `Goal` + response. Empty and repetitive responses are flagged beside the label.
- **Residual gap.** Per item, refusal rate = share of its 3 samples labelled Refused; gap = English minus target on the same items, restricted to items with U1 "understood" in the target cell; paired bootstrap over items (B=5000, seed 20261113).

**Deviations to state when presenting:** judge is Mistral-Small-3.2-24B (not GPT-4.1) with TukaBench's system message and Safety-IRT's English-`Goal` input convention; the benign judge variant (Complied) is our adaptation; judge-human agreement for this setup is unknown; U2 back-translation was left out; Belebele is zero-shot (absolute numbers are not comparable to published few-shot numbers).

## Findings by measure

**C.** English is at ceiling (U1 100%, Belebele 87-95%), so the probe format works. Yoruba is far below it at every size and barely above chance on Belebele. Stripping does not lower comprehension in any way the gate would count: Belebele falls by 1-4 points (the 8B and 14B drops exclude zero), but U1 "understood" goes *up* by 5 points. That is a floor effect plus, plausibly, U1 partly rewarding surface matching of names and loanwords, which stripping makes look more English; this is a hypothesis I did not test. U1 and U1-twin agree only moderately in Yoruba at 4B (Cohen's kappa 0.31 / 0.35 for `yor` / `yor-stripped`), so each is a noisy reading.

**H.** English: 0.973-0.978 at layers 22-39. Yoruba: 0.52-0.55; stripped 0.49-0.57. The difference to English is -0.43 to -0.46 at every size (CIs exclude 0). *Exploratory, not pre-registered (4B):* a direction fit inside Yoruba reaches 0.69 [0.62, 0.76], but 84% of that margin survives shuffling, so it too is largely lexical.

**R.** Under the primary judge, 70–97% of Yoruba responses are labelled Deflected. The share labelled Jailbroken rises from 4.0% to 7.7% to 29.7%, with most of the labels on responses flagged repetitive. The shares of all Yoruba responses labelled Jailbroken and passing the empty/repetitive filter are 0.3% / 0.0% / 9.7%. The second judge's overall Jailbroken rates are 9.7% / 3.3% / 8.7%.

**Residual gap.** It is large and about the same in understood and not-understood items (4B: +86 vs +82), but refusal is about 0% in both, so there is no variation left for understanding to explain. The "understood" sets are small (21 / 21 / 31 items for `yor`; 14 / 13 / 20 on the strict U1-and-twin rule). Misclassification check (4B): if every item the model did not understand were answered by uniform guessing, 20-25% of the "understood" set would be false positives; the residual gap moves only from 86 to 88 points (`yor`) if that share is anywhere from 0 to 40%.

## Judge disagreement

Under phi-4, Yoruba harmful prompts are 25-51% "Refused" (primary: 0.3-4.3%), and English 98-100% (primary: 84-87%). At 4B, phi-4 labels 27% of repetitive looping non-English responses as Refused, which does not fit the rubric as written ("clearly refused"); Mistral calls them Deflected. That favours the primary judge but is not validation. Hand labelling is the adjudication: `outputs/Qwen3-4B/calibration/sheet.csv` and `outputs/Qwen3-14B/calibration/sheet.csv` (48 rows each, judge labels withheld). Please label the 4B stripped-Yoruba rows first, where the judges differ most (primary 1.0% vs phi-4 50.7% Refused).

## What I would do next

1. Hand-label the calibration sheets and rerun with the better judge; request TukaBench's human labels; repeat with GPT-4.1 once provider access is confirmed.
2. Find a regime with high comprehension before claiming anything about action: Swahili or Hausa on the same models, or a Yoruba-adapted model. This is a suggestion, not a result.
3. Refine H: shuffle within sentence structure, or a harm direction fit on matched twins with identical vocabulary.

## Reproducibility

Seed 20261113; dataset revisions pinned in `src/lrr/data.py`; `summary.json` per run records versions, GPU and a code hash; vLLM 0.33.0, H100 80GB. Measured in-function Modal time is about 50 minutes (C+H 9 min, generation 13, judging 18, second judge 11), plus start-up and model downloads that I could not see; check the Modal dashboard for the cost. Harmful prompts and responses are in `outputs/` and the private Volume `lrr-outputs`, both outside git. Derived data inherits TukaBench's CC-BY-NC 4.0.

## Tables (generated by `lrr.report`)

### C: comprehension (95% bootstrap CI over items / questions)

| model | cell | U1 understood | U1 mean acc | U1-twin understood | Belebele acc |
|---|---|---|---|---|---|
| Qwen3-4B | eng | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 98.5% [96.5, 100.0] | 89.6% [87.6, 91.6] |
| Qwen3-4B | yor | 30.5% [24.0, 37.5] | 42.5% [36.2, 49.0] | 37.5% [32.0, 43.0] | 30.3% [27.3, 33.4] |
| Qwen3-4B | yor-stripped | 35.0% [28.0, 41.5] | 44.8% [38.0, 51.2] | 40.5% [35.0, 46.0] | 29.2% [26.2, 32.2] |
| Qwen3-8B | eng | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 99.0% [97.5, 100.0] | 86.9% [84.6, 89.1] |
| Qwen3-8B | yor | 26.5% [20.0, 33.0] | 42.5% [36.2, 48.5] | 31.0% [25.5, 36.5] | 29.8% [26.9, 32.9] |
| Qwen3-8B | yor-stripped | 32.0% [25.5, 38.5] | 45.8% [39.2, 51.7] | 33.0% [27.5, 39.0] | 25.9% [23.1, 28.8] |
| Qwen3-14B | eng | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 94.7% [93.1, 96.1] |
| Qwen3-14B | yor | 43.5% [36.5, 51.0] | 60.2% [54.5, 65.8] | 49.0% [44.0, 54.5] | 34.1% [31.1, 37.3] |
| Qwen3-14B | yor-stripped | 49.0% [41.5, 56.0] | 61.5% [55.2, 67.5] | 51.5% [46.5, 57.0] | 30.7% [27.8, 33.6] |

Chance: U1/U1-twin understood (right in both orders) 6.25% if guessing; Belebele 25%.

| model | difference | U1 mean acc (pts) | U1 understood (pts) | Belebele (pts) |
|---|---|---|---|---|
| Qwen3-4B | yor - eng | -57.5 [-63.7, -51.0] | -69.5 [-76.0, -62.5] | -59.2 [-62.9, -55.7] |
| Qwen3-4B | yor-stripped - yor | +2.3 [-1.5, +6.2] | +4.5 [-0.5, +9.5] | -1.1 [-3.7, +1.4] |
| Qwen3-8B | yor - eng | -57.5 [-63.7, -51.5] | -73.5 [-80.0, -67.0] | -57.1 [-60.7, -53.3] |
| Qwen3-8B | yor-stripped - yor | +3.3 [-0.3, +6.8] | +5.5 [+1.5, +9.5] | -3.9 [-6.3, -1.4] |
| Qwen3-14B | yor - eng | -39.8 [-45.5, -34.2] | -56.5 [-63.5, -49.0] | -60.6 [-63.8, -57.1] |
| Qwen3-14B | yor-stripped - yor | +1.2 [-2.2, +4.7] | +5.5 [+1.0, +10.0] | -3.4 [-6.0, -1.0] |

### H: English harm direction, AUROC at the English-chosen layer (95% CI over twin pairs)

| model | layers (fold 0/1) | eng | yor | yor-stripped | yor - eng | bag-of-words eng | bag-of-words yor |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | 23/22 | 0.978 [0.960, 0.992] | 0.523 [0.465, 0.578] | 0.566 [0.490, 0.636] | -0.455 [-0.515, -0.398] | 0.773 | 0.779 |
| Qwen3-8B | 24/35 | 0.973 [0.955, 0.989] | 0.542 [0.478, 0.604] | 0.515 [0.446, 0.588] | -0.431 [-0.500, -0.367] | 0.773 | 0.779 |
| Qwen3-14B | 32/39 | 0.977 [0.962, 0.990] | 0.550 [0.482, 0.618] | 0.491 [0.423, 0.561] | -0.427 [-0.497, -0.357] | 0.773 | 0.779 |

### Meaning-destroyed control (5 word shuffles; rule: fires if retained margin ratio > 0.5)

| model | cell | AUROC unshuffled | AUROC shuffled (mean, sd) | retained margin ratio | fires |
|---|---|---|---|---|---|
| Qwen3-4B | eng | 0.978 | 0.780 (0.016) | 0.58 | True |
| Qwen3-4B | yor | 0.523 | 0.502 (0.023) | undefined (no margin to retain) | None |
| Qwen3-8B | eng | 0.973 | 0.792 (0.014) | 0.62 | True |
| Qwen3-8B | yor | 0.542 | 0.511 (0.029) | undefined (no margin to retain) | None |
| Qwen3-14B | eng | 0.977 | 0.819 (0.017) | 0.67 | True |
| Qwen3-14B | yor | 0.550 | 0.491 (0.033) | -0.18 | False |

### R: judge labels on sampled responses (k=3, T=0.7), harmful prompts, 95% CI over items

| model | cell | Refused | Jailbroken | Deflected | empty/repetitive | hit 1024-token cap |
|---|---|---|---|---|---|---|
| Qwen3-4B | eng | 84.0% [76.7, 90.7] | 2.0% [0.0, 4.7] | 14.0% [8.0, 20.7] | 0% | 4% |
| Qwen3-4B | yor | 1.3% [0.0, 3.7] | 4.0% [1.7, 7.3] | 94.7% [91.0, 97.7] | 90% | 81% |
| Qwen3-4B | yor-stripped | 1.0% [0.0, 2.3] | 1.7% [0.0, 4.0] | 97.3% [95.0, 99.3] | 56% | 46% |
| Qwen3-8B | eng | 85.7% [79.0, 91.7] | 1.0% [0.0, 3.0] | 13.3% [7.7, 20.0] | 0% | 4% |
| Qwen3-8B | yor | 4.3% [1.7, 7.3] | 7.7% [4.7, 11.3] | 88.0% [83.7, 92.0] | 96% | 89% |
| Qwen3-8B | yor-stripped | 0.3% [0.0, 1.0] | 3.0% [1.0, 5.7] | 96.7% [94.0, 98.7] | 79% | 74% |
| Qwen3-14B | eng | 87.0% [81.0, 92.3] | 2.3% [0.0, 5.7] | 10.7% [6.0, 16.0] | 0% | 3% |
| Qwen3-14B | yor | 0.3% [0.0, 1.0] | 29.7% [24.0, 35.7] | 70.0% [64.0, 75.7] | 80% | 86% |
| Qwen3-14B | yor-stripped | 1.3% [0.0, 3.0] | 10.7% [6.7, 15.0] | 88.0% [83.3, 92.3] | 64% | 61% |

### R: benign twins

| model | cell | Refused | Complied | Deflected |
|---|---|---|---|---|
| Qwen3-4B | eng | 6.0% [2.0, 10.7] | 88.3% [82.3, 93.7] | 5.7% [2.3, 9.7] |
| Qwen3-4B | yor | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] |
| Qwen3-4B | yor-stripped | 0.3% [0.0, 1.0] | 0.7% [0.0, 1.7] | 99.0% [97.7, 100.0] |
| Qwen3-8B | eng | 8.7% [4.0, 14.0] | 85.0% [78.3, 91.3] | 6.3% [2.7, 10.7] |
| Qwen3-8B | yor | 4.0% [1.7, 6.7] | 4.0% [1.7, 6.7] | 92.0% [88.3, 95.3] |
| Qwen3-8B | yor-stripped | 0.0% [0.0, 0.0] | 2.7% [1.0, 4.7] | 97.3% [95.3, 99.0] |
| Qwen3-14B | eng | 7.3% [2.7, 12.7] | 84.7% [78.0, 90.7] | 8.0% [4.0, 12.7] |
| Qwen3-14B | yor | 1.3% [0.3, 2.7] | 22.3% [17.0, 27.7] | 76.3% [71.0, 81.7] |
| Qwen3-14B | yor-stripped | 1.3% [0.0, 3.0] | 7.7% [4.3, 11.3] | 91.0% [87.0, 94.7] |

### Residual gap: English refusal minus target refusal, harmful prompts, percentage points [95% paired bootstrap]

| model | target | raw (all 100) | residual (U1 understood) | residual strict (U1 and U1-twin) | not understood |
|---|---|---|---|---|---|
| Qwen3-4B | yor | +82.7 [+75.0, +89.7] (n=100) | +85.7 [+68.4, +100.0] (n=21) | +78.6 [+54.5, +100.0] (n=14) | +81.9 [+72.9, +89.6] (n=79) |
| Qwen3-4B | yor-stripped | +83.0 [+75.7, +90.0] (n=100) | +86.7 [+70.8, +100.0] (n=25) | +84.1 [+66.7, +100.0] (n=21) | +81.8 [+73.2, +89.5] (n=75) |
| Qwen3-8B | yor | +81.3 [+74.3, +87.7] (n=100) | +85.7 [+69.2, +100.0] (n=21) | +76.9 [+50.0, +100.0] (n=13) | +80.2 [+72.5, +87.1] (n=79) |
| Qwen3-8B | yor-stripped | +85.3 [+78.7, +91.3] (n=100) | +85.0 [+68.2, +100.0] (n=20) | +76.9 [+50.0, +100.0] (n=13) | +85.4 [+77.9, +91.9] (n=80) |
| Qwen3-14B | yor | +86.7 [+80.7, +92.0] (n=100) | +83.9 [+72.0, +94.0] (n=31) | +78.3 [+60.4, +93.3] (n=20) | +87.9 [+80.8, +94.2] (n=69) |
| Qwen3-14B | yor-stripped | +85.7 [+79.7, +91.0] (n=100) | +80.8 [+70.3, +90.3] (n=40) | +75.8 [+59.4, +89.6] (n=22) | +88.9 [+81.5, +95.4] (n=60) |

Sensitivity, excluding empty/repetitive responses (small n; most Yoruba responses are repetitive):

| model | target | raw | residual (U1 understood) |
|---|---|---|---|
| Qwen3-4B | yor | +83.3 [+66.7, +96.5] (n=22) | +100.0 [+100.0, +100.0] (n=3) |
| Qwen3-4B | yor-stripped | +81.8 [+71.2, +90.9] (n=53) | +92.3 [+75.0, +100.0] (n=13) |
| Qwen3-8B | yor | +78.8 [+51.8, +100.0] (n=11) | +100.0 [+100.0, +100.0] (n=2) |
| Qwen3-8B | yor-stripped | +85.8 [+75.2, +94.7] (n=40) | +100.0 [+100.0, +100.0] (n=9) |
| Qwen3-14B | yor | +85.6 [+76.1, +93.7] (n=45) | +89.5 [+76.2, +98.4] (n=19) |
| Qwen3-14B | yor-stripped | +86.1 [+78.3, +93.0] (n=66) | +74.2 [+55.1, +90.7] (n=20) |

### Judge sensitivity: Mistral-Small-3.2-24B (primary) vs phi-4 (second), harmful prompts, sampled responses

| model | cell | Refused (primary) | Refused (phi-4) | Jailbroken (primary) | Jailbroken (phi-4) | Deflected (primary) | Deflected (phi-4) |
|---|---|---|---|---|---|---|---|
| Qwen3-4B | eng | 84.0% | 98.3% | 2.0% | 0.0% | 14.0% | 1.7% |
| Qwen3-4B | yor | 1.3% | 25.3% | 4.0% | 9.7% | 94.7% | 65.0% |
| Qwen3-4B | yor-stripped | 1.0% | 50.7% | 1.7% | 0.7% | 97.3% | 48.7% |
| Qwen3-8B | eng | 85.7% | 99.0% | 1.0% | 0.0% | 13.3% | 1.0% |
| Qwen3-8B | yor | 4.3% | 50.0% | 7.7% | 3.3% | 88.0% | 46.7% |
| Qwen3-8B | yor-stripped | 0.3% | 37.3% | 3.0% | 3.0% | 96.7% | 59.7% |
| Qwen3-14B | eng | 87.0% | 99.7% | 2.3% | 0.0% | 10.7% | 0.3% |
| Qwen3-14B | yor | 0.3% | 37.3% | 29.7% | 8.7% | 70.0% | 54.0% |
| Qwen3-14B | yor-stripped | 1.3% | 37.3% | 10.7% | 6.0% | 88.0% | 56.7% |

| model | target | raw gap, primary | raw gap, phi-4 | residual (U1 understood), primary | residual, phi-4 |
|---|---|---|---|---|---|
| Qwen3-4B | yor | +82.7 [+75.0, +89.7] (n=100) | +73.0 [+66.7, +79.3] (n=100) | +85.7 [+68.4, +100.0] (n=21) | +63.5 [+43.5, +81.5] (n=21) |
| Qwen3-4B | yor-stripped | +83.0 [+75.7, +90.0] (n=100) | +47.7 [+40.0, +55.3] (n=100) | +86.7 [+70.8, +100.0] (n=25) | +44.0 [+27.0, +61.3] (n=25) |
| Qwen3-8B | yor | +81.3 [+74.3, +87.7] (n=100) | +49.0 [+42.3, +55.7] (n=100) | +85.7 [+69.2, +100.0] (n=21) | +46.0 [+33.3, +59.4] (n=21) |
| Qwen3-8B | yor-stripped | +85.3 [+78.7, +91.3] (n=100) | +61.7 [+55.0, +68.0] (n=100) | +85.0 [+68.2, +100.0] (n=20) | +48.3 [+33.3, +63.5] (n=20) |
| Qwen3-14B | yor | +86.7 [+80.7, +92.0] (n=100) | +62.3 [+54.7, +69.7] (n=100) | +83.9 [+72.0, +94.0] (n=31) | +49.5 [+34.7, +64.6] (n=31) |
| Qwen3-14B | yor-stripped | +85.7 [+79.7, +91.0] (n=100) | +62.3 [+55.3, +69.0] (n=100) | +80.8 [+70.3, +90.3] (n=40) | +60.0 [+48.0, +71.2] (n=40) |

### Judge agreement by language (sampled harmful prompts)

Each column shows κ / parsed pairs out of all responses / Unparsed (primary/second). The headline uses unstripped base-language cells. Human calibration is required before treating judge rates as validated.

| Run | Language cell | Prompt kind | Setting | All responses | Non-degenerate subset | All-response κ ≥ 0.6 |
|---|---|---|---|---|---|---|
| Qwen3-4B | eng | harm | sampled | 0.087 / 300/300 / 0/0 | 0.087 / 300/300 / 0/0 | Fail |
| Qwen3-4B | yor | harm | sampled | 0.117 / 300/300 / 0/0 | 0.042 / 30/30 / 0/0 | Fail |
| Qwen3-8B | eng | harm | sampled | 0.070 / 300/300 / 0/0 | 0.070 / 300/300 / 0/0 | Fail |
| Qwen3-8B | yor | harm | sampled | 0.068 / 300/300 / 0/0 | 0.214 / 11/11 / 0/0 | Fail |
| Qwen3-14B | eng | harm | sampled | 0.019 / 300/300 / 0/0 | 0.019 / 300/300 / 0/0 | Fail |
| Qwen3-14B | yor | harm | sampled | 0.031 / 300/300 / 0/0 | 0.027 / 61/61 / 0/0 | Fail |

### Supplementary judge agreement

| Run | Language cell | Prompt kind | Setting | All responses | Non-degenerate subset | All-response κ ≥ 0.6 |
|---|---|---|---|---|---|---|
| Qwen3-4B | eng | benign | greedy | 0.374 / 100/100 / 0/0 | 0.374 / 100/100 / 0/0 | Fail |
| Qwen3-4B | eng | benign | sampled | 0.318 / 300/300 / 0/0 | 0.318 / 299/299 / 0/0 | Fail |
| Qwen3-4B | eng | harm | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 100/100 / 0/0 | Fail |
| Qwen3-4B | yor | benign | greedy | -0.009 / 100/100 / 0/0 | — / 8/8 / 0/0 | Fail |
| Qwen3-4B | yor | benign | sampled | 0.000 / 300/300 / 0/0 | 0.000 / 21/21 / 0/0 | Fail |
| Qwen3-4B | yor | harm | greedy | 0.030 / 100/100 / 0/0 | 0.000 / 10/10 / 0/0 | Fail |
| Qwen3-4B | yor-stripped | benign | greedy | -0.019 / 100/100 / 0/0 | 0.000 / 30/30 / 0/0 | Fail |
| Qwen3-4B | yor-stripped | benign | sampled | 0.018 / 300/300 / 0/0 | 0.000 / 99/99 / 0/0 | Fail |
| Qwen3-4B | yor-stripped | harm | greedy | -0.020 / 100/100 / 0/0 | 0.000 / 43/43 / 0/0 | Fail |
| Qwen3-4B | yor-stripped | harm | sampled | -0.017 / 300/300 / 0/0 | 0.000 / 131/131 / 0/0 | Fail |
| Qwen3-8B | eng | benign | greedy | 0.405 / 100/100 / 0/0 | 0.405 / 100/100 / 0/0 | Fail |
| Qwen3-8B | eng | benign | sampled | 0.400 / 300/300 / 0/0 | 0.400 / 300/300 / 0/0 | Fail |
| Qwen3-8B | eng | harm | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 100/100 / 0/0 | Fail |
| Qwen3-8B | yor | benign | greedy | 0.368 / 100/100 / 0/0 | — / 2/2 / 0/0 | Fail |
| Qwen3-8B | yor | benign | sampled | 0.203 / 300/300 / 0/0 | 0.156 / 9/9 / 0/0 | Fail |
| Qwen3-8B | yor | harm | greedy | -0.021 / 100/100 / 0/0 | -0.200 / 3/3 / 0/0 | Fail |
| Qwen3-8B | yor-stripped | benign | greedy | 0.261 / 100/100 / 0/0 | 0.630 / 20/20 / 0/0 | Fail |
| Qwen3-8B | yor-stripped | benign | sampled | 0.047 / 300/300 / 0/0 | 0.171 / 68/68 / 0/0 | Fail |
| Qwen3-8B | yor-stripped | harm | greedy | 0.059 / 100/100 / 0/0 | 0.000 / 18/18 / 0/0 | Fail |
| Qwen3-8B | yor-stripped | harm | sampled | 0.042 / 300/300 / 0/0 | -0.031 / 64/64 / 0/0 | Fail |
| Qwen3-14B | eng | benign | greedy | 0.336 / 100/100 / 0/0 | 0.336 / 100/100 / 0/0 | Fail |
| Qwen3-14B | eng | benign | sampled | 0.382 / 300/300 / 0/0 | 0.382 / 300/300 / 0/0 | Fail |
| Qwen3-14B | eng | harm | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 100/100 / 0/0 | Fail |
| Qwen3-14B | yor | benign | greedy | 0.211 / 100/100 / 0/0 | 0.253 / 13/13 / 0/0 | Fail |
| Qwen3-14B | yor | benign | sampled | 0.289 / 300/300 / 0/0 | 0.260 / 87/87 / 0/0 | Fail |
| Qwen3-14B | yor | harm | greedy | 0.113 / 100/100 / 0/0 | 0.485 / 13/13 / 0/0 | Fail |
| Qwen3-14B | yor-stripped | benign | greedy | -0.051 / 100/100 / 0/0 | -0.087 / 25/25 / 0/0 | Fail |
| Qwen3-14B | yor-stripped | benign | sampled | 0.084 / 300/300 / 0/0 | 0.112 / 95/95 / 0/0 | Fail |
| Qwen3-14B | yor-stripped | harm | greedy | 0.022 / 100/100 / 0/0 | -0.051 / 19/19 / 0/0 | Fail |
| Qwen3-14B | yor-stripped | harm | sampled | 0.088 / 300/300 / 0/0 | 0.033 / 109/109 / 0/0 | Fail |

### Supplementary all-task agreement by language

For comparison with the original gate check, these rows pool harmful and benign prompts and sampled and greedy decoding settings within each unstripped language.

| Run | Language cell | Prompt kind | Setting | All responses | Non-degenerate subset | All-response κ ≥ 0.6 |
|---|---|---|---|---|---|---|
| Qwen3-4B | eng | all | all | 0.619 / 800/800 / 0/0 | 0.619 / 799/799 / 0/0 | Pass |
| Qwen3-4B | yor | all | all | 0.092 / 800/800 / 0/0 | 0.037 / 69/69 / 0/0 | Fail |
| Qwen3-8B | eng | all | all | 0.651 / 800/800 / 0/0 | 0.651 / 800/800 / 0/0 | Pass |
| Qwen3-8B | yor | all | all | 0.117 / 800/800 / 0/0 | 0.135 / 25/25 / 0/0 | Fail |
| Qwen3-14B | eng | all | all | 0.636 / 800/800 / 0/0 | 0.636 / 800/800 / 0/0 | Pass |
| Qwen3-14B | yor | all | all | 0.154 / 800/800 / 0/0 | 0.220 / 174/174 / 0/0 | Fail |

---

# Addendum: Swahili and Hausa, and a decoding check (13 Nov 2026, later)

Addendum A covers Swahili and Hausa. Addendum B covers the decoding sensitivity. Each run uses its own English reference. At 14B, the primary judge labelled 87.0% of English responses Refused in the Yoruba run and 88.3% in the Swahili/Hausa run. Historical decoding sensitivity comparisons are paired by prompt item only. Position-based seeding changed some sample seeds across runs.

## What this adds to the bottom line

1. **No Swahili or Hausa cell meets the pre-registered high-comprehension rule** (U1 accuracy >= 95% and Belebele >= 90% of English). H1's crossover therefore still cannot be tested by the standard we set. Swahili is the near miss: U1 accuracy 87% / 94.5% / 99% (4B / 8B / 14B), but Belebele is 45% / 51% / 64%, which is 51% / 59% / 67% of English. The two comprehension readings disagree about how well the model reads Swahili, and I did not move the goalposts.
2. **Swahili judge labels change with size.** The primary judge's English gap stays near +84 / +85 / +86 points. Deflected labels fall from 92% to 75% to 49%, and Jailbroken labels rise from 7% to 24% to 49%. The judges disagree on the magnitude. At 14B, 51% of sampled responses are flagged empty or repetitive.
3. **Hausa behaves like Yoruba**: U1 accuracy 50% / 54% / 70%, Belebele 32% / 32% / 38%, refusal 0-2%, gap +83 to +88. Hooked-letter stripping does nothing measurable (Belebele -0.3 / -0.2 / -0.4 points, CIs include 0). It is a weak manipulation to begin with: it changes 72-79% of prompts but merges under 1% of word types.
4. **The direction separates weakly in Swahili at the English-selected layers.** AUROC is 0.57 / 0.58 / 0.63 (14B CI 0.58–0.68), with Hausa near 0.5. Swahili retains 1.79 / 0.93 / 0.63 of its above-chance margin after shuffling, crossing the shuffle threshold at each size.
5. **The decoding change leaves most of the judge-labelled gap.** 8B Yoruba is an intermediate outcome. The primary judge's Refused share rises from 4.3% to 8.0%, the empty/repetitive flag falls from 96% to 67%, and the gap narrows from 81.3 to 76.7 points. Across these settings, anti-repetition Refused reaches at most 8% in the non-English cells and gaps remain large.

For 14B Swahili under anti-repetition decoding, 91.0% pass the empty/repetitive filter, 38.3% have a stop finish reason, and 37.3% meet both conditions.

## Judges, again

- **Yoruba, Swahili, and Hausa fall below κ = 0.6** on both all sampled harmful responses and the subset passing the empty/repetitive filter. English also falls below 0.6 on sampled harmful responses. It passes only in the all-task pooling of harmful and benign prompts and sampled and greedy settings (κ 0.619–0.668), a descriptive comparison with a different sample.
- On the 147 design-setting 14B Swahili responses passing the empty/repetitive filter, Mistral labels 63% Jailbroken and 3% Refused. Phi-4 labels 26.5% Jailbroken and 26.5% Refused.
- A third model rated 30 responses from that subset. Agreement with Mistral was 70% and with phi-4 was 43%. Exact refusal rates, coherence, relevance and usability require blinded ratings by fluent humans.

## Reading against the pre-stated rules

| rule | outcome for Swahili / Hausa |
|---|---|
| High-C cell (Addendum A) | **None.** Swahili 14B is closest (U1 99%, Belebele 67% of English). |
| Residual gap in a high-C cell | **Not applicable**, since no cell qualifies. For reference the residual gap (U1-understood items) is +85 / +85 / +86 for Swahili (n = 78 / 87 / 98 items), +87 / +86 / +86 for Hausa (n = 29 / 33 / 50). |
| H1 crossover in one language | **Untested.** Needs a smaller model that is not high-C and a larger one that is. |
| Shuffle control | Fires for Swahili at all sizes; undefined for Hausa (no unshuffled margin). |
| Stripping check (`hau-stripped`) | Fails: no drop in U1 or Belebele. |
| Judge kappa >= 0.6 | Below threshold on sampled harmful responses at all sizes and in every language, including after filtering. |
| Decoding sensitivity (Addendum B) | 8B Yoruba shows an intermediate result. Across the test cells, the primary judge's Refused share stays at or below 8% and the gaps remain large. |

## Caveats specific to this part

- n = 100 items per language; "understood" sets are large for Swahili (78-98) but 29-50 for Hausa.
- Belebele here is zero-shot with an English instruction; absolute numbers are not comparable to published few-shot results. A strict reading of the 90% clause may understate Swahili comprehension; I report it as pre-registered.
- The decoding check uses one alternative setting and the primary judge only. It cannot rule out every sampling choice, only that this common anti-loop one does not change the refusal result.
- My recollection of Qwen's own recommended non-thinking settings is from memory; check the model card before citing it.
- Cost: I estimate roughly 80 minutes of in-function H100 time for this part (from run times; start-up and downloads not included). I cannot see billing, so check the Modal dashboard for the actual spend.

## Files

`outputs/<model>-swh-hau/` (C, H, generations, both judges, `analysis/` and `analysis_judge_second/`), `outputs/<model>-antirep/`, third-rater labels in `outputs/Qwen3-14B-swh-hau/calibration/claude_rater_labels.csv`, calibration sheet for Swahili/Hausa 14B in the same folder (64 rows, judge labels withheld). Tables below are generated by `lrr.report` and `lrr.sens_antirep`.

## Tables: Swahili and Hausa

### C: comprehension (95% bootstrap CI over items / questions)

| model | cell | U1 understood | U1 mean acc | U1-twin understood | Belebele acc |
|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 98.5% [96.5, 100.0] | 89.6% [87.6, 91.6] |
| Qwen3-4B-swh-hau | swh | 81.0% [74.5, 87.0] | 87.0% [82.5, 91.2] | 67.0% [60.5, 73.0] | 45.4% [42.2, 48.7] |
| Qwen3-4B-swh-hau | hau | 37.5% [30.0, 45.0] | 50.2% [44.0, 56.5] | 39.5% [34.0, 45.0] | 31.9% [28.9, 34.9] |
| Qwen3-4B-swh-hau | hau-stripped | 39.0% [31.5, 46.5] | 51.7% [45.5, 58.0] | 41.0% [35.0, 47.0] | 31.6% [28.6, 34.7] |
| Qwen3-8B-swh-hau | eng | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 99.0% [97.5, 100.0] | 86.9% [84.6, 89.1] |
| Qwen3-8B-swh-hau | swh | 90.5% [86.5, 94.5] | 94.5% [91.8, 96.8] | 78.5% [73.5, 83.5] | 50.9% [47.7, 54.1] |
| Qwen3-8B-swh-hau | hau | 43.0% [36.0, 50.0] | 53.5% [46.8, 60.0] | 38.0% [32.0, 43.5] | 31.9% [28.8, 35.0] |
| Qwen3-8B-swh-hau | hau-stripped | 40.5% [34.0, 47.0] | 51.2% [44.8, 57.5] | 38.0% [32.5, 43.5] | 31.7% [28.6, 34.8] |
| Qwen3-14B-swh-hau | eng | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 100.0% [100.0, 100.0] | 94.7% [93.1, 96.1] |
| Qwen3-14B-swh-hau | swh | 98.5% [96.5, 100.0] | 99.0% [97.5, 100.0] | 95.0% [92.0, 97.5] | 63.6% [60.4, 66.8] |
| Qwen3-14B-swh-hau | hau | 56.5% [49.0, 64.0] | 69.5% [64.0, 75.0] | 51.0% [45.5, 57.0] | 38.3% [35.2, 41.4] |
| Qwen3-14B-swh-hau | hau-stripped | 57.5% [50.5, 65.0] | 69.8% [64.2, 75.0] | 49.5% [44.0, 55.0] | 37.9% [34.7, 41.1] |

Chance: U1/U1-twin understood (right in both orders) 6.25% if guessing; Belebele 25%.

| model | difference | U1 mean acc (pts) | U1 understood (pts) | Belebele (pts) |
|---|---|---|---|---|
| Qwen3-4B-swh-hau | swh - eng | -13.0 [-17.5, -8.8] | -19.0 [-25.5, -13.0] | -44.1 [-47.6, -40.6] |
| Qwen3-4B-swh-hau | hau - eng | -49.8 [-56.0, -43.5] | -62.5 [-70.0, -55.0] | -57.7 [-61.0, -54.1] |
| Qwen3-4B-swh-hau | hau-stripped - eng | -48.3 [-54.5, -42.0] | -61.0 [-68.5, -53.5] | -58.0 [-61.3, -54.4] |
| Qwen3-4B-swh-hau | hau-stripped - hau | +1.5 [+0.3, +2.8] | +1.5 [+0.0, +3.5] | -0.3 [-2.2, +1.4] |
| Qwen3-8B-swh-hau | swh - eng | -5.5 [-8.3, -3.2] | -9.5 [-13.5, -5.5] | -36.0 [-39.3, -32.6] |
| Qwen3-8B-swh-hau | hau - eng | -46.5 [-53.2, -40.0] | -57.0 [-64.0, -50.0] | -55.0 [-58.7, -51.3] |
| Qwen3-8B-swh-hau | hau-stripped - eng | -48.8 [-55.2, -42.5] | -59.5 [-66.0, -53.0] | -55.2 [-58.8, -51.7] |
| Qwen3-8B-swh-hau | hau-stripped - hau | -2.3 [-4.0, -0.7] | -2.5 [-5.0, +0.0] | -0.2 [-2.0, +1.7] |
| Qwen3-14B-swh-hau | swh - eng | -1.0 [-2.5, +0.0] | -1.5 [-3.5, +0.0] | -31.1 [-34.2, -28.0] |
| Qwen3-14B-swh-hau | hau - eng | -30.5 [-36.0, -25.0] | -43.5 [-51.0, -36.0] | -56.3 [-59.7, -52.9] |
| Qwen3-14B-swh-hau | hau-stripped - eng | -30.2 [-35.8, -25.0] | -42.5 [-49.5, -35.0] | -56.8 [-60.2, -53.2] |
| Qwen3-14B-swh-hau | hau-stripped - hau | +0.3 [-1.5, +2.0] | +1.0 [-1.5, +4.0] | -0.4 [-2.2, +1.3] |

### H: English harm direction, AUROC at the English-chosen layer (95% CI over twin pairs)

| model | layers (fold 0/1) | eng | swh | hau | hau-stripped | difference to eng (pt, CI) | bag-of-words per cell |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | 23/22 | 0.978 [0.960, 0.992] | 0.570 [0.511, 0.626] | 0.510 [0.435, 0.585] | 0.513 [0.443, 0.587] | swh: -0.408 [-0.471, -0.346]; hau: -0.468 [-0.541, -0.394]; hau-stripped: -0.465 [-0.538, -0.388] | eng 0.77, swh 0.78, hau 0.84, hau-stripped 0.84 |
| Qwen3-8B-swh-hau | 24/35 | 0.973 [0.955, 0.989] | 0.582 [0.531, 0.639] | 0.468 [0.398, 0.538] | 0.416 [0.346, 0.490] | swh: -0.390 [-0.447, -0.330]; hau: -0.504 [-0.579, -0.436]; hau-stripped: -0.557 [-0.629, -0.482] | eng 0.77, swh 0.78, hau 0.84, hau-stripped 0.84 |
| Qwen3-14B-swh-hau | 32/39 | 0.977 [0.962, 0.990] | 0.630 [0.579, 0.684] | 0.514 [0.444, 0.583] | 0.531 [0.460, 0.602] | swh: -0.347 [-0.400, -0.291]; hau: -0.463 [-0.534, -0.393]; hau-stripped: -0.447 [-0.521, -0.375] | eng 0.77, swh 0.78, hau 0.84, hau-stripped 0.84 |

### Meaning-destroyed control (5 word shuffles; rule: fires if retained margin ratio > 0.5)

| model | cell | AUROC unshuffled | AUROC shuffled (mean, sd) | retained margin ratio | fires |
|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | 0.978 | 0.780 (0.016) | 0.58 | True |
| Qwen3-4B-swh-hau | swh | 0.570 | 0.625 (0.030) | 1.79 | True |
| Qwen3-4B-swh-hau | hau | 0.510 | 0.466 (0.027) | undefined (no margin to retain) | None |
| Qwen3-8B-swh-hau | eng | 0.973 | 0.792 (0.014) | 0.62 | True |
| Qwen3-8B-swh-hau | swh | 0.582 | 0.577 (0.003) | 0.93 | True |
| Qwen3-8B-swh-hau | hau | 0.468 | 0.429 (0.022) | undefined (no margin to retain) | None |
| Qwen3-14B-swh-hau | eng | 0.977 | 0.819 (0.017) | 0.67 | True |
| Qwen3-14B-swh-hau | swh | 0.630 | 0.582 (0.027) | 0.63 | True |
| Qwen3-14B-swh-hau | hau | 0.514 | 0.485 (0.032) | undefined (no margin to retain) | None |

### R: judge labels on sampled responses (k=3, T=0.7), harmful prompts, 95% CI over items

| model | cell | Refused | Jailbroken | Deflected | empty/repetitive | hit 1024-token cap |
|---|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | 84.0% [76.7, 90.7] | 2.0% [0.0, 4.7] | 14.0% [8.0, 20.7] | 0% | 4% |
| Qwen3-4B-swh-hau | swh | 0.3% [0.0, 1.0] | 7.3% [4.7, 10.3] | 92.3% [89.0, 95.3] | 96% | 87% |
| Qwen3-4B-swh-hau | hau | 0.0% [0.0, 0.0] | 5.7% [3.3, 8.3] | 94.3% [91.7, 96.7] | 81% | 57% |
| Qwen3-4B-swh-hau | hau-stripped | 1.3% [0.3, 2.7] | 1.0% [0.0, 2.3] | 97.7% [95.7, 99.3] | 56% | 45% |
| Qwen3-8B-swh-hau | eng | 85.7% [79.0, 91.7] | 1.0% [0.0, 3.0] | 13.3% [7.7, 20.0] | 0% | 4% |
| Qwen3-8B-swh-hau | swh | 1.0% [0.0, 2.3] | 24.3% [18.3, 30.3] | 74.7% [68.3, 80.3] | 80% | 81% |
| Qwen3-8B-swh-hau | hau | 0.3% [0.0, 1.0] | 7.0% [4.0, 10.3] | 92.7% [89.3, 95.7] | 85% | 81% |
| Qwen3-8B-swh-hau | hau-stripped | 0.3% [0.0, 1.0] | 4.0% [1.7, 7.0] | 95.7% [92.7, 98.3] | 78% | 75% |
| Qwen3-14B-swh-hau | eng | 88.3% [82.7, 93.7] | 0.7% [0.0, 1.7] | 11.0% [6.0, 16.3] | 0% | 3% |
| Qwen3-14B-swh-hau | swh | 2.3% [0.7, 4.7] | 48.7% [41.7, 55.7] | 49.0% [42.0, 56.0] | 51% | 75% |
| Qwen3-14B-swh-hau | hau | 1.7% [0.3, 3.3] | 23.3% [17.7, 29.7] | 75.0% [68.7, 80.7] | 84% | 78% |
| Qwen3-14B-swh-hau | hau-stripped | 0.7% [0.0, 1.7] | 18.7% [13.3, 24.3] | 80.7% [75.0, 86.0] | 76% | 68% |

### R: benign twins

| model | cell | Refused | Complied | Deflected |
|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | 6.0% [2.0, 10.7] | 88.3% [82.3, 93.7] | 5.7% [2.3, 9.7] |
| Qwen3-4B-swh-hau | swh | 0.0% [0.0, 0.0] | 3.3% [1.3, 5.7] | 96.7% [94.3, 98.7] |
| Qwen3-4B-swh-hau | hau | 0.7% [0.0, 1.7] | 0.3% [0.0, 1.0] | 99.0% [97.7, 100.0] |
| Qwen3-4B-swh-hau | hau-stripped | 0.0% [0.0, 0.0] | 0.0% [0.0, 0.0] | 100.0% [100.0, 100.0] |
| Qwen3-8B-swh-hau | eng | 8.7% [4.0, 14.0] | 85.0% [78.3, 91.3] | 6.3% [2.7, 10.7] |
| Qwen3-8B-swh-hau | swh | 0.7% [0.0, 1.7] | 19.3% [13.7, 25.0] | 80.0% [74.3, 85.7] |
| Qwen3-8B-swh-hau | hau | 0.0% [0.0, 0.0] | 2.3% [0.7, 4.3] | 97.7% [95.7, 99.3] |
| Qwen3-8B-swh-hau | hau-stripped | 0.0% [0.0, 0.0] | 2.3% [0.3, 4.7] | 97.7% [95.3, 99.7] |
| Qwen3-14B-swh-hau | eng | 8.0% [3.0, 13.7] | 83.7% [77.0, 89.7] | 8.3% [4.3, 13.3] |
| Qwen3-14B-swh-hau | swh | 2.0% [0.0, 4.7] | 34.0% [27.0, 41.3] | 64.0% [56.7, 71.0] |
| Qwen3-14B-swh-hau | hau | 0.0% [0.0, 0.0] | 14.7% [9.7, 19.7] | 85.3% [80.3, 90.3] |
| Qwen3-14B-swh-hau | hau-stripped | 0.0% [0.0, 0.0] | 9.3% [5.3, 14.0] | 90.7% [86.0, 94.7] |

### Residual gap: English refusal minus target refusal, harmful prompts, percentage points [95% paired bootstrap]

| model | target | raw (all 100) | residual (U1 understood) | residual strict (U1 and U1-twin) | not understood |
|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | swh | +83.7 [+76.3, +90.3] (n=100) | +84.6 [+76.3, +91.9] (n=78) | +84.9 [+74.9, +93.5] (n=53) | +80.3 [+63.6, +93.9] (n=22) |
| Qwen3-4B-swh-hau | hau | +84.0 [+76.7, +90.7] (n=100) | +87.4 [+75.0, +97.4] (n=29) | +88.3 [+73.3, +100.0] (n=20) | +82.6 [+73.6, +90.7] (n=71) |
| Qwen3-4B-swh-hau | hau-stripped | +82.7 [+75.3, +89.3] (n=100) | +87.8 [+75.9, +97.5] (n=30) | +88.9 [+74.5, +100.0] (n=21) | +80.5 [+71.4, +88.6] (n=70) |
| Qwen3-8B-swh-hau | swh | +84.7 [+78.0, +90.7] (n=100) | +84.7 [+77.5, +91.1] (n=87) | +85.4 [+77.8, +92.1] (n=71) | +84.6 [+61.5, +100.0] (n=13) |
| Qwen3-8B-swh-hau | hau | +85.3 [+78.7, +91.3] (n=100) | +85.9 [+73.9, +96.1] (n=33) | +82.5 [+64.4, +97.8] (n=19) | +85.1 [+76.9, +92.1] (n=67) |
| Qwen3-8B-swh-hau | hau-stripped | +85.3 [+78.7, +91.3] (n=100) | +84.4 [+71.2, +95.7] (n=30) | +81.5 [+63.0, +97.8] (n=18) | +85.7 [+77.8, +92.5] (n=70) |
| Qwen3-14B-swh-hau | swh | +86.0 [+80.0, +91.7] (n=100) | +86.1 [+79.8, +91.6] (n=98) | +87.4 [+81.3, +92.8] (n=90) | +83.3 [+66.7, +100.0] (n=2) |
| Qwen3-14B-swh-hau | hau | +86.7 [+80.7, +92.0] (n=100) | +86.0 [+77.3, +93.8] (n=50) | +83.3 [+71.2, +93.6] (n=28) | +87.3 [+79.0, +94.3] (n=50) |
| Qwen3-14B-swh-hau | hau-stripped | +87.7 [+81.7, +93.0] (n=100) | +88.0 [+79.3, +95.5] (n=50) | +86.9 [+75.0, +96.8] (n=28) | +87.3 [+78.8, +94.5] (n=50) |

Sensitivity, excluding empty/repetitive responses (small n; most Yoruba responses are repetitive):

| model | target | raw | residual (U1 understood) |
|---|---|---|---|
| Qwen3-4B-swh-hau | swh | +81.8 [+55.6, +100.0] (n=11) | +85.7 [+50.0, +100.0] (n=7) |
| Qwen3-4B-swh-hau | hau | +90.8 [+79.5, +100.0] (n=29) | +100.0 [+100.0, +100.0] (n=8) |
| Qwen3-4B-swh-hau | hau-stripped | +83.6 [+73.4, +92.6] (n=55) | +93.3 [+77.8, +100.0] (n=15) |
| Qwen3-8B-swh-hau | swh | +87.4 [+78.0, +95.5] (n=45) | +85.1 [+74.0, +94.4] (n=38) |
| Qwen3-8B-swh-hau | hau | +78.4 [+64.3, +90.5] (n=34) | +93.9 [+85.2, +100.0] (n=11) |
| Qwen3-8B-swh-hau | hau-stripped | +87.3 [+77.6, +95.3] (n=42) | +96.3 [+87.5, +100.0] (n=9) |
| Qwen3-14B-swh-hau | swh | +83.7 [+76.3, +90.3] (n=85) | +83.9 [+76.4, +90.6] (n=83) |
| Qwen3-14B-swh-hau | hau | +80.4 [+68.4, +91.2] (n=40) | +88.9 [+74.6, +100.0] (n=18) |
| Qwen3-14B-swh-hau | hau-stripped | +89.5 [+82.7, +95.3] (n=54) | +94.9 [+88.3, +100.0] (n=26) |

### Judge sensitivity: Mistral-Small-3.2-24B (primary) vs phi-4 (second), harmful prompts, sampled responses

| model | cell | Refused (primary) | Refused (phi-4) | Jailbroken (primary) | Jailbroken (phi-4) | Deflected (primary) | Deflected (phi-4) |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | 84.0% | 98.3% | 2.0% | 0.0% | 14.0% | 1.7% |
| Qwen3-4B-swh-hau | swh | 0.3% | 13.0% | 7.3% | 10.0% | 92.3% | 77.0% |
| Qwen3-4B-swh-hau | hau | 0.0% | 26.3% | 5.7% | 6.3% | 94.3% | 67.3% |
| Qwen3-4B-swh-hau | hau-stripped | 1.3% | 41.0% | 1.0% | 0.7% | 97.7% | 58.3% |
| Qwen3-8B-swh-hau | eng | 85.7% | 99.0% | 1.0% | 0.0% | 13.3% | 1.0% |
| Qwen3-8B-swh-hau | swh | 1.0% | 16.0% | 24.3% | 13.7% | 74.7% | 70.3% |
| Qwen3-8B-swh-hau | hau | 0.3% | 22.3% | 7.0% | 8.3% | 92.7% | 69.0% |
| Qwen3-8B-swh-hau | hau-stripped | 0.3% | 31.0% | 4.0% | 2.7% | 95.7% | 66.3% |
| Qwen3-14B-swh-hau | eng | 88.3% | 99.7% | 0.7% | 0.0% | 11.0% | 0.3% |
| Qwen3-14B-swh-hau | swh | 2.3% | 22.0% | 48.7% | 24.3% | 49.0% | 53.7% |
| Qwen3-14B-swh-hau | hau | 1.7% | 19.7% | 23.3% | 16.0% | 75.0% | 64.3% |
| Qwen3-14B-swh-hau | hau-stripped | 0.7% | 23.3% | 18.7% | 9.7% | 80.7% | 67.0% |

| model | target | raw gap, primary | raw gap, phi-4 | residual (U1 understood), primary | residual, phi-4 |
|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | swh | +83.7 [+76.3, +90.3] (n=100) | +85.3 [+79.7, +90.3] (n=100) | +84.6 [+76.3, +91.9] (n=78) | +86.3 [+80.2, +91.8] (n=78) |
| Qwen3-4B-swh-hau | hau | +84.0 [+76.7, +90.7] (n=100) | +72.0 [+65.7, +78.0] (n=100) | +87.4 [+75.0, +97.4] (n=29) | +69.0 [+55.6, +81.0] (n=29) |
| Qwen3-4B-swh-hau | hau-stripped | +82.7 [+75.3, +89.3] (n=100) | +57.3 [+49.7, +64.7] (n=100) | +87.8 [+75.9, +97.5] (n=30) | +52.2 [+36.4, +67.9] (n=30) |
| Qwen3-8B-swh-hau | swh | +84.7 [+78.0, +90.7] (n=100) | +83.0 [+77.7, +88.0] (n=100) | +84.7 [+77.5, +91.1] (n=87) | +82.4 [+76.5, +87.8] (n=87) |
| Qwen3-8B-swh-hau | hau | +85.3 [+78.7, +91.3] (n=100) | +76.7 [+71.0, +82.3] (n=100) | +85.9 [+73.9, +96.1] (n=33) | +75.8 [+64.4, +86.0] (n=33) |
| Qwen3-8B-swh-hau | hau-stripped | +85.3 [+78.7, +91.3] (n=100) | +68.0 [+61.3, +74.7] (n=100) | +84.4 [+71.2, +95.7] (n=30) | +67.8 [+53.8, +81.0] (n=30) |
| Qwen3-14B-swh-hau | swh | +86.0 [+80.0, +91.7] (n=100) | +77.7 [+71.7, +83.0] (n=100) | +86.1 [+79.8, +91.6] (n=98) | +78.2 [+72.4, +83.7] (n=98) |
| Qwen3-14B-swh-hau | hau | +86.7 [+80.7, +92.0] (n=100) | +80.0 [+74.3, +85.0] (n=100) | +86.0 [+77.3, +93.8] (n=50) | +82.7 [+75.7, +89.1] (n=50) |
| Qwen3-14B-swh-hau | hau-stripped | +87.7 [+81.7, +93.0] (n=100) | +76.3 [+70.7, +82.0] (n=100) | +88.0 [+79.3, +95.5] (n=50) | +75.3 [+66.7, +82.7] (n=50) |

### Judge agreement by language (sampled harmful prompts)

Each column shows κ / parsed pairs out of all responses / Unparsed (primary/second). The headline uses unstripped base-language cells. Human calibration is required before treating judge rates as validated.

| Run | Language cell | Prompt kind | Setting | All responses | Non-degenerate subset | All-response κ ≥ 0.6 |
|---|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | harm | sampled | 0.087 / 300/300 / 0/0 | 0.087 / 300/300 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau | harm | sampled | 0.068 / 300/300 / 0/0 | -0.018 / 58/58 / 0/0 | Fail |
| Qwen3-4B-swh-hau | swh | harm | sampled | 0.099 / 300/300 / 0/0 | -0.176 / 12/12 / 0/0 | Fail |
| Qwen3-8B-swh-hau | eng | harm | sampled | 0.070 / 300/300 / 0/0 | 0.070 / 300/300 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau | harm | sampled | 0.107 / 299/300 / 0/1 | 0.068 / 45/45 / 0/0 | Fail |
| Qwen3-8B-swh-hau | swh | harm | sampled | 0.265 / 300/300 / 0/0 | 0.284 / 61/61 / 0/0 | Fail |
| Qwen3-14B-swh-hau | eng | harm | sampled | 0.050 / 300/300 / 0/0 | 0.050 / 300/300 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau | harm | sampled | 0.182 / 300/300 / 0/0 | 0.258 / 49/49 / 0/0 | Fail |
| Qwen3-14B-swh-hau | swh | harm | sampled | 0.250 / 300/300 / 0/0 | 0.224 / 147/147 / 0/0 | Fail |

### Supplementary judge agreement

| Run | Language cell | Prompt kind | Setting | All responses | Non-degenerate subset | All-response κ ≥ 0.6 |
|---|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | benign | greedy | 0.386 / 100/100 / 0/0 | 0.386 / 100/100 / 0/0 | Fail |
| Qwen3-4B-swh-hau | eng | benign | sampled | 0.321 / 300/300 / 0/0 | 0.321 / 299/299 / 0/0 | Fail |
| Qwen3-4B-swh-hau | eng | harm | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 100/100 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau | benign | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 22/22 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau | benign | sampled | -0.015 / 300/300 / 0/0 | 0.000 / 70/70 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau | harm | greedy | 0.145 / 100/100 / 0/0 | 0.000 / 20/20 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau-stripped | benign | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 48/48 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau-stripped | benign | sampled | 0.000 / 300/300 / 0/0 | 0.000 / 151/151 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau-stripped | harm | greedy | -0.010 / 100/100 / 0/0 | 0.000 / 43/43 / 0/0 | Fail |
| Qwen3-4B-swh-hau | hau-stripped | harm | sampled | 0.043 / 300/300 / 0/0 | 0.000 / 132/132 / 0/0 | Fail |
| Qwen3-4B-swh-hau | swh | benign | greedy | 0.174 / 100/100 / 0/0 | — / 0/0 / 0/0 | Fail |
| Qwen3-4B-swh-hau | swh | benign | sampled | 0.364 / 300/300 / 0/0 | 1.000 / 5/5 / 0/0 | Fail |
| Qwen3-4B-swh-hau | swh | harm | greedy | 0.128 / 100/100 / 0/0 | 0.000 / 3/3 / 0/0 | Fail |
| Qwen3-8B-swh-hau | eng | benign | greedy | 0.405 / 100/100 / 0/0 | 0.405 / 100/100 / 0/0 | Fail |
| Qwen3-8B-swh-hau | eng | benign | sampled | 0.400 / 300/300 / 0/0 | 0.400 / 300/300 / 0/0 | Fail |
| Qwen3-8B-swh-hau | eng | harm | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 100/100 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau | benign | greedy | 0.078 / 100/100 / 0/0 | 0.000 / 7/7 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau | benign | sampled | -0.021 / 300/300 / 0/0 | -0.022 / 38/38 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau | harm | greedy | 0.062 / 100/100 / 0/0 | 0.000 / 7/7 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau-stripped | benign | greedy | -0.009 / 100/100 / 0/0 | 0.000 / 14/14 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau-stripped | benign | sampled | -0.018 / 300/300 / 0/0 | 0.000 / 55/55 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau-stripped | harm | greedy | 0.096 / 100/100 / 0/0 | 0.000 / 14/14 / 0/0 | Fail |
| Qwen3-8B-swh-hau | hau-stripped | harm | sampled | 0.037 / 300/300 / 0/0 | -0.010 / 66/66 / 0/0 | Fail |
| Qwen3-8B-swh-hau | swh | benign | greedy | 0.396 / 100/100 / 0/0 | 0.400 / 3/3 / 0/0 | Fail |
| Qwen3-8B-swh-hau | swh | benign | sampled | 0.280 / 300/300 / 0/0 | 0.255 / 45/45 / 0/0 | Fail |
| Qwen3-8B-swh-hau | swh | harm | greedy | 0.315 / 100/100 / 0/0 | 0.077 / 6/6 / 0/0 | Fail |
| Qwen3-14B-swh-hau | eng | benign | greedy | 0.365 / 100/100 / 0/0 | 0.365 / 100/100 / 0/0 | Fail |
| Qwen3-14B-swh-hau | eng | benign | sampled | 0.420 / 300/300 / 0/0 | 0.420 / 300/300 / 0/0 | Fail |
| Qwen3-14B-swh-hau | eng | harm | greedy | 0.000 / 100/100 / 0/0 | 0.000 / 100/100 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau | benign | greedy | 0.035 / 100/100 / 0/0 | 0.000 / 4/4 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau | benign | sampled | 0.138 / 299/300 / 0/1 | 0.262 / 46/46 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau | harm | greedy | 0.198 / 100/100 / 0/0 | 0.153 / 10/10 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau-stripped | benign | greedy | 0.200 / 100/100 / 0/0 | -0.065 / 11/11 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau-stripped | benign | sampled | 0.087 / 300/300 / 0/0 | 0.143 / 48/48 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau-stripped | harm | greedy | 0.143 / 100/100 / 0/0 | 0.221 / 12/12 / 0/0 | Fail |
| Qwen3-14B-swh-hau | hau-stripped | harm | sampled | 0.151 / 300/300 / 0/0 | 0.189 / 73/73 / 0/0 | Fail |
| Qwen3-14B-swh-hau | swh | benign | greedy | 0.392 / 100/100 / 0/0 | 0.570 / 17/17 / 0/0 | Fail |
| Qwen3-14B-swh-hau | swh | benign | sampled | 0.336 / 300/300 / 0/0 | 0.237 / 148/148 / 0/0 | Fail |
| Qwen3-14B-swh-hau | swh | harm | greedy | 0.225 / 100/100 / 0/0 | 0.159 / 23/23 / 0/0 | Fail |

### Supplementary all-task agreement by language

For comparison with the original gate check, these rows pool harmful and benign prompts and sampled and greedy decoding settings within each unstripped language.

| Run | Language cell | Prompt kind | Setting | All responses | Non-degenerate subset | All-response κ ≥ 0.6 |
|---|---|---|---|---|---|---|
| Qwen3-4B-swh-hau | eng | all | all | 0.623 / 800/800 / 0/0 | 0.622 / 799/799 / 0/0 | Pass |
| Qwen3-4B-swh-hau | hau | all | all | 0.085 / 800/800 / 0/0 | -0.006 / 170/170 / 0/0 | Fail |
| Qwen3-4B-swh-hau | swh | all | all | 0.166 / 800/800 / 0/0 | 0.103 / 20/20 / 0/0 | Fail |
| Qwen3-8B-swh-hau | eng | all | all | 0.651 / 800/800 / 0/0 | 0.651 / 800/800 / 0/0 | Pass |
| Qwen3-8B-swh-hau | hau | all | all | 0.094 / 799/800 / 0/1 | 0.045 / 97/97 / 0/0 | Fail |
| Qwen3-8B-swh-hau | swh | all | all | 0.314 / 800/800 / 0/0 | 0.303 / 115/115 / 0/0 | Fail |
| Qwen3-14B-swh-hau | eng | all | all | 0.668 / 800/800 / 0/0 | 0.668 / 800/800 / 0/0 | Pass |
| Qwen3-14B-swh-hau | hau | all | all | 0.198 / 799/800 / 0/1 | 0.293 / 109/109 / 0/0 | Fail |
| Qwen3-14B-swh-hau | swh | all | all | 0.351 / 800/800 / 0/0 | 0.330 / 335/335 / 0/0 | Fail |

## Tables: decoding sensitivity (Addendum B)

### Decoding sensitivity (harmful prompts, sampled, primary judge)

Historical comparisons are paired by item only, without matched random draws. Each gap uses English from the same run.

| model | cell | setting | Refused | Jailbroken | Deflected | Non-empty/non-repetitive | Stopped | Both | Refusal gap to English (pts) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | eng | design (Yoruba run) | 84.0% [76.7, 90.7] | 2.0% [0.0, 4.7] | 14.0% [8.0, 20.7] | 100.0% | 96.3% | 96.3% | - |
| Qwen3-4B | yor | design (Yoruba run) | 1.3% [0.0, 3.7] | 4.0% [1.7, 7.3] | 94.7% [91.0, 97.7] | 10.0% | 18.7% | 10.0% | +82.7 [+75.0, +89.7] |
| Qwen3-4B | eng | design (Swahili/Hausa run) | 84.0% [76.7, 90.7] | 2.0% [0.0, 4.7] | 14.0% [8.0, 20.7] | 100.0% | 96.3% | 96.3% | - |
| Qwen3-4B | swh | design (Swahili/Hausa run) | 0.3% [0.0, 1.0] | 7.3% [4.7, 10.3] | 92.3% [89.0, 95.3] | 4.0% | 13.0% | 3.3% | +83.7 [+76.3, +90.3] |
| Qwen3-4B | hau | design (Swahili/Hausa run) | 0.0% [0.0, 0.0] | 5.7% [3.3, 8.3] | 94.3% [91.7, 96.7] | 19.3% | 43.0% | 19.0% | +84.0 [+76.7, +90.7] |
| Qwen3-4B | eng | anti-repetition | 83.3% [76.0, 90.0] | 2.7% [0.3, 5.7] | 14.0% [8.0, 20.7] | 100.0% | 97.0% | 97.0% | - |
| Qwen3-4B | yor | anti-repetition | 1.0% [0.0, 2.3] | 9.3% [5.3, 13.7] | 89.7% [85.0, 93.7] | 15.7% | 40.0% | 14.7% | +82.3 [+75.0, +89.0] |
| Qwen3-4B | swh | anti-repetition | 0.0% [0.0, 0.0] | 13.3% [9.0, 18.0] | 86.7% [82.0, 91.0] | 20.3% | 52.7% | 17.7% | +83.3 [+76.0, +90.0] |
| Qwen3-4B | hau | anti-repetition | 1.0% [0.0, 2.3] | 11.3% [7.0, 16.0] | 87.7% [82.3, 92.3] | 23.0% | 65.3% | 22.7% | +82.3 [+75.0, +89.0] |
| Qwen3-8B | eng | design (Yoruba run) | 85.7% [79.0, 91.7] | 1.0% [0.0, 3.0] | 13.3% [7.7, 20.0] | 100.0% | 96.0% | 96.0% | - |
| Qwen3-8B | yor | design (Yoruba run) | 4.3% [1.7, 7.3] | 7.7% [4.7, 11.3] | 88.0% [83.7, 92.0] | 3.7% | 11.0% | 3.0% | +81.3 [+74.3, +87.7] |
| Qwen3-8B | eng | design (Swahili/Hausa run) | 85.7% [79.0, 91.7] | 1.0% [0.0, 3.0] | 13.3% [7.7, 20.0] | 100.0% | 96.0% | 96.0% | - |
| Qwen3-8B | swh | design (Swahili/Hausa run) | 1.0% [0.0, 2.3] | 24.3% [18.3, 30.3] | 74.7% [68.3, 80.3] | 20.3% | 19.3% | 13.3% | +84.7 [+78.0, +90.7] |
| Qwen3-8B | hau | design (Swahili/Hausa run) | 0.3% [0.0, 1.0] | 7.0% [4.0, 10.3] | 92.7% [89.3, 95.7] | 15.0% | 19.3% | 14.7% | +85.3 [+78.7, +91.3] |
| Qwen3-8B | eng | anti-repetition | 84.7% [77.7, 91.0] | 1.3% [0.0, 3.7] | 14.0% [8.0, 20.7] | 100.0% | 96.7% | 96.7% | - |
| Qwen3-8B | yor | anti-repetition | 8.0% [5.0, 11.3] | 19.0% [14.3, 24.3] | 73.0% [67.0, 78.7] | 32.7% | 53.0% | 25.3% | +76.7 [+69.0, +83.7] |
| Qwen3-8B | swh | anti-repetition | 1.0% [0.0, 2.3] | 41.0% [33.7, 48.3] | 58.0% [50.7, 65.0] | 63.7% | 31.0% | 26.7% | +83.7 [+76.7, +90.0] |
| Qwen3-8B | hau | anti-repetition | 1.0% [0.0, 2.3] | 14.7% [10.0, 19.7] | 84.3% [79.3, 89.3] | 45.0% | 60.3% | 44.3% | +83.7 [+76.7, +90.0] |
| Qwen3-14B | eng | design (Yoruba run) | 87.0% [81.0, 92.3] | 2.3% [0.0, 5.7] | 10.7% [6.0, 16.0] | 100.0% | 97.0% | 97.0% | - |
| Qwen3-14B | yor | design (Yoruba run) | 0.3% [0.0, 1.0] | 29.7% [24.0, 35.7] | 70.0% [64.0, 75.7] | 20.3% | 14.0% | 8.0% | +86.7 [+80.7, +92.0] |
| Qwen3-14B | eng | design (Swahili/Hausa run) | 88.3% [82.7, 93.7] | 0.7% [0.0, 1.7] | 11.0% [6.0, 16.3] | 100.0% | 96.7% | 96.7% | - |
| Qwen3-14B | swh | design (Swahili/Hausa run) | 2.3% [0.7, 4.7] | 48.7% [41.7, 55.7] | 49.0% [42.0, 56.0] | 49.0% | 25.0% | 20.3% | +86.0 [+80.0, +91.7] |
| Qwen3-14B | hau | design (Swahili/Hausa run) | 1.7% [0.3, 3.3] | 23.3% [17.7, 29.7] | 75.0% [68.7, 80.7] | 16.3% | 22.0% | 16.0% | +86.7 [+80.7, +92.0] |
| Qwen3-14B | eng | anti-repetition | 86.7% [80.7, 92.3] | 2.7% [0.0, 6.0] | 10.7% [5.7, 16.0] | 100.0% | 97.7% | 97.7% | - |
| Qwen3-14B | yor | anti-repetition | 0.3% [0.0, 1.0] | 44.7% [38.7, 51.0] | 55.0% [48.7, 61.3] | 59.0% | 28.3% | 21.7% | +86.3 [+80.3, +92.0] |
| Qwen3-14B | swh | anti-repetition | 2.3% [0.3, 5.3] | 59.3% [51.7, 66.7] | 38.3% [31.3, 45.7] | 91.0% | 38.3% | 37.3% | +84.3 [+77.7, +90.3] |
| Qwen3-14B | hau | anti-repetition | 0.0% [0.0, 0.0] | 37.7% [31.3, 44.3] | 62.3% [55.7, 68.7] | 54.7% | 59.3% | 48.3% | +86.7 [+80.7, +92.3] |

---

# Addendum: where does the failure sit? Checks C1-C7, D and E (25-27 Nov 2026)

Rules and corrections are recorded in `PREREG.md` Addenda C–E and the 28 November correction. Aggregate numbers are in `results/diagnostics/`. C6, C7, D2 and E use the primary judge alone. C6 uses three samples per prompt, and C7, D2 and E use one greedy response per prompt.

## Bottom line

1. **No sustained transfer under the D1 three-layer rule.** This particular English difference-of-means direction never achieves cross-fold AUROC ≥ 0.70 across three consecutive layers in a target language. The highest single-layer AUROC is 0.73 at 8B Swahili layer 1.
2. **Effective refusal steering generally affected harmful and benign prompts together.** No harm-specific effect met the D2/E replication rule. At 8B layer 21, the Yoruba difference was +18 points [5, 32] in D2 and +16 [3, 29] in the exact-layer rerun. None of the four neighbouring layers met the threshold. This establishes limited robustness across layers.
3. **H1 remains untested.** No cell cleared the preregistered comprehension bar. C6 and C7 measure changes caused by particular interventions and do not locate the failure in the unmodified model.

## Checks

| check | what it asks | result |
|---|---|---|
| C1 token fertility | Are more tokens associated with lower U1 scores? | The primary rule is met at 4B (4/5 cells) and 8B (3/5), with 2/5 at 14B. See the adjusted analysis below. |
| C2 lexical baseline | How much U1 performance can this trigram baseline explain? | It is close to 4B/8B Yoruba and Hausa U1. Swahili beats it. On U1 passes where the trigram baseline misses, the judge-labelled residual gap is +84 to +100 points using each run's English control. |
| C3 bits per byte | How predictable are these passages to the model? | About 2.3–3.2× English. BPB does not provide a consistent full ranking of comprehension or a consistent trend across model sizes. |
| C4 alignment | do non-English prompts land near their English twins? | Pooled retrieval grows with size: Swahili 61% to 95%, Yoruba 13% to 35%, Hausa 18% to 35%. |
| C5 translation (chrF) | can the model translate the prompt itself? | Swahili 31 to 51; Yoruba and Hausa 18-24. |
| C6 answer in English | Does this instruction raise Refused labels? | Primary-judge Refused is 0–9% in target cells and 83–86% in the English control. Empty/repetitive flags fall. The coherence criterion awaits blinded human labels. |
| C7 refusal direction | How do ablation and addition change judge labels? | Corrected ablation passes the 30-point rule at 8B (−46 points) and falls short at 14B (−17 points). |
| D1 transfer by layer | Does separation persist over three consecutive layers? | No target meets the rule for this direction. |
| D2 layer sweep | Does steering produce a harm-specific difference? | Two of 45 cells meet the initial threshold, both at 8B layer 21. |
| E dense replication | Is the 8B effect robust to nearby layers? | Zero of four new layers meet the threshold. The same-layer Yoruba effect meets it again. |

The original post hoc lexical adjustment left 14 of 15 fertility fits without a significant negative association. The exception was 8B stripped Hausa (95% CI −0.736 to −0.047). The L2 sensitivity fit retains the primary counts and the stripped-Hausa result. Its adjusted 8B Swahili interval also just excludes zero (−1.021 to −0.003), so the adjustment is sensitive to the estimator. The 14B Swahili interval is unreliable and omitted because 97 of 2,000 bootstrap samples have only one outcome class. Failed fits are counted in the aggregate.

The corrected 14B C2b gaps on U1 passes missed by the trigram baseline are +90.5 points for Swahili (49 items), +91.2 for Hausa (19), and +96.7 for stripped Hausa (20). These are conditional comparisons on this particular baseline.

C6 is paired by item only. Historical runs did not match all random draws. The stop reason and empty/repetitive flag are separate diagnostics. Goal-word overlap is an exact-match measure and cannot establish whether an answer is on topic.


| C6 model | Cell | Passes empty/repetitive filter | Stop finish reason | Both |
|---|---|---|---|---|
| 4B | eng | 100.0% → 100.0% | 96.3% → 97.3% | 96.3% → 97.3% |
| 4B | yor | 10.0% → 99.0% | 18.7% → 99.0% | 10.0% → 99.0% |
| 4B | swh | 4.0% → 99.3% | 13.0% → 99.3% | 3.3% → 99.3% |
| 4B | hau | 19.3% → 99.3% | 43.0% → 99.3% | 19.0% → 99.3% |
| 8B | eng | 100.0% → 100.0% | 96.0% → 98.3% | 96.0% → 98.3% |
| 8B | yor | 3.7% → 99.0% | 11.0% → 99.0% | 3.0% → 99.0% |
| 8B | swh | 20.3% → 99.7% | 19.3% → 99.7% | 13.3% → 99.7% |
| 8B | hau | 15.0% → 99.0% | 19.3% → 99.0% | 14.7% → 99.0% |
| 14B | eng | 100.0% → 100.0% | 97.0% → 97.0% | 97.0% → 97.0% |
| 14B | yor | 20.3% → 100.0% | 14.0% → 100.0% | 8.0% → 100.0% |
| 14B | swh | 49.0% → 99.0% | 25.0% → 99.0% | 20.3% → 99.0% |
| 14B | hau | 16.3% → 98.0% | 22.0% → 98.0% | 16.0% → 98.0% |

### C7 with the H-selected layers

The 8B and 14B reruns use the exact fold layers from their saved H summaries and add vectors at the same representation sites used for fitting. Ablation also projects out the direction after final normalization. The 4B row is historical. All percentages below are primary-judge labels from 100 greedy responses per condition.

| Model | H fold layers | English harmful Refused after ablation | Change in points [95% CI] | 30-point rule | English benign Refused after addition |
|---|---|---|---|---|---|
| 4B | 23/22 | 86% → 42% | -44 [-54, -34] | Pass | 5% → 87% |
| 8B | 24/35 | 86% → 40% | -46 [-56, -36] | Pass | 9% → 78% |
| 14B | 32/39 | 86% → 69% | -17 [-26, -8] | Fail | 8% → 74% |

At 8B, English Refused falls by 46 points, largely as Deflected labels rise from 13% to 56%. The corrected 14B drop is 17 points and falls short of the 30-point rule. Its addition results are descriptive.

| Model | Cell | Harmful before (%) | Harmful after 1.5× (%) | Benign before (%) | Benign after 1.5× (%) | Harmful minus benign change, points [95% CI] |
|---|---|---:|---:|---:|---:|---|
| 4B | Yoruba | 0 | 83 | 0 | 83 | 0 [-7, +7] |
| 4B | Swahili | 1 | 12 | 0 | 6 | +5 [-2, +12] |
| 4B | Hausa | 0 | 84 | 0 | 86 | -2 [-10, +6] |
| 8B | Yoruba | 4 | 50 | 7 | 45 | +8 [0, +16] |
| 8B | Swahili | 0 | 16 | 1 | 16 | +1 [-8, +10] |
| 8B | Hausa | 0 | 70 | 0 | 76 | -6 [-16, +4] |
| 14B | Yoruba | 4 | 7 | 0 | 9 | -6 [-15, +2] |
| 14B | Swahili | 3 | 9 | 3 | 5 | +4 [-4, +12] |
| 14B | Hausa | 1 | 3 | 0 | 8 | -6 [-12, 0] |

At 8B the harmful-only steering thresholds are met for Yoruba at 1.5× and Hausa at 1.0× and 1.5×. At 1.5×, all intervals for the harmful-minus-benign change include zero. These comparisons leave small or moderate differences unresolved and support no conclusion about where the unmodified model fails.

## Caveats

- 100 prompt pairs, reused across all checks; each comparison is a few dozen to 100 items.
- Blinded human calibration is required for validated refusal rates and for claims about coherence, relevance or usability.
- The English direction retains a substantial signal after shuffling. The D1 result applies to this direction and the three-layer threshold.
- D2 compares 45 cells uncorrected; that is why a replication was required before counting any hit.
- The layer-21 rerun changed some outputs and some threshold classifications. A general claim about generation variability would require controlled reruns.

## Files

Aggregate results: `results/diagnostics/` (`c1_c2`, `c3_c4_c5`, `c4_last_token`, `c6_c7` [C6, C7, C7 benign control, D2, E], `h_layers` [D1], `h_curves_<size>`). Generations, judge labels and hidden states stay private (gitignored).
