# Failure Analysis

Protocol: dump every golden example any system got wrong (`reports/errors.jsonl`, 536 rows
across all four systems), cluster by intent-pair and route-error type, then read them.
I read **66 agent errors** individually — 44 intent errors and 33 routing errors (11
examples are wrong on both).

All examples below are real, PII-scrubbed rows with their ids. Nothing here is invented.

**Headline:** the agent's 66 errors are not evenly spread. Two intent pairs account for
21 of 44 intent errors, and **one routing reason code accounts for 14 of 26 needless
escalations**. The routing problem is concentrated enough to be worth fixing; the intent
problems are mostly taxonomy-boundary disputes where my own label is arguable.

---

## Mode 1 — Frustration read as distress: over-firing `abusive_or_distressed_language`

**Frequency: 14 of 26 needless escalations (54%), 6.4% of the whole golden set.** This is
the single largest driver of the agent's routing cost. Every one came from the **model
layer**, not the deterministic rules.

**Real examples:**
> id=2916668 — "I don't pay $10 a month for this shit <URL>"
> → gold: auto (`praise_or_chitchat`), agent: **escalate** (`abusive_or_distressed_language`)

> id=578821 — "Why would i pay for you if i cant listen Beyonce, bitches?"
> → gold: auto (`content_missing_or_metadata`), agent: **escalate** (`abusive_or_distressed_language`)

**Hypothesis.** The router prompt lists "abusive, or genuinely distressed rather than
merely annoyed" as an escalation trigger, but the model keys on **profanity** rather than
on target or severity. Swearing *about a product* is ordinary Twitter register for this
brand's audience; it is not abuse directed at an agent, and it is not distress. My
deterministic rule for this code is deliberately narrow (self-harm language only), so all
14 come from the model's judgement layer.

**Cheap test that would confirm or kill it.** Take the 14 cases, strip profanity tokens
while preserving meaning, and re-run the router. If escalations collapse, the trigger is
lexical profanity rather than genuine distress. **Cost: 14 LLM calls.**

**Fix if confirmed** (attempted in Phase 8): tighten the prompt to distinguish *profanity
about the product* (auto) from *abuse directed at a person* or *genuine distress* (escalate).

---

## Mode 2 — `feature_request_or_complaint` vs `content_missing_or_metadata`

**Frequency: 8 of 44 intent errors (18%).** My codebook names this pair as the most
confusable and gives an explicit tie-break rule — "asking for **music** → content; asking
for **software** → feature request" — and the agent still gets it wrong.

**Real examples:**
> id=574720 — "hey guys could you get #Hellveto split with galeón?"
> → gold: `feature_request_or_complaint`, agent: `content_missing_or_metadata` (conf 0.80)

> id=466813 — "why isn't <USER>'s Where are you Christmas on here?? It would make my Christmas playlist a 12/10🔥🔥🔥"
> → gold: `feature_request_or_complaint`, agent: `content_missing_or_metadata` (conf 0.80)

**Hypothesis.** These are genuinely **label disputes, not model errors** — and I think the
agent is arguably right on both. "Please add this album" is a catalogue request, which my
own codebook assigns to `content_missing_or_metadata`. The Pass-A labeller put them in
`feature_request_or_complaint` because they are phrased as *requests* rather than as
reports of absence. The boundary is defined by **phrasing**, not by what a support agent
would do — and a support agent answers both identically. Note the agent's confidence is
0.80 on both: it is not confused, it disagrees.

**Cheap test.** Have a human adjudicate these 8. If the human sides with the agent, the
error is in my labels and the correct fix is to merge or re-draw the boundary, not to
touch the model. **Cost: 8 items in `label_cli.py`, ~4 minutes.** This is exactly what the
human review queue exists for.

---

## Mode 3 — `praise_or_chitchat` collapsing into `other`

**Frequency: 8 of 44 intent errors (18%), the single largest confusion pair.** Notably,
agent confidence on these is **0.3–0.4**, far below its usual 0.8 — the model knows it is
guessing.

**Real examples:**
> id=538061 — "Yo <USER> try this out during 2018 and compensate me <URL>"
> → gold: `praise_or_chitchat`, agent: `other` (conf 0.30)

> id=2915271 — "I enjoy my default pattern when wondering 'is service X down' is to search for it on <USER> and see if others are complaining. Looks li…"
> → gold: `praise_or_chitchat`, agent: `other` (conf 0.40)

**Hypothesis.** Both labels are defensible and the distinction carries **no routing
consequence** — both are auto-handled with a light reply. The examples are rambling social
posts that are neither praise nor a support request, so they sit exactly on the
`praise_or_chitchat` / `other` seam. My codebook's rule ("`other` is a real label, not a
confidence signal") is too weak to separate them.

**Cheap test.** Check whether these 8 differ in *routing* outcome. If they never do
(they don't, in this sample), the two classes are functionally identical for the product
and macro-F1 is being penalised for a distinction that does not matter. **Cost: zero, it
is a query over `errors.jsonl`.**

**Implication.** This is an argument for reporting a **routing-weighted** intent metric
alongside macro-F1 — a confusion that changes no decision should not cost the same as one
that does.

---

## Mode 4 — `app_bug_or_crash` predicted as `playback_streaming_issue`

**Frequency: 5 of 44 intent errors (11%).** The other pair my codebook explicitly flags,
with the rule "app process fails → bug; audio fails while the app runs → playback".

**Real examples:**
> id=587519 — "My <USER> music stops every time I open snapchat. I guess this is the end to my food snaps 😭"
> → gold: `app_bug_or_crash`, agent: `playback_streaming_issue` (conf 0.80)

> id=453448 — "When <USER> / just doesn't work anymore after being in Denmark for 4 months <URL>"
> → gold: `app_bug_or_crash`, agent: `playback_streaming_issue` (conf 0.70)

**Hypothesis.** The customer describes the **symptom** ("music stops") and the label
requires inferring the **cause** (an OS-level audio-focus bug triggered by another app).
That inference is not present in the text. The tie-break rule is well-posed but
unanswerable from a single tweet — it asks the classifier to know something the customer
did not say.

**Cheap test.** Re-classify these 5 with the *brand's actual reply* appended (the reply
often reveals whether Spotify treated it as a bug). If accuracy jumps, the information is
genuinely absent from the customer turn alone, and the taxonomy is asking for more than
the input contains. **Cost: 5 LLM calls.**

---

## Mode 5 — Missed escalations: billing and security phrased calmly

**Frequency: 7 of 220 (3.2%).** The most consequential mode, because under the 10:1 cost
model these 7 dominate the agent's expected cost. Notably **0 of 32 hard-flagged examples
were missed** — all 7 came from the "easy" slice, which is the opposite of what I expected.

**Real examples:**
> id=2881633 — "hey guys, sent in an email almost a week ago. How long do you normally take to reply? Billing issue."
> → gold: **escalate** (`billing_dispute`), agent: **auto** (`informational_answer`), intent correctly `billing_charge_dispute`

> id=993168 — "hello Spotify has taken 9.99 out my bank when I have the student spotify?"
> → gold: **escalate** (`billing_dispute`), agent: **auto** (`known_troubleshooting_flow`), intent correctly `billing_charge_dispute`

**Hypothesis.** In every one of the 7, **the intent classifier was right and the router
still auto-handled it.** The deterministic billing rule requires explicit phrasing
("charged twice", "double charge", "unauthorized charge"); "has taken 9.99 out my bank
when I have the student spotify" is a genuine charge dispute that matches none of those
patterns, so it fell through to the model layer, which read it as a calm question. The
failure is that **the rule layer keys on phrasing while the intent head already knows the
category** — the two are not talking to each other.

**Cheap test / fix.** Add a rule: if `intent ∈ {billing_charge_dispute,
account_access_login}` **and** the message contains a monetary amount or a security noun,
escalate regardless of phrasing. Re-run and count missed escalations. **Cost: one rule, one
eval re-run.** This is the Phase 8 fix, because it targets the metric that matters most
and it exploits information the pipeline already has but discards.

---

## What I did not find

- **No hallucinated policy in the sample I read.** Judge safety scores average 4.92/5 and
  the drafter's hard constraints appear to hold. I would not claim this generalises from
  220 examples.
- **No parse failures at all** — 0/220 on classify, 0/220 on draft, 0/216 on route. The
  structured-output path is reliable at this scale.
- **No length-cap violations** — 0/220 replies exceeded 280 characters.
- **The hard slice was not harder.** Accuracy on hard-flagged examples was 84.4% vs 79.3%
  on easy, and hard examples produced 0 missed escalations. My hard-case heuristics select
  for *sarcasm, brevity and escalation keywords*, and escalation keywords make routing
  **easier**, not harder — the deterministic rules fire on exactly those. The quota
  therefore did not do what I intended, and I report it rather than quietly dropping it.
