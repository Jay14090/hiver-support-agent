# Intent Codebook — SpotifyCares

The label definitions used for (a) the classifier prompt, (b) both pre-label passes, and
(c) human adjudication. Every example below is a **real, PII-scrubbed message** from the
corpus split (never dev/test — those are held out).

**How this taxonomy was built.** 3,000 corpus messages were embedded with
`all-MiniLM-L6-v2` and clustered. HDBSCAN found **0 clusters (100% noise)**; KMeans
silhouette was **~0.04 at every k from 6 to 20**, i.e. there is very little natural
cluster structure in short support tweets. k=9 was chosen inside the target band and the
clusters were then **edited by hand**. The clusters established the topic *mass*
(content-missing alone is ~50% of the sample) and produced three of the intents below;
the remaining intents were confirmed by targeted keyword probes over the corpus
(`scripts/intent_evidence.py`) and by reading messages. This is stated plainly because
"induced from the data" should not be allowed to imply more statistical support than
actually exists.

**The tie-break rules are the load-bearing part.** Most disagreement between two
annotators on this data is not "which topic" but "which of two adjacent topics", so each
intent names its most confusable neighbour and how to resolve it.

**Global rules, applied before any per-intent rule:**
1. Label what the customer **wants resolved**, not how they feel. Anger is not an intent.
2. **Multi-intent → pick what a support agent would act on first.** A message that is
   both a billing problem and a playback complaint is billing: money outranks annoyance.
3. **Thread context counts.** "still not working" inherits the intent of the earlier turn.
4. Use `other` only when nothing else fits — not as a dumping ground for "unclear".

---

## `playback_streaming_issue`
**Definition:** Music will not play, stops, skips, stutters, buffers, has no sound, or
downloaded/offline content will not play or has vanished.

**Positive examples (real):**
- "My music keeps stuttering/skipping ever since I upgraded to iOS 11.. tried reinstalling but it doesn't fix it"
- "the web player randomly skips it mid song"
- "Why does my downloaded music magically disappear every few days!!??"

**Near-miss negatives:**
- "Why isn't the Chronic on Spotify??" → `content_missing_or_metadata` (the track was never there; nothing is broken)
- "app just crashes when I hit play" → `app_bug_or_crash` (the application fails, not the stream)

**Tie-break vs `content_missing_or_metadata`:** if the customer previously *could* play
it and now cannot, it is playback. If it was never available, it is content.
**Tie-break vs `app_bug_or_crash`:** does the app keep running? If the app survives and
the audio fails → playback. If the app dies → bug.

---

## `account_access_login`
**Definition:** Cannot log in, locked out, password/email problems, or the account is
suspected compromised.

**Positive examples (real):**
- "It's been a while now since I can't log in to my Spotify account. It's about 2 months now."
- "I think my account got hacked! I got an email that it changed to a diff rando email?"
- "Do you really do NO verification of email changes? My account was hacked"

**Near-miss negatives:**
- "I can't renew my Student Premium because Spotify won't verify I'm a student" → `plan_or_premium_management` (they are logged in; the *plan* is the problem)
- "my premium isn't active after paying" → `plan_or_premium_management`

**Tie-break vs `plan_or_premium_management`:** can they get *into* the account? If yes,
it is a plan problem, not an access problem.
**Escalation note:** suspected compromise is a mandatory escalation regardless of intent.

---

## `billing_charge_dispute`
**Definition:** A specific charge is wrong — charged twice, charged after cancelling,
unauthorised charge, wrong amount.

**Positive examples (real):**
- "i got charged twice this month??"
- "I do not have an account with you guys, but I have recently had unauthorized charges on my bank account"
- "signed up for the $4.99 student discount with hulu and just got charged $9.99 twice..I need my money."

**Near-miss negatives:**
- "how do I cancel my subscription" → `subscription_cancel_refund` (no disputed charge yet)
- "how much is premium in India" → `plan_or_premium_management`

**Tie-break vs `subscription_cancel_refund`:** a dispute is about money **already taken**;
a cancellation is about **future** money.
**⚠ Rarity warning:** only 3 clear matches in 6,791 corpus messages. Billing is
overwhelmingly handled in DM, so this intent is severely under-represented in public
data. Per-class metrics for it will be unstable and are reported with support counts.

---

## `subscription_cancel_refund`
**Definition:** Wants to cancel, unsubscribe, stop being billed, or asks for a refund
where no specific charge is being disputed.

**Positive examples (real):**
- "i need cancel my account with U, but i dont know how, in my spotify profile page, i dont see any option to do that"
- "where is the 'if you ever recommend another edited song to me I'm gonna cancel my account' button?"
- "should I cancel my premium and go elsewhere?"

**Near-miss negatives:**
- "I got charged twice, I want my money back" → `billing_charge_dispute` (a specific charge)
- "I'm cancelling if you don't add an LG TV app" → `feature_request_or_complaint` (a threat used as leverage; the *want* is the feature)

**Tie-break vs `feature_request_or_complaint`:** is cancelling the **request** or the
**threat**? Only the request lands here.
**Escalation note:** an explicit cancellation intent is a mandatory escalation (retention).

---

## `plan_or_premium_management`
**Definition:** Anything about which plan the customer is on — student verification,
Family/Duo membership, upgrading or switching, Premium not activating after payment.

**Positive examples (real):**
- "i have been verified for student account but my premium is yet to be active, and the $5 have been [taken]"
- "I cannot see or change my family plan!"
- "is it possible to add my student discount to my existing Spotify premium account?"

**Near-miss negatives:**
- "I got charged twice for premium" → `billing_charge_dispute`
- "can't log into my account" → `account_access_login`

**Tie-break vs `billing_charge_dispute`:** is the complaint that money moved **wrongly**
(dispute), or that the **entitlement** did not follow the money (plan management)?
*This distinction is the single most confusable pair in the taxonomy and was the reason
`plan_management_family_duo` was widened to `plan_or_premium_management`.*

---

## `content_missing_or_metadata`
**Definition:** A song, album or artist is absent from the catalogue, or catalogue
metadata is wrong (wrong artist page, wrong album art, mis-attributed track).

**Positive examples (real):**
- "WHY IS THE CHRONIC NOT ON SPOTIFY?? what is going on here? 😡"
- "you added an incorrect album to this artists page - this is Lydia (band, USA) and the album is from Lydia (singer, Japan)"
- "Hmm, I can't seem to find [artist]'s new album. Why is that?"

**Near-miss negatives:**
- "my downloaded songs disappeared" → `playback_streaming_issue` (they had it; now it will not play)
- "please make Spotify available in India" → `feature_request_or_complaint` (market availability, not catalogue)

**Tie-break vs `playback_streaming_issue`:** see that intent's rule — did it ever work?
**Volume note:** this is the single largest intent in the public data (~50% of the
induced cluster mass). It is also the one Spotify can least often *fix*, which is why it
matters that the router does not treat volume as importance.

---

## `device_integration_issue`
**Definition:** Spotify will not work with a specific external device or platform — car
/ CarPlay / Android Auto, Bluetooth speakers, Alexa, Google Home, Sonos, Chromecast,
PlayStation, Xbox, smart TVs, wearables.

**Positive examples (real):**
- "my Spotify app is not working on my PlayStation 4 I'm having trouble connecting to the servers help HELP"
- "can't control Spotify on my home screen since iOS software update"
- "your apps are incompatible and that's super lame" (re: a paired device)

**Near-miss negatives:**
- "why don't you have an Apple Watch app?" → `feature_request_or_complaint` (asking for something that does not exist)
- "the phone app crashes" → `app_bug_or_crash` (no second device involved)

**Tie-break vs `feature_request_or_complaint`:** does the integration **exist and fail**
(device issue) or **not exist yet** (feature request)? This is the rule that keeps the
large "please make an Apple Watch app" cluster out of this intent.

---

## `app_bug_or_crash`
**Definition:** The application itself misbehaves — crashes, freezes, will not open,
black screen, UI elements broken or missing.

**Positive examples (real):**
- "there's an intermittent glitch on the iOS app where the bottom taskbar disappears if Waze is also in use"
- "everytime I seem to go on a flight my Spotify crashes and I lose all my saved songs"
- "Please fix your app for Windows Phone 10 ... your app blows"

**Near-miss negatives:**
- "songs keep skipping" → `playback_streaming_issue`
- "I wish the app had a sleep timer" → `feature_request_or_complaint`

**Tie-break vs `playback_streaming_issue`:** app process fails → bug. Audio fails while
the app runs → playback.

---

## `feature_request_or_complaint`
**Definition:** Asks for something that does not exist, or complains about how an
existing, *working* feature is designed. Includes market/geographic availability
requests and "please make an app for X".

**Positive examples (real):**
- "I wish Spotify would allow you to add a new song to the top of a playlist instead of the bottom."
- "please make your services available in INDIA. There will be no other competition here."
- "your shuffle for playlist suck, only flaw though"

**Near-miss negatives:**
- "shuffle stopped working yesterday" → `app_bug_or_crash` (it worked before; now broken)
- "please add Trout Mask Replica by Captain Beefheart" → `content_missing_or_metadata` (a catalogue request, not a product feature)

**Tie-break vs `app_bug_or_crash`:** working-as-designed but disliked → feature request.
Previously working, now broken → bug.
**Tie-break vs `content_missing_or_metadata`:** asking for **music** → content. Asking
for **software** → feature request.

---

## `praise_or_chitchat`
**Definition:** Thanks, compliments, jokes, or social chatter with no support request at
all.

**Positive examples (real):**
- "THANK YOU SO MUCH! I ALWAYS PLAY THAT PLAYLIST DON'T WORRY!"
- "cheers for the follow guys! Love Spotify!"
- "you guys are the best 💚"

**Near-miss negatives:**
- "Love Spotify! but can you add a remove-all button?" → `feature_request_or_complaint` (praise wrapping a request)
- "thanks, that fixed it!" mid-thread → still `praise_or_chitchat` (the request was resolved earlier in the thread)

**Tie-break rule:** if removing the pleasantries leaves **any** actionable request, it is
not chitchat. Praise is a wrapper, not an intent.

---

## `other`
**Definition:** A genuine support message that fits none of the above, or is too vague
to assign even with thread context.

**Positive examples (real):**
- "I have an issue I can't find answer to on the Spotify website. Need help please"
- "You ok spotify ??"
- "I still have no fucking idea how to use Spotify."

**Rule:** `other` is a real label, not a confidence signal. If the message is clearly a
playback problem but poorly written, it is still `playback_streaming_issue`. Reserve
`other` for messages whose *topic* is genuinely unidentifiable, and expect the router to
escalate most of them.
