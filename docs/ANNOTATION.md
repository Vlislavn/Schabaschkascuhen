# Schabaschkascuhen — Annotation Guide

This is how you teach the job-finder what a great job looks like for **you** — all on one local
web page, with one click per job. The system learns from every click and gets sharper over time.
Nothing is sent anywhere; it all stays on this Mac.

There are **two views of the same page**:

- **`/annotate`** — the **rating queue**: every job the system has scored but you haven't rated
  yet. This is how you teach it from scratch — rate a batch whenever you have a few minutes. Rated
  jobs disappear from the queue.
- **`/`** — the **daily slate**: today's top ~10 (8 best + 2 wildcards). Your everyday view.

Both show the **same card** and the **same buttons**. (There used to be a 100-row Excel pack for
the cold start — it's been retired; the rating queue replaces it.)

---

## Your rubric — one click per job

| Button | Meaning | Records |
|:------:|---------|:-------:|
| 💻🐀 | not for me («офисная мышь», office mouse) | score 2 |
| 😎 | yes, interesting | score 4 |
| 👸✨🧚 | dream job («шабашка», the dream/jackpot gig) | score 5 |
| **applied** *(daily slate only)* | I applied to this | a flag (keeps your score) |

The score chip on each card runs the same scale — **💻🐀 «офисная мышь» (1) → 👸✨🧚 «шабашка» (5)** —
on a grey→gold gradient, so you read the verdict at a glance.

What a **👸✨🧚** looks like is profile-specific: strong fit, a genuinely exciting domain or role,
acceptable location/work mode, and no hard blocker. A **💻🐀** is a clear miss: hidden requirements,
remote/work-mode mismatch, staffing-agency spam, slop text, or just a boring office-mouse role.

Your **profile** in `config/profile.yaml` defines the current strengths, target roles, magnets, and
repellents. Magnets and repellents show on the card as the "why" tag — the system assigns them; your
💻🐀/😎/👸✨🧚 click is the real signal.

> 💡 Don't overthink it. ~10–20 seconds per job. Your gut reaction is exactly what it's learning.

---

## How to use it

**1. Start the page (once per terminal, activate the env first):**
```
source .venv/bin/activate
python -m schabasch.cli serve
```

**2. Open it in your browser:**
```
http://localhost:8787/annotate     # the rating queue — start here to teach the system
http://localhost:8787/              # today's slate
http://localhost:8787/eval         # validation: how well matches track YOUR ratings (live)
http://localhost:8787/gaps         # recurring skill gaps across jobs you WANT (😎/👸✨🧚/applied)
http://localhost:8787/funnel       # pipeline funnel (scrape→…→slate counts, scraper health)
```

**3. What's on each card:** the **💻🐀→👸✨🧚 score chip** (gradient by 1–5) · title · company · city ·
work mode · the posting date (*published N days ago* — or *found N days ago* when the board gives no posting
date) · a collapsible **🎯 Skills {%}** breakdown (✓ have / ◐ partial / ✗ missing per requirement).
A red **⛔** banner means you don't meet a hard requirement (e.g. a PhD you don't have) — the one
"stop" signal. Top picks also show a **🔎 verified** line: company size, salary, "English-speaking team",
and a **deterministic** listing check — "open ✓" / "⚠ listing closed" (confirmed 404) /
"ℹ listing not checked" (couldn't check — never a false "closed").

**4. Click one button per card.** The card dims and shows a ✓; a counter at the top tracks your
progress (e.g. *Marked 7/30*). Misclicked? Click **↶ change** to re-enable the card and pick
again. On the daily slate, rated cards won't come back tomorrow.

- **applied** (daily slate) is a flag *on top of* your score — clicking it doesn't invent a 5. If
  you want to record how good it was, click 😎/👸✨🧚 as well.
- Fewer than 10 cards on the slate is normal — the system never pads the list with junk.
- If a scraper died overnight, a red banner appears at the top so you know the list might be thin.
- Press `Ctrl-C` in the terminal to stop the page when you're done.

---

## What happens after you click

- Your labels go into a private `label` table (your "golden dataset") — nothing leaves the Mac.
- The **judge** (the LLM that scores jobs) learns from your 👸✨🧚 (dream) and 💻🐀 (not-for-me) examples.
- Once you've given **~30** labels, a small **ranking model** starts learning your taste and pushes
  the bottom-of-the-barrel jobs out before they ever waste a slot. Around **50–100** the judge gets
  a calibration check (`schabasch cv`).
- The **`/eval`** page shows, live, how well the matcher's ranking tracks *your* ratings — the more
  you rate, the more reliable that number (it tells you when there's enough).
- The more you click, the better tomorrow's list.

---

## Quick reference

| I want to… | Command / URL |
|------------|---------------|
| Rate the queue (teach from scratch) | `python -m schabasch.cli serve` → http://localhost:8787/annotate |
| Open today's slate | http://localhost:8787/ |
| See how well matches track my ratings | http://localhost:8787/eval |
| See recurring skill gaps (what to add/learn) | http://localhost:8787/gaps |
| Run a fresh nightly collection | `python -m schabasch.cli tick` |
| See the pipeline funnel | http://localhost:8787/funnel |

> Activate the environment once per terminal: `source .venv/bin/activate`

---

## FAQ

**How many should I rate?** The more the better. ~30 is where the ranking model kicks in; ~50–100
gives a solid start. Do it in small batches from the `/annotate` queue.

**What if I'm unsure?** Pick 💻🐀 if it's clearly not for you, 😎 if you'd be curious, 👸✨🧚 only for a
real dream job. When torn between 😎 and 💻🐀, lean 💻🐀 — the system errs toward fewer, better jobs.

**A job is in a magnet domain but through a staffing agency — what then?** 💻🐀. A repellent
(temp-agency) beats a magnet.

**Can I change a rating I already gave?** Yes — click **↶ change** on the card and pick a different
button, or just rate it again next time it appears.

**Is anything uploaded?** No. Everything is local: the database and the LLM.
