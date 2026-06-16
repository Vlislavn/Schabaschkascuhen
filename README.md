# Schabaschkascuhen

A local ($0) nightly dream-job finder for **Alina** — the Heidelberg/Frankfurt market. Every night:
collect jobs → geo/hard filters → dedup → **a cheap ML gate** drops the junk → LLM normalizes each into a
card → an LLM judge scores 1–5 against her rubric → **CV↔job matching** (`fit_score = 0.7·HyRE +
0.3·bge-m3 sparse`; a cross-encoder + per-requirement LLM coverage shown on the card, plus an eligibility
gate; the magnet judge is de-conflated from fit) → an agentic deep-search of the top jobs → a morning
slate of 10 on a **💻🐀 “office mouse” → 👸✨🧚 «шабашка»** scale. Every click teaches the system.
Everything is local (ollama qwen3:8b + bge-m3 + bge-reranker).

> **«шабашка»** (*shabashka*) is Russian slang for the jackpot side-gig — here, the dream job; the product
> is named after it (see the [Glossary](#glossary)). The web UI ships **English by default with a 🇷🇺
> Russian toggle** in the top-right of every page.

Docs: [docs/USE_CASE.md](docs/USE_CASE.md) · [docs/ANNOTATION.md](docs/ANNOTATION.md) (how to rate) ·
[config/profile.example.yaml](config/profile.example.yaml) (profile/rubric/settings template → copy to `config/profile.yaml`).

## Interface

The daily web surface (English shown; click **Русский** top-right to switch). Regenerate these images with
`python -m scripts.gen_screenshots` — they render from **synthetic demo data** (no real listings).

**`/` — daily slate** (8 best + 2 "explore" wildcards; themed 💻🐀→👸✨🧚 score chip, collapsible skills
breakdown, deep-dive, deterministic listing check; one click per card)

![Daily slate](docs/screenshots/slate.png)

**`/annotate` — rating queue** (every judged-but-unrated job; teach the system from scratch, one click each)

![Annotation queue](docs/screenshots/annotate.png)

**`/eval` — live match validation** (how well the ranking tracks **your** ratings — pairwise / NDCG@10 /
spearman per signal) · **`/gaps` — recurring skill gaps** (which requirements keep coming up missing across
the jobs you want — what to add to the CV or learn)

![Validation](docs/screenshots/eval.png)
![Skill gaps](docs/screenshots/gaps.png)

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,v2]"                 # schabasch + bge-m3/LightGBM + tests
#   (or, faster, from the lockfile:  uv sync --extra dev --extra v2)
pip install "git+https://github.com/Vlislavn/JobSpy.git"   # Indeed/LinkedIn (public fork)
ollama pull qwen3:8b                        # local LLM (server on :11434)
cp config/profile.example.yaml config/profile.yaml         # → fill in your profile (profile.summary)
# optional (agentic deep-dive investigate/discover; everything works without it — the step degrades gracefully):
# pip install -e ~/code/work/KatherLab/prototype-internal-KL
```
> The first `tick`/`features` run downloads bge-m3 + bge-reranker (~1.5 GB) once — slow that time, cached after.

## Daily use

One command, one surface — a web page. Full guide: [docs/ANNOTATION.md](docs/ANNOTATION.md).
```bash
schabasch serve                            # http://127.0.0.1:8787
```
- **`/`** — the morning slate of ~10 (8 best + 2 wildcards), **fresh only** (last_seen ≤ `fresh_days`,
  14 days). Card: the **💻🐀→👸✨🧚** score chip (gradient over 1–5) · "why" · ⛔ eligibility / ⚠ gap ·
  date ("posted" or "found N days ago" when the board gives no posting date) · **🎯 skills breakdown**
  (llm_cov % + ✓/◐/✗ per requirement) · 🔎 verified (company/salary/English team + a **deterministic**
  "open / closed / not checked" listing check). Buttons 💻🐀/😎/👸✨🧚/applied — every click teaches it.
- **`/annotate`** — the single annotation surface: the queue of judged-but-unrated jobs (same cards and
  💻🐀/😎/👸✨🧚 buttons; **no freshness ceiling** — it's the backlog). The system learns from scratch here;
  rated jobs leave the queue. (There used to be an xlsx pack — retired in favour of this one surface.)
- **`/eval`** — match validation against YOUR real labels (live, updates as you rate): pairwise/NDCG/
  spearman per signal. "Clean" signals (fit_score/cross-encoder/coverage/eligibility) don't see the labels
  — an honest estimate; the judge/triage are marked "⚠ trains on labels". A synthetic GOLD stays CLI-only
  (`python -m eval.match_eval`; `--real-labels` uses the labels).
- **`/gaps`** — which skills are regularly missing for the jobs you WANT (😎/👸✨🧚/applied): an aggregate
  over the requirement breakdown (`llm_cov_reqs`), ranked by "missing most often" — candidates for the CV / to learn.

After ~30 labels the ML gate (LightGBM) kicks in; ~50–100 calibrates the judge (`schabasch cv`).

## Nightly cycle

```bash
schabasch tick               # canary→scrape→filters→dedup→features→triage→normalize→judge→investigate→slate
schabasch tick --german      # with the German query matrix
schabasch funnel             # the funnel + canaries (dead scraper vs empty market)
```
Scheduler (macOS launchd, 03:00) — fill in your paths (`__REPO_DIR__`/`__VENV_PYTHON__` in the plist) and install:
```bash
sed -e "s|__REPO_DIR__|$PWD|" -e "s|__VENV_PYTHON__|$PWD/.venv/bin/python|" \
  deploy/com.schabasch.nightly.plist > ~/Library/LaunchAgents/com.schabasch.nightly.plist
launchctl load -w ~/Library/LaunchAgents/com.schabasch.nightly.plist
```

## Individual steps (optional)

```bash
schabasch candidate "<description or path to a CV>"   # extract the candidate profile (for matching)
schabasch features          # bge-m3 embeddings + aspect CV↔job matching
schabasch triage            # ML gate: must/should/could/drop (cheap cut before the LLM)
schabasch triage-train      # train the LightGBM gate on labels (≥30 labels)
schabasch cv                # 5-fold judge↔labels agreement (the daily-run gate)
schabasch discover          # agentic search beyond the boards (opt-in, needs kl_agent_builder)
schabasch investigate       # deep-search the top-N (company/salary + deterministic listing check)
schabasch rerank            # cross-encoder + LLM coverage → fit_score (de-conflates the slate ranking)
schabasch gaps              # recurring skill gaps across the jobs you want (😎/👸✨🧚)
```

> **Matching (fit-gate):** the real CV↔job signal `fit_score = 0.7·HyRE + 0.3·bge-m3 sparse` is computed by
> `rerank` (HyRE = cosine to an ideal résumé; bge-m3 sparse = deterministic lexical; a cross-encoder and
> per-requirement LLM coverage are also computed and shown on the card). The nightly `tick` runs `rerank`
> before `slate`; if you call `slate` standalone, run `rerank` first, else fit=0 and ranking falls back to the judge.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q       # 348 tests, no network/ollama (LLM and bge-m3 are mocked)
```

## Architecture

```
candidate.py  candidate profile (LLM extraction from the CV + aspect texts)
features.py   bge-m3 + bge-reranker: fit_score = 0.7·HyRE + 0.3·bge-m3 sparse; cross-encoder + llm_cov on the card
aspects.py    job segmentation (EN+DE) + aspect scoring
eligibility.py  hard requirements (education/PhD/language) → eligibility gate (de-conflated from fit)
triage.py     LightGBM gate (cold-start = match_score), calibration, drop-bucket before the LLM
normalize.py  qwen3:8b → card (language reality, hybrid, slop-score)
judge.py      qwen3:8b 1–5 against the rubric + few-shot from the extreme labels (💻🐀=2 / 👸✨🧚=5)
investigate.py  deep-search agent: company/salary/culture + DETERMINISTIC listing check
i18n.py / locales/   data-driven UI translations (en/ru) — add a language by dropping a locales/<lang>.json
metrics.py / validation.py / gaps.py   ranking metrics · /eval (vs real labels) · /gaps (skill gaps)
slate.py / feedback_app.py   slate 8+2 + /annotate + /eval + /gaps + FastAPI feedback; 💻🐀→👸✨🧚 scale, EN/RU toggle
sources/      jobspy (Indeed/LinkedIn) · arbeitsagentur · agent_discovery (opt-in) · tertiary
pipeline.py / cli.py   the nightly tick, spike import, funnel, gaps
```

Status FSM: `new → prefiltered | described → normalized → filtered | scored → slated → labeled | expired`.
Config — `config/profile.yaml`. Notes on agents/tokens are in `schabasch/agent_runtime.py` comments.

## Glossary

The product's flavour is Russian slang, kept on purpose (it's the brand) and explained here:

- **«шабашка» (*shabashka*)** — a lucrative side-gig / jackpot job; here **the dream job**. Score **5**, emoji **👸✨🧚**.
- **office mouse** (Russian «офисная мышь») — a dull, boxed-in cubicle job. Score **1**, emoji **💻🐀**.
- The 1→5 scale runs **💻🐀 → 🐭 → 😐 → 😎 → 👸✨🧚** on a grey→gold gradient.
- **magnet** — a domain Alina is drawn to (space · animals · military-security · public-sector ·
  complex-projects · new-domain); **repellent** — a turn-off (hidden-German · biotech · slop-text ·
  boring-role · remote-only · temp-agency). The system assigns them; her 💻🐀/😎/👸✨🧚 click is the real signal.
