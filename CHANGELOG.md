# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Triage gate: ship only a model that beats its forward test; out of the slate ranking (2026-06-18)
`/eval` showed triage at 42% / −0.14 (worse than a coin-flip). Investigation (receipts): the model's
OWN temporal holdout is negative (saved artifact `spearman_rho −0.31`) yet `train()` saved + served it
anyway, and the eval row was computed over a 2/3-cold_start stored-score sediment (random-CV flatters
it to +0.39; the temporal split is the honest forward read). Fixes: **(R1)** NEW deploy gate
`triage._should_deploy` — `train()` KEEPS the prior model rather than overwrite it with one whose
temporal-holdout spearman ≤ `triage.deploy_min_spearman` (default 0.0; NaN/None = uncomputable → don't
block, only a measured failure rejects). **(R2)** triage REMOVED from the slate sort tie-break
(`slate.build_slate`) — it's a drop-filter, not a ranker; fit-led order unchanged. **(R3)** the `/eval`
triage row now measures ONLY actual ML decisions (`model_version != 'cold_start'`), relabeled
"ML-гейт (triage, drop-фильтр)", omitted when <10 ML-scored — so it reads as honest gate-health
(≈random) instead of a broken-matcher number. Tests: `test_should_deploy_gate` + deploy-gate
reject/ship end-to-end.

### Stale & expired vacancies kept off the slate (2026-06-18)
The morning slate carried expired Indeed jobs + a 14-month-old Arbeitsagentur posting. Fixes:
(1) NEW `freshness.too_old` + an INGESTION date gate (`search.max_post_age_days`, default
`ceil(hours_old/24)`) drops a posting older than the window in `arbeitsagentur.search` /
`jobspy_source.scrape` (null date passes) — kills the AA `veröffentlichtseit` flood (`ingest_stale_skip`
funnel). (2) `slate.fresh_days` 14→7. (3) NEW `pipeline.verify_liveness` re-verifies stale top
SCORED/SLATED cards and EXPIREs the confirmed-gone — AA via its API, Indeed via NEW
`sources.indeed.check_open(jk)` (jobspy TLS session + the embedded `"expired"` JSON boolean; the
`abgelaufen` banner is React boilerplate → a false positive, spike-confirmed against the user's
labels). `None` never false-closes; bounded by `retention.liveness_recheck_max`. (4) the tick/​fetch
slate is now `rebuild=True` so mid-day fresh jobs surface (labels preserved). (5b) a `/feedback` note
matching an expiry pattern (`abgelaufen`/expired/too-late/`не могу податься`/…) → EXPIRED, not just
dedup-hidden. (5c) NEW `slate.recent_reject_penalty` demotes role-kinds the user keeps scoring ≤2
(engineer/intern) in the SORT key only (`slate.pref_penalty_days/floor`; effective_score +
quality_floor + /eval stay pure). Regression tests: `test_freshness`, `test_indeed_liveness`,
`test_expire` (verify_liveness), `test_feedback_app` (auto-expire), `test_slate` (pref penalty).

### Real-usage fixes: direction feedback · /more backlog · honest refetch delta (2026-06-18)
From 4 Telegram-usage problems (laws-of-UX gated). ② NEW `/feedback action=direction` (🧭) — "wrong
specifics, right direction": low score removes THIS vacancy from the slate while a positive
`role_feedback fits=1` + the magnet why_tag boost the DOMAIN; web card + bot button (row 2, Hick's).
③ NEW `/backlog.json` (reuses `annotation_batch`) → the bot's `/more` walks the judged-but-unrated
pool beyond the ≤10 daily slate. ①④ `nightly_tick` now computes an honest `delta` {scraped, NEW,
reseen, slate_size, slate_new, wall_seconds} (new = first_seen this tick; no upsert signature change),
surfaced in `/fetch-status` → the bot replaces "0 шабашек (0 новых)" with "scraped X, NEW 0, slate
unchanged" + `/why`. Economy: opt-in `search.linkedin_hours_old` (tighter LinkedIn window). M3 verify
fix: rerank xenc-skip uses one query, not N× `feature_row`. (`models.FEEDBACK_TO_SCORE` += direction,
additive.)

### Browsing adapters completed: keyless search + config-driven agent backend (2026-06-17)
Finished the deferred browsing adapters. `browsing/search.py` (keyless ddgs/DuckDuckGo, optional
self-hosted SearXNG via `browsing.searxng_url`) — live-verified (real Kununu results). `extract.clean`
live-verified on a real page (terma.com, 15.6k chars). `agent_runtime.build_agent` now sets the agent's
search backend from config (`browsing.search_backend`, default ddg — NEVER tavily; no API keys).
`browsing/registry.py` added but the `deutschland` (Handelsregister) backend is REJECTED for cause —
it downgrades numpy 2→1.26 + pulls an OCR stack, breaking the bge-m3/lightgbm pipeline (see
docs/IMPORT_AUDIT.md); the adapter degrades to a no-op, grounding stands on Wikidata + legal-suffix.
`docker-compose.searxng.yml` for opt-in multi-engine search.

### SOTA employer research: Wikidata entity resolution + hard-before-soft grounding (2026-06-17)
New `schabasch/browsing/` keyless adapter layer (imported packages behind a stable boundary, not
hand-rolled scrapers): `entity.resolve` (Wikidata wbsearchentities/wbgetentities — typed, name-matched
entity → canonical identity + official_site/country/employees/inception) and `extract.clean`
(trafilatura). `validate_company` is now a ladder Wikidata→Wikipedia(guarded)→legal-suffix, and
`_investigate_row` resolves the employer identity BEFORE the agent and injects it (hard-before-soft) so
the tight agent stops guessing who the company is. Fixes entity disambiguation at the source (Terma→
"Therme" gone; Phoenix Medical now a typed medical co with a country flag). `company_knowledge` gains
wikidata_qid/official_site/country/employees/inception; the backfill re-grounds all employers via the
ladder. llm.http_get_json now retries 429/502/503/504 (transient) — keyless APIs rate-limit in bursts.

### Seed employer DB from existing data + tighten Wikipedia name-match guard (2026-06-17)
`scripts/backfill_company_knowledge.py` lifts durable employer facts already in `investigation` into
the new `company_knowledge` (18 employers; idempotent, no model/network). Reviewing the data caught
bare-opensearch wrong matches (Terma Group→"Therme Group" spa; Phoenix Medical→"Phoenix Media"), so
`investigate._name_matches` now requires every significant query token to appear in the Wikipedia
article title — wired into `_wikipedia_company` (rejects early) + re-validates stored descriptions in
the backfill (drops the bad ones → live re-research). Reuse gate also ignores thin (no-description)
records so they don't suppress a proper investigation.

### Less redundant pipeline work + persistent employer knowledge base (2026-06-17)
Audit found the ReAct agent (the costliest stage) re-investigating done jobs and re-researching the
same employer per vacancy/tick. Fixes: investigate_top now skips vacancies already in `investigation`
(covers closed jobs too); new persistent `company_knowledge` sidecar (key=normalize_company) caches
durable employer facts — researched once, reused everywhere — with only fresh news/reputation
re-asked past a TTL, folded into the one agent call (config `investigate.company_*`). rerank_scored
skips candidates with a fresh `xenc_full` and won't load the 2GB reranker when none need scoring.

### Fix: features-stage torch/OpenMP deadlock (CAPA) (2026-06-17)
The `features` stage hung indefinitely (47+ min, 0% CPU): a parallel `copy_kernel` on the background
tick thread lost-wakeup-deadlocked with every thread parked in `_pthread_cond_wait` (verified by
`sample`; trigger = ~86% swap). **Corrective:** `features.torch_num_threads` (default 1) caps torch's
OpenMP intra-op pool via `_cap_torch_threads` in `extract_features`/`rerank_scored` (1 = inline
`parallel_for`, no pool, cannot deadlock) + `OMP_NUM_THREADS` set from the same knob at serve start.
**Preventive:** `memory_guard` watchdog now flags pressure on a swap GROWTH-delta
(`memory.swap_growth_floor_mb`, default 512) — macOS free-RAM% is blind to swap saturation.

### Freshness re-rank + accent posting-date badge (2026-06-16)
`build_slate` sorts on `effective · recency_mult` (floor 0.6, halflife 7d) → recent jobs on top;
reorders only (effective_score + quality_floor stay pure fit, so fresh-slightly-worse beats stale-great
without disqualifying it, /eval preserved). Posting date → accent freshness badge by the title (green
≤2d). Config `slate.recency_{halflife_days,floor}`; floor 1.0 = off.

### Can't-qualify floors — clearance/citizenship hard-blocker + 0%-skill floor (MEASURED, shipped) (2026-06-16)
A "Senior Cyber Threat Analyst / Active TS-SCI" (US-citizen-only, 0/8 hard reqs met) topped the slate
via the military-security magnet — a job Alina literally can't qualify for. Two measured ranking floors:
- **`eligibility.jd_hard_blocker`** — deterministic JD regexes (`eligibility.hard_blockers`, e.g. TS/SCI,
  security clearance, US-citizenship) → STRUCTURAL floor like a PhD-position, applied live in
  `recompute_live`. Empty list = off.
- **0%-skill floor** — `slate.zero_cov_mult` multiplies down any card with `llm_cov==0` (meets ZERO
  extracted hard requirements) in `effective_score`; `cov=None` (uncomputed) is never floored. A HARD
  rule at exactly-0, distinct from the cov BLEND that regressed.
- **Measured on the 50 real labels:** guardrail held/improved — blockers neutral (0 labeled matches),
  0-cov floor 0.862→**0.871** pairwise (9 labeled 0-cov jobs now rank correctly low). Enabled in
  profile.yaml (`zero_cov_mult: 0.0`, clearance/citizenship `hard_blockers`). Cyber job → off the slate.

### serve: skip the startup fetch when data is fresh (2026-06-16)
`schabasch serve` no longer re-scrapes on every restart: if a fetch completed within
`serve.refetch_after_hours` (default 12h, off when 0) it reuses the recent slate and the bot greets
immediately. `--refetch` forces a fresh fetch; `--no-fetch` never fetches. Freshness = the last
`slate` funnel stage timestamp (`_hours_since_last_fetch`/`_refetch_guard`).
- **"Not updated → refetch?" prompt in BOTH surfaces:** when the startup fetch is skipped,
  `_FETCH_STATE.fetch_skipped`/`data_age_hours` (exposed on `/fetch-status`) drive (a) a UI alert on `/`
  ("Данные от N ч назад — не обновлялись; нажми обновить, ~15-20 мин") above the existing fetch button,
  and (b) a bot-greeting note + a `🔄 Обновить вакансии` inline button → `POST /fetch` → "обновляю…" →
  re-greets the owner when the fetch finishes (`refetch_cb`/`_notify_refetch_done`).

### Rich startup/per-stage console logging (2026-06-16)
`schabasch serve` was near-silent during its ~15-30 min startup pipeline. Now: a startup BANNER (mode,
pending scale, the stage plan with ⏳ model markers, ETA note), per-stage `▶ start / ✓ done (result, Ns)`
lines with a 30s heartbeat so slow stages (LinkedIn scrape, qwen) never look frozen, phase headers
`[1/3] retrain / [2/3] fetch / [3/3] investigate`, per-card `🔎 [i/N] company …` in the progressive pass,
the slate-ready line, and a `✓ complete in Xm Ys`. `pipeline.VERBOSE` (default on) + `serve --quiet` to
mute. Logging only — counts/funnel/FSM unchanged. (Tip: `curl /fetch-status` shows a live run's stage.)

### Two-axis feedback — "good domain, WRONG role" (Delphi-panel design, P0–P3) (2026-06-16)
A 1–5 rating conflated DOMAIN interest with ROLE fit (Alina rated a VINFAST ML-Engineer 4 = "cool
domain, wrong role" → the matcher learned "wants engineer jobs"). Now the role axis is separate:
- **P0** new `role_feedback.py` sidecar (`label_role`: vacancy_id, role_kind, fits, source) + `POST
  /role-feedback` (classifies role server-side; `--dry` = no-write). Bot: a POSITIVE rating on an
  ambiguous role (engineer/junior/lead) surfaces a conditional `[✅ role fits][🙅 wrong role]` row;
  `role_cb` posts it then advances. `slate.build_slate` cards now carry `role_kind`.
- **P1** `validation.eval_report` adds a DIRECTIONAL `effective_role` row (raw gold masked by 🙅 votes);
  the raw-label `effective` row stays the decisive guardrail (self-confirming row is `clean=False`, never a gate).
- **P2** `role_kind.multiplier(kind,cfg,con)` LEARNS a Beta-smoothed per-kind rate from `label_role`
  when `slate.role_kind_learn.enabled` — replacing the hardcoded 0.7; falls back to the static default
  below `n_min` (behaviour-preserving: off ⇒ today's 0.862/0.571 baseline exactly).
- **P3** `eval/role_ablation.py` gate runner: learning OFF vs ON on real labels; ships only if the raw
  guardrail doesn't regress. Demo (8 seeded votes): engineer mult 0.7→0.555, directional 0.745→0.754,
  guardrail 0.862 held. Frozen contracts untouched (additive sidecar). Reviewed → SHIP, no blockers.

### Slate de-dup + per-mode chat lock + measured matching experiment (2026-06-16)
- **De-dup:** the daily slate no longer re-serves a vacancy you already RATED (`label` row); a shown-but-
  unrated card decays out after `slate.max_reshows` (default 2) days. `_load_scored` gains the exclusion
  (daily path only; /annotate keeps the full backlog). `build_slate(..., rebuild=True)` + `schabasch slate
  --rebuild` drop today's saved slate so it applies now. (Fixes 8/10 of the slate being already-rated jobs.)
- **Per-mode bot chat lock:** `serve` (prod) locks the bot to `telegram.chat_id`; `serve --dry` to
  `telegram.chat_id_debug` — debug clicks never reach the real user's golden labels.
- **Matching experiment (MEASURED → not shipped):** wired `llm_cov` into a shared `slate.effective_score`
  (used by `validation.eval_report` too, so the eval can't drift) + an llm_cov gate on the eligibility
  soft-lift. On real labels the cov-blend REGRESSED ranking (pairwise 0.862→~0.76 — the labels reward
  aspiration jobs), so the knobs (`slate.cov_weight`, `eligibility.soft_lift_cov_min`) ship **off (0)**.
  The shared `effective_score` correctly includes role-kind now, raising the honest eval baseline to
  0.862/0.571. Recorded in `docs/MATCHING_SOTA.md`.


### Telegram bot hook — `/slate.json` + optional bot spawn (2026-06-16)
- New `GET /slate.json` on the feedback app: today's slate as JSON (the cards the HTML index renders),
  the only read the Telegram bot needs. `default=float` shim guards any numpy `fit_score`.
- `feedback_app.serve()` optionally spawns `python -m schabasch_bot` (separate repo) when
  `telegram.enabled` + a token are set; soft-skips if uninstalled/disabled; terminates the child on exit.
- Additive `telegram:` config block (example yaml). Bot lives in `schabasch-tg-bot`.
  `chat_id: 0` = auto-lock the bot to the first `/start` (no manual id needed); a set id hard-locks.
- `schabasch serve --dry` — testing mode: POST /feedback validates + acks `{ok, dry}` but does NOT
  write to the `label` table (golden labels untouched). Lets you exercise the bot end-to-end safely.
- `schabasch enrich` — new entrypoint to run `enrichment.enrich_slate` on today's slate on demand
  (clean re-parse + pros/cons + company review). Previously enrich only ran inside a full nightly_tick,
  so manually-assembled DBs never got the `vacancy_enrichment` table; this gives a standalone command.
- **`schabasch serve` now retrains + fetches on start** (daemon thread; `--no-fetch` / `serve.fetch_on_start`
  to skip): `triage.retrain_checkpointed` archives the previous model (+ metrics + date) to
  `data/models/archive/` and `registry.jsonl` then retrains on new feedback (no-op if unchanged);
  `nightly_tick(run_investigate=False)` does the full fetch; the top-2 are investigated to seed, then
  the rest run TOP→DOWN ('on the go'). `/fetch-status` gains `slate_ready` (the bot's greet trigger).
- `investigate.investigate_one(cfg, con, vid)` — idempotent single-card agentic investigation
  (progressive path); `investigate_top` refactored to share `_investigate_row`. `nightly_tick` gains
  `run_investigate` flag.

### Per-feature ablation harness — `eval/feature_ablation.py` (2026-06-16)
- New **model-free** ablation (reads cached `feature_json` vs the 50 real labels; no LLM/bge-m3/35B; ~1.3s):
  MODE 1 standalone ranking power, MODE 2 leave-one-out of the production fit blend, MODE 3 add-one-in
  5-fold **held-out (same-subset Δ)** — the earns-its-place test. Bootstrap 95% CIs. Run:
  `python -m eval.feature_ablation --real-labels`. Documented in `docs/MATCHING_SOTA.md`.
- **First run (n=49):** production fit correctly relies on both `hyre` (Δ+0.084) and `sparse` (Δ+0.059) —
  no dead weight; **almost nothing adds beyond fit** held-out; the LLM agent booleans
  (`requirements_verified` −0.03, `company_known` −0.09) *hurt* as ranking features (correctly used as
  gates/display, not matcher inputs). Cautionary tale: `title_log_len` tops standalone (0.775) but hurts
  in-blend (−0.043) — standalone power ≠ earns-its-place. n=49 → CIs wide; |Δ|<~0.05 is noise.
- No production change (measurement tool). 36 touched-suite tests green.

### Frontier eval — LLM-canonical-JD → embed (decision record; NOT shipped) (2026-06-16)
- New `eval/canonical_jd_experiment.py`: tests the purest "LLM-extracts-for-ML" form — LLM-extract a
  canonical JD skills/requirements list → embed (bge-m3) → CV-cosine signal — vs the raw-JD baseline +
  production `fit_score`, with **bootstrap 95% CIs** over the user's 50 real labels + a **5-fold held-out blend**.
- **Result:** extractor strength is the lever — **35B canon→cv_full 0.713 pairwise** (clears raw-JD baseline
  0.511; ≈ HyRE 0.724) vs **qwen canon 0.582** (weak). Confirms "complex extraction wants the bigger model".
- **Decision: NOT wired into `fit_weights`** — the held-out blend does NOT improve (`fit+canon` 0.536 <
  fit-alone 0.609; α_fit→1.0): it's **redundant with HyRE** (already LLM→embed). n=50 → CIs wide; revisit at
  ≥75–100 labels. Validates the architecture thesis without a production change. (Phase-separated extract→eval
  so the 22GB 35B never co-loads with the embedder.)

### English-first repo + bilingual UI toggle + new «шабашка» emoji + README screenshots (2026-06-16)
- **English-first.** README + docs translated to English; «шабашка»/«office mouse» kept as glossed
  signature terms (README Glossary). UI strings extracted to a data-driven i18n layer.
- **Bilingual UI.** New `schabasch/i18n.py` + `schabasch/locales/{en,ru}.json` (164 keys); `slate.py`/
  `feedback_app.py` thread `lang`; English default + 🇷🇺 toggle via `?lang=`. Add a locale = drop a JSON.
- **Emoji.** Score-5 «шабашка» `💅💸` → **👸✨🧚** everywhere (code/docs/tests); 💻🐀/🐭/😐/😎 unchanged.
- **Screenshots.** `scripts/gen_screenshots.py` renders the 4 pages from synthetic demo data → PNGs in
  `docs/screenshots/` (no PII, reproducible) embedded on the README front page.
- **Matcher doc fix.** README now states the real `fit_score = 0.7·HyRE + 0.3·bge-m3 sparse` (was a stale 0.6/0.4).
- Tests: `test_i18n` (locale key-parity) + a slate bilingual toggle test; ~10 RU assertions → EN. 348 green.

### Agent → local MLX 35B (verified sota-grade) · escalation collapsed to a single small tier (2026-06-16)
- **`llm.roles.agent` → MLX 35B (`:8082`)** as the investigate/discover model. Verified live: ~25s/
  fetchable card, finalizes, **honest** (caught Hamilton Barnes = recruiter; ABB = Swiss; no fabricated
  salary) — beats the small models, which hallucinated employers/salaries.
- **Auto-fallback (unattended-safe).** `agent_client_params` probes the agent's localhost endpoint; a
  dead `:8082` (MLX not started) → falls back to ollama `qwen3:8b`, so a `tick` still investigates and
  never co-loads the 22GB 35B with the bulk pool. Remote roles used as-is (no ping). Helper `_endpoint_reachable`.
- **Escalation stays within small models.** `enrichment._deep_chain` drops `sota` + slop-escalation →
  single `deep_reasoning` tier + ollama `normalizer` fallback; `deep.enable_sota: false`.
- The 35B is supervised/exclusive (22GB; can't co-reside) — see plan runbook (evict ollama → serve_mlx → investigate).
- Tests: `test_llm_clients` (reachability fallback round-trip), `test_enrichment` (single-tier cascade). 343 green.

### ReAct agent finalize-protocol fix — strong (sota) + qwen agents now finalize cleanly (2026-06-16)
- **Root cause (CAPA, deterministic).** The kl ReAct loop maps a *bare* result object/array — which
  the investigate/discover prompts ask the model to "return (no prose, no fences)" — to a
  `final_answer` whose `content` is `""`, because kl reads content from the `content`/`answer` key,
  absent in a schema object (prototype-internal-KL `shared/parsing.py::action_from_mapping`). An empty
  final_answer is rejected as DEGENERATE (`safety.py::is_substantive`); kl's give-up guards only fire
  at 8 but schabasch caps `max_turns=6`, so the loop **never accepts it** → exhausts turns → ungrounded
  salvage prose → our parser failed. Verified: kather `sota` emitted a clean final_answer on turn 4
  and the loop rejected it 3× → `[max turns exhausted]`. (The qwen path separately surfaced
  `[budget exceeded]` reaching `parse_json_output` as a confusing *"forgot a comma"* `SyntaxError`.)
- **Fix (`agent_runtime.py`, generalizable across investigate + discover).**
  - `build_agent` appends `_FINALIZE_CONTRACT` to every agent's system prompt — teaching kl's action
    protocol so the task's JSON is emitted as a STRING inside `final_answer.content` (grounded in kl's
    own `_HARMONY_REPROMPT`). No more bare-object → empty-content degenerate trap. `/no_think` still
    leads for qwen.
  - `parse_json_output` now recognizes ALL kl non-finalize sentinels (`[budget exceeded]`,
    `[max turns exhausted]`, the salvage preamble) → a clean *"agent did not finalize"* error instead
    of a misleading `SyntaxError`.
- **Rejected (measured, recorded so it isn't re-litigated):** raising `max_turns`→10 for non-qwen —
  it REGRESSED sota **4/5 → 3/5** (extra turns invite wandering past coverage-complete →
  `completion_guard_force_synthesized`/`budget_exceeded` → ungrounded salvage). Tight bounds + the
  contract win; both tiers keep `max_turns=6`.
- **Measured** (`investigate_top(top_n=5)`, real top-5 SCORED cards, supervised on DB copies):
  **qwen 2/5 → 5/5** (now finalizes in 2-3 turns); **sota 3/5 → 4/5**, finalizing GROUNDED
  (`stop=final_answer` at turn ~5) instead of via salvage. The lone residual sota miss is an
  un-fetchable posting (vacancy 7) that ALSO fails under qwen — a data problem, not the harness.
- Tests: `tests/test_agent_runtime.py` (12) — sentinel parsing, contract injection, budget bounds,
  qwen `/no_think` ordering. Full suite **338 green**.

### Supervised qwen3.5 verification + laws-of-UX-gated fleet UX redesign (2026-06-16)
- **qwen3.5 measured live (supervised, on a DB copy), measure-then-ship:**
  - **deep_reasoning → qwen3.5:4b (SHIPPED as the local default).** `enrich_slate` on qwen3.5:4b
    enriched 3/3 slate cards in **66s** with high-quality, profile-aware pros/cons (independently
    caught the «стажировка/internship» objection + the «работать головой не руками» alignment). It's
    local, ~3.4GB, fast, reliable → replaces the 35B-MLX-needs-serving default for the enrichment
    tier (the 35B stays an opt-in upgrade via `scripts/serve_mlx.sh` + a one-line config swap).
  - **agent → kept on qwen3:8b (negative result).** qwen3.5:4b on the investigate ReAct agent
    finalized only **1/3** (2/3 kl-ReAct parse errors: "unterminated string", "invalid syntax") over
    736s. Same failure class as qwen3:8b AND kather `sota` → the gap is the **kl_agent_builder ReAct
    action-parser (harness), not the model** (confirmed across 3 models). `agent_runtime.py` gained
    `strong_*` budgets (max_turns 10 > kl's give-up guard 8 so its salvage net can fire) for the
    non-qwen agent path — the harness fix continues under the spawned task.
- **Fleet UX/UI deeper redesign — staffed by the `engineering-skills` plugin, laws-of-UX-gated.**
  Ran `laws-of-ux-gate` FIRST (subtraction list before additions); then `senior-frontend` (redesign),
  `a11y-audit` (+ its `contrast_checker.py`, all tokens verified ≥6.3:1 AA), `adversarial-reviewer`
  (found 1 real defect — see below). All in `schabasch/slate.py` (single-file renderer). **Subtractions:**
  demoted the competing bold-blue `<summary>` on skills/enrichment → neutral (Von Restorff: only the
  score badge + ⛔ stay strong); removed the filter chips from `/annotate` (Tesler/Jakob — meaningless
  on an unlabeled queue); capped enrichment snippets 5→2; moved the note `<textarea>` from mid-card to
  a `+ заметка` toggle AFTER the buttons (Selective Attention). **Additions:** `@media ≤480px`
  responsive (meta→2 lines, pros/cons stack, **metrics tables reflow to label:value card-lists**,
  ≥44px tap targets); WCAG-AA contrast tokens; `:focus-visible` rings; `aria-label` on every emoji
  button + `role="img" aria-label="оценка N/5"` on the score badge + `<label>` on the note; chips
  `<span>`→`<button>` (keyboard-accessible, WCAG 2.1.1); a compact **collapsed emoji legend** in the
  header (Jakob/Mental Model). **Keyboard shortcuts DEFERRED by the gate** (learning cost > value for a
  non-technical user). `adversarial-reviewer` caught the metrics tables' bare `<tr><th>` header (no
  `<thead>`) → the mobile card-list hide never fired → **fixed by wrapping headers in `<thead>`**.
  Verified live in the browser (Preview MCP, DB copy) at **780px + 380px**: legend, chips-as-buttons,
  role flags, demoted summaries, the note-toggle/open logic, and the **table→card-list reflow** all
  render correctly; console error-free; +14 `tests/test_ux_render.py` invariants. 339 tests green.

### Model cascade · comment→task tracker · transparent+memory-safe fetch · Zotero-style card (2026-06-16)
- **Model cascade (the "smarter model" ask).** New `schabasch/llm_clients.py` adds an `OpenAIClient`
  (any OpenAI-compatible `/chat/completions`: mlx_lm.server, api.kather.ai) alongside the frozen
  `OllamaClient`, plus `make_llm_client(cfg, role)` — a config-driven router (`llm.roles`). Tier-0 qwen3:8b
  (ollama) for bulk normalize/judge **unchanged**; Tier-1 local **Qwen3.6-35B-OptiQ-4bit** (mlx_lm.server
  `:8082`, `scripts/serve_mlx.sh`); Tier-2 **api.kather.ai `sota`** for the hardest reasoning (key read
  from `OPENAI_API_KEY` via the configured `.env` — **never stored in the repo**). The ReAct
  investigate/discover agent (`agent_runtime.py`) is now role-routed too (fix for the "underpowered with
  qwen3:8b" verdict). **Verified live:** api.kather.ai produced grounded pros/cons + a clean re-parse of the
  MAM "AI-слоп" JD ("…роль финансового контролёра в IT, а не бизнес-аналитика").
- **Memory safety for the UI fetch** (the "подключить модуль с управлением памятью" ask). Vendored
  `schabasch/memory_guard.py` (stdlib-only, from IVAI; macOS `memory_pressure -Q`): the `/fetch` worker
  starts a watchdog + `require_headroom` before the run, and `pipeline.nightly_tick` gates every
  model-loading stage (`features`/`normalize`/`judge`/`rerank`/`investigate`/`enrich`) — a low-RAM moment
  **skips** the stage (logged `memory_skip`, tick continues) instead of a swap death spiral; the button
  shows «мало памяти — закрой тяжёлые приложения». Config `memory.{guard_enabled,hard_floor_pct,soft_floor_pct}`.
- **Transparent fetch progress** (the "написано scrape, непонятно что это" fix). `/fetch-status` now maps
  the live funnel stage to a plain-RU description + a `🧠` heavy-stage marker + an ordered 12-step
  checklist with the current step `(n/12)`; the button shows e.g. «⏳ Читаю описания моделью qwen3:8b → карточки (8/12) 🧠».
- **Review comments → tracked tasks** (the "учитывать ОБЯЗАТЕЛЬНО, пометить что учли" ask). New
  `schabasch/tasks.py` (`session_comment_task` sidecar) + `scripts/ingest_comment_tasks.py` ingest EVERY
  comment (20: 16 `label.why_freetext` + 4 session-md-only) idempotently, theme-tag each (engineer/junior/
  slop/degree/hidden-de/duplicate/gap/pref) with an open|accounted|wontfix status. New `GET /tasks` page +
  `POST /task-status` toggle. **Acted on the gaps**, measured on the 50 real labels: a deterministic
  `role_kind` classifier (`schabasch/role_kind.py`) soft-down-ranks hands-on-engineer (×0.7) and intern/
  working-student (×0.5) roles in `slate._effective` + flags them (`🛠 hands-on`, `🎓 стажёр`) — never a
  hard drop (still explore-eligible). Receipt: engineer roles mean **2.09**, junior **2.00** vs neutral
  **3.00** on her real labels → the down-rank tracks her ratings; only 1/25 down-ranked jobs is high-rated
  (the VINFAST engineer she flagged "прикольный, НО инженер"), softly demoted not hidden.
- **Zotero-style rich card** (the "слишком мало информации" ask). New `schabasch/enrichment.py`
  (`vacancy_enrichment` sidecar, keyed by content_hash): a deterministic EXTRACTIVE pass (bge-reranker
  ranks JD sentences against goal queries → key snippets) + an ABSTRACTIVE pass via the cascade (35B-MLX
  default, kather `sota` escalation for `slop_score ≥ deep.slop_escalate_thr`) producing pros/cons, a deep
  company review, and a clean re-parse of muddy ads — all grounded (anti-fabrication). Runs on the slate
  set only (`deep.enrich_top_n`), memory-gated, after rerank; degrades to snippets-only when no abstractive
  tier is reachable. The card shows a collapsible «📄 Глубокий обзор вакансии» with `model_used` provenance.
- 313 tests green (+33: `test_llm_clients`, `test_tasks`, `test_role_kind`, `test_memory_fetch`,
  `test_enrichment`). All work is additive — `models.py`/`db.py` schema/`llm.py` signatures/`config.py` untouched.

### Skill-feature foundation, unsuitable-job filters, triage fix, validated company agent, UI fetch (2026-06-16)
- **Fixed a triage-pipeline crash + measured it (don't wire).** `feature_vector` / `triage._load_labeled`
  built inconsistent-width vectors (1053-dim when a bge embedding existed, 29-dim when not) → `np.stack`
  crashed on a mixed labeled set and a trained model couldn't score un-embedded rows. Now always
  `[dense(1024, zero-padded if missing) ++ named]`. With the fix, `triage-eval` runs (n=49): spearman
  0.26 / **Cohen-κ −0.125** (worse than chance) — at n≈50 the LGBM is not predictive, and training would
  enable hard-drop, so **triage stays cold-start (not wired)**; the fit-led ranking (effective 0.819)
  remains the matcher. Revisit at ≥75–100 labels. Verified the CV profile is faithful (senior/bachelor/BA).
- **Filter unsuitable vacancies** (recall-first): `normalize._filter_card` now hard-drops
  `temp_agency_guess` (Zeitarbeit the scraper flag missed; reuses `FilterReason.TEMP_AGENCY`); the slate
  gained a measured **`slate.quality_floor = 0.45`** — an exploit slot must clear it on `effective`
  (measured on 37 real labels: drops 11 label-2 unsuitable jobs from exploit, **0 high-rated dropped**;
  below-floor jobs stay explore-eligible, not hidden); and a **`🙈 скрыть слабые` filter chip** on `/`
  + `/annotate`. (slop stays judge-penalized, not hard-dropped — noisy + `FilterReason` is frozen.)
- **Deeper, validated company agent (W4).** `investigate.py` now returns a `company_description` +
  `german_rooted`, cross-validated on a KNOWN site independent of qwen: `validate_company` queries
  **Wikipedia** (keyless, de→en, with a UA header + an org-indicator guard so "SCHOTT" no longer matches
  "Schottland") and the **German legal-suffix registry signal** (GmbH/AG/… → rooted-in-Germany). Stores
  `company_description`/`german_rooted`/`company_verified`/`validation_source` in the `investigation`
  sidecar, patches `vacancy_feature` (`company_known` = independently verified; new `german_rooted` →
  integration). Card shows the description + `🇩🇪 укоренена в Германии` + `✓ Wikipedia`. SOTA card replenished.
- **UI fetch trigger + dedup visibility (W5).** `POST /fetch` runs the full pipeline async on a
  background worker with a **single-flight lock** (409 if running — no double model-load); `GET
  /fetch-status` polls progress; a `🔄 обновить вакансии` button on `/` + `/annotate`. De-dup is now
  visible: `/funnel` parses the `dedup_fuzzy` candidates, and the slate header shows `🔁 дедуп: N
  похожих` (+ the per-card `также:` collapse). Answers "как запустить фетч из UI" + "происходит ли дедуп".
  Runtime-verified end-to-end (browser drove the button/chips; the worker really ran the pipeline; 409
  single-flight held). Hardened the worker: the lock-release `finally` now wraps `db.connect`/import too,
  so an early failure can't wedge `/fetch` into a permanent 409 (`running=True` forever).

### Real SOTA hybrid upgrade + workflow verification (2026-06-16)
- **Shipped the bge-m3 SPARSE-lexical hybrid** — the matcher was dense-only (HyRE); bge-m3 is a native
  dense+sparse+ColBERT model. Measured on the 37 real labels (`eval/hybrid_measure.py`, one foreground
  bge-m3 load via the native `compute_score(weights_for_different_modes)`): fusing HyRE with full-doc
  sparse beats HyRE-only on **all three** metrics — `fit_score 0.803→0.814 pairwise / 0.539→0.584
  ndcg@10 / 0.408→0.427 spearman`; production `effective 0.793→0.819 pairwise / 0.483→0.584 ndcg@10`.
  Sparse is **deterministic** (no qwen run-variance, unlike HyRE) → also a stability anchor. **ColBERT
  was measured and did NOT help** (every colbert mode ≤ sparse); must-have-segment pairing lost to
  whole-document. `features.fit_weights = {hyre: 0.7, sparse: 0.3}`, `features.sparse_norm = 0.45`
  (fixed per-vacancy divisor — no set-relative min-max). New `features.bgem3_sparse_scores` (computed
  in `rerank_scored`, reusing the loaded model) + `_blend_fit`/`fit_from_feature` extended; backfilled
  the eval gold + slate pool via `scripts/backfill_bgem3_sparse.py`. Tests in `tests/test_fit.py`.
  SOTA card replenished: `sota-pattern-index/.../bge-m3/native-hybrid-fusion.md`.
- **Verified the new-vacancy FETCH + de-dup workflow (live):** `schabasch scrape` added **54 net-new
  vacancies** (indeed 152 / arbeitsagentur 721 / linkedin 296 raw, URL-deduped on insert); `dedup`
  logged 22 cross-source near-dup candidates; a deliberate cross-source dup (indeed≡linkedin, same
  company, gender-tag normalized) was caught at sim 100.0 with **statuses unchanged** (logged-not-merged
  contract); `tests/test_dedup_fuzzy.py` + `test_dedup_models.py` green (18).
- **Verified the agentic work-detail search (live):** `schabasch investigate --top-n 2` produced fresh
  real enrichment for one card (Thorn SDS BDM-Space: company_size mid, english_team true, €60–85k,
  verified_requirements, verdict ok), patched `vacancy_feature` (requirements_verified=1.0), renders the
  🔎 line; the other card hit the documented qwen3:8b max-turns limit — bounded, retried, counted as an
  error, no hang. 267 tests green.

### SOTA-signal audit follow-up (2026-06-15 PM)
- **Measured, then declined to ship the SOTA hybrid/liked signals** (`eval/hybrid_probe.py`, audited +
  reproduced, read-only over caches — no model load). On the 37 real labels (only 5–6 positives) none
  beats the shipped HyRE-led blend on ndcg@10: liked-similarity 0.43–0.65 (n=6 library, data-starved),
  CV↔JD dense cosine 0.41, `RRF(hyre,xenc)` 0.822 pairwise *but* ndcg 0.373 with bootstrap 95% CI
  [−0.099,+0.250] (straddles zero) → slate-unsafe. **Kept fit_hyre-led ranking unchanged.** Sparse-
  lexical + ColBERT fusion DEFERRED (need a coordinated bge-m3 re-embed; ColBERT vecs aren't cached,
  `return_colbert_vecs=False`). Audit surfaced that `xenc≈0` for ~every CV↔JD pair (cross-encoder
  saturates low on this genre → the 0.2 xenc weight is nearly inert) and `fit_score` has a tiny dynamic
  range (0.607–0.677); a coverage-floor on `llm_cov` was REJECTED because it would have demoted her #1
  job (439, llm_cov 0.125). `eval/hybrid_probe.py` is kept as a read-only probe harness.
- **Backfilled her 15 June session notes into `label.why_freetext`** (`scripts/backfill_session_notes.py`,
  idempotent, DB backed up first; `tests/test_backfill_notes.py`). 0/37 labels carried a note, so the
  judge few-shot learned nothing from her real reasoning; 7 session jobs (Merz/EUMETSAT/DuPont/MAM/
  Merck/SCHOTT/Westinghouse) mapped to exact vids, notes filled (scores untouched). 4 now seed
  `judge.build_fewshot` (the score≤2/=5 ones: "Master Data ≠ degree", "инженер нет — хочу головой/lead",
  "AI слоп", "требуют интерна"). Effect materializes on the next coordinated re-judge.

### Fixed (audit follow-up)
- **Soft-degree lift could rescue a job she can't qualify for.** Because `fit≈0.64` for ~every JD here,
  the `fit ≥ soft_lift_threshold` lift always fired. A **2-step degree gap (Bachelor missing a PhD) is
  now STRUCTURAL (red ⛔) and never lifted**; only a 1-step gap (Bachelor missing a Master) stays soft +
  liftable (`eligibility.py`; `tests/test_eligibility.py::test_phd_prose_2step_is_structural_not_lifted`).
- **Graded slop penalty** — `judge.build_system_prompt` now penalizes `slop_score ≥ 45` (soft) /
  `≥ 60` (hard), not just the ≥60 cliff (MAM-797 read slop≈45 and slipped through).
- **Boring-domain carrier** — judge prompt now defines what makes a domain boring (GMP/GxP/compliance/
  routine controlling → `boring-role` tag, soft minus; her "GMP/GxP кажется скучным").
- **Conditional-German in the normalizer** — `normalize.py` `language_reality` rule now maps a
  "local language / German if the role is based in Germany" conditional for a DE-located role to `de`
  (defense-in-depth behind the `hardfilters.GERMAN_REQ` fix).

### Changed
- **Match quality RE-TUNED on the user's 37 REAL labels** (2026-06-15; `eval/match_eval --real-labels`,
  n=37/188 pairs). The blend was tuned on a synthetic Opus gold; on her real clicks that ranking
  inverts. Production `effective` ranking: **pairwise 0.564 → 0.793, ndcg@10 0.247 → 0.483** — beats
  judge_only (0.572), the old effective (0.564) and old fit_score (0.644), approaching the best single
  signal (HyRE 0.803). 263 tests green; eval harness is read-only over stored features (no model load).
  - **Headline fit is HyRE-led** — `features.fit_weights = {hyre: 0.8, xenc: 0.2}` (was `{xenc: 0.6,
    llm_cov: 0.4}`). On real labels HyRE is the best single signal and `llm_cov` the weakest, so
    `llm_cov` is dropped from the headline (still computed + shown as the card's ✓/◐/✗ breakdown).
    `features._blend_fit` now takes `fit_hyre`; new `features.fit_from_feature` is the single blend
    source, recomputed live so a re-tune takes effect without a heavy rerank.
  - **Effective ranking is FIT-LED** — `slate._effective = fit_score · (1 + β·judge_norm) · elig_score`
    (`slate.judge_blend_beta`, default **0**). The old `(judge + λ·triage) · fit_gate · elig` let the
    near-random magnet judge lead (≈ random on real labels). The magnet now differentiates
    comparable-fit jobs as the tie-break + drives explore-slot selection + the card emoji. Mirrored in
    `validation.eval_report` and `eval/match_eval` so the benchmark matches production.
  - `eval/experiment.py` extended with `--real-labels` (searches blends + the fit-led effective over
    `validation.label_gold`, recomputing eligibility live per blend).

### Fixed
- **Eligibility false positives were net-NEGATIVE on real labels** (multiplying by `elig_score`
  collapsed the best blend 0.803 → 0.59 because the gate demoted jobs she likes). Both bugs from the
  15 June session fixed in `schabasch/eligibility.py`:
  - **"Master Data" / "Scrum Master" ≠ a master's degree** — negative-context guard (`_master_is_degree`
    + the `_apply_overrides` Master-Data guard) nulls a phantom `education_required=master` when every
    "master" mention in the JD is non-degree. Fixes her #1 job (Merz-439, was falsely ⛔). Applied over
    cached extractions so a bad cache self-corrects on read.
  - **⛔ is structural-only; a prose-degree gap is soft + high-fit-lifted** — `eligibility_gate` now
    returns a `severity` (`structural` red ⛔ vs `soft` amber) and accepts `fit_score`; a SOFT degree
    factor is lifted to 1.0 when `fit_score ≥ eligibility.soft_lift_threshold` (her SCHOTT ask). Stored
    as `feature_json.elig_severity`; `slate._card_block` picks red vs amber from it. Recomputed live
    (`features.recompute_live`, `eligibility.req_from_cache`) so the fix applies without a rerank.
- **Hidden conditional German** (`hardfilters.GERMAN_REQ`) — DuPont-67's "Fluency in the local
  language … German if the role is based in Germany" slipped past the window heuristics; added
  `local language … German`, `German if … (based|located|role)`, `Deutsch … (wenn|sofern) Deutschland`
  alternations. Regression added to `tests/test_hardfilters.py`.

### Added
- **Cross-account repost collapse (display only)** — `slate._collapse_reposts` merges the same
  role+city posted under different recruiter names (455 Laveer ≡ 906 Westinghouse) to one card with an
  `also_at` note, with a specificity guard (only ≥3-token titles, or same-company dups) so generic
  titles at different employers aren't false-merged. Pipeline `dedup` (blocks by company) is unchanged.
- **Far-but-in-Germany jobs shown + marked, not dropped** (WS4) — `geo.geo_class` / `geo.geo_mark`;
  `geo.prefilter` and `normalize._filter_card` no longer drop far-DE onsite/hybrid jobs; `slate`
  marks them `📍 далеко · ~N км до <anchor>`, prefers them for the explore slots, and
  `triage.select_for_normalize` orders near-cities-first so far jobs don't starve the normalize budget.
- **Free-text feedback on every card** (WS2) — `slate._card_block` adds a per-card `<textarea>`; the
  `fb()` JS sends it; `feedback_app.Feedback.note` maps it to `label.why_freetext` (both rating and
  applied branches; COALESCE keeps a prior note/score). It already flows into the judge few-shot
  (`judge.render_fewshot` NOTE: line for score≤2 / =5) — corrections now change future scoring.
- **Smart filter chips** (WS3, client-side, no new endpoint) — `slate._chip_row` + `_JS.applyFilter`:
  time window (`data-days`), order (перспективные = fit DOM order / свежие), and a far toggle, over
  cards stamped with `data-*`. On `/` and `/annotate`. Default shows the full curated slate; chips narrow.
- **Judge persona + slop rubric** — `config/profile.yaml: profile.summary` encodes her taste
  (head-not-hands, likes lead/principal, dislikes monthly/deadlines/routine); `judge.build_system_prompt`
  penalizes high `slop_score` (≥60 → slop-text tag) and the head-not-hands/lead/monthly preferences.

### Added (prior)
- **Fresh, dated slate + skill-gap stats + themed scale** (verified on a live EN+German `tick`:
  245 tests green, fit_score NDCG@10 0.96 held; dates render; deep-search produced fresh enrichment):
  - **Posting dates always show** — `slate._posted_ago(date_posted, first_seen)` falls back to
    "найдено N дн." when a board gives no posting date (LinkedIn never does); fresh AA scrapes now
    populate `date_posted` (1081/1449). `first_seen` carried through `_load_scored`/`build_slate`.
  - **Daily-slate freshness ceiling** — `_load_scored(…, max_age_days)` bounds the daily slate to
    jobs re-seen within `slate.fresh_days` (14); `/annotate` (annotation_batch) keeps the full
    backlog. Stops a judged job from re-entering the slate forever.
  - **Skill-gap dashboard `/gaps` + `gaps` CLI** (`schabasch/gaps.py::gap_report`) — aggregates the
    per-requirement `llm_cov_reqs` across WANTED jobs (`label.score≥4 OR applied=1`) into the
    recurring missing/partial skills, ranked worst-first, framed vs the CV's skills.
  - **Deep-search reach** — `slate.investigate_top_n` 3 → 6 (more top cards researched per tick).
  - **Themed score scale** — `slate._score_badge`: 1 «офисная мышь» = 💻🐀 → 5 «шабашка» = 💅💸 on a
    grey→gold gradient encoding magnitude; feedback buttons re-themed 💻🐀 / 😎 / 💅💸 (tooltips kept).
- **Per-skill match breakdown on every card.** `features._llm_coverage` now persists the
  per-requirement list (`present`/`partial`/`missing`) it already computes — new `requirements`
  column on `llmcov_cache` (PRAGMA-guarded migration) + `feat["llm_cov_reqs"]`. `slate._skills_html`
  renders a collapsible `🎯 Навыки {llm_cov:.0%} · n✓·n◐·n✗` with the ✓/◐/✗ requirement list
  (matched first). Headline % = `llm_cov` (the honest skill-coverage), NOT the xenc-compressed
  `fit_score`; it supersedes the one-line fit ⚠ note when present (Occam). Live on the real CV: GRO
  Data Analytics shows 92% with ✓ Power BI / ✓ process analysis / ✓ stakeholder mgmt. Tests in
  `tests/test_fit.py`.
- **Reliable, deterministic listing check** (`investigate._check_still_open`) — replaces the qwen
  agent's still-open *guess*, which reported false "no active listing" on anti-bot-**blocked**
  Indeed links. Arbeitsagentur: re-query the API by refnr (`arbeitsagentur.check_open`); boards:
  HTTP status (`llm.http_get_status`) — `404/410`→closed, `2xx/3xx`→open, `403`/timeout→**unknown**
  (never a false "closed"). `slate._verified_html` renders three honest states: "открыта ✓" /
  "⚠ вакансия закрыта" (only on a confirmed 404/410/AA-gone) / "ℹ листинг не проверён" (couldn't
  verify). The agent no longer judges closure (dropped from `_SYSTEM_PROMPT`). Live-verified
  (AA refnr→open, bogus→closed, Indeed→200). Tests in `tests/test_agent_discovery.py`.
- **Live validation page `/eval`.** Match-quality metrics (pairwise / NDCG@10 / Spearman) computed
  against the user's **real labels** (the `label` table, `score_1_5` with `applied→5`), so the numbers
  update as she rates in `/annotate` — no manual code re-point, and it works at any label count
  (banner nudges to `/annotate` below `slate.eval_min_pairs`, default 15). Honest leakage handling:
  the label-independent fit signals (`fit_score`/`xenc_full`/`llm_cov`/`elig_score`) are shown as
  clean; `judge_only`/`effective`/`triage` train on labels and are flagged "⚠ обучается на метках".
  Metric helpers were consolidated out of `eval/match_eval.py` into a shared `schabasch/metrics.py`
  (reused by both the CLI harness and the page; CLI synthetic-gold numbers unchanged — fit_score
  0.768 / effective 0.852); new `schabasch/validation.py`; `eval/match_eval.py` gained a
  `--real-labels` flag. New `slate.render_eval_html`, `feedback_app` `GET /eval`,
  `tests/test_validation.py`.
- **Single web annotation surface `/annotate`.** A "rating queue" page (FastAPI `GET /annotate`):
  every judged-but-unrated vacancy, the same card and the same 👎/👍/⭐ buttons as the daily slate,
  writing to the same `label` table via `POST /feedback`. Rated jobs leave the queue (Goal-Gradient);
  it replaces the Excel pack as the cold-start path. New `slate.annotation_batch` /
  `slate.render_annotate_html`; tests in `tests/test_annotate.py`.

### Changed
- **Daily-card UX (Laws-of-UX gate, subtraction-first).** Von Restorff: the only strong-emphasis
  channels are now the blue score and the red ⛔ eligibility "stop"; the fit-gap ⚠, the green
  "verified" box, the teal why-badge, and the explore border are demoted to neutral grey so the ⛔
  stands out. Working Memory: the `👎=2 · 👍=4 · ⭐=5` legend moved off the header onto each button's
  tooltip; the header now carries live progress (N/M) and a link to `/annotate`. Added a client-side
  undo (↶ изменить) and an end-of-list completion state (Peak-End). `render_html` and the annotate
  page now share one `_card_block` / `_page`.
- **Judge few-shot now learns from 👎.** `judge.build_fewshot` widened its extreme-anchor filter from
  `score IN (1,5)` to `(score<=2 OR score=5)`. The only web negative signal is 👎=2 (there is no "1"
  button), so under the old filter downvotes never entered few-shot and the judge never learned
  repellents from real labels. Regression:
  `tests/test_annotate.py::test_fewshot_picks_up_downvote_low_anchor`.

### Removed
- **Excel bootstrap annotation pack.** Retired the `build-bootstrap` and `labels-import` CLI
  commands, `pipeline.import_bootstrap_labels`, `normalize.normalize_ids` (its only caller), and
  `schabasch/bootstrap.py` (the openpyxl xlsx writer + stratified sampler). The `/annotate` queue is
  now the single annotation surface — one mental model (Tesler / Occam). This supersedes the earlier
  unreleased "bootstrap labels survive the `card_json` join" fix, which is moot now the xlsx path is
  gone.

### Fixed
- **Matcher was matching the WRONG person — re-grounded on the user's real CV.** The stored profile
  (and a memory note) said "ML engineer"; her actual CV is **Senior Business Analyst** (Bachelor in
  Business Informatics, no master; skills = business analysis / BPMN / process design / target
  operating models + Python/SQL/Tableau/Power BI/AWS). This had INVERTED the ranking — her real
  strong-fit roles (business-analyst / process-owner / data-analytics-BI / program-PM / consulting)
  were scored low and pure-ML roles high. Fix: re-extracted `candidate_profile` via `candidate
  --cv-path` from the real CV; corrected `config/profile.yaml` `profile.summary` (magnets kept as
  ASPIRATIONAL pivot domains per USE_CASE); re-labeled the `eval/match_eval.py` GOLD; re-ran
  `rerank`. **Opus-graded benchmark, before(wrong-CV)→after(real-CV) vs the corrected gold:**
  `fit_score` pairwise 0.725→**0.872**, NDCG@10 .558→**0.960**, Spearman .463→**0.752**; `llm_cov`
  pairwise 0.531→**0.869** (was ≈random). 235 tests green; profile/config/GOLD edits don't touch
  tested code paths.
- **Re-tuned `slate.fit_gate_floor` 0.5 → 0.25.** With fit corrected, the magnet judge over-fired on
  aspiration domains she can't do (BDM-Space / Cyber-Threat: judge=5, fit≈0). A `fit_gate_floor`
  sweep vs the gold was monotonic (pure fit×elig best, NDCG 0.96); 0.25 makes **fit dominant**
  (match) while the magnet still differentiates do-able jobs (pivot space), without the overfit
  extreme of 0.0. Effective ranking 0.725→**0.79** pairwise, NDCG .799→**.869**. Tuned on the Opus
  proxy gold — to re-validate on the user's real `/annotate` labels. `eval/match_eval.py` effective now
  reads the configured floor (benchmark == production), not a hardcoded 0.5.
- **Removed all silent failures flagged by the IVAI dead-code / AI-slop scan** (`python -m
  scripts.deadcode scan schabasch`; score 20 → 57, error findings 9 → 0, behavior preserved at 235
  tests green). Each broad/`pass`-only `except` was replaced with fail-fast-or-record handling per
  IVAI-D001: `candidate._read_cv` PDF-backend fallback now catches only `ImportError` (real
  extraction errors propagate); the ESCO normalisation `except (ImportError, Exception): pass` →
  `ImportError`-only (a real ESCO bug surfaces); `eligibility.extract_requirements` evicts a corrupt
  cache row instead of silently re-failing; `investigate._patch_feature_row` narrows to
  `(sqlite3.Error, TypeError, ValueError, json.JSONDecodeError)` and records the skip in the funnel;
  `triage._compute_metrics` catches only `ImportError`/`ValueError` and records absent optional
  metrics as `None`; `slate._verified_html` salary formatting validates via a `_to_int` helper
  instead of a swallowing try/except; `validation.eval_report` and `eval/match_eval.py` skip a
  malformed `feature_json` row explicitly.
- **Redundancy / idiom:** `dedup.find_fuzzy_candidates` now iterates `itertools.combinations(group,
  2)` instead of a manual `range(len(...))` double index loop (clearer, same pairs); removed a
  trivial restating comment in `aspects.py`.
