# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Entries are deliberately terse;
full rationale, receipts, and rejected alternatives live in git history and `docs/`.

## [Unreleased]

### 2026-07-21
- Scrubbed internal endpoint/install-path references → generic placeholders (`api.example.com`,
  `<local prototype-internal-KL checkout>`); no behavior change (`sota` role is disabled).
- Changelog condensed to one-liners; details in git history.
- AI-slop scan (IVAI deadcode): 2 swallowed-exception errors fixed — `ats_boards` SmartRecruiters
  detail fetch now logs the failure + sets the fallback explicitly; `llm_clients._extract_json_obj`
  parse ladder restructured without `except: pass`. Score 32→41, 0 errors; hardcoded-url warnings
  remain classified FP (adapter identity), size warnings deferred.

### 2026-07-04
- Per-user `search.remote_worldwide`: extra LinkedIn-only pass (`is_remote=True` + Worldwide);
  user-2 overlay refocused (startup/agentic-AI queries, remote taste_rules).

### 2026-07-03
- New keyless opt-in sources: social person-posts (Bluesky/Mastodon/Telegram, keyword-gated before
  the LLM), monthly HN Who-is-hiring (region-filtered), public ATS boards (7 providers, slug-probe
  cached in `ats_board`); jobspy passes `glassdoor` through `search.sources`.
- Bot `/all` + `GET /pool.json` (`slate.full_pool`): all judged-but-unrated vacancies, best first.
- De-personalization: judge magnets/repellents/`taste_rules` from `cfg.profile`; hard drops gate on
  `profile.repellents`; `role_kind` defaults neutral; enrichment uses the user's own summary.
  User-2 re-judged under his rubric; 11 remote-only jobs resurrected.
- Registration live-run fixes: RU→DE city aliases + multi-city + «вся Германия» (`geo.anchors:
  null`), CV file+text merge, `POST /update-queries` + `GET /queries`, per-user cfg-cache fix,
  empty-DB refetch-guard fix, /fetch-status polls filtered from the access log.
- Self-service Telegram registration (`POST /register`, `registration.py`, `MAX_USERS=2`): CV +
  city + queries (or CV-derived) → overlay yaml + isolated DB + candidate extraction, full rollback
  on failure; latent example.yaml geo/queries bugs fixed; `pypdf` declared.
- Multi-user: `users.py` — overlay yaml per user (`config/users/<key>.yaml`), isolated data dirs,
  `?user=` threading across web/bot, `tick` loops all users.

### 2026-07-01
- `/gaps` reconciles false "skill gaps" via deterministic eligibility ordinals (ISCED edu, CEFR
  lang) — 14/46 false gaps removed live; genuine gaps kept.
- Slate carries the last non-empty slate forward across midnight (no empty morning board).

### 2026-06-30
- Rerank cold-start starvation fixed: freshness moved into the candidate WHERE (`xenc_full IS
  NULL`), so new SCORED jobs get fit instead of the stale top-30.
- Rerank budget ordered `last_seen DESC, judge_score DESC` — the slate's fresh window ranks first.

### 2026-06-25
- llm_cov false-missing cut: coverage judge fed an explicit SKILLS/LANGUAGES block + full CV
  (cache `c2`). Measured (12 JDs): missing −23%, false-missing −25%; deterministic token guard
  rejected by the non-overfit gate.

### 2026-06-22
- HyRE null-vs-zero fix: `fit_hyre==0.0` treated as absent in `fit_from_feature` (phantom zero
  tanked labeled gold outside the rerank pool). Effective pairwise 0.804→0.814; naive backfill
  rejected (regressed). `slate.quality_floor` 0.45→0.37.
- Engineer flood (config-only): search queries retargeted off engineer titles;
  `role_kind_mult.hands_on_engineer` 0.7→0.45 — rank-neutral on 100 labels, exploit slots cleared.

### 2026-06-21
- Investigate-agent finalize-block fixed (CAPA): free text moved from `AgentInput.task` →
  `context` (kl's facet gate scans only `task`); agent search swapped to the snippet-returning
  ddgs adapter. Measured 0/5→5/5 finalize, slate 9/9.

### 2026-06-18
- Triage honesty: deploy gate `_should_deploy` (negative temporal-holdout keeps the prior model),
  triage removed from the slate tie-break, `/eval` row counts only real ML decisions.
- Staleness: ingestion date gate (`search.max_post_age_days`), `slate.fresh_days` 14→7,
  `pipeline.verify_liveness` (AA API + Indeed embedded JSON) expires confirmed-gone, expiry-pattern
  feedback → EXPIRED, `recent_reject_penalty` sort-only demotion.
- Usage fixes: 🧭 direction feedback (domain yes / specifics no), `/backlog.json` + bot `/more`,
  honest fetch delta in `/fetch-status`, opt-in `search.linkedin_hours_old`.

### 2026-06-17
- Browsing adapters finished: keyless ddgs/SearXNG search + trafilatura extract, config-driven
  agent search backend (never tavily); `deutschland` package rejected (numpy downgrade).
- Employer grounding: Wikidata entity resolution (`browsing/entity.py`) heads the
  Wikidata→Wikipedia→legal-suffix ladder; identity resolved BEFORE the agent and injected;
  `http_get_json` retries transient 429/5xx.
- Employer KB seeded from existing `investigation` rows (backfill script); Wikipedia name-match
  now requires every significant token (kills Terma→"Therme"-type wrong matches).
- Redundancy: investigate skips done vacancies; persistent `company_knowledge` sidecar caches
  durable employer facts (TTL for news); rerank skips fresh `xenc_full` rows.
- features-stage torch/OpenMP deadlock fixed (CAPA): `features.torch_num_threads=1` + swap-growth
  delta in the memory watchdog.

### 2026-06-16
- Freshness re-rank: slate sorts on `effective · recency_mult` (halflife 7d, floor 0.6) + a
  posting-date badge; effective_score/eval stay pure.
- Can't-qualify floors (measured, 0.862→0.871 pairwise): `eligibility.jd_hard_blocker` regexes
  (clearance/citizenship) + `slate.zero_cov_mult` on `llm_cov==0`.
- `serve` skips the startup fetch when data is fresher than `serve.refetch_after_hours` (12h);
  both UI and bot surface an explicit "data is N h old — refetch?" prompt.
- Rich startup/per-stage console logging with a 30s heartbeat (`serve --quiet` mutes).
- Two-axis feedback (domain vs role): `role_feedback` sidecar + `POST /role-feedback` + bot role
  buttons; optional Beta-smoothed learned role multiplier (off by default) + ablation gate.
- Slate de-dup (rated never re-served; unrated decay after `slate.max_reshows`), per-mode bot chat
  lock (prod vs `--dry`), cov-blend experiment measured and shipped OFF (regressed 0.862→~0.76).
- Telegram hook: `GET /slate.json`, optional bot spawn, `serve --dry`, `schabasch enrich`,
  retrain+fetch on serve start (checkpointed model archive), `investigate_one` progressive path.
- `eval/feature_ablation.py` (model-free, bootstrap CIs): hyre+sparse both earn their place;
  nothing else adds held-out; agent booleans hurt as ranking features (stay gates/display).
- `eval/canonical_jd_experiment.py` (decision record): 35B canonical-JD→embed hits 0.713 pairwise
  but is redundant with HyRE held-out — NOT wired.
- English-first repo: README/docs in EN, `i18n.py` + en/ru locales (🇷🇺 toggle), 👸✨🧚 score-5
  emoji, reproducible README screenshots from synthetic data.
- Agent → local MLX 35B (`:8082`, measured sota-grade) with auto-fallback to ollama qwen3:8b when
  down; enrichment collapsed to a single small tier (`deep.enable_sota: false`).
- ReAct finalize-protocol fix (CAPA): `_FINALIZE_CONTRACT` teaches `final_answer.content`;
  `parse_json_output` knows all kl sentinels. qwen 2/5→5/5, sota 3/5→4/5; raising max_turns→10
  rejected (regressed 4/5→3/5).
- qwen3.5 verified live: `deep_reasoning`→qwen3.5:4b shipped; agent kept on qwen3:8b (gap is the
  kl ReAct parser, not the model). Laws-of-UX-gated slate redesign: one focal point, ≤480px
  card-list reflow, WCAG-AA, aria, `<thead>` fix; browser-verified at 780/380px.
- Model cascade (`llm_clients.py`: `OpenAIClient` + config-driven role router), memory-safe fetch
  (`memory_guard` watchdog + per-stage gates), transparent 12-step fetch progress, comment→task
  tracker (`tasks.py`, `/tasks`), deterministic `role_kind` down-rank (engineer ×0.7, intern ×0.5,
  never a hard drop), Zotero-style card enrichment (`enrichment.py` sidecar, extractive +
  abstractive, grounded).
- Triage vector width fixed (dense zero-pad) but stays cold-start (κ −0.125 @ n=49 — not wired);
  temp-agency hard drop + `quality_floor 0.45` + hide-weak chip; validated company agent
  (Wikipedia + legal-suffix → `german_rooted`); UI `POST /fetch` with single-flight lock + visible
  dedup.
- REAL SOTA hybrid shipped: bge-m3 sparse fused with HyRE (`fit_weights {hyre:0.7, sparse:0.3}`,
  `sparse_norm 0.45`) → effective 0.819 pairwise / 0.584 ndcg@10; ColBERT measured, didn't help.
  Fetch/dedup/investigate workflows live-verified (+54 vacancies, dup caught at sim 100.0).

### 2026-06-15
- Measured & declined: liked-similarity, CV↔JD cosine, RRF(hyre,xenc) (slate-unsafe CI); HyRE-led
  blend kept. June-15 session notes backfilled into `label.why_freetext` (seed judge few-shot).
- Audit fixes: 2-step degree gap is structural (never lifted); graded slop penalty (soft ≥45);
  boring-domain rubric; conditional-German normalizer rule.
- RE-TUNED on 37 real labels: fit HyRE-led (`{hyre:0.8, xenc:0.2}`), effective FIT-LED
  (`fit·(1+β·judge)·elig`, β=0). Pairwise 0.564→0.793, ndcg@10 0.247→0.483.
- Eligibility false positives fixed: "Master Data"/"Scrum Master" ≠ degree; ⛔ structural-only,
  soft gaps high-fit-lifted. Hidden conditional-German regexes added to `hardfilters`.
- Added: cross-account repost collapse (display-only), far-DE jobs marked not dropped, per-card
  free-text feedback → judge few-shot, client-side filter chips, judge persona + slop rubric.

### 2026-06-14 and earlier
- Fresh dated slate (posting dates + `fresh_days` ceiling), `/gaps` dashboard, themed 1–5 emoji
  scale, per-skill ✓/◐/✗ breakdown on cards, deterministic listing-open check (HTTP/AA API, never
  a false "closed"), live `/eval` page vs real labels, single `/annotate` surface (Excel pack
  removed), card UX pass (subtraction-first), judge few-shot learns from 👎.
- Matcher re-grounded on the REAL CV (Senior Business Analyst, not ML-engineer): fit pairwise
  0.725→0.872, NDCG@10 0.558→0.960; `slate.fit_gate_floor` 0.5→0.25 (fit-dominant).
- IVAI dead-code/AI-slop scan cleanup: 9 silent-failure errors → 0 (score 20→57); broad excepts →
  fail-fast-or-record; `itertools.combinations` idiom in dedup.
