---
type: usecase
id: UC-001
title: Daily vacancy slate with learnable «office mouse → шабашка» ranking
status: definition
benefit: 5
uncertainty: 3
scope: mvp
goal_level: user-goal
owner: Vlad
updated: 2026-06-12
---

# Schabaschkascuhen — daily dream-job search with learnable ranking

## User & Context

**Alina** — **Senior Business Analyst** (Bachelor in Business Informatics, **no master's degree**; lives in Heidelberg, German **A2** — works in English, wants to learn German). Real skills (from her CV): business analysis (requirements, user stories, **BPMN 2.0 / UML**, process design, target operating models, UAT, change mgmt), delivery/PM (backlog, roadmap, stakeholder/vendor mgmt) + **Python, SQL, Tableau, Power BI, AWS, Jira**. **NOT an ML engineer** (that was an error in the old memory). Key characteristic: she **doesn't know what her dream job looks like** («шабашка» — the dream/jackpot gig) — there are no reference examples; she wants to **pivot into a new field** building on strong analytical skills. The system must help her *find* it, not match against a sample.

**Preference profile (`config/profile.yaml`):**
- **Magnets (aspirational domains for the pivot, NOT current skills):** animals, space, military/security, large complex projects, public sector / local projects, a new field. Best match = her analytical/process/BI strengths × one of these domains.
- **Companies:** locally rooted in Germany (IBM/Airbus/Bundeswehr — yes; a purely American startup with no German roots — a miss).
- **Location:** Heidelberg or Frankfurt, hybrid. Not remote.
- **Integration score:** does the job give integration points in Germany — language, path to citizenship.
- **Repellers:** a hidden German requirement in an English-language vacancy; biotech; slop/AI-generated text; a boring role; remote-only; Zeitarbeit.
- **Role:** business / process / systems analysis, program/project management, data/BI — intersecting with an aspirational domain.

**User limits:** ~10 vacancies per day on the daily slate; the `/annotate` queue grows as the nightly passes run (no hard ceiling — there used to be an xlsx batch of 100, retired).

## Goal & Business Driver

Today the search is manual reading of variously-formatted descriptions with no filters on hidden parameters (first of all the "de-facto" German language requirement); estimated ~an hour a day, mostly on garbage. Goal: **≤10 minutes a day**, the system filters out garbage, explains its scores, and learns from every decision — including detecting repellers that Alina has not yet put into words herself.

## Job (JTBD)

> When I'm looking for a job in a field new to me and don't know what my «шабашка» looks like, I want the system to show me up to 10 best candidates a day with an explanation and to learn from my reactions, so that I find my dream job without spending hours reading garbage vacancies.

## Trigger & Frequency

- **Step 0 — DONE 2026-06-12** (see [spike/SPIKE_REPORT.md](spike/SPIKE_REPORT.md)): verdict **go-with-changes**. Measured: Indeed ✅ (213 rows, 100% descriptions), Arbeitsagentur API ✅ (314 rows — found a co-primary that wasn't in the plan), LinkedIn ⚠️ (289 rows, 1.7% descriptions), Glassdoor ❌ / Google ❌ (dead; Google dies **silently**). Bootstrap volume confirmed: 731 unique vacancies in one evening (2.4× the target of 300).
- **Step 0.5 — gate before annotation (mandatory, before bootstrap):** (a) 3 consecutive nights of the same probes via cron — bans show up on the 2nd–3rd night + measure the real churn; (b) LLM pilot: 20 real descriptions through qwen → card validity, translation, language_reality, throughput; (c) decision on LinkedIn (TBC-8); (d) health-counter canaries implemented and tested against Google's "silent zero" signature.
- **Bootstrap (one-off):** collect ≥300 (already have 731) → annotate 100.
- **Daily:** cron every night (03:00) — scrape and score; in the morning Alina opens the slate. Every day.
- **Weekly:** taste report + judge revalidation.

## Actors & Stakeholders

| Actor | Role | Interest / responsibility |
|-------|------|---------------------------|
| Alina | initiator, sole user, annotator | find a шабашка in ≤10 min/day |
| Pipeline | nightly vacancy collection | **Indeed + LinkedIn via the Vlislavn/JobSpy fork (1.1.82 + #343) + native Arbeitsagentur fetcher** (v4/jobs, X-API-Key `jobboerse-jobsuche`); incrementality via `hours_old=24–48`; Should: Arbeitnow JSON API, GermanTechJobs RSS; canary + min-row assertion per source |
| Desc-Fetcher | obtaining the full description (a new explicit stage) | per-board policy: Indeed — already 100%; Arbeitsagentur — `/pc/v3/jobdetails` ~1 req/s (v2 dead); LinkedIn — per TBC-8 |
| Normalizer (LLM, local qwen3:8b — per the pilot) | raw description → unified **card** | role, company, domain, location/hybrid, language reality, integration potential, 2-line summary, slop-score; translation from German |
| Judge (LLM, local qwen3:8b) | scores the card 1–5 + "why" | rubric from `config/profile.yaml` + few-shot from labels; tag or free-text |
| Slate | renders up to 10 + receives feedback | local FastAPI page, buttons write to SQLite |
| Sources (external) | data | **complement, not redundancy**: board overlap is 2.9% — losing a board = losing its whole layer; degradation is the norm |

## Preconditions

- Step 0 (spike) passed ✅; step 0.5 (3 nights + LLM pilot + canaries + LinkedIn decision) passed before annotation begins.
- The query matrix is run with German terms too (Raumfahrt, Verteidigung, Tierpflege, öffentlicher Dienst…) — TBC-1.
- `config/profile.yaml` created from the interview (a bootstrap artifact, not implied knowledge).
- LLM: local stack (ollama qwen3.5:4b / qwen3:8b; bge-m3 in the HF cache) — TBC-3 closed, $0.
- jobspy is installed from **Vlad's fork** `Vlislavn/JobSpy` @ `89b0b3d` (= 1.1.82 + LinkedIn date-fix #343 + interactive CLI): `pip install -e "~/code/from GH/JobSpy"`; spike scripts kept as regression canaries.
- SQLite database initialized.

## Inputs

- Search query matrix: magnet domains × roles × {Heidelberg, Frankfurt} — drafted before the spike, calibrated afterward.
- `config/profile.yaml` — preference profile and judge rubric.
- Alina's labels (`/annotate` + daily feedback on the slate) — golden dataset.

## Main Success Scenario (daily)

1. Cron 03:00 launches the pipeline. → Collection: Indeed + LinkedIn (jobspy fork, **incrementally via `hours_old=24–48`** — server-side freshness filtering instead of a full re-scrape) and Arbeitsagentur (native fetcher, pagination size=50, filter on `datumErsteVeroeffentlichung`) over the query matrix; write to SQLite; **canary query + min-row assertion** per source.
2. The pipeline deduplicates **conservatively** (refnr for Arbeitsagentur; exact URL + normalized company+title across boards). → fuzzy candidates are only logged, not merged; new ones get `status=new`.
3. **Geocoded** geo-filter: an allowlist of cities/PLZ with real distance to Heidelberg/Frankfurt; don't trust the board's radius (miles, measured leak: Stuttgart ~85 km in the Heidelberg@50 results). → `status=prefiltered` with a reason.
4. Desc-Fetcher obtains full descriptions where they're missing: Arbeitsagentur `/pc/v3/jobdetails` (~1 req/s), LinkedIn — per the TBC-8 policy. → `status=described`; a details failure → the vacancy waits, picked up tomorrow.
5. The Normalizer builds a card for each described vacancy (budget K/night per the pilot results; overflow → queued to the next night). → `status=normalized`.
6. Authoritative filter **by card**: remote-only, outside Heidelberg/Frankfurt hybrid, `language_reality=de`; the structural `is_remote` is **only a hint** (measured: 29 Indeed rows False-but-hybrid, LinkedIn constantly False). → `status=filtered` with a reason.
7. The Judge scores each surviving card: a 1–5 rating, a "why" tag (from a dictionary, by rating polarity) **or free-text**, 1 sentence of explanation. → `status=scored`.
8. The system assembles the slate: **8 exploit** (top by rating, ≤3 per company/domain) + **2 explore** (random/uncertainty sample from scored outside the top and the reserve of past days, marked explore). → page render; there may be fewer than 10 vacancies — you must not pad with garbage.
9. Alina opens the local page in the morning. → sees the cards: rating, tag, explanation, link to the original.
10. For each one Alina presses 💻🐀 / 😎 / 👸✨🧚 / applied (or skips). → write to the golden dataset in ≤1 s (scale mapping: 💻🐀=2, 😎=4, 👸✨🧚=5; applied — a separate flag on top of the rating); the vacancy no longer appears in the slate.
11. Weekly: `/eval` — agreement / matching quality against Alina's real labels; `/gaps` — top recurring skill gaps for the desired roles; the judge's few-shot is topped up with edge labels. → Alina updates `config/profile.yaml` if she wants.

## Bootstrap Flow (sub-use case, one-off) — now via `/annotate`

1. Spike night (step 0): run the query matrix over all boards → a "board × volume × quality" report → go / reshape the matrix.
2. The system collects vacancies, filters (geo/hard), deduplicates, normalizes and judges — the scored ones land in the **`/annotate` queue**.
3. Alina opens `/annotate` (**the only annotation surface** — the `build-bootstrap`/`labels-import` xlsx batch is retired): a queue of scored-but-unlabeled cards, the same buttons **💻🐀 / 😎 / 👸✨🧚** (score 2/4/5) as on the daily slate. She labels by chunk when she has time; labeled items leave the queue.
4. After ~30 labels the ML gate (LightGBM) turns on; ~50–100 — the judge is calibrated (`schabasch cv`, agreement ≥75% against her labels); few-shot is topped up with edge labels (💻🐀=2 / 👸✨🧚=5).
5. **`/eval`** shows matching quality against her REAL labels (live, updated as labeling proceeds); **`/gaps`** — which skills are regularly lacking for the desired roles.

## Extensions / Alternate Flows

- **1a.** A source returned 0 / went down: the canary with the min-row assertion distinguishes a "dead scraper" from an "empty market" (measured case: Google returned 0 rows **without an exception** — HTTP 200, JS shell); alert in the slate header, not silence. Remember: boards are a complement (2.9% overlap), losing a board = losing its layer.
- **1b.** All sources returned 0: the slate is assembled from the reserve (scored items from past days, not shown); a "stale" header + a log entry.
- **4a.** Arbeitsagentur's details endpoint returned 403/empty (precedent: v2 died silently): the vacancy stays `new`, retry tomorrow; counter to the log.
- **5a.** The LLM endpoint is unavailable: the night skips normalization/scoring, the queue is preserved; slate from the reserve with a marker; retry the next night.
- **7a.** The Judge returned invalid JSON: retry ×2 with backoff → the card stays `normalized`, picked up tomorrow; to the log.
- **9a.** Alina didn't open the slate: nothing is lost; tomorrow the slate is reassembled accounting for the new ones, the unviewed compete on equal footing.
- **10a.** 💻🐀 on everything for ≥5 days in a row: the system shows a "what I've learned about your taste" report and proposes expanding the query matrix (exploration mode, Should).

## Edge Cases

| Case | Resolution | Why |
|--------|---------|------|
| The same job under different URLs (aggregators) | conservative dedup; fuzzy pairs to the log for manual review | a false merge silently destroys a live vacancy — worse than a duplicate |
| Vacancy disappeared (404) | `status=expired`, removed from the slate | don't waste Alina's click |
| Description in German | Normalizer translates, sets a language flag; the *description's* language ≠ the *job's* language | a German-language vacancy about an English-language job exists |
| English text, but "fließend Deutsch erforderlich" | `language_reality=de` → filtered with a reason in the log | pain #1 from the interview; **measured: 18.1% (27/149) of English descriptions are a trap**; those 27 are a ready test set for the detector |
| Recruiter spam / agency postings | Arbeitsagentur: deterministic flag `istArbeitnehmerUeberlassung`; Indeed/LinkedIn: tag filter (Should) | the official API gives the flag for free — more accurate than any prompt |
| Silent zero from a source | canary + min-row assertion (Extension 1a) | Google died exactly like this: 0 rows, 0 exceptions |

## Error Handling

- No stage failure brings down the pipeline: vacancies stay in their previous finite-state-machine status and are picked up the next night (idempotency).
- All LLM calls — retry with exponential backoff; classified errors to the log.
- Each nightly run writes a funnel report: collected / dedup / prefiltered / normalized / filtered / scored / in slate / errors — the numbers show where it's leaking.

## Output & Post-Success State

- **Output:** the morning slate (local page, up to 10 cards) · a growing golden dataset (CSV export) · a weekly taste report · the funnel log.
- **Post-state:** labels recorded; the judge's few-shot is fresh; on 👸✨🧚/applied Alina applies via the link herself. The user's next step is to actually apply.

## Acceptance Criteria (Given / When / Then)

| Scenario | Given | When | Then |
|----------|-------|------|------|
| Happy path | the nightly run succeeded, ≥10 scored | Alina opens the slate | up to 10 cards (8 exploit + 2 explore marked), each with a rating, "why", explanation, link; review ≤10 min |
| Hidden German | an English vacancy with "fließend Deutsch erforderlich" | normalization | `language_reality=de`, doesn't enter the slate, reason in the funnel log |
| Thin day | scored < 10 | slate assembly | the slate is shorter than 10; not padded with garbage |
| Board degradation | one source returned 0, the rest are alive | the nightly run | the canary/min-row assertion distinguishes a dead scraper from an empty market; alert in the slate header; the pipeline made it through |
| Total zero | all boards returned 0 | slate assembly | slate from the reserve + a stale banner + a log entry |
| Feedback | Alina pressed 💻🐀 | write | label to the golden dataset in ≤1 s; the vacancy doesn't return to the slate |
| Judge validation | 100 bootstrap labels | 5-fold CV | agreement on the binary intent (rating ≥4 ↔ "interview=yes") ≥75%; otherwise the rubric is revised before launching daily |

## Non-Functional Requirements

- **Performance:** the nightly run ≤2 h (verify with the pilot: first night ~731 cards × median ~1,081 tok.; steady-state per the churn measurement, estimate 37–110 new/night); the slate is precomputed — opens instantly (Doherty); a feedback click is recorded in ≤1 s.
- **Cost:** **$0 — fully local stack** (ollama qwen3.5:4b / qwen3:8b — already downloaded; bge-m3 4.3GB + bge-reranker-v2-m3 2.1GB — already in the HF cache; M4 Pro / 48GB handles it with room to spare). API keys (names in zotero-summarizer/.env) — fallback only.
- **Privacy:** everything local (SQLite + files + local LLM); nothing leaves the machine; the Arbeitsagentur API is the only external call besides scraping.
- **Reliability:** the pipeline is idempotent; statuses form a finite-state machine `new → prefiltered | described → normalized → filtered | scored → slated → labeled | expired`; restart is safe; jobspy from the fork `Vlislavn/JobSpy` @ 89b0b3d (editable install — fixes under Vlad's control), spike probes — regression canaries after any bump.
- **Autonomy (agentic):** read-only risk tier; no auto-applies; the deeper-investigation agent (Could) — read-only of public sources.
- **Observability:** a funnel log per run; board health counters; the judge↔Alina agreement history (`/eval`).
- **Tesler:** the complexity of reading variously-formatted descriptions is carried by the system (cards), not Alina.

## Data Entities & Fields

| Entity | Field | Type | Req | Validation / Notes |
|--------|-------|------|-----|--------------------|
| Vacancy | id, url, title, company, location, remote_type, source_board, refnr, is_temp_agency | — | yes | url is unique; refnr — Arbeitsagentur's stable key; company+title normalized for dedup; is_temp_agency from `istArbeitnehmerUeberlassung` |
| Vacancy | description_raw, description_lang | text | yes | the description's language ≠ the job's language |
| Vacancy | card_json | json | after norm. | role, domain, hybrid, language_reality, integration_est, summary, slop_flag |
| Vacancy | status | enum | yes | finite-state machine (see NFR) |
| Vacancy | first_seen, last_seen | ts | yes | for expired |
| Label | vacancy_id, score_1_5, why_tag, why_freetext, interview_bool, applied_bool, source | — | yes | source ∈ {bootstrap, slate}; unified scale: 💻🐀=2, 😎=4, 👸✨🧚=5 |
| JudgeScore | vacancy_id, score, tag, freetext_why, explanation, model, rubric_version, ts | — | yes | rubric_version — for comparing rubric versions |
| SlateEntry | date, vacancy_id, rank, slot_type, feedback | — | yes | slot_type ∈ {exploit, explore} |

## Scope (MoSCoW)

- **Must (MVP):** ~~step 0 spike~~ ✅ done · **step 0.5: 3 consecutive nights + LLM pilot of 20 cards + canaries + LinkedIn decision** · nightly collection **Indeed + Arbeitsagentur API** (co-primary) with canaries and min-row assertions · conservative dedup (refnr + URL + company/title) · **geocoded** geo-filter (allowlist of cities/PLZ) · **description-fetching stage** (AA v3-details ~1 req/s; LinkedIn per TBC-8) · normalization into cards (local LLM, budget K from the pilot) · authoritative filter by card (incl. language_reality; is_remote — a hint) · LLM judge on all survivors (1–5 + tag/free-text + explanation) · cross-encoder + LLM coverage → `fit_score` + eligibility gate (de-conflating magnet from fit) · slate of up to 10 = 8 exploit + 2 explore, fresh only (`slate.fresh_days` 14) · web surfaces: `/` feedback (💻🐀/😎/👸✨🧚/applied → SQLite) · `/annotate` (annotation queue) · `/eval` (validation against real labels) · `/gaps` (skill gaps) · `config/profile.yaml` as an explicit artifact · golden dataset export · funnel log · pin `python-jobspy==1.1.82` + spike probes as regression canaries.
- **Should:** German terms in the query matrix (TBC-1; the thin magnets "animals" 5.9% and "public sector" 5.5% starve without them) · Arbeitnow JSON API + GermanTechJobs RSS as tertiary fetchers (~30 LOC each, legally clean, 0% overlap with the boards) · a heuristic hidden-German detector (a second line; the test set of 27 is ready) · a vacancy-text slop detector · auto integration score · a weekly taste report + a proposal for new tags · a recruiter-spam tag filter for Indeed/LinkedIn (for AA — the structural flag is already in Must) · exploration mode · churn measurement via a 48–72 h repeat run.
- **Could:** an ML gate on embeddings (bge-m3 already in the cache) — **only if** the funnel log shows consistently >100 survivors/night · a deeper-investigation agent for top candidates (company website, kununu) · an applications kanban · StepStone v2 — only if a coverage analysis shows the Indeed crawler misses its exclusives.
- **Won't (v1):** auto-applies and sending anything outward · a hosted/packaged UI (a local single-page page is not a "UI app") · multi-user · mobile · **Glassdoor and Google** (measured: dead under jobspy 1.1.82; Google — silently) · **scraping StepStone** (Akamai + an explicit ToS ban; content partly arrives via Indeed) · **Xing** (API dead since 2022, JS shell, AGB/GDPR risks) · **any salary features** (Indeed DE 0%, LinkedIn 0%, AA ~10% — no data).

## Success Metrics

- Time on the daily review ≤10 min/day (self-measured; baseline — estimated ~60 min, not measured).
- ≥1 😎/👸✨🧚 reaction in the slate on ≥50% of days (rolling 4-week window).
- Share of 😎+👸✨🧚 in the slate: the 4-week rolling average ≥ the level of the first two weeks and not degrading.
- Judge validation: 5-fold CV agreement on the binary intent ≥75% on 100 bootstrap labels (the daily-launch gate) **and above the majority-class baseline + 10 pp** (check on the first ~30 labels — with skewed labels 75% may be trivial); Spearman is reported diagnostically, it is not a threshold.
- language_reality detector: ≥25/27 on the ready hidden-German mini-test-set from the spike (`spike/data/indeed.csv`).
- Bootstrap: 100 labels collected (recommendation — 5 sessions of ≤30 min).
- **North Star:** ≥3 `applied` presses on vacancies rated 5 in the first 2 months of the daily regime.

## Relationships

- **Reuse source:** `~/code/personal/zotero-summarizer` — the audit confirmed it (see spike report §4), strategy **fork-and-gut**: take the LLM client (`integrations/llm.py`, LLMClient protocol), daemon/tick (`services/triage/feeds/_loop.py` + `_tick.py`), the feedback surface (`frontend/src/components/VerdictPanel.jsx` + `storage/_repo_verdicts.py`/`_repo_feedback.py`, ~250 lines, PRIORITIES → bad/good/star/applied), the goals.yaml+pydantic config pattern. **Do NOT port** `services/golden/goldenset.py` (Zotero-entangled) — write our own label ingestion from slate events. The classifier gate is SPECTER2-dependent — relevant only on a Could trigger, swap the embedder for bge-m3.
- **Conceptual source of the slop detector:** the aislop skill/repository (detection of AI-generated text).
- **Upstream — measured statuses (2026-06-12, jobspy 1.1.82):** Indeed ✅ (0 blocks, 100% descriptions); LinkedIn ⚠️ (0×429 at depth 1–3 pages without a proxy; descriptions only via fetch, not tested at volume); Glassdoor ❌ (HTTP 400 + GraphQL refusal); Google ❌ (silent zero, parser rotted). **Arbeitsagentur Jobsuche API** ✅ — official, static key `jobboerse-jobsuche`, no registration; details v3 only. Tertiary: Arbeitnow API ✅, GermanTechJobs RSS ✅ (1,590 items). Full numbers: [spike/SPIKE_REPORT.md](spike/SPIKE_REPORT.md).
- **Vlad's fork and playground:** `~/code/from GH/JobSpy` = the fork `Vlislavn/JobSpy` @ 89b0b3d (1.1.82 + upstream fixes #343/#337 + his own `jobspy_cli.py`). `jobspy_playground.ipynb` — a working prototype: the 2026-06-11 run independently confirmed the spike (salary 0/30, Heidelberg 1/30); `render_job()`/`show_jobs()` with FIELD_LABELS — a ready card skeleton for slate rendering; `hours_old` — the mechanism for incremental nightly collection.

## Glossary & Enums

| Term | Meaning | Allowed values |
|--------|----------|---------------------|
| Шабашка | the dream job, the best in the dataset | rating = 5 |
| Офисная мышь (office mouse) | the worst job: boredom + slop + gives nothing | rating = 1 |
| Card | normalized LLM representation of a vacancy | card_json |
| Slate | the morning selection of up to 10 | 8 exploit + 2 explore |
| Judge | LLM judge by rubric | rating 1–5, tag/free-text, explanation |
| Integration score | integration points in Germany | 0–2 (auto-scored — Should) |
| Hidden German | English text, de-facto needs German | language_reality=de |
| Golden dataset | all of Alina's labels, a unified scale | CSV export |
| Slate feedback | local-page buttons | 💻🐀=2 · 😎=4 · 👸✨🧚=5 · applied (flag) |

## Open Questions

| ID | Question | Owner | Needed by | Resolution |
|----|--------|-------|---------|-----------|
| TBC-1 | Final query matrix — add German terms (Raumfahrt, Verteidigung, Tierpflege, öffentlicher Dienst, Prozessverbesserung…); the English base is already measured | Vlad + system | before bootstrap | partial: the English matrix works (731 in an evening); the German run remains |
| TBC-2 | Alina's resume (for Should matching and refining the role) | Alina | ✅ closed: CV loaded via `candidate --cv-path` | — |
| TBC-3 | LLM: local vs API | — | — | **decided 2026-06-12: local, $0** — qwen3.5:4b (Normalizer) / qwen3:8b (Judge), bge-m3 in the cache; quality is verified by the pilot (TBC-7) |
| TBC-4 | Is a proxy needed for LinkedIn | system | — | **decided for shallow**: 0×429 at 25/request without a proxy; deep pagination and fetch-at-volume → TBC-8 |
| TBC-5 | Name of category 1 | Vlad | — | decided: «офисная мышь» (office mouse) |
| TBC-6 | Real nightly churn (new uniques/night) — currently a guess of 5–15% of 731 | system | step 0.5 (3 nights) | — |
| TBC-7 | Local LLM quality on real cards | — | — | **decided 2026-06-12 (pilot of 28 cards):** JSON 28/28, language_reality 19/20 (hidden German 5/5), ~7 s/card, 731 = 1.55 h; **Normalizer and Judge = qwen3:8b** (Russian in the summary 8/8 vs 7/20 for 4b). Fix with the prompt: the "2 lines" format; further-test slop/temp discrimination |
| TBC-8 | LinkedIn's role | — | — | **decided 2026-06-12: pipeline-grade nightly.** 300/300 descriptions in 7.8 min, zero 429 without a proxy (jobspy's built-in pacing ~0.64 req/s); safety net: 5 s between requests + stop after 2 consecutive failures |
| TBC-9 | Robustness of dedup keys across days (reposts under new URLs) and link-rot of slate links | system | repeat run 48–72 h | — |

## Broad Implementation Strategy (forward-looking, not a tech spec)

- **Reuse (audit-verified, files named in Relationships):** LLMClient + daemon/tick + VerdictPanel/verdict-storage + the config pattern from zotero-summarizer; the fork `Vlislavn/JobSpy` (editable) for Indeed/LinkedIn; `render_job()`/`show_jobs()` from `jobspy_playground.ipynb` as the slate-card prototype; spike scripts (`spike/*.py`) as the basis for production fetchers and as regression canaries.
- **To build:** the native Arbeitsagentur fetcher (search v4 + details v3, ~1 req/s) · the geocoder allowlist (an offline table of cities/PLZ) · canaries + min-row assertions · the normalizer prompt + the card schema · `config/profile.yaml` from the interview · the bootstrap-table generator · the judge prompt with few-shot · the slate renderer + the FastAPI feedback page · label ingestion from slate events (our own, goldenset.py is not ported) · the funnel log.
- **Plan:** 1) ~~spike~~ ✅ → 2) step 0.5: 3 cron nights + LLM pilot of 20 cards + LinkedIn decision + canaries (all the scripts already exist, ~zero cost); 3) the German run of the matrix (TBC-1) + collection; 4) the bootstrap table from a stratified 100 (dry-run the stratification on the already-collected 731); 5) labeling 100 + judge calibration + the CV gate (with the majority-baseline check); 6) the daily cycle (slate + feedback); 7) Should features as the data arrives.

## Revision History

| Date | Version | Author | Change |
|------|---------|--------|--------|
| 2026-06-12 | 0.1 | Vlad + Claude | Initial definition: interview → brainstorm → UX gate (Tesler/Hick/Cognitive Load/Goal-Gradient) → adversarial review (architect + skeptic), edits applied |
| 2026-06-12 | 0.2 | Claude (spike, 10 agents) | Empirical spike done (see spike/SPIKE_REPORT.md): + Arbeitsagentur API co-primary; + a description-fetching stage; + a geocoded geo-filter; + canaries; − Glassdoor/Google (dead); TBC-3/4 closed; added step 0.5 (3 nights + LLM pilot); TBC-6…9 opened |
| 2026-06-12 | 0.2.1 | Vlad + Claude | Accounted for the Vlislavn/JobSpy fork @ 89b0b3d (install editable instead of a PyPI pin) and jobspy_playground.ipynb: `hours_old=24–48` as the mechanism for incremental nightly collection; LinkedIn fetch_description confirmed on n=15; render_job/show_jobs — the slate-card prototype |
| 2026-06-12 | 0.2.2 | Claude (workflow, 9 agents) | TBC-7/8 closed by measurements (LLM pilot 28/28; LinkedIn 300/300 without bans; AA 298/298). Created: docs/ROADMAP.md, docs/LITERATURE_REVIEW.md (lit + GitHub import list + SOTA comparison), data/annotation/batch_001.xlsx (100 rows, LI 34/IN 33/AA 33). Normalizer → qwen3:8b. Judge plan B (setwise/best-worst) documented — the decision is Vlad's if the CV gate fails |
