# Schabaschkascuhen — Roadmap

> Why this file exists: [USE_CASE.md](../USE_CASE.md) answers "what and for whom", this document — "in what
> order, why in this order, and how we know a phase is done". Each phase has a **gate** — a measurable
> exit condition. Without passing the gate, the next phase does not begin.
> Companion documents: [spike/SPIKE_REPORT.md](../spike/SPIKE_REPORT.md) (measured reality),
> [docs/LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) (literature + SOTA + import list).

## North Star and why we are building this

Vlad is looking for a «шабашка» — the dream/jackpot gig in a field new to him (not biotech), Heidelberg/Frankfurt,
hybrid, English + integration into Germany. The key feature: **there are no reference examples** — the system
must *discover* his taste from his ratings, not match against a sample. Today's pain: ~an hour a day spent
reading mixed-format junk, hidden German requirements (measured: **18.1%** of English-language
vacancies are a trap).

**North Star:** ≥3 `applied` clicks on vacancies rated 5 within the first 2 months of daily mode.
**Proxy metrics:** ≤10 min/day; ≥1 😎/👸✨🧚 in the slate ≥50% of days; judge↔Vlad agreement ≥75% (5-fold CV,
above majority-baseline + 10 pp).

## Principles (fixed by the project's goal)

1. **Write as little ourselves as possible — import everything**: a fork/import/copy of a pattern always takes
   priority over our own code. Every component in the map below carries an explicit source tag.
2. **$0 and local**: ollama (qwen3.5:4b / qwen3:8b) + bge-m3 from the HF cache; API — fallback only.
3. **Gates with measurements, not opinions**: a phase closes on a number (the way the spike closed step 0).
4. **Annotation is the core of the product, not prep work**: every click by Vlad grows the golden dataset.
5. **Vacancy freshness does not matter for annotation** — the material comes from the already-collected pool.

## Component map: build vs import

| Component | Source | Status |
|---|---|---|
| Indeed/LinkedIn scrape | **fork `Vlislavn/JobSpy`** @ 89b0b3d, editable | ✅ wired in |
| Arbeitsagentur fetcher (v4 search + v3 details) | `sources/arbeitsagentur.py` | ✅ implemented (verified against live API) |
| Tertiary fetchers (Arbeitnow API, GermanTechJobs RSS) | `sources/tertiary.py`, strict region filter | ✅ implemented |
| Geocoder allowlist (city/PLZ → distance) | `geo.py` (offline table ~95 cities) | ✅ implemented |
| LLM client (retry/backoff, structured JSON) | `llm.py` (zotero `integrations/llm.py` pattern) | ✅ implemented |
| Daemon/tick orchestration | `pipeline.py` (`_loop.py`+`_tick.py` pattern) | ✅ implemented |
| Feedback page (4 buttons → SQLite) | `feedback_app.py` (VerdictPanel pattern) | ✅ implemented |
| Slate card render | `slate.py` (`render_job()` style) | ✅ implemented |
| Config (profile/rubric.yaml + .env) | `config.py` + `config/profile.yaml` | ✅ implemented |
| Annotation — web queue `/annotate` (xlsx retired 2026-06-15) | `slate.annotation_batch` + `feedback_app` route | ✅ implemented |
| Judge CV calibration (fold-SEM, N=3, baseline) | `calibration.py` (are/multi-run-variance) | ✅ implemented |
| ML gate (embeddings + log-reg) | **copy** of classifier.py from zotero-summarizer, embedder → bge-m3 | Could, on trigger |
| Import list from the GitHub research | see [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) §3 | ✅ ready |

**Import list v1 (result of the GitHub research, 2026-06-12):**
1. **JobSpy PR [#347](https://github.com/speedyapply/JobSpy/pull/347)** — cherry-pick into the fork: 13 lines, fixes the dead Glassdoor (CSRF URL after the Next.js migration). Optionally PR #366 (Xing scraper, 227 lines) after a smoke test.
2. **[bundesAPI/jobsuche-api](https://github.com/bundesAPI/jobsuche-api)** — openapi.yaml as the spec for our AA client (the README recommends `/pc/v4/jobdetails`; our v3 works — check v4).
3. **RapidFuzz** (MIT, active) — fuzzy dedup following the archived JobFunnel `filters.py` pattern (TF-IDF threshold + duplicate taxonomy + seen-lockfile): we import the library, copy the pattern.
4. **[ielab/llm-rankers](https://github.com/ielab/llm-rankers)** (Apache-2.0, pip) — setwise/pairwise ranking of the slate (see the judge's Plan B in Phase 2).
5. **QuickApply** (MIT) + **VincenzoImp/job-search-tool** (MIT) — copy of the review/exclude/track-cycle pattern and the FastAPI+React skeleton for the feedback page.
6. **Rejected**: the entire AIHawk line (archived + AGPL), jobspy-api wrappers, dedupe.io, ojd_daps_skills (Anglo-centric), HN tooling. Upstream JobSpy has not been pushed since 2026-02 — the fork remains authoritative.

## Phases

### Phase 0 — Spike: what is really available ✅ CLOSED 2026-06-12

Verdict go-with-changes. Indeed ✅ 213/100% descriptions · **Arbeitsagentur API ✅ 314 (co-primary,
found by the spike)** · LinkedIn ⚠️ (descriptions only via fetch) · Glassdoor/Google ❌ dead ·
board overlap 2.9% · hidden German 18.1% · LLM stack local $0 · pool of 731 unique
vacancies in one evening. Full numbers: [SPIKE_REPORT.md](../spike/SPIKE_REPORT.md).

### Phase 0.5 — Trust gate (in progress)

**Why:** one night ≠ every night; the LLM has not yet produced a single card.

| Test | Status | Result |
|---|---|---|
| LLM pilot: 28 cards via qwen (TBC-7) | ✅ **PASSED 2026-06-12** | JSON-valid **28/28 (100%)**, language_reality **19/20** (hidden German 5/5), ~7 s/card; 731 cards = **1.55 h**, a 50–110 night = **6–14 min**. Verdict: **ship local, qwen3:8b** also for the Normalizer (Russian in the summary 8/8 vs 7/20 for 4b). Weaknesses: the "2 lines" format is not respected (fixable via prompt), slop/temp flags not discriminated (no positives in the pilot) |
| Arbeitsagentur at full volume (TBC-9 part) | ✅ **PASSED** | **298/298** HTTP 200 @0.93 req/s, zero 403/429, p50 latency 0.07 s; externeURL rows (28) — all with full descriptions. Production-ready |
| LinkedIn fetch at scale (TBC-8) | ✅ **PASSED** | **300/300** descriptions in 7.8 min, zero 429 (jobspy's built-in pacing ~0.64 req/s is sufficient); 243 rows, 100% descriptions, median 3,894 chars. Verdict: **pipeline-grade nightly** (insurance: 5 s between requests + stop after 2 consecutive failures — already in `spike/scripts/a1_runner.py`) |
| 3 nights in a row of cron probes + churn measurement (TBC-6) | remaining | 3/3 nights without blocks; churn measured |
| Canaries + min-row assertions (signature of a "silent Google") | Phase 3 code | unit test on a simulated zero |

### Phase 1 — Material for annotation ✅ COLLECTED 2026-06-12

**Why:** quickly give Vlad 100 cards — annotation must not wait for the pipeline. Freshness does not matter.

- ✅ Pool with full descriptions: Indeed 213 + **LinkedIn 243** (collected today, 100% descriptions) +
  **Arbeitsagentur 298** (details collected today, 100%) = **754 annotation-grade vacancies**.
- ✅ Artifact: [`data/annotation/batch_001.xlsx`](../data/annotation/batch_001.xlsx) — exactly 100,
  stratification **LinkedIn 34 / Indeed 33 / AA 33** (the LinkedIn quota — Vlad considers their postings
  higher quality), with known negatives; an instruction sheet, dropdowns, the feature-3 formula,
  a progress counter; a CSV mirror alongside.
- **Remaining gate:** Vlad opens the file and confirms "annotation is comfortable" on the first 5.

### Phase 2 — Annotation + judge calibration

**Why:** this is the creation of ground truth — the most valuable part of the system.

- Vlad: 100 × 3 features (recommended 5 sessions × 20, ~30 min each).
- In parallel: `profile/rubric.md` gets extended with tags/findings from the annotation.
- Judge calibration: few-shot from extreme examples (1 and 5); 5-fold CV.
- **Gate:** agreement ≥75% on binary intent AND ≥ majority-baseline + 10 pp, counted honestly:
  mean±SEM across folds, unjudgeable outside the denominator, N=3 judge runs for verdict stability.
- **Plan B (if the gate is not cleared within 3 rubric/few-shot iterations):** the literature is unambiguous — on
  small models a comparative protocol is more reliable than pointwise 1–5 (PRP/Setwise). Then: the judge
  internally switches to setwise tournaments (`llm-rankers` is already in the import list), the 1–5 scale stays
  only as a display, anchored to labels; label collection — best-worst quadruples (same budget,
  ×5 pairwise constraints). This is **Vlad's decision**, since it changes the annotation UX — see
  [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) §2 and §6.

### Phase 3 — Daily-cycle MVP

**Why:** close the loop "night → slate → clicks → dataset".

- Nightly pipeline: cron 03:00 → Indeed+LinkedIn (`hours_old=24–48`, incremental) + AA
  (filter by posting date) → dedup (refnr/URL/company+title) → geocode filter → fetch
  descriptions → Normalizer → card filter → Judge → slate 8 exploit + 2 explore.
- Surface: a local FastAPI page (copy of VerdictPanel), buttons 💻🐀/😎/👸✨🧚/applied ≤1 s.
- Canaries, funnel log, FSM statuses, idempotency.
- **Engineering disciplines from the SOTA comparison** (full list — LITERATURE_REVIEW §5):
  (1) **hard-before-soft**: a declarative ladder of deterministic checkers (regex German with
  exoneration guards "von Vorteil/is a plus", AA temp flag, geocode, remote) BEFORE any LLM call +
  content-hash short-circuit for reposts; (2) **pin the grader-tuple**: model + ollama digest +
  rubric_version + fewshot_hash in every JudgeScore row; (3) **classified error
  envelope**: a closed enum of rejection/filter reasons (filtered_geo / filtered_language_de /
  llm_invalid_json / …) + lossless details — the funnel becomes diagnosable; (4) a single
  retry wrapper (Retry-After → capped backoff + jitter) for all network and LLM calls;
  (5) slop — density-smoothed 0–100 with tiers, not a binary flag (aislop pattern).
- **Gate:** 7 mornings in a row the slate is ready by 08:00 without manual intervention; every click lands in
  the golden dataset; the canary catches a simulated "silent zero".

### Phase 4 — Should: signal quality

- German terms in the matrix (TBC-1) — they rescue subtle magnets (animals 5.9%, public sector 5.5%).
- Heuristic hidden-German detector (second line; test set of 27 examples ready) — gate ≥25/27.
- Slop detector for the vacancy text (aislop pattern: density scoring, not a binary flag).
- Auto integration score; weekly taste report + a suggestion of new tags; Arbeitnow + GTJ fetchers.
- **Gate:** the share of 😎+👸✨🧚 in the slate (rolling 4 weeks) no lower than the bootstrap baseline; zero vacancies
  with real German in the slate over 2 weeks.

### Phase 5 — Could: by trigger, not by plan

| Feature | Inclusion trigger |
|---|---|
| ML gate (bge-m3 + log-reg from zotero-summarizer) | funnel steadily >100 survivors/night |
| Agentic re-investigator (company website, kununu) | ≥3 weeks of stable slate and Vlad asks for depth on the top picks |
| Application Kanban | ≥10 applied accumulated |
| StepStone v2 | coverage analysis shows a miss of Indeed-crawler exclusives |

## Risks (top 5, from the critique and the spike)

1. **Quality of the local judge against Vlad's taste** — the main product risk; nothing yet has
   shown that a 4B/8B model will hit 75%. Mitigation: pilot before annotation, API fallback, pairwise Plan B.
2. **Bans on nights 2–3** (Indeed Cloudflare, LinkedIn 429 at fetch volume) — mitigation: the 3-night
   test, AA as a legally clean co-primary, throttling.
3. **Churn unknown** (5–15% — a guess) — the size of K and the ≤2 h window may drift; mitigation: measure it.
4. **jobspy drift** (Glassdoor/Google rotted under 1.1.82) — mitigation: an editable fork under control,
   spike probes as regression canaries.
5. **Annotation fatigue** (300 clicks) — mitigation: cards instead of raw texts, sessions of 20,
   auto-fill of feature 3, a progress bar.

## Open questions

They live in [USE_CASE.md → Open Questions](../USE_CASE.md#open-questions): TBC-1 (German matrix),
TBC-2 (resume), TBC-6 (churn), TBC-7 (pilot — closing today), TBC-8 (LinkedIn — closing
today), TBC-9 (reposts/link-rot).

## History

| Date | Event |
|---|---|
| 2026-06-12 | Use case v0.1 → v0.2.1 (interview → brainstorm → UX gate → adversarial review → spike of 10 agents → fork/notebook) |
| 2026-06-12 | Roadmap v1; Phase 0 closed; Phase 0.5 and Phase 1 launched (workflow: LinkedIn@scale, AA@volume, LLM pilot, research ×4, assembly of the annotation package + lit review) |
| 2026-06-13 | **Codebase v1 implemented** per the `schabasch/INTERFACES.md` contracts: all Must modules (sources/jobspy+AA, geo, hardfilters, normalize, judge, slate, feedback_app, pipeline, cli) + Phase 2 (calibration CV gate) + Phase 1 (bootstrap generator). 51 tests green (no network/ollama). **Run end-to-end on the spike pool:** import 754 → geo −122 → hard −269 (235 hidden-DE / 34 Zeitarbeit) → normalize (qwen3:8b, ~6 s/card, 0 JSON errors) → judge (tag+score, 0 errors) → slate 8+2 → HTML → FastAPI serve + POST /feedback → golden.csv. AA client verified against the live API (16 vacancies + details). German detector: 41/43 on the spike set (gate ≥25). |
| 2026-06-14 | **Import-over-build audit + finalization:** (1) RapidFuzz fuzzy dedup implemented (`dedup.py`, 11 tests, found 10 real cross-board duplicates in the spike pool); (2) AA details migrated v3→v4 (live probe: identical response); (3) Tertiary regional filter tightened (dist is not None); (4) macOS launchd scheduler (`deploy/com.schabasch.nightly.plist`, plutil-valid, ship-only); (5) `docs/IMPORT_AUDIT.md` — full conformance table. 66 tests green. tick --tertiary validated live. Remaining: Vlad annotates 100 → CV gate ≥75% → launchctl load → Phase 3 (7 nights). |
