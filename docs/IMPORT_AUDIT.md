# Import-over-build Audit — v1

**Date:** 2026-06-14. **Auditor:** implementation pass vs. `docs/LITERATURE_REVIEW.md` §3 + §6.

Project principle #1: *import/fork/copy a pattern first; only write from scratch when nothing fits.*
This document records the conformance verdict for every import-list item and SOTA-pattern discipline.

---

## §3 Import-list v1 — per-item verdicts

| # | Item | Verdict | Evidence |
|---|---|---|---|
| 1 | **Cherry-pick JobSpy PR #347** (Glassdoor CSRF fix, 13 lines) | **PARTIAL** — fork wired, PR #343 merged; PR #347 not cherry-picked (Glassdoor outside v1 search matrix, spike measured it as dead). To apply: `git -C "$(pip show python-jobspy | grep Location | cut -d' ' -f2)/jobspy" cherry-pick <commit>` | `sources/jobspy_source.py`; spike measured 0 Glassdoor hits. |
| 2 | **AA client on `/pc/v4/jobdetails`** (bundesAPI/jobsuche-api README) | **DONE** — migrated from v3→v4 (2026-06-14). Live probe confirmed both versions return HTTP 200 with identical `stellenangebotsBeschreibung`; v4 is now the active endpoint. | `sources/arbeitsagentur.py:102` — `f"{API}/pc/v4/jobdetails/{b64}"` |
| 3 | **RapidFuzz** fuzzy dedup, JobFunnel `filters.py` pattern | **DONE** — `rapidfuzz` declared in `pyproject.toml:15`, now imported and used in `schabasch/dedup.py`. Logged-not-merged contract. Live run found 10 real cross-board near-dups in the spike pool (identical titles across indeed/linkedin). | `schabasch/dedup.py`; `tests/test_dedup_fuzzy.py` (11 tests, all green) |
| 4 | **llm-rankers** (setwise/pairwise) for slate ranking | **CORRECTLY DEFERRED** — Plan B judge; triggered only if CV gate fails ×3 iterations + Vlad decides to change UX. Roadmap §Phase 2 Plan B. | Not installed; `judge.py:9` comment flags it. |
| 5 | **QuickApply + VincenzoImp/job-search-tool** UI pattern | **DONE** — `feedback_app.py` follows VerdictPanel pattern (4 POST endpoints, direct SQLite write, zero impedance mismatch). Surface 2 annotation tool. | `feedback_app.py`; §4 annotation-platform analysis chose this as winner. |

---

## §5 SOTA-pattern disciplines — conformance table

| Pattern | Source | Verdict | File:line |
|---|---|---|---|
| Hard-before-soft filter ladder + content-hash short-circuit | ARE `tool_judge.py:591` | **DONE** | `hardfilters.py:apply_hard_filters`, `normalize.py:normalize_pending` (hash check), `geo.py:prefilter` |
| Classified error envelope (closed enum + lossless details) | claude-code `error-envelope.md` | **DONE** | `models.py:ErrorClass`, `db.py:set_error`, all callers |
| Central `with_retry` + Retry-After + capped jitter | claude-code `retry-backoff.md` | **DONE** | `llm.py:with_retry`, `llm.py:http_get_json` |
| Pinned grader-tuple (model+digest+rubric_version+fewshot_hash per row) | ARE `pinned-judge-model.md` | **DONE** | `db.py:judge_score` schema, `judge.py:judge_pending` |
| Density-smoothed slop_score 0–100 (not binary) | aislop `slop-scoring.md` | **DONE** | `models.py:Card.slop_score`, `Card.from_llm_json:101-102` (coerces legacy bool) |
| Tri-state failure taxonomy (True/False/None=unjudgeable) | ARE `failure-taxonomy.md` | **DONE** | `calibration.py:cross_validate` excludes unjudgeable from denominator; `judge.py` parks invalid-JSON rows |
| FP-guard exoneration cascade (aislop pattern) | aislop `fp-guards.md` | **DONE** | `hardfilters.py:german_required` (EXON_LEAD/TRAIL, STRONG_REQ override); `dedup.py:_disqualified` |
| Few-shot prompt assembly in code (`<example>` blocks) | claude-code `tool-prompts-fewshot.md` | **DONE** | `judge.py:render_fewshot`, `build_fewshot`, `build_system_prompt` |
| Fold-level SEM, N=3 verdict stability | ARE `multi-run-variance.md` | **DONE** | `calibration.py:cross_validate` (runs=3, fold SEM, Pass^k verdict stability) |
| Canary closed-enum verdicts (dead_scraper/degraded/empty_market) | ARE `failure-taxonomy.md` | **DONE** | `models.py:CanaryVerdict`, `sources/jobspy_source.py:canary` |
| Content-hash short-circuit before LLM | ARE `hard-before-soft-judge.md` | **DONE** | `normalize.py:normalize_pending` → `db.card_by_hash` |
| memdir-style taste profile (file-per-fact, supersession) | claude-code `memdir-file-memory.md` | **PARTIAL** — `config/profile.yaml` is a single file, not file-per-fact; acceptable for v1 scale (1 user, ~15 facts). Revisit if rubric grows beyond ~30 facts. | `config/profile.yaml` |

---

## Correctly deferred items (not gaps)

These are absent by design — each has an explicit roadmap trigger:

| Item | Trigger for activation |
|---|---|
| setwise/pairwise judge (`llm-rankers`) | CV gate fails ×3 rubric iterations AND Vlad approves UX change |
| bge-m3 + LightGBM/SetFit learned ranker | Funnel stably >100 survivors/night AND ≥200 Vlad labels |
| GATE interview (LLM-driven elicitation) | Before first slate (can do any time, not yet scheduled) |
| ESCO/ISCO occupation mapping | After bge-m3 embeddings introduced (Could tier) |
| Binoculars perplexity slop engine | After local model pair validated on German job-ad domain |
| Best-worst 4-tuple annotation format | If 1–5 scale CV agreement stagnates below 75% after 2 rubric iterations |
| JobSpy PR #347 (Glassdoor) | If Glassdoor re-appears in search matrix |
| Xing scraper (PR #366) | After smoke-test passes on live XING corpus |

---

## Nightly scheduler

`deploy/com.schabasch.nightly.plist` — macOS launchd, runs `cli tick --tertiary` at 03:00.
`plutil -lint` verified valid. **Not loaded** — enable when Phase-3 gate begins:

```bash
# Enable
cp deploy/com.schabasch.nightly.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.schabasch.nightly.plist

# Disable
launchctl unload -w ~/Library/LaunchAgents/com.schabasch.nightly.plist

# Logs
tail -f /tmp/schabasch_nightly.log
tail -f /tmp/schabasch_nightly_err.log
```
