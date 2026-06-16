# Schabaschkascuhen — project rules

## Frozen contracts (promoted from project memory 2026-06-15)

`schabasch/INTERFACES.md` is the source of truth for every module's **public signatures**.
Private internals are free to change; public signatures are not.

The four data-contract files are frozen — marked "Готово и НЕ менять":
- `models.py` — Status / FilterReason / ErrorClass / CanaryVerdict enums, Card dataclass, dedup primitives
- `db.py` — SQLite schema + FSM helpers
- `llm.py` — OllamaClient + `with_retry` + `http_get_json`
- `config.py`

Rules:
- **Additive-only.** New features add **sidecar tables** (e.g. `vacancy_feature`, `triage_decision`,
  `*_cache`) and new modules — never alter frozen schemas or change a public signature.
- **Narrow exception.** A data-contract file may be edited **only for correctness/reliability**
  (e.g. `db.py` `PRAGMA busy_timeout`, `COALESCE` on label upsert, dedup regexes) — **never** to
  change a public signature. Tests must stay green (behavior-preservation gate).
- The SOTA patterns from `docs/LITERATURE_REVIEW.md` §5 are already baked into these contracts
  (hard-before-soft filter ladder, content-hash short-circuit before LLM, classified error envelope,
  pinned grader-tuple, density-smoothed slop_score, typed tri-state failure taxonomy, central retry
  wrapper) — reuse them, don't reinvent.

Full record + reasoning: project memory `interfaces-contract.md`.
