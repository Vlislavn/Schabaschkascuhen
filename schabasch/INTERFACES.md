# Контракты модулей (для параллельной сборки)

Готово и НЕ менять: `models.py`, `db.py`, `llm.py`, `config.py`, `config/profile.yaml`.
Каждый модуль реализует РОВНО эти публичные сигнатуры (приватное — как угодно).
Все модули принимают `cfg` (dict из `config.load()`) и `con` (sqlite3.Connection из `db.connect()`).

## sources/jobspy_source.py
```python
def scrape(cfg: dict, con, *, queries: list[str] | None = None,
           sources: list[str] | None = None, hours_old: int | None = None) -> dict[str, int]
    # Indeed+LinkedIn через ФОРК (он установлен в venv editable). На каждый (query, city):
    # scrape_jobs(...); LinkedIn — linkedin_fetch_description=True, 5s между запросами,
    # стоп источника после 2 сбоев подряд (паттерн spike/scripts/a1_runner.py).
    # Каждую строку -> db.upsert_vacancy (source, url, title, company, city,
    # is_remote_hint, description (у Indeed есть сразу), query_term, query_city).
    # Возвращает {source: inserted_count}. Пишет db.log_funnel('scrape', n, source).

def canary(cfg: dict, con) -> dict[str, str]
    # 1 канарный запрос на источник ('machine learning' @ Frankfurt, results_wanted=5).
    # min-row assertion: 0 строк без исключения => CanaryVerdict.DEAD_SCRAPER (кейс Google).
    # Пишет db.log_canary; возвращает {source: verdict}.
```

## sources/arbeitsagentur.py
```python
API = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
HEADERS = {"X-API-Key": "jobboerse-jobsuche"}

def search(cfg: dict, con, *, queries: list[str] | None = None) -> int
    # GET {API}/pc/v4/jobs?was=..&wo=..&umkreis=50&size=50 (+пагинация page=N),
    # фильтр свежести по aktualitaet/veroeffentlichtseit при наличии. Throttle 1 req/s.
    # upsert_vacancy(source='arbeitsagentur', refnr=..., url=f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{refnr}").
    # Возвращает число новых.

def fetch_details(cfg: dict, con, *, limit: int = 400) -> int
    # Для vacancy WHERE source='arbeitsagentur' AND status='new':
    # GET {API}/pc/v3/jobdetails/{base64(refnr)} @1 req/s, llm.http_get_json (404/410 -> EXPIRED).
    # title=stellenangebotsTitel, employer=firma, city=stellenlokationen[0].adresse.ort,
    # description=stellenangebotsBeschreibung, is_temp_agency=istArbeitnehmerUeberlassung (NaN->NULL).
    # Описание есть -> status DESCRIBED (db.upsert_vacancy сделает это сам при повторном upsert,
    # но проще прямой UPDATE description/desc_hash/is_temp_agency/status). Возвращает число описанных.
```

## geo.py
```python
def geo_check(city: str | None, cfg: dict) -> tuple[bool, float | None]
    # Embedded таблица ~80 городов/общин BW+Hessen+RLP {name: (lat, lon)} (+ нормализация
    # 'Frankfurt am Main, HE, DE' -> 'frankfurt am main'). haversine до ближайшего якоря.
    # Возврат: (в радиусе?, расстояние_км). Город неизвестен -> (True, None) — НЕ режем, пометка.
def prefilter(cfg: dict, con) -> dict[str, int]
    # Все NEW+DESCRIBED без card: geo_check; вне радиуса -> set_status(PREFILTERED, FilterReason.GEO).
    # Возвращает {'kept': n, 'prefiltered': m}. log_funnel.
```

## hardfilters.py  (hard-before-soft: ДО любого LLM)
```python
GERMAN_REQ = re.compile(r"(fließend|verhandlungssicher|Deutschkenntnisse|German.{0,40}(required|fluent|mandatory|C1|B2|native)|fluent in German)", re.I)
EXONERATION = re.compile(r"(nice to have|is a plus|von Vorteil|wünschenswert|a plus|not required)", re.I)
def german_required(description: str) -> bool
    # match GERMAN_REQ, но окно ±80 символов вокруг матча НЕ содержит EXONERATION.
def apply_hard_filters(cfg: dict, con) -> dict[str, int]
    # Для DESCRIBED: german_required(desc) -> FILTERED/LANGUAGE_DE;
    # is_temp_agency==1 -> FILTERED/TEMP_AGENCY. Возвращает счётчики по причинам. log_funnel.
```
Регресс-тест: spike/data/indeed.csv — 27 строк матчат сырой GERMAN_REQ; german_required
обязан поймать >=25 из них (см. tests/).

## normalize.py
```python
SYSTEM_PROMPT: str  # на базе spike/llm_pilot/run_pilot.py; «ровно 2 строки по-русски» — few-shot пример.
def normalize_pending(cfg: dict, con, *, budget: int | None = None) -> dict[str, int]
    # DESCRIBED -> карточка: сначала db.card_by_hash(desc_hash) (short-circuit репостов),
    # иначе OllamaClient(normalizer_model).chat_json; Card.from_llm_json валидирует;
    # set_status(NORMALIZED, card_json=...). LLMError -> db.set_error + остаётся DESCRIBED.
    # Затем фильтр по карточке: work_mode=='remote' -> FILTERED/REMOTE_ONLY;
    # language_reality=='de' -> FILTERED/LANGUAGE_DE; иначе остаётся NORMALIZED.
    # Возвращает {'normalized': n, 'cached': c, 'filtered': f, 'errors': e}. log_funnel.
```

## judge.py
```python
def build_fewshot(con, max_n: int) -> tuple[str, str]   # (fewshot_text, fewshot_hash) из крайних label (score<=2 и =5 — 💻🐀-якорь + 💅💸)
def judge_pending(cfg: dict, con) -> dict[str, int]
    # NORMALIZED -> оценка: rubric из cfg['profile'] (scale+magnets+repellents+summary),
    # few-shot блок <example>+<commentary>; ответ {"score":1-5,"why_tag":str|null,
    # "why_freetext":str|null,"explanation":str}; why_tag валидировать по models.WHY_TAGS (иначе null + текст во freetext).
    # insert_judge_score с ПОЛНЫМ grader-tuple (model, model_digest, rubric_version, fewshot_hash);
    # set_status(SCORED). Возвращает счётчики. log_funnel.
```

## slate.py
```python
def build_slate(cfg: dict, con, slate_date: str) -> list[dict]
    # FIT-LED: топ-8 по effective = fit_score·(1+β·judge_norm)·elig_score (β=slate.judge_blend_beta,
    #   default 0); tie-break: judge score, triage, integration, -slop. fit_score + eligibility
    #   recomputed LIVE (features.recompute_live) under current weights/gate (no model load).
    #   ≤3 на компанию; cross-account reposts collapsed (_collapse_reposts → also_at); +2 explore
    #   (far-but-in-DE preferred; seed=slate_date). INSERT slate_entry, set_status(SLATED).
    # Card keys: vacancy_id, rank, slot_type, title, company, city, url, score, why, explanation,
    #   summary, work_mode, date_posted, first_seen, investigation, fit_score, fit_note, llm_cov,
    #   llm_cov_reqs, elig_score, elig_note, elig_severity, far, dist_km, geo_anchor, also_at, user_note.
def _load_scored(con, rubric_version=None, *, max_age_days=None) -> list[dict]
    # latest judge score per vacancy (SCORED/SLATED, active rubric); max_age_days → freshness ceiling
    # on last_seen (daily slate passes slate.fresh_days; annotation_batch passes None = full backlog).
def _posted_ago(date_posted, first_seen=None) -> str  # "опубл. N дн" else "найдено N дн" (first_seen fallback)
def _score_badge(score) -> str   # themed chip 1 💻🐀 «офисная мышь» → 5 💅💸 «шабашка», grey→gold gradient
def _skills_html(e) -> str       # collapsible "🎯 Навыки {llm_cov%} · ✓/◐/✗" from feature_json.llm_cov_reqs
def render_html(slate: list[dict], slate_date: str, alerts: list[str] | None = None) -> str
    # Self-contained HTML; one _card_block (shared) + _page. Strong accent only on the score chip
    # (💻🐀→💅💸 gradient) + red ⛔. Buttons 💻🐀/😎/💅💸/applied -> POST /feedback; header = progress N/M.
def annotation_batch(cfg, con, slate_date) -> tuple[list[dict], int]
    # Очередь разметки = _load_scored(max_age_days=None) (SCORED/SLATED, LABELED отпадает), ≤ annotate_batch.
def render_annotate_html(items, slate_date, *, total_pending: int) -> str
    # Единственная поверхность разметки (xlsx ретайрнут): тот же _card_block, кнопки 💻🐀/😎/💅💸 (без applied).
def render_eval_html(report: dict) -> str
    # Дашборд валидации (validation.eval_report): headline = fit_score (чистый сигнал) + таблица
    # сигналов; «⚠ обучается на метках» на judge/effective/triage; баннер «нужно ~N пар» → /annotate.
def render_gaps_html(report: dict) -> str
    # Дашборд пробелов навыков (gaps.gap_report): повторяющиеся missing/partial требования по ЖЕЛАННЫМ.
```

## metrics.py
```python
# Ранжирующие метрики, общие для CLI eval и /eval. gold — всегда явный {vacancy_id: relevance}.
def pairwise_accuracy(scores, gold) -> tuple[float, int]   # % правильно упорядоченных пар (ties skip)
def ndcg_at_k(scores, gold, k=10) -> float
def spearman(scores, gold) -> float                        # scipy опционален → 0.0 если нет
def evaluate(scores, gold, *, name="") -> dict             # {name,pairwise_acc,ndcg@10,spearman,n,n_pairs}
def top_bottom(scores, gold, *, rationales=None, k=8) -> str

## validation.py
```python
def label_gold(con) -> dict[int, int]      # gold из РЕАЛЬНЫХ меток: score_1_5 (applied→5), max по источникам
def eval_report(cfg, con) -> dict
    # Метрики matcher-сигналов против реальных меток. Чистые (CV↔JD, без утечки): fit_score/xenc_full/
    # llm_cov/elig_score (clean=True). Текут на метках (clean=False): judge_only/effective/triage.
    # → {n_labels, n_comparable_pairs, min_pairs (cfg.slate.eval_min_pairs), reliable, headline, rows}.
```

## gaps.py
```python
def gap_report(cfg, con) -> dict
    # WANTED = label.score_1_5>=4 OR applied=1; aggregate feature_json.llm_cov_reqs across them →
    # per-requirement {missing, partial, present, jobs}, ranked by (missing + 0.5*partial) desc.
    # → {n_wanted, n_jobs_with_reqs, rows, reliable, candidate_skills}. Drives /gaps + CLI `gaps`.
```

## feedback_app.py
```python
def create_app(cfg: dict) -> fastapi.FastAPI
    # GET /         -> HTML сегодняшнего slate (build_slate если ещё нет, иначе из slate_entry)
    # GET /annotate -> HTML очереди разметки (annotation_batch + render_annotate_html)
    # GET /eval     -> HTML валидации против реальных меток (validation.eval_report + render_eval_html)
    # GET /gaps     -> HTML пробелов навыков по желанным вакансиям (gaps.gap_report + render_gaps_html)
    # POST /feedback {vacancy_id:int, action:'bad'|'good'|'star'|'applied', note?:str|null}
    #   -> models.FEEDBACK_TO_SCORE; applied => label.applied=1, сохраняя прежний score_1_5;
    #      note => label.why_freetext (COALESCE: пустой note не стирает прежний; → judge few-shot);
    #      db.insert_label(source='slate'); UPDATE slate_entry.feedback. Ответ {"ok":true} <=1 c.
    # GET /funnel   -> воронка + канарейки + dedup_candidates (parsed dedup_fuzzy) + dedup_count (json).
    # POST /fetch   -> async full pipeline (nightly_tick) on a background worker, SINGLE-FLIGHT
    #                  (409 if running — no double model-load). GET /fetch-status -> {running, stage, …}.
def serve(cfg: dict)  # uvicorn на cfg['slate']['port']
```

## pipeline.py
```python
def nightly_tick(cfg, con, *, german_queries=False, budget=None, tertiary=False) -> dict
    # Порядок: canary -> scrape(jobspy)+arbeitsagentur.search [+tertiary] -> fetch_details ->
    # expire_stale -> geo.prefilter -> hardfilters -> dedup -> features -> triage ->
    # normalize_pending(budget) -> judge_pending -> rerank_scored -> investigate_top(top_n) ->
    # build_slate(today). Каждый шаг в try/except — сбой шага не валит tick. Возвращает воронку.
def import_spike_data(cfg: dict, con) -> dict[str, int]
    # Разовый импорт уже собранного: spike/data/indeed.csv, linkedin_described.csv,
    # arbeitsagentur_details.csv -> upsert_vacancy (с описаниями => DESCRIBED).
# RETIRED: import_bootstrap_labels / normalize_ids / bootstrap.py — xlsx-разметка заменена
# единой веб-поверхностью /annotate (slate.annotation_batch + render_annotate_html).
```

## cli.py  (typer)
```
schabasch import-spike | tick [--german] [--budget N] | scrape | details | normalize | judge |
slate | serve | canary | export-golden | funnel | rerank | features | triage | cv | gaps | investigate
```
Разметка теперь в вебе: `serve` -> `/annotate` (очередь) · `/` (дневной slate) · `/eval` · `/gaps`;
xlsx-команды `build-bootstrap` / `labels-import` ретайрнуты.

## config/profile.yaml (контрактные ключи)
```
slate.fresh_days: 14          # daily slate: only last_seen ≤ N days (annotate keeps backlog)
slate.investigate_top_n: 6    # deep-search runs on top-N SCORED each tick
slate.quality_floor: 0.45     # an exploit slot must clear this `effective` (drops unsuitable; below-
                              #   floor stays explore-eligible). Measured: drops 11 label-2, 0 high-rated.
slate.judge_blend_beta: 0.0   # effective = fit_score·(1+β·judge_norm)·elig_score (FIT-LED; β=0 best on
                              #   real labels — magnet judge near-random; it tie-breaks instead). REAL-tuned.
slate.fit_gate_floor: 0.25    # LEGACY (old judge-led gate); unused by the fit-led _effective.
slate.annotate_batch: 30      # /annotate page size
slate.eval_min_pairs: 15      # /eval reliability banner threshold
features.fit_weights: {hyre: 0.7, sparse: 0.3}    # fit_score = HyRE + bge-m3 SPARSE hybrid (REAL-label
                              #   tuned; beats HyRE-only on all 3 metrics; sparse is deterministic). _blend_fit.
features.sparse_norm: 0.45    # fixed divisor mapping bge-m3 sparse ≈[0,1] (per-vacancy, no min-max)
features.hybrid_sparse: true  # compute the bge-m3 sparse-lexical signal (bgem3_sparse) in rerank_scored
eligibility.floor: 0.35 / .mid: 0.6               # hard-qualification gate multipliers
eligibility.soft_lift_threshold: 0.55            # SOFT (prose-degree) gap lifted to 1.0 when fit ≥ this
```
features._llm_coverage → (coverage, missing, requirements[]); persists llmcov_cache.requirements.
features.fit_from_feature(feat, weights) → live fit_score; features.recompute_live(con, vid, cfg,
  cand_quals) → {fit_score, fit_note, elig_score, elig_note, elig_severity} from caches (no model).
eligibility.eligibility_gate(req, cand, *, floor, mid, fit_score=None, soft_lift_threshold) →
  (mult, reason, severity) where severity ∈ {"structural"(red ⛔), "soft"(amber, lifted by high fit)}.
eligibility.req_from_cache(con, content_hash, jd_text) → cached req + guards (no LLM); None on miss.
geo.geo_class(city, cfg) → "near"|"far"|"unknown"; geo.geo_mark(city, cfg) → {far, dist_km, anchor}.
  prefilter no longer drops far-DE (returns near/far/unknown counts; nothing PREFILTERED).
features.bgem3_sparse_scores(model, cv, jds) → bge-m3 sparse-lexical scores via native compute_score
  (FlagEmbedding m3.py:686-699); used by rerank_scored (batched) + scripts/backfill_bgem3_sparse.py.
feature_json keys: fit_score, xenc_full, fit_hyre, bgem3_sparse, llm_cov, llm_cov_missing, llm_cov_reqs,
  elig_score, elig_note, elig_severity, requirements_verified, company_known, german_rooted.
features.feature_vector: ALWAYS [dense(1024, zero-padded if no embedding) ++ named] — fixed width so
  triage train (_load_labeled) and score share a shape (a 1053-vs-29 mix crashed np.stack).
investigate.validate_company(name, agent_enrichment, cache) → {company_description, german_rooted,
  company_verified, validation_source, wiki_url} — keyless Wikipedia (de→en, UA + org-guard) + German
  legal-suffix; cross-checks the agent. investigation enrichment adds those fields; patches feature
  company_known (=verified) + german_rooted.
slate.build_slate item keys add: far, dist_km, geo_anchor, also_at, elig_severity, user_note.
entry: `python -m schabasch.cli ...` (и console_script `schabasch`).

## tests/ (pytest, без сети и без ollama — мокать llm.OllamaClient.chat_json)
- test_hardfilters.py: 27-строчный регресс-сет из spike/data/indeed.csv (>=25 пойманы), exoneration-кейсы.
- test_dedup_models.py: normalize_company/title, dedup_key, content_hash.
- test_db_fsm.py: upsert идемпотентен; статусные переходы; insert_label conflict-update.
- test_geo.py: Stuttgart (~85 км) режется, Mannheim (~20 км от HD) — нет, неизвестный город — не режется.
- test_slate.py: 8+2, ≤3/компанию, детерминизм по seed=date (мок-данные в sqlite :memory:).
```

## Additive modules / endpoints / config (2026-06-16) — frozen files untouched
```python
# schabasch/llm_clients.py — model cascade (OpenAI-compatible alongside the frozen OllamaClient)
class OpenAIClient: chat_json(system, user) -> dict   # mlx_lm.server / api.kather.ai; reasoning_content fallback
make_llm_client(cfg, role) -> OllamaClient | OpenAIClient    # role ∈ normalizer|judge|candidate|agent|deep_reasoning|sota
role_available(cfg, role) -> bool          # openai roles need a key (localhost assumed up); skips keyless sota
agent_client_params(cfg) -> {provider, base_url, model, api_key}   # for agent_runtime (role 'agent')

# schabasch/memory_guard.py — vendored from IVAI (stdlib; macOS memory_pressure -Q)
require_headroom(context)  # raises MemoryHeadroomError below hard floor; start_watchdog(); memory_under_pressure(probe=)
configure_from_cfg(cfg)    # maps cfg['memory'].{hard_floor_pct,soft_floor_pct,watchdog_interval_seconds} → env

# schabasch/role_kind.py — deterministic role classifier (engineer/junior down-rank + flags)
classify(title, summary=None) -> "hands_on_engineer"|"junior"|"lead"|""   # multiplier(kind, cfg); flag(kind)

# schabasch/tasks.py — review-comment tracker (sidecar session_comment_task)
ingest_from_db(con, *, extra=None) -> dict   # idempotent (NULL-safe); theme_for(text); upsert_task; set_status; all_tasks; summary

# schabasch/enrichment.py — Zotero-style card (sidecar vacancy_enrichment, keyed by content_hash)
enrich_slate(cfg, con, *, slate_date, top_n=None) -> dict   # extractive (bge-reranker) + abstractive (cascade)
enrichments(con) -> {vid: {key_snippets, pros, cons, company_review, clean_summary, model_used}}

# feedback_app.py endpoints (additive): GET /tasks, POST /task-status {task_id,status}; /fetch-status enriched
#   with {stage_human, heavy, stage_index, n_stages, stages[]}.
# slate.py: render_tasks_html(tasks, summary); _card_block adds the «📄 Глубокий обзор» enrich block + role flag.
# pipeline.nightly_tick: heavy stages wrapped in _heavy_step (memory-gated); new final 'enrich' step.
```
```yaml
llm.roles.{normalizer,judge,candidate,agent,deep_reasoning,sota}: {client, model, base_url, api_key_env, ...}
llm.env_file: <path to .env>          # source of OPENAI_API_KEY for the sota tier (never committed)
deep.{enable, slop_escalate_thr, enable_sota, enrich_top_n}   # W3 abstractive enrichment knobs
memory.{guard_enabled, hard_floor_pct, soft_floor_pct, watchdog_interval_seconds}   # Apple-silicon swap safety
slate.role_kind_mult: {hands_on_engineer: 0.7, junior: 0.5, ...}   # role-kind soft down-rank (real-label tuned)
```
Sidecars added (additive, db.py schema frozen): `session_comment_task`, `vacancy_enrichment`.

## UX redesign (2026-06-16, laws-of-UX gated) — internal to slate.py render, no public-signature change
- `slate.py`: new `_legend_html()` (collapsed emoji legend); `_card_block` note moved behind a
  `toggleNote(id)` JS toggle AFTER the buttons; chips are `<button>` not `<span>`; meta split into
  `.meta`+`.meta2`; score badge `role=img aria-label`; emoji buttons get `aria-label`; `_CSS` gained
  `@media (max-width:480px)` (tables→card-list via `td[data-label]`+`<thead>`), `.sr-only`,
  `:focus-visible`, AA contrast tokens. `render_annotate_html` no longer emits the filter chips.
- `config/profile.yaml`: `llm.roles.deep_reasoning` DEFAULT is now `{client: ollama, model: qwen3.5:4b}`
  (measured local enrichment win); the 35B MLX is an opt-in swap. `agent` stays qwen3:8b.
- `agent_runtime._AGENT_DEFAULTS` gained `strong_max_turns`/`strong_max_tool_calls`/
  `strong_max_wall_time_seconds` for the non-qwen agent path (kl give-up guard fires at 8).
