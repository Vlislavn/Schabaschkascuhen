# Schabaschkascuhen

Локальный ($0) ночной поиск работы мечты («шабашки») для Алины — рынок Heidelberg/Frankfurt.
Каждую ночь: сбор вакансий → гео/hard-фильтры → дедуп → **дешёвый ML-гейт** отсекает мусор →
LLM-нормализация в карточки → LLM-судья 1–5 по рубрике → **матчинг CV↔вакансия** (cross-encoder
`fit_score = 0.6·xenc + 0.4·LLM-покрытие` + eligibility-гейт, де-конфляция магнита фитом) →
агентный deep-search топ-вакансий → утренний slate из 10 со шкалой **💻🐀 «офисная мышь» →
💅💸 «шабашка»**. Каждый клик дообучает систему. Всё локально (ollama qwen3:8b + bge-m3 + bge-reranker).

Документы: [docs/USE_CASE.md](docs/USE_CASE.md) · [docs/ANNOTATION.md](docs/ANNOTATION.md) (как размечать) ·
[config/profile.example.yaml](config/profile.example.yaml) (шаблон профиля/рубрики/настроек → скопируй в `config/profile.yaml`).

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,v2]"                 # schabasch + bge-m3/LightGBM + тесты
#   (или быстрее, по lock-файлу:  uv sync --extra dev --extra v2)
pip install "git+https://github.com/Vlislavn/JobSpy.git"   # Indeed/LinkedIn (публичный форк)
ollama pull qwen3:8b                        # локальный LLM (сервер на :11434)
cp config/profile.example.yaml config/profile.yaml         # → впиши свой профиль (profile.summary)
# опционально (агентный deep-dive investigate/discover; без него всё работает — шаг мягко деградирует):
# pip install -e ~/code/work/KatherLab/prototype-internal-KL
```
> Первый запуск `tick`/`features` один раз скачает bge-m3 + bge-reranker (~1.5 ГБ) — это долго; дальше из кеша.

## Ежедневное использование

Одна команда, одна поверхность — веб-страница. Полный гайд: [ANNOTATION.md](ANNOTATION.md).
```bash
schabasch serve                            # http://127.0.0.1:8787
```
- **`/`** — утренний slate из ~10 (8 лучших + 2 wildcard), **только свежие** (last_seen ≤ `fresh_days`,
  14 дн.). Карточка: оценка-чип **💻🐀→💅💸** (градиент по 1–5) · «почему» · ⛔ eligibility / ⚠ разрыв ·
  дата («опубл.» либо «найдено N дн.», если борд не дал дату публикации) · **🎯 разбор навыков**
  (llm_cov % + ✓/◐/✗ по требованиям) · 🔎 verified (компания/зарплата/англ-команда + **детерминированная**
  проверка «открыта / закрыта / не проверён»). Кнопки 💻🐀/😎/💅💸/applied — каждый клик дообучает.
- **`/annotate`** — единственная поверхность разметки: очередь оценённых судьёй, но ещё не
  размеченных вакансий (те же карточки и кнопки 💻🐀/😎/💅💸; **без потолка свежести** — это бэклог).
  С неё система учится с нуля; размеченное исчезает из очереди. Раньше был xlsx-пакет
  (`build-bootstrap`/`labels-import`) — ретайрнут в пользу одной веб-поверхности.
- **`/eval`** — валидация матчинга против ТВОИХ реальных меток (живёт, обновляется по мере разметки):
  pairwise/NDCG/spearman для каждого сигнала. «Чистые» сигналы (fit_score/cross-encoder/покрытие/
  eligibility) не видят метки — честная оценка; судья/triage помечены «⚠ обучается на метках».
  Синтетический GOLD остаётся только в CLI (`python -m eval.match_eval`; `--real-labels` — по меткам).
- **`/gaps`** — какие навыки регулярно отсутствуют под ЖЕЛАННЫЕ вакансии (😎/💅💸/applied): агрегат
  по разбору требований (`llm_cov_reqs`), отсортирован по «чаще всего не хватает» — кандидаты в резюме/на обучение.

После ~30 меток включается ML-гейт (LightGBM); ~50–100 — калибруется судья (`schabasch cv`).

## Ночной цикл

```bash
schabasch tick               # canary→scrape→фильтры→dedup→features→triage→normalize→judge→investigate→slate
schabasch tick --german      # с немецкой матрицей запросов
schabasch funnel             # воронка + канарейки (мёртвый скрейпер vs пустой рынок)
```
Планировщик (macOS launchd, 03:00) — подставь свои пути (`__REPO_DIR__`/`__VENV_PYTHON__` из плиста) и установи:
```bash
sed -e "s|__REPO_DIR__|$PWD|" -e "s|__VENV_PYTHON__|$PWD/.venv/bin/python|" \
  deploy/com.schabasch.nightly.plist > ~/Library/LaunchAgents/com.schabasch.nightly.plist
launchctl load -w ~/Library/LaunchAgents/com.schabasch.nightly.plist
```

## Отдельные шаги (по желанию)

```bash
schabasch candidate "<описание или путь к CV>"   # извлечь профиль кандидата (для матчинга)
schabasch features          # bge-m3 эмбеддинги + аспектный матчинг CV↔вакансия
schabasch triage            # ML-гейт: must/should/could/drop (дешёвый отсев до LLM)
schabasch triage-train      # обучить LightGBM-гейт на метках (≥30 меток)
schabasch cv                # 5-fold agreement судья↔метки (gate запуска daily)
schabasch discover          # агентный поиск вне бордов (opt-in, нужен kl_agent_builder)
schabasch investigate       # deep-search топ-N (компания/зарплата + детерминированная проверка листинга)
schabasch rerank            # cross-encoder + LLM-покрытие → fit_score (de-conflate ранжирования slate)
schabasch gaps              # повторяющиеся пробелы навыков под желанные (😎/💅💸) вакансии
```

> **Матчинг (fit-gate):** реальный сигнал соответствия CV↔вакансия `fit_score = 0.6·cross-encoder
> + 0.4·LLM-покрытие требований` считает `rerank`. Ночной `tick` запускает `rerank` перед `slate`;
> если зовёшь `slate` отдельно — сначала `rerank`, иначе fit=0 и ранжирование откатится к судье.

## Тесты

```bash
.venv/bin/python -m pytest tests/ -q       # 245 тестов, без сети/ollama (LLM и bge-m3 мокаются)
```

## Архитектура

```
candidate.py  профиль кандидата (LLM-извлечение из CV + аспектные тексты)
features.py   bge-m3 + bge-reranker: fit_score = 0.6·cross-encoder + 0.4·LLM-покрытие требований (llm_cov)
aspects.py    сегментация вакансии (EN+DE) + аспектный скоринг
eligibility.py  жёсткие требования (образование/PhD/язык) → eligibility-гейт (де-конфляция фитом)
triage.py     LightGBM-гейт (cold-start = match_score), калибровка, drop-bucket до LLM
normalize.py  qwen3:8b → карточка (язык-реальность, гибрид, slop-score)
judge.py      qwen3:8b 1–5 по рубрике + few-shot из крайних меток (💻🐀=2 / 💅💸=5)
investigate.py  агент deep-search: компания/зарплата/культура + ДЕТЕРМИНИРОВАННАЯ проверка листинга
metrics.py / validation.py / gaps.py   метрики ранжирования · /eval (vs реальные метки) · /gaps (пробелы навыков)
slate.py / feedback_app.py   slate 8+2 + /annotate + /eval + /gaps + FastAPI-фидбек; шкала 💻🐀→💅💸
sources/      jobspy (Indeed/LinkedIn) · arbeitsagentur · agent_discovery (opt-in) · tertiary
pipeline.py / cli.py   ночной tick, импорт спайка, воронка, gaps
```

FSM статусов: `new → prefiltered | described → normalized → filtered | scored → slated → labeled | expired`.
Конфиг — `config/profile.yaml`. Заметки по агентам/токенам — в комментариях `schabasch/agent_runtime.py`.
