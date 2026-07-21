# SOTA résumé↔JD matching — research + adoption plan

**Problem (2026-06-14):** the slate surfaces magnet-domain jobs (space/defense) the candidate
can't actually do. Root cause is two-part:
1. **Coverage is a blob cosine.** `aspects.py: cov_musthave_maxsim = cosine(CV-skills-vector, JD-must-have-vector)` — a single cosine between two *aggregated* documents. Any two professional docs cosine ≈ 0.4–0.6, so it never detects "the user lacks the required skills." The only per-skill signal (`n_musthave_missing`) is crude keyword matching, penalized just −0.025 each; ColBERT coverage is never wired (0.0).
2. **The magnet judge dominates ranking.** Slate sorts on `judge_score + 0.5·triage + 0.3·xenc`; the qwen3:8b judge scores by domain/magnet (space=5) and overrides any fit signal.

A 7-angle SOTA research sweep (41 repos, 33 techniques, adversarially verified) **converged unanimously**: stop ranking on blob-cosine + domain-judge; rank on **real per-requirement skill coverage** with the **qualification as a hard gate applied before** the magnet judge.

## Verified tools (real page fetches, June 2026)

| Tool | What | Evidence | Verdict |
|---|---|---|---|
| **bge-reranker-v2-m3** (already cached; `features._DirectReranker`) | cross-encoder fit CV↔must-haves | local, working | **use now** — restores discrimination, no training |
| **HyRE** (ConFit-v2, arXiv:2502.12361, ACL 2025) | LLM writes the *ideal résumé* per JD; compare the user's CV vs that (same genre → discriminative). Paper: +13.8% recall / +17.5% nDCG over ConFit | repo `jasonyux/ConFit-v2` **10★, MIT, HyRE code in `scripts/hyre/`, NO checkpoint** | **borrow the HyRE pattern** (reimplement w/ qwen3 + bge-m3) |
| **CareerBERT** (arXiv:2503.02056, ESWA 2025) | fine-tuned SBERT, résumé↔ESCO shared embedding space; **ships a pretrained checkpoint** `lwolfrum2/careerbert-jg` on HF | repo `julianrosenberger/careerbert`, open-science | **evaluate/vendor** — the one SOTA matcher with a *downloadable model* |
| `esco-skill-extractor` (pip) | turnkey ESCO skill-set extraction (all-MiniLM-L6-v2 → ESCO URIs) | `KonstantinosPetrakis/...` **25★, MIT, CPU, PyPI 0.1.18** | **vendor/dep** — fills the empty `esco.py` slot |
| jjzha `jobbert_skill_extraction` + `jobbert_knowledge_extraction` (HF) | BERT NER (SkillSpan, 14.5K sent); **knowledge spans = hard reqs** (degree/orbital-mechanics tells) | ~0.1B params (CPU), ~2.9k dl/mo | borrow — granular extractor |
| nestauk `ojd_daps_skills` | ESCO skill extraction, maps **unseen** skills (React→ESCO) | Nesta, real | alt extractor |
| ConFit-v3 "non-negotiable requirements" judge prompt | per-requirement qualification check before any domain bonus | technique | **borrow prompt** |
| MoritzLaurer `deberta-v3-base-zeroshot` / PyLate `answerai-colbert-small` | NLI entailment / ColBERT MaxSim per requirement | HF, CPU | optional signals |
| srbhr/Resume-Matcher | **27.4k★** but confirmed keyword+embedding similarity + UI; not a fit engine | fetched README | **do NOT adopt for matching** |

**Architectural source (de-conflation):** "De-conflating Preference and Qualification" / **JobRec**, arXiv:2602.03097 (USC). Quantifies the user's exact bug: training on *preference only* gives preference R@5=0.827 but **qualification R@5 collapses to 0.233**; separating the two recovers Qual R@5=0.767. → our magnet judge models "would the user apply", not "can the user get/do it"; the two must be separated. Code: `github.com/brycekan123/DualOptimization_jobrec`.

ConFit ships **no checkpoints** → reimplement HyRE + the gate prompt. CareerBERT DOES ship a checkpoint, so it's the one matcher worth trying off-the-shelf.

## ❌ REJECTED — blending llm_cov into the slate rank (measured 2026-06-16)

User flagged a VINFAST "ML Optimization Engineer" (Skills 12%, master-required) at slate #1 for a
Business Analyst. Hypothesis: rank ignores honest coverage — blend `llm_cov` into the effective score
(`fit_eff = (1-λ)·fit + λ·llm_cov`) + gate the degree soft-lift on `llm_cov` + harder engineer down-rank.

**Measured on the 50 REAL labels** (`validation.eval_report`, effective pairwise/ndcg@10):

| config | pairwise | ndcg@10 |
|---|---|---|
| baseline (λ=0, eng 0.7) | **0.862** | 0.571 |
| λ=0.3 cov_min .3 eng .5 | 0.787 | 0.412 |
| λ=0.4 cov_min .3 eng .5 | 0.776 | 0.413 |
| cov-gate only (λ=0, cov_min .3, eng .5) | 0.808 | 0.538 |
| engineer 0.7→0.3 (λ=0) | 0.860 | 0.571 |

Every cov-blend variant **regressed** pairwise by 0.05–0.10. **Root cause: the labels reward the
aspiration jobs** — VINFAST's `user_label = 4` (the user rated it "really cool, but an engineer"). The
ranking is faithfully obeying the label; no formula fixes the complaint without contradicting the data.
**Shipped: knobs OFF** (`slate.cov_weight=0`, `eligibility.soft_lift_cov_min=0`). The real fix was
**de-dup** (a rated job is excluded → VINFAST drops off the slate) + re-labelling aspiration jobs.
Side win: the shared `slate.effective_score` now includes role-kind (the old eval omitted it) → honest
baseline rose 0.819→**0.862** pairwise.

## ⚠️ CORRECTION — re-tuned on 37 REAL labels (2026-06-15 PM)

The Tier-1 notes below were tuned on a **synthetic Opus gold**. Once the user hand-labelled 37 real jobs
(`label` table; `15JuneSession.md`), benchmarking on the user's ACTUAL clicks **inverted the synthetic
ranking** and **falsified the "HyRE flat ~0.8, excluded" claim** (`python -m eval.match_eval
--real-labels`, n=37, 188 pairs):

| signal (REAL labels) | pairwise | ndcg@10 | spearman | synthetic said |
|---|---|---|---|---|
| **fit_hyre** | **0.803 ★** | **0.539** | 0.408 | "flat, exclude" — WRONG |
| 0.8·hyre+0.2·xenc (shipped) | 0.793 | 0.483 | 0.392 | — |
| xenc_full | 0.739 | 0.409 | 0.338 | weak alone (synth 0.587) |
| llm_cov | 0.649 | 0.324 | 0.212 | strongest single — WRONG (weakest on real) |
| OLD fit_score (xenc .6/cov .4) | 0.644 | 0.276 | 0.201 | NDCG 0.96 on synthetic |
| OLD effective (judge-led) | 0.564 ✗ | 0.247 | 0.084 | — (≈ random on real) |

**Three measured changes shipped:**
1. **Headline fit is now HyRE-led:** `fit_weights = {hyre: 0.8, xenc: 0.2}` (`features._blend_fit`).
   HyRE (qwen-written ideal résumé, even on *raw* bge-m3) is the single best predictor of the user's real
   clicks; `llm_cov` is DROPPED from the headline (every blend with it fell to ~0.68) but still
   computed + shown as the per-requirement card breakdown. NOTED-REJECTED: pure HyRE measures ~0.01
   higher (0.803) but loses the deterministic xenc stability anchor against HyRE's qwen run-variance.
2. **Effective is FIT-LED, not judge-led** (`slate._effective`): `fit · (1 + β·judge_norm) · elig`,
   `β = slate.judge_blend_beta`, **default 0** — the magnet judge is near-random on the user's labels
   (judge_only 0.572), so on this set β=0 measures best; the magnet instead DIFFERENTIATES
   comparable-fit jobs as the tie-break + drives explore + the card emoji. Raise β when 4/5 labels accrue.
3. **The eligibility gate was net-NEGATIVE on real labels** (`fit·elig` collapsed HyRE 0.803→0.59):
   its false positives — the "Master Data"≠degree bug (the user's #1 job 439) and a prose-degree minimum the user would
   ignore (SCHOTT 1164) — demoted jobs the user likes. Fixed by the Master-Data negative-context guard +
   a high-fit soft-lift (`eligibility.py`); elig then stops hurting (HyRE back to 0.803 under `fit·elig`).

**Result (production `effective`, mirrored in `validation.eval_report` + `eval/match_eval`):**
pairwise **0.564 → 0.793**, ndcg@10 **0.247 → 0.483** — beats judge_only (0.572), the old effective
(0.564) and the old fit_score (0.644), approaching the best single signal (0.803). Lead number is the
honest real-label one; synthetic GOLD remains a CLI-only regression floor. **Card to update:** the
DualOptimization_jobrec de-conflate card — add that the **preference head can be LESS predictive than
the qualification head for a given user**, so *fit must be able to lead* (Ref: this repo
`schabasch/slate.py` `_effective`, `eval/experiment.py --real-labels`).

## ✅ REAL SOTA HYBRID SHIPPED (2026-06-16) — dense-HyRE + bge-m3 sparse

The deferred hybrid is now shipped + measured (`eval/hybrid_measure.py`, one foreground bge-m3 run via
the native `compute_score(weights_for_different_modes)` — FlagEmbedding `.../encoder_only/m3.py:686-699`).
bge-m3 is a native dense+sparse+ColBERT model; the matcher was dense-only. **Full-doc SPARSE-lexical
fused with HyRE beats HyRE-only on all three real-label metrics:** `0.7·hyre + 0.3·(sparse/0.45)` →
fit `0.803→0.814` pairwise / `0.539→0.584` ndcg@10 / `0.408→0.427` spearman; production `effective
0.793→0.819 / 0.483→0.584 ndcg`. Sparse is **deterministic** (no qwen run-variance — the anchor `xenc≈0`
failed to give). **ColBERT measured & did NOT help** (every colbert mode ≤ sparse); the must-have-segment
pairing lost to whole-document. Shipped via `features.bgem3_sparse_scores` (in `rerank_scored`, reusing
the loaded model) + `fit_weights {hyre:0.7, sparse:0.3}` + `sparse_norm 0.45` (fixed per-vacancy divisor,
no set-relative min-max). Card: `sota-pattern-index/.../bge-m3/native-hybrid-fusion.md`. Lesson: a
*model-deferred* "no signal helps" (the earlier `hybrid_probe.py`) is NOT a negative — run the model first.

## Per-feature ablation — does each feature earn its place? (`eval/feature_ablation.py`, 2026-06-16)

**Run:** `python -m eval.feature_ablation --real-labels`  (model-free — reads cached `vacancy_feature.feature_json`
vs the `label` table; no LLM/bge-m3/35B; ~1.3s). Optional `--boot=N` (bootstrap resamples, default 500).

Three modes, all with bootstrap 95% CIs over the real labels:
- **MODE 1 — standalone ranking power:** each feature alone as a ranker (oriented by spearman sign) → pairwise/ndcg/spearman + CI. "Has *any* label signal?"
- **MODE 2 — leave-one-out of the production fit blend** (`{hyre:0.7, sparse:0.3}`): drop each component, recompute `features.fit_from_feature`, report Δpairwise → what the matcher *relies on*.
- **MODE 3 — add-one-in, 5-fold HELD-OUT (same-subset Δ):** blend `α·fit + (1−α)·feature`, α picked on train folds, eval on the held-out fold, baseline computed on the *same* id-subset/folds → does the feature ADD orthogonal signal? **This is the earns-its-place test.**

**Decision rule:** a feature earns a place in the matcher only if its **MODE-3 held-out Δ > 0**. Standalone power (MODE 1) ≠ earns-its-place — e.g. `title_log_len` *tops* MODE 1 (0.775) yet **hurts in-blend (−0.043)**, a textbook small-n spurious correlation.

**First run (n=49, 2026-06-16):** MODE 2 → production correctly leans on both `hyre` (Δ+0.084) and `sparse` (Δ+0.059) — no dead weight. MODE 3 → **almost nothing adds beyond fit**; the LLM agent booleans (`requirements_verified` −0.03, `company_known` −0.09) *hurt* as ranking features (they belong as gates/display, where they live). The two nominal "ADDS" (`is_remote_hint` +0.074, `has_structure` +0.059) are binary metadata on ~8-item held-out folds → small-n-fragile, not ship signals. **Binding caveat: n=49 → CIs wide (random ≈0.61 [0.44,0.80]); |Δ|<~0.05 is noise.** Model-based permutation importance on the LightGBM gate is the richer ablation but is deferred until ≥75–100 labels (the gate is cold-start at n=50).

## Adoption plan (ROI order, all local/$0)

> **✅ Tier 1 IMPLEMENTED & live-verified (2026-06-15).** `features.rerank_scored` now also computes
> **HyRE** (`fit_hyre`, cached in `hyre_cache` per JD) + a blended `fit_score` over the SCORED+SLATED
> pool; `slate.build_slate` de-conflates via a multiplicative **fit-gate** (`(judge+λ·triage)·gate`,
> floor 0.5) + a ⚠ skill-gap note. Live result: a judge-5 *space* job the user can't do (Thorn "BDM –
> Space") sank to explore; ML/engineering jobs the user can do lead the slate. 207 tests green.
> **Empirical finding:** the **cross-encoder (bge-reranker) is the genuine fit signal** (xenc 0.02–0.93,
> discriminates correctly). Raw-bge-m3 **HyRE cosine is flat (~0.8)** without a fine-tuned encoder
> (ConFit fine-tuned theirs) → weighted low (`fit_weights xenc:0.7/hyre:0.15/cov:0.15`); HyRE becomes
> useful after the Tier-3 contrastive fine-tune. Cards harvested: ConFit-v2/hyre, DualOptimization/deconflate.

**Tier 1 — quick wins, no training:**
- Replace `cov_musthave_maxsim` blob cosine with the **cached cross-encoder** (`features.rerank` → CV vs must-have section) as the headline fit signal.
- **HyRE**: qwen3:8b synthesizes an "ideal résumé" per JD → embed (bge-m3) → cosine the user's CV vs that. Add as a strong `fit_hyre` feature.
- **De-conflate ranking** (`slate.py`): the magnet judge orders, but a **low-fit DOWN-RANK** (decided 2026-06-14, not a hard-drop) pushes unqualified-but-interesting jobs to the bottom / explore slots with a visible **"⚠ big skill gap"** note — never hidden (recall > precision, since skill extraction is imperfect). i.e. fit becomes a strong term that can override the magnet, but a magnet job is shown-with-warning, not deleted.

> **Decision (2026-06-14):** gate = **down-rank, not hard-drop**.
> **IMPLEMENTED & live-verified (2026-06-15):** `fit_score = 0.6·cross-encoder(xenc_full) +
> 0.4·LLM-coverage(llm_cov)`; eligibility gate; de-conflate `fit_gate_floor = 0.25` (fit-dominant,
> magnet differentiates do-able jobs). Profile re-grounded on the user's REAL **Senior Business Analyst**
> CV (was wrongly "ML engineer"). Validated: fit_score NDCG@10 0.96 / pairwise 0.87 vs Opus gold.
> HyRE kept as a stored feature only (flat ~0.8 on raw bge-m3). ESCO set-coverage tested & rejected.

**Tier 2 — real skill coverage:**
- Build `esco.py` on `esco-skill-extractor` (or jjzha NER): extract discrete ESCO skill sets from CV + JD must-have → **asymmetric coverage** `|CV ∩ required| / |required|` + an explicit **missing-required-skills list** (surfaced on the card). Replaces `cov_musthave_maxsim` math in `aspects.py`. Improve `candidate.py` to extract granular skills from the user's real CV (currently 7 coarse skills).
- Per-requirement **structured LLM judge** (qwen3:8b strict JSON: present/partial/missing + CV-evidence quote) as the gate.

**Tier 3 — heavier:** ConFit-style contrastive fine-tune of bge-m3 with hard negatives (reference-only).

**Sources:** github.com/jasonyux/ConFit-v2 · github.com/KonstantinosPetrakis/esco-skill-extractor · huggingface.co/jjzha/jobbert_skill_extraction · github.com/techwolf-ai/workrb · github.com/nestauk/ojd_daps_skills · arXiv:2505.24640 (ConTeXT), ConFit-v2 (HyRE), "De-conflating Preference and Qualification".

---

## Role-kind soft down-rank — measured on 50 REAL labels (2026-06-16)

The user's review comments are emphatic & repeated: "no engineers / I don't want to work with my hands, I want to work with my head"
(≥4 jobs) + intern/working-student rejections. A deterministic title classifier (`schabasch/role_kind.py`)
adds a multiplicative down-rank to `slate._effective` (`× role_kind_mult`), **never a hard drop** (recall-first):

| role_kind          | n  | mean label | multiplier |
|--------------------|----|-----------|-----------|
| hands_on_engineer  | 22 | **2.09**  | 0.7       |
| junior (intern/WS) | 3  | **2.00**  | 0.5       |
| lead (principal)   | 2  | 3.50      | 1.0 (the user LIKES lead) |
| neutral (BA/PM/owner) | 23 | **3.00** | 1.0       |

The down-rank tracks the user's ratings cleanly (engineer/junior ≈2.0 vs neutral ≈3.0). Only **1 of 25**
down-ranked jobs is high-rated — the VINFAST "ML Optimization Engineer" (4) the user flagged "really cool,
but an engineer"; it's softly demoted (×0.7), **still explore-eligible**, not hidden. A `lead`/`principal`
engineering role is head-not-hands work the user likes → NOT penalized. Cards show a quiet `🛠 hands-on` / `🎓 intern` flag.

### Re-tune on the "мусорные инженеры" complaint — 100 REAL labels (2026-06-22)

A fresh run still surfaced too many engineer vacancies. Two root causes, both config-only:

1. **Ingestion (primary).** `search.queries_en/de` were ~70% engineer titles (`machine learning
   engineer`, `software engineer`, `systems engineer`, …) fed straight to the scrapers
   (`pipeline.py:162`) → the fetched pool was engineer-dominated, and a slate down-rank only
   RE-ORDERS the pool. Retargeted the queries to her employable roles (business/data/BI analyst,
   project/program/process/product manager, IT consultant) + the strongest aspiration domains
   (`aerospace project manager`, `defense program manager`, `public sector` / `Raumfahrt
   Projektmanager`). **No bare engineer/Ingenieur/Entwickler title remains.**

2. **Ranking (safety net).** `role_kind_mult.hands_on_engineer` tightened **0.7 → 0.45** (config
   override; `_DEFAULT_MULT` unchanged). On the 100 real labels the engineer cohort is **30× rated-2
   garbage + 3× rated-4 gems** (none 1/3/5). The mult is **rank-NEUTRAL** on the eval guardrail —
   `effective` pairwise **0.804** / ndcg@10 **0.44** / spearman **0.541** are identical from 0.7 down
   to 0.35 (so no regression). Its real effect is **exploit-slot composition**: at 0.45 **all 33**
   labeled engineers fall below `quality_floor` and leave the exploit slots (vs **21/33** at 0.7),
   while the 3 gems (VINFAST ML / Ground Segment / Human Factors) stay **explore-eligible** — the
   user's «давить, но не прятать». `quality_floor` left at 0.45 (at mult 0.45 the floor is moot for
   engineers — all already below it). The 2 non-engineer ≥4 jobs below the floor are pre-existing
   low-FIT misses, unaffected by this change.

### The 2 non-engineer rated-4 misses — null-vs-zero HyRE bug + floor re-tune (2026-06-22)

The two ≥4 jobs flagged above (3321 "Project Manager Budget to Report", 3069 "Associate Director Data
Science") scored fit ≈ 0.12 / 0.15. **Root cause: a null-masquerading-as-zero.** `extract_features` writes
`fit_hyre = 0.0` (it's in `FEATURE_NAMES`) before `rerank_scored` runs, and `rerank_scored` only
*overwrites* `fit_hyre` when HyRE actually generates. The 16 LABELED gold jobs never enter the
SCORED/SLATED rerank pool (only the sparse backfill touched them) → they kept the `0.0` default. Since
HyRE carries the heaviest weight (0.7), that phantom zero dragged the headline fit to ~0. A real HyRE
score is `_cosine01 = (cos+1)/2 ∈ (0,1]` — **exactly 0.0 only on a degenerate/missing vector**.

**Fix (generalizable, no model load):** `features.fit_from_feature` now treats `fit_hyre == 0.0` as
ABSENT → the blend renormalizes over the present signals (sparse). Measured on the 100 real labels
(`validation.eval_report`): **effective pairwise 0.804→0.814, spearman 0.540→0.558, ndcg@10 0.44 flat**
(no regression). Rescues 3069 (fit 0.15→0.50, clears the floor on its own sparse signal).

**❌ REJECTED — backfilling the missing HyRE.** Simulated (assign the median 0.797 to the 16 jobs):
effective **regressed 0.804→0.787, ndcg 0.44→0.358**. Raw-bge-m3 HyRE is a near-constant (`n=83`: median
0.797, std **0.019**, spearman-vs-gold **0.099**) → a backfill just adds a uniform +0.56 to all 16 jobs,
inflating the 14 gold-2 jobs above better ones. HyRE is mostly an additive bias; the discriminative
content of `fit_score` is the 0.3-weight sparse term. So "backfill the missing signal" is the *wrong* fix.

**Floor re-tune `quality_floor` 0.45→0.37 (config).** 3321's honest sparse-only fit is 0.39 (a finance/
PM role, lexically distant from a BA/Python CV) — below the 0.45 floor. The floor does **not** enter the
eval `effective` metric (it only partitions exploit/explore), so lowering it cannot regress the
guardrail. Engineers cap at effective **0.343** (×0.45 role-mult), so any floor in (0.343, 0.45] excludes
all 33 equally — the role-mult, not the floor, keeps engineers out. 0.45 was over-tight: in the live pool
only **3** jobs cleared it (exploit starved 3/8). At 0.37, **9** clear (0 engineers), seating both rated-4
jobs with margin above the engineer cap. Below-floor jobs stay explore-eligible (recall-first).

## Model cascade for hard reasoning (2026-06-16)

`schabasch/llm_clients.py` routes per role: Tier-0 ollama qwen3:8b (bulk, unchanged) → Tier-1 local
Qwen3.6-35B MLX (`:8082`) → Tier-2 remote OpenAI-compatible `sota`. The Zotero-style card enrichment
(`schabasch/enrichment.py`) uses the cascade for pros/cons + deep company + a **clean re-parse of muddy
"AI-slop" ads** (escalates to `sota` when `slop_score ≥ deep.slop_escalate_thr`). Extractive snippets via
bge-reranker are deterministic (always available); abstractive degrades gracefully. Verified live: the remote
`sota` re-parsed the MAM IT-Controlling JD into "a financial-controller role in IT, not a business analyst".
