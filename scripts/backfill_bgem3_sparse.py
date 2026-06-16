"""Backfill the bge-m3 SPARSE-lexical hybrid signal (`bgem3_sparse`) into feature_json for the
eval gold + slate pool, then recompute the stored fit_score under the new HyRE+sparse blend.

Why: the SOTA hybrid (features.bgem3_sparse_scores) is computed going forward in rerank_scored, but
that only touches SCORED/SLATED vacancies — the LABELED gold (37) needs it too so eval/match_eval +
the /eval page reflect production. This one-shot fills LABELED ∪ SCORED ∪ SLATED with a SINGLE
foreground bge-m3 load (idempotent — re-running recomputes the same scores). Back up the DB first.

Run (foreground, swap-gated): .venv/bin/python -m scripts.backfill_bgem3_sparse
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from schabasch import config, db, features as _features
from schabasch.candidate import candidate_doc
from schabasch.models import Status


def main():
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    _features._ensure_schema(con)
    f_cfg = cfg.get("features", {})
    fit_w = f_cfg.get("fit_weights", {"hyre": 0.7, "sparse": 0.3})
    scale = float(f_cfg.get("sparse_norm", _features._SPARSE_NORM_SCALE))

    cv = (candidate_doc(con) or "")[:3000]
    if not cv:
        print("no candidate CV — abort"); return

    rows = con.execute(
        """SELECT v.id, v.title, v.description
           FROM vacancy v JOIN vacancy_feature vf ON vf.vacancy_id = v.id
           WHERE v.status IN (?, ?, ?) AND v.description IS NOT NULL""",
        (Status.LABELED.value, Status.SCORED.value, Status.SLATED.value),
    ).fetchall()
    print(f"vacancies to backfill (LABELED∪SCORED∪SLATED with a feature row): {len(rows)}")
    if not rows:
        return

    print("loading bge-m3 (foreground, single-instance) …")
    model = _features._load_model(f_cfg.get("model", "BAAI/bge-m3"))
    jds = [f"{r['title'] or ''}\n{r['description'] or ''}" for r in rows]
    print("computing bge-m3 sparse-lexical scores …")
    sp = _features.bgem3_sparse_scores(model, cv, jds)

    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for i, r in enumerate(rows):
        vid = int(r["id"])
        feat = _features.feature_row(con, vid) or {}
        feat["bgem3_sparse"] = float(sp[i])
        feat["fit_score"] = _features.fit_from_feature(feat, fit_w, sparse_scale=scale)
        con.execute(
            "UPDATE vacancy_feature SET feature_json = ?, computed_at = ? WHERE vacancy_id = ?",
            (json.dumps(feat, ensure_ascii=False), now, vid),
        )
        n += 1
    con.commit()
    print(f"backfilled bgem3_sparse + recomputed fit_score for {n} vacancies "
          f"(scale={scale}, weights={fit_w})")


if __name__ == "__main__":
    main()
