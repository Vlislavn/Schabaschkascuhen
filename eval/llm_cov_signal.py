"""LLM per-requirement coverage (ConFit-v3 'non-negotiable requirements' pattern): qwen lists each
must-have and judges present/partial/missing vs the CV. fit = (present + 0.5*partial)/total.
Compute on the gold set, persist to _extra.llm_cov, validate on the harness."""
from __future__ import annotations

import json
import re

from schabasch import config, db, candidate
from schabasch.llm import OllamaClient
from eval.match_eval import GOLD, evaluate

con = db.connect(config.load()["paths"]["db"])
cv = candidate.candidate_doc(con) or ""

_SYS = (
    "You assess whether a CANDIDATE can DO a job. From the JOB, extract the 3–8 genuine MUST-HAVE "
    "requirements (core skills, tools, experience, qualifications needed to perform the role — "
    "IGNORE nice-to-haves, perks, and company boilerplate). For EACH, judge strictly from the "
    "CANDIDATE only: present | partial | missing. Be honest — a skill not evidenced is missing. "
    'Return ONLY JSON: {"requirements":[{"requirement": str, "verdict": "present|partial|missing"}], '
    '"missing_summary": "<short, the key missing must-haves>"}'
)


def llm_cov(title: str, desc: str) -> tuple[float, str]:
    client = OllamaClient(model=config.load()["llm"]["judge_model"], num_ctx=8192, temperature=0)
    user = f"CANDIDATE:\n{cv[:1500]}\n\nJOB:\n{title}\n{(desc or '')[:3000]}"
    try:
        obj = client.chat_json(_SYS, user)
        reqs = obj.get("requirements") or []
        if not reqs:
            return 0.5, ""
        v = [str(r.get("verdict", "")).lower() for r in reqs if isinstance(r, dict)]
        tot = len(v) or 1
        cov = (v.count("present") + 0.5 * v.count("partial")) / tot
        return float(cov), str(obj.get("missing_summary") or "")
    except Exception:
        return 0.5, ""   # neutral on failure (don't gate/boost)


rows = con.execute(
    "SELECT v.id, v.title, v.description, vf.feature_json FROM vacancy v "
    "JOIN vacancy_feature vf ON vf.vacancy_id=v.id WHERE v.id IN (%s)"
    % ",".join(str(i) for i in GOLD)).fetchall()

scores = {}
for r in rows:
    cov, miss = llm_cov(r["title"] or "", r["description"] or "")
    scores[int(r["id"])] = cov
    feat = json.loads(r["feature_json"])
    feat.setdefault("_extra", {})["llm_cov"] = cov
    feat["_extra"]["llm_cov_missing"] = miss[:120]
    con.execute("UPDATE vacancy_feature SET feature_json=? WHERE vacancy_id=?",
                (json.dumps(feat, ensure_ascii=False), int(r["id"])))
    con.commit()
    print(f"  cov={cov:.2f} gold={GOLD[int(r['id'])][0]}  {GOLD[int(r['id'])][1][:48]}  miss={miss[:50]}")

print("\n", evaluate(scores, name="llm_cov"))
