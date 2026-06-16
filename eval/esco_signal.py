"""Compute ESCO discrete-skill coverage for the gold jobs, persist to feature_json['_extra'],
and evaluate on the gold set. Validates that ESCO coverage discriminates her-fit from adjacent."""
from __future__ import annotations

import json
import time

from schabasch import config, db, candidate
from eval.match_eval import GOLD, evaluate

con = db.connect(config.load()["paths"]["db"])
cv_text = candidate.candidate_doc(con) or ""

t = time.time()
from esco_skill_extractor import SkillExtractor
se = SkillExtractor(model="all-MiniLM-L6-v2", skills_threshold=0.5, device="cpu")
print(f"SkillExtractor loaded in {time.time()-t:.0f}s")


def skills(text: str) -> set[str]:
    try:
        return set(se.get_skills([text or ""])[0])
    except Exception:
        return set()


cv_skills = skills(cv_text)
print(f"CV ESCO skills: {len(cv_skills)}")

rows = con.execute(
    "SELECT v.id, v.title, v.description, vf.feature_json FROM vacancy v "
    "JOIN vacancy_feature vf ON vf.vacancy_id=v.id WHERE v.id IN (%s)"
    % ",".join(str(i) for i in GOLD)
).fetchall()

esco_cov = {}
for r in rows:
    jd = skills(r["description"] or "")
    cov = len(cv_skills & jd) / len(jd) if jd else 0.0
    esco_cov[int(r["id"])] = cov
    feat = json.loads(r["feature_json"])
    feat.setdefault("_extra", {})["esco_cov"] = cov
    feat["_extra"]["esco_missing_n"] = len(jd - cv_skills)
    con.execute("UPDATE vacancy_feature SET feature_json=? WHERE vacancy_id=?",
                (json.dumps(feat, ensure_ascii=False), int(r["id"])))
con.commit()

print("\nESCO coverage by gold label:")
for i in sorted(esco_cov, key=lambda x: -esco_cov[x]):
    print(f"  cov={esco_cov[i]:.2f} gold={GOLD[i][0]}  {GOLD[i][1][:55]}")
print("\n", evaluate(esco_cov, name="esco_cov"))
