"""P3 — the role-fit learning GATE (Delphi panel). Compare the slate rank with role_kind learning
OFF vs ON on the REAL labels, and report the ship decision.

DECISIVE guardrail = the raw-label `effective` row (independent of the role votes) — it must NOT
regress when learning is ON, or we keep it OFF. The `effective_role` row is the veto-aware DIRECTIONAL
signal (self-confirming → it may veto a ship, never justify one). Run once enough golden 🙅 'wrong
role' votes accrue, to decide whether to flip `slate.role_kind_learn.enabled`.

    python -m eval.role_ablation
"""
from __future__ import annotations

import copy

from schabasch import config, db, role_feedback, validation


def _rows(cfg, con) -> dict:
    return {r["name"]: r for r in validation.eval_report(cfg, con)["rows"]}


def _g(rows: dict, name: str):
    r = rows.get(name, {})
    return r.get("pairwise_acc"), r.get("ndcg@10")


def main() -> None:
    cfg0 = config.load()
    con = db.connect(cfg0["paths"]["db"])
    n_votes = len(role_feedback.veto_map(con, source="slate"))
    print(f"golden role-fit votes (source='slate'): {n_votes}")

    off = copy.deepcopy(cfg0)
    off.setdefault("slate", {}).setdefault("role_kind_learn", {})["enabled"] = False
    on = copy.deepcopy(cfg0)
    on.setdefault("slate", {}).setdefault("role_kind_learn", {})["enabled"] = True

    ro, rn = _rows(off, con), _rows(on, con)
    print(f"  GUARDRAIL  effective        OFF={_g(ro,'effective')}   ON={_g(rn,'effective')}")
    print(f"  directional effective_role  OFF={_g(ro,'effective_role')}   ON={_g(rn,'effective_role')}")

    op, _ = _g(ro, "effective")
    npair, _ = _g(rn, "effective")
    if op is not None and npair is not None:
        ok = npair >= op - 1e-9
        print(f"  SHIP GATE (guardrail not regressed): {'PASS — safe to enable' if ok else 'FAIL — keep learning OFF'}")
    if n_votes == 0:
        print("  note: 0 votes → learning ON == OFF (neutral). Re-run after Alina tags 🙅 wrong-role cards.")


if __name__ == "__main__":
    main()
