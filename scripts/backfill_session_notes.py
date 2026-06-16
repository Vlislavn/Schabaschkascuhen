"""Backfill Alina's hand-written 15 June session feedback into label.why_freetext.

Why: she wrote rich per-job reasoning (schabasch/15JuneSession.md), but it lived only in that .md —
0/37 labels carried a note, so judge.build_fewshot learned NOTHING from her actual reasons (the
WS2 textarea→why_freetext loop only captures FUTURE clicks). This one-shot, idempotent backfill maps
each session-feedback job to its labelled vacancy id (EXACT match, verified against the label table —
no fuzzy matching) and writes her verbatim note as label.why_freetext via db.insert_label, which
COALESCEs: it ONLY fills the note, never changes her score/applied flag. Re-running is a no-op.

Effect: on the NEXT (coordinated, model-loading) re-judge tick, build_fewshot surfaces these as
NOTE: lines — the score<=2 negatives (1071 EUMETSAT "инженер нет", 797 MAM "AI слоп", 906
Westinghouse "требуют интерна") and the score=5 positive (439 Merz "Master Data ≠ degree") teach the
judge her real taste. Provenance for every note: schabasch/15JuneSession.md.

Run: .venv/bin/python -m scripts.backfill_session_notes        (PYTHONPATH=repo root, or run from it)
"""
from __future__ import annotations

from schabasch import config, db

# vacancy_id -> (verbatim-faithful note from 15JuneSession.md, the session line it came from)
# Mapping verified against the live label table (company+title) — see the docstring.
SESSION_NOTES: dict[int, str] = {
    439: "Ни слова про master как degree — только Master Data, это другое. "
         "Неверная интерпретация требования.",                                      # Merz, L5
    1071: "Инженер — нет. Не хочу работать руками, хочу головой (могу работать С "
          "разработчиками, но не быть одним из них). Люблю когда в вакансии lead. "
          "Не люблю monthly/дедлайны — это стресс; дедлайны и рутина — минус.",      # EUMETSAT, L2
    67:  "Скрытый немецкий: «German if the role is based in Germany» — для роли в "
         "Германии это де-факто требование.",                                       # DuPont, L4
    797: "Ничего не понятно, какой-то AI-слоп текст.",                              # MAM, L2
    597: "Интересно, но не проходит из-за master degree.",                         # Merck, L4
    1164: "Круто, много мэтча по резюме — их требование по master degree я бы "
          "здесь проигнорила. GMP/GxP кажется скучным.",                            # SCHOTT, L4
    906: "Не actually bad, но требуют интерна. Продублировано — та же позиция у "
         "другого аккаунта (Laveer / Westinghouse).",                              # Westinghouse, L2
}


def backfill(con) -> dict[str, int]:
    out = {"written": 0, "skipped_no_label": 0, "already": 0}
    for vid, note in SESSION_NOTES.items():
        row = con.execute(
            "SELECT score_1_5, applied, interview, why_freetext FROM label "
            "WHERE vacancy_id = ? AND source = 'slate'", (vid,)).fetchone()
        if row is None:
            out["skipped_no_label"] += 1
            print(f"  vid {vid}: NO slate label — skipped (mapping drift?)")
            continue
        if (row["why_freetext"] or "").strip() == note.strip():
            out["already"] += 1
            print(f"  vid {vid}: note already present — no-op")
            continue
        # COALESCE upsert: score_1_5=None keeps her rating, applied=0 keeps it (MAX), only the note fills.
        db.insert_label(con, vid, {
            "score_1_5": None, "applied": int(row["applied"] or 0), "source": "slate",
            "interview": row["interview"], "why_freetext": note,
        })
        out["written"] += 1
        print(f"  vid {vid}: note set (score kept = {row['score_1_5']})")
    return out


def main():
    cfg = config.load()
    con = db.connect(cfg["paths"]["db"])
    print(f"Backfilling {len(SESSION_NOTES)} session notes into label.why_freetext …")
    res = backfill(con)
    n_ft = con.execute(
        "SELECT COUNT(*) FROM label WHERE why_freetext IS NOT NULL AND TRIM(why_freetext) != ''"
    ).fetchone()[0]
    print(f"done: {res} | labels with a note now: {n_ft}")


if __name__ == "__main__":
    main()
