"""Regenerate the README screenshots from synthetic demo data.

The generated images are intentionally compact README thumbnails, not full-page
retina dumps. The HTML files are kept next to the PNGs for inspection.

    python -m scripts.gen_screenshots            # English (default; what the README embeds)
    python -m scripts.gen_screenshots --lang ru  # Russian, to eyeball the toggle

Re-run after any UI change — that's the point (the screenshots never drift from the code).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from schabasch import slate

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CAPTURES = {
    "slate": (900, 900),
    "annotate": (900, 720),
    "eval": (900, 430),
    "gaps": (900, 440),
}


def _card(**kw) -> dict:
    """A demo slate entry with safe defaults (fake company/title — nothing real)."""
    base = {
        "vacancy_id": kw.get("vacancy_id", 1), "title": "", "company": "", "city": "Heidelberg",
        "url": "https://example.com/job", "score": 4, "why": "", "summary": "",
        "work_mode": "hybrid", "date_posted": "2026-06-12", "first_seen": "2026-06-12",
        "investigation": None, "enrichment": None, "fit_score": 0.0, "fit_note": "",
        "llm_cov": None, "llm_cov_reqs": [], "elig_note": "", "elig_severity": "structural",
        "far": False, "dist_km": None, "geo_anchor": None, "also_at": [], "user_note": "",
        "slot_type": "exploit", "feedback": None,
    }
    base.update(kw)
    return base


# --- synthetic slate: a dream-gig, a solid match with a deep-dive, an explore far job -------------
_SLATE = [
    _card(vacancy_id=1, title="Business Analyst — Satellite Programs", company="Aurora Space GmbH",
          city="Heidelberg", score=5, why="space", summary="Shape requirements for ground-segment "
          "software on Europe's next earth-observation constellation.", fit_score=0.82, llm_cov=0.8,
          llm_cov_reqs=[{"requirement": "Business analysis", "verdict": "present"},
                        {"requirement": "BPMN 2.0", "verdict": "present"},
                        {"requirement": "Stakeholder management", "verdict": "present"},
                        {"requirement": "Aerospace domain", "verdict": "partial"},
                        {"requirement": "German C1", "verdict": "missing"}],
          investigation={"company_size": "large", "salary_eur_min": 70000, "salary_eur_max": 90000,
                         "english_team_signal": True, "german_rooted": True, "still_open": True,
                         "company_description": "Mid-cap European space-systems integrator."}),
    _card(vacancy_id=2, title="Process Owner — Public Sector Digitalization", company="Demo Verwaltung AG",
          city="Frankfurt am Main", score=4, why="public-sector", summary="Own target operating "
          "models for a state agency's case-management rollout.", fit_score=0.66, llm_cov=0.7,
          llm_cov_reqs=[{"requirement": "Process design", "verdict": "present"},
                        {"requirement": "UAT / change mgmt", "verdict": "present"},
                        {"requirement": "Public-sector experience", "verdict": "missing"}],
          enrichment={"clean_summary": "A change-management role driving a multi-year digitalization "
                      "program; analyst-led, not hands-on.",
                      "pros": ["English-friendly team", "Hybrid, Frankfurt"],
                      "cons": ["Some German helpful"], "company_review": "Stable public-sector contractor.",
                      "key_snippets": [{"goal": "requirements", "snippet": "Define target operating model"}],
                      "model_used": "qwen3.5:4b"}),
    _card(vacancy_id=3, title="Data Analyst (Werkstudent)", company="Beispiel Biotech",
          city="Heidelberg", score=2, why="biotech", summary="Part-time student role in a lab-software team.",
          slot_type="explore", fit_score=0.34, fit_note="big skill gap",
          elig_note="role is a working-student / intern position", elig_severity="soft"),
]

_EVAL = {"n_labels": 42, "reliable": True, "min_pairs": 15, "n_comparable_pairs": 120,
         "headline": {"pairwise_acc": 0.81, "ndcg@10": 0.58},
         "rows": [
             {"name": "fit_score", "label": "fit_score (CV↔job)", "pairwise_acc": 0.81, "ndcg@10": 0.58,
              "spearman": 0.43, "n": 42, "clean": True},
             {"name": "cross-encoder", "label": "cross-encoder", "pairwise_acc": 0.74, "ndcg@10": 0.51,
              "spearman": 0.36, "n": 42, "clean": True},
             {"name": "llm_cov", "label": "skill coverage", "pairwise_acc": 0.65, "ndcg@10": 0.44,
              "spearman": 0.28, "n": 42, "clean": True},
             {"name": "judge", "label": "magnet judge", "pairwise_acc": 0.72, "ndcg@10": 0.49,
              "spearman": 0.33, "n": 42, "clean": False}]}

_GAPS = {"n_wanted": 18, "n_jobs_with_reqs": 16, "reliable": True, "rows": [
    {"requirement": "German (C1)", "missing": 11, "partial": 3, "present": 4, "jobs": 18},
    {"requirement": "Public-sector experience", "missing": 7, "partial": 2, "present": 1, "jobs": 10},
    {"requirement": "Aerospace / space domain", "missing": 6, "partial": 4, "present": 0, "jobs": 10},
    {"requirement": "Power BI", "missing": 3, "partial": 1, "present": 9, "jobs": 13}]}


def render(lang: str) -> dict[str, str]:
    return {
        "slate": slate.render_html([_SLATE[0], _SLATE[2]], "2026-06-16", lang=lang),
        "annotate": slate.render_annotate_html(_SLATE[:2], "2026-06-16", total_pending=37, lang=lang),
        "eval": slate.render_eval_html(_EVAL, lang=lang),
        "gaps": slate.render_gaps_html(_GAPS, lang=lang),
    }


def capture(html_path: Path, png_path: Path, *, width: int, height: int) -> bool:
    if not Path(CHROME).exists():
        return False
    subprocess.run([CHROME, "--headless", f"--screenshot={png_path}",
                    f"--window-size={width},{height}", "--hide-scrollbars",
                    "--force-device-scale-factor=1", "--default-background-color=FFFFFFFF",
                    html_path.as_uri()], check=True, capture_output=True)
    return True


def main(lang: str = "en") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pages = render(lang)
    for name, html in pages.items():
        hp = OUT / f"{name}.html"
        hp.write_text(html, encoding="utf-8")
        width, height = CAPTURES[name]
        ok = capture(hp, OUT / f"{name}.png", width=width, height=height)
        print(f"  {name}: wrote {hp.name}" + (f" + {name}.png" if ok else " (no Chrome → HTML only)"))
    print(f"done ({lang}) → {OUT}")


if __name__ == "__main__":
    lang = "en"
    if "--lang" in sys.argv:
        lang = sys.argv[sys.argv.index("--lang") + 1]
    main(lang)
