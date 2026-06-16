"""B (UX redesign) render invariants — the laws-of-ux-gate subtractions + a11y additions are real.

Pure render assertions (no DB): the gate's decisions must be VISIBLE in the shipped HTML (the gate's
verify step — "Hick's Law → cut to 3 means the UI ships 3"). Reuses slate.render_* with dict fixtures.
"""
from __future__ import annotations

import schabasch.slate as S

_CARD = {
    "vacancy_id": 1, "title": "Senior Engineer", "company": "ACME", "city": "Frankfurt",
    "work_mode": "hybrid", "score": 4, "slot_type": "exploit", "summary": "строка1\nстрока2",
    "url": "http://x", "user_note": "",
    "llm_cov": 0.5, "llm_cov_reqs": [{"requirement": "Python", "verdict": "present"}],
    "enrichment": {"key_snippets": [{"goal": "g", "snippet": f"s{i}"} for i in range(5)],
                   "pros": ["p1"], "cons": ["c1"], "company_review": "rev",
                   "clean_summary": "", "model_used": "sota"},
}
_EXPLORE = {**_CARD, "vacancy_id": 2, "slot_type": "explore"}


def _slate():
    return S.render_html([_CARD, _EXPLORE], "2026-06-16")


def _annotate():
    return S.render_annotate_html([_CARD], "2026-06-16", total_pending=1)


# ── subtractions (laws-of-ux gate) ────────────────────────────────────────────────────────────
def test_annotate_drops_filter_chips_slate_keeps_them():
    assert 'class="chips"' not in _annotate()        # Tesler/Jakob: wrong on an unlabeled queue
    assert 'class="chips"' in _slate()


def test_competing_blue_summaries_demoted():
    """Von Restorff: skills/enrich <summary> must NOT carry the focal blue (#1F4E79) — only the
    score badge + ⛔ are strong. The focal blue stays for the .progress, not the card summaries."""
    css = S._CSS
    assert "details.skills summary{cursor:pointer;color:#2a2a2a" in css
    assert "details.enrich summary{cursor:pointer;color:#2a2a2a" in css
    assert "#1F4E79" not in css.split("details.skills summary")[1].split("}")[0]


def test_enrichment_snippets_capped_at_two():
    html = S._enrichment_html(_CARD["enrichment"])   # 5 snippets in → at most 2 rendered
    assert html.count('class="ej-goal"') == 2


def test_note_field_is_after_buttons_behind_a_toggle():
    h = S._card_block(_CARD)
    assert h.index('class="btns"') < h.index('id="note-1"')   # note comes AFTER the buttons
    assert 'class="note-toggle"' in h                          # hidden behind a "+ заметка" toggle
    assert 'id="note-1"' in h and 'style="display:none"' in h  # collapsed by default (no prior note)


def test_prior_note_opens_by_default():
    h = S._card_block({**_CARD, "user_note": "важно"})
    assert "важно" in h and 'class="note-toggle"' not in h     # a saved note shows, no toggle needed


# ── additions (gate-justified) ────────────────────────────────────────────────────────────────
def test_responsive_and_a11y_css_tokens_present():
    css = S._CSS
    assert "@media (max-width:480px)" in css
    assert ".sr-only{" in css
    assert "focus-visible" in css
    assert "table.metrics td::before{content:attr(data-label)" in css   # mobile card-list reflow
    assert "min-height:44px" in css                                     # Fitts tap targets


def test_emoji_buttons_have_aria_labels():
    h = S._card_block(_CARD)                              # default render = English
    assert 'aria-label="Not for me — office mouse"' in h
    assert 'aria-label="Interesting"' in h and 'aria-label="шабашка — the dream"' in h
    assert 'aria-label="Applied"' in h


def test_score_badge_has_accessible_name():
    assert 'role="img" aria-label="score 4 of 5"' in S._score_badge(4)


def test_chips_are_keyboard_focusable_buttons():
    sl = _slate()
    assert '<button type="button" class="chip' in sl        # not <span> → keyboard-accessible
    assert "<span class=\"chip\"" not in sl


def test_legend_present_on_card_pages():
    assert "legend" in _slate() and "legend" in _annotate()
    assert "💻🐀 office mouse" in _slate()


def test_explore_slot_has_clear_label():
    assert "interest check" in S._card_block(_EXPLORE)


def test_meta_row_split_for_mobile_stacking():
    assert 'class="meta2"' in S._card_block(_CARD)   # secondary meta → own line on mobile


def test_metrics_tables_carry_data_labels():
    rep = {"n_labels": 2, "reliable": True, "headline": {"pairwise_acc": 0.8, "ndcg@10": 0.5},
           "n_comparable_pairs": 9, "min_pairs": 15,
           "rows": [{"name": "fit", "pairwise_acc": 0.8, "ndcg@10": 0.5, "spearman": 0.4,
                     "n": 5, "clean": True}]}
    html = S.render_eval_html(rep)
    assert 'data-label="pairwise"' in html
    # header in <thead> so the @media `thead{position:absolute}` mobile-hide actually fires
    # (bare <tr><th> headers would render as a broken card on phones — adversarial-review catch)
    assert "<thead>" in html and "<tbody>" in html
