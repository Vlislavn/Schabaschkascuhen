"""Регресс детектора скрытого немецкого: 27-строчный spike-сет (gate ≥25) + exoneration."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from schabasch.hardfilters import german_required

ROOT = Path(__file__).resolve().parent.parent
INDEED = ROOT / "spike" / "data" / "indeed.csv"

# Тот же эвристический ground-truth, что дал измерение 18.1% скрытого немецкого в спайке.
HIDDEN_DE = re.compile(
    r"(?i)(german[^.\n]{0,45}(c1|c2|b2|fluent|native|mandatory|required|must|business level))"
    r"|((fluent|fluency|proficient|proficiency|good command|excellent command|business[- ]fluent|very good)"
    r"[^.\n]{0,35}german)|deutschkenntnisse"
)
_DE = set("und der die das für mit wir sie nicht werden sind eine einen einem bei auf als auch oder "
          "zu im sowie unsere unserer ihre dich dir du wird haben kenntnisse erfahrung aufgaben "
          "profil bieten".split())
_EN = set("the and you with for our are will of to in is on as we your this that have team work "
          "skills experience role about from be or an at".split())


def _lang(t: str) -> str:
    w = re.findall(r"[a-zäöüß]+", str(t).lower())
    return "de" if sum(x in _DE for x in w) > sum(x in _EN for x in w) else "en"


def test_obvious_requirements_caught():
    assert german_required("Fluent German (C1) is required for this position.")
    assert german_required("Verhandlungssichere Deutschkenntnisse erforderlich.")
    assert german_required("We need someone fluent in German and English.")
    assert german_required("Native German speaker, business English.")


def test_exoneration_blocks_optional_german():
    # «a plus / von Vorteil / nice to have» рядом с German → НЕ требование
    assert not german_required("English is essential; German is a plus.")
    assert not german_required("Gute Deutschkenntnisse sind von Vorteil.")
    assert not german_required("German language skills are a plus, not required.")
    assert not german_required("Fluent in English. German would be a plus.")


def test_strong_req_beats_far_exoneration():
    # сильный сигнал требования + далёкая «a plus» про ДРУГИЕ языки → всё ещё требование
    assert german_required(
        "Fluency in German and English is essential; additional European languages are a plus."
    )


def test_empty_and_none():
    assert not german_required("")
    assert not german_required(None)  # type: ignore[arg-type]


@pytest.mark.skipif(not INDEED.exists(), reason="spike indeed.csv not present")
def test_regression_hidden_german_gate():
    import pandas as pd

    df = pd.read_csv(INDEED)
    df = df[df["description"].notna()].copy()
    df["lang"] = df["description"].map(_lang)
    en = df[df["lang"] == "en"]
    testset = en[en["description"].map(lambda d: bool(HIDDEN_DE.search(str(d))))]
    caught = testset["description"].map(lambda d: german_required(str(d))).sum()
    # Гейт спайка: поймать ≥25 из ~27 скрыто-немецких англоязычных вакансий.
    assert caught >= 25, f"caught only {caught}/{len(testset)}"


def test_dupont_67_conditional_local_language_german():
    """WS1d / DuPont-67: the conditional 'local language … German if the role is based in Germany'
    phrasing slipped through (exceeds the 35-char window, no required token near 'German'). For Alina
    the role IS in Germany → German is de-facto hard → must be caught."""
    jd = ("Requirements:\nFluent English (written and spoken)\n"
          "Fluency in the local language, depending on location:\n"
          "German if the role is based in Germany")
    assert german_required(jd) is True
    # German conditional variant
    assert german_required("Deutsch, sofern die Stelle in Deutschland angesiedelt ist.") is True
    # but a genuinely optional 'local language a plus' must NOT trip
    assert german_required("Fluency in the local language is a plus, not required.") is False


def test_german_required_beats_other_language_a_plus():
    """Regression: 'Fluent German required. <X> a plus' must flag (the 'a plus' refers to a
    different language, not German) — this was a false-negative leaking hidden-German."""
    from schabasch.hardfilters import german_required
    assert german_required("Fluent German required. English is a plus") is True
    assert german_required("Verhandlungssicheres Deutsch erforderlich") is True
    # negation must stay exonerated
    assert german_required("German not required, English fluent") is False
    assert german_required("Deutschkenntnisse nicht erforderlich") is False
