"""i18n catalog integrity: every locale has the SAME key set (translation-drift guard) + t() behavior."""
from __future__ import annotations

from schabasch import i18n


def test_locales_have_identical_key_sets():
    cats = i18n._catalogs()
    assert "en" in cats and "ru" in cats
    base = set(cats["en"])
    for lang, cat in cats.items():
        assert set(cat) == base, f"{lang} differs: missing={base - set(cat)}, extra={set(cat) - base}"


def test_default_lang_available_and_first():
    langs = i18n.available_langs()
    assert i18n.DEFAULT_LANG in langs and langs[0] == i18n.DEFAULT_LANG


def test_normalize_lang_falls_back():
    assert i18n.normalize_lang("ru") == "ru"
    assert i18n.normalize_lang("de") == i18n.DEFAULT_LANG   # unknown → default
    assert i18n.normalize_lang(None) == i18n.DEFAULT_LANG


def test_t_formats_and_falls_back_to_key():
    assert i18n.t("en", "posted.days_ago", verb="posted", days=3) == "posted 3d ago"
    assert i18n.t("en", "no.such.key") == "no.such.key"   # missing key → the key itself
    # «шабашка» stays literal in both locales (signature term); a {lang} placeholder is allowed
    assert "шабашка" in i18n.t("en", "label.shabashka")
    assert i18n.t("ru", "label.shabashka") == "шабашка"
    assert "lang=ru" in i18n.t("en", "eval.empty", lang="ru")
