"""Контракты данных: статусы FSM, причины фильтрации, классы ошибок, карточка.

Единственный источник правды для всех модулей. Менять — только вместе с db.py.
"""
from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
from typing import Any


class Status(str, enum.Enum):
    """FSM вакансии: new → prefiltered | described → normalized → filtered | scored → slated → labeled | expired."""

    NEW = "new"
    PREFILTERED = "prefiltered"      # срезана грубым гео-фильтром (до описания)
    DESCRIBED = "described"          # полное описание добыто
    NORMALIZED = "normalized"        # карточка построена
    FILTERED = "filtered"            # срезана авторитетным фильтром по карточке
    SCORED = "scored"                # оценена судьёй
    SLATED = "slated"                # показана в slate
    LABELED = "labeled"              # получила метку Влада
    EXPIRED = "expired"              # 404/410 на источнике


class FilterReason(str, enum.Enum):
    """Закрытый enum причин отсечения — воронка должна быть диагностируемой."""

    GEO = "filtered_geo"
    REMOTE_ONLY = "filtered_remote"
    LANGUAGE_DE = "filtered_language_de"
    TEMP_AGENCY = "filtered_temp_agency"
    NO_DESCRIPTION = "filtered_no_description"
    EXPIRED_GONE = "expired_gone"


class ErrorClass(str, enum.Enum):
    """Классифицированный error envelope (паттерн claude-code/error-envelope):
    закрытый enum + lossless details рядом."""

    INVALID_JSON = "invalid_json"
    SCHEMA_VIOLATION = "schema_violation"
    EMPTY_OUTPUT = "empty_output"
    TIMEOUT = "timeout"
    ENDPOINT_DOWN = "endpoint_down"
    OVERSIZED_INPUT = "oversized_input"
    HTTP_ERROR = "http_error"
    UNKNOWN = "unknown"


class CanaryVerdict(str, enum.Enum):
    """Канарейка источника: отличаем мёртвый скрейпер от пустого рынка (кейс «тихого Google»)."""

    OK = "ok"
    DEGRADED = "degraded"            # часть запросов пустые/упали
    DEAD_SCRAPER = "dead_scraper"    # канарный запрос дал 0 — скрейпер/парсер мёртв
    EMPTY_MARKET = "empty_market"    # канарейка жива, но рынок пуст по матрице


# Кнопки slate → единая шкала (👎=2, 👍=4, ⭐=5; applied — флаг поверх; direction=2 «не эта вакансия,
# но направление интересно» — низкий score убирает из показа, плюс отдельный role-fit=1 буст домена).
FEEDBACK_TO_SCORE = {"bad": 2, "good": 4, "star": 5, "direction": 2}

WHY_TAGS = [
    # magnets — profile-specific aspirational domains (USE_CASE.md)
    "animals", "space", "military-security", "complex-projects", "public-sector", "new-domain",
    # repellents
    "hidden-german", "biotech", "slop-text", "boring-role", "remote-only", "temp-agency",
]


@dataclasses.dataclass
class Card:
    """Нормализованная карточка вакансии (выход Normalizer, вход фильтров/судьи)."""

    role: str
    company: str
    domain: str
    city: str
    work_mode: str               # remote | hybrid | onsite | unknown
    language_posting: str        # de | en
    language_reality: str        # de | en | unknown — язык, нужный для РАБОТЫ
    integration_potential: int   # 0..2
    summary_2lines: str          # по-русски
    slop_score: int              # 0..100, density-smoothed (паттерн aislop), НЕ бинарный
    temp_agency_guess: bool

    @classmethod
    def from_llm_json(cls, d: dict[str, Any]) -> "Card":
        """Валидация ответа LLM. Бросает ValueError (→ ErrorClass.SCHEMA_VIOLATION)."""
        try:
            wm = str(d["work_mode"]).lower()
            if wm not in ("remote", "hybrid", "onsite", "unknown"):
                wm = "unknown"
            lr = str(d["language_reality"]).lower()
            if lr not in ("de", "en", "unknown"):
                raise ValueError(f"language_reality={lr!r}")
            slop = d.get("slop_score", d.get("slop_flag", 0))
            slop = int(slop) if not isinstance(slop, bool) else (60 if slop else 0)
            return cls(
                role=str(d["role"])[:200],
                company=str(d["company"])[:200],
                domain=str(d["domain"])[:100],
                city=str(d.get("city", ""))[:100],
                work_mode=wm,
                language_posting=str(d["language_posting"]).lower()[:2],
                language_reality=lr,
                integration_potential=max(0, min(2, int(d["integration_potential"]))),
                summary_2lines=str(d["summary_2lines"])[:600],
                slop_score=max(0, min(100, slop)),
                temp_agency_guess=bool(d.get("temp_agency_guess", False)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"card schema violation: {e}") from e

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def normalize_company(name: str) -> str:
    """Нормализация имени компании для дедуп-ключа (срезать правовые суффиксы)."""
    import re
    s = (name or "").lower().strip()
    # (?![\w-]) — не срезать 'co' внутри составного/дефисного имени (Co-operative, Coherent):
    # раньше \bco\.?\b превращал 'Co-operative' → 'operative', разваливая дедуп-ключ.
    s = re.sub(r"\b(?:gmbh|ag|se|kg|kgaa|mbh|inc|ltd|llc|co|& co)\.?(?![\w-])", "", s)
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def normalize_title(title: str) -> str:
    """Нормализация тайтла: срезать гендер-теги (m/w/d) и пунктуацию."""
    import re
    s = (title or "").lower()
    s = re.sub(r"\((?:[mwfdx]\s*[/|]\s*){1,3}[mwfdx]\)", "", s)
    s = re.sub(r"\(\s*gn\s*\)", "", s)
    # also strip UNPARENTHESIZED gender tags (e.g. 'engineer m/w/d') — these survived and split
    # the dedup key between '... (m/w/d)' and '... m/w/d' variants of the same posting.
    s = re.sub(r"\b[mwfdx](?:\s*[/|]\s*[mwfdx]){1,3}\b", "", s)
    s = re.sub(r"\ball genders\b", "", s)
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()


def dedup_key(company: str, title: str) -> str:
    return f"{normalize_company(company)}::{normalize_title(title)}"


def content_hash(description: str) -> str:
    """Hash описания: short-circuit повторной нормализации репостов (hard-before-soft)."""
    return hashlib.sha256((description or "").strip().encode("utf-8")).hexdigest()[:16]
