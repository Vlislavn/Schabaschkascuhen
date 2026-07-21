"""Жёсткие детерминированные чекеры ДО любого LLM (паттерн are/hard-before-soft-judge).

Боль №1 Влада: англоязычная вакансия с де-факто требованием немецкого (измерено 18.1%).
Детектор немецкого = broad pattern + exoneration-каскад (паттерн aislop/fp-guards):
сырой матч требования Deutsch/German, затем short-circuit, если в окне ±80 символов есть
аффирмативный сигнал НЕ-требования («nice to have», «von Vorteil», «is a plus»). Бюджет
точности — в гардах. Регресс-гейт: spike/data/indeed.csv, ≥25 из 27 сырых матчей.
"""
from __future__ import annotations

import re

from . import db
from .models import FilterReason, Status

# Broad pattern: требование владения немецким. Покрывает DE- и EN-формулировки.
GERMAN_REQ = re.compile(
    r"(fließend|verhandlungssicher|Deutschkenntnisse|"
    r"German.{0,40}(required|fluent|mandatory|C1|C2|B2|native|proficien)|"
    r"(fluent|fluency|proficien|business.?level|good command|excellent command|"
    r"very good|native).{0,35}German|"
    r"fluent in German|sehr gute? Deutsch|gute? Deutschkenntnisse|"
    # CONDITIONAL / "local language" hidden-German (DuPont 67: "Fluency in the local language,
    # depending on location: German if the role is based in Germany"). The "Fluency…German" span
    # exceeds 35 chars and "German if…" carries no required/fluent token nearby — these alternations
    # catch the conditional phrasing directly. For the user the role IS in DE → German is de-facto hard.
    r"local language.{0,80}(German|Deutsch)|"
    r"German\s+if\b.{0,60}(German|Deutschland|based|located|role)|"
    r"Deutsch.{0,30}(wenn|sofern|falls).{0,40}Deutschland)",
    re.I,
)
# Exoneration: аффирмативные сигналы, что немецкий НЕ требуется (FP-гард).
EXONERATION = re.compile(
    r"(nice to have|is a plus|are a plus|von Vorteil|wünschenswert|\ba plus\b|not required|"
    r"would be a plus|desirable|of advantage|nicht erforderlich|kein.{0,10}Deutsch|"
    r"we offer German|German classes|learn German|\boptional\b)",
    re.I,
)
# Сильный сигнал требования — побеждает exoneration (кейс «German essential; other langs a plus»).
# БЕЗ голого «required»: оно ловится в «not required» и ломает негацию — за него отвечает GERMAN_REQ.
STRONG_REQ = re.compile(
    r"(essential|mandatory|vorausgesetzt|verhandlungssicher|"
    r"must (have|speak|be)|zwingend|fließend|"
    # 'German required/needed/erforderlich' is a hard requirement and must beat an 'a plus'
    # exoneration about a DIFFERENT language; negation (not/nicht/kein) keeps it weak.
    r"(?<!not )(?<!nicht )(?<!kein )(required|needed|erforderlich))",
    re.I,
)
# Окно асимметрично и КОРОТКОЕ: «a plus» сразу после «German» = опционально; далеко = про другие
# языки. Exoneration ищется в [match_start − LEAD, match_end + TRAIL].
EXON_LEAD = 45
EXON_TRAIL = 28


def german_required(description: str) -> bool:
    """True, если текст требует немецкий. Exoneration в окне снимает требование, КРОМЕ случаев
    с сильным сигналом (essential/erforderlich/fließend) — тогда «a plus» про другие языки игнорим."""
    if not description:
        return False
    text = str(description)
    for m in GERMAN_REQ.finditer(text):
        lo = max(0, m.start() - EXON_LEAD)
        hi = min(len(text), m.end() + EXON_TRAIL)
        window = text[lo:hi]
        if EXONERATION.search(window) and not STRONG_REQ.search(window):
            continue  # оправдано (опциональный немецкий) — этот матч не считается требованием
        return True
    return False


def active_repellents(cfg: dict | None) -> set[str]:
    """Which HARD DROPS are on for THIS user — from cfg.profile.repellents (multi-user fix:
    the drops used to be unconditional, i.e. user #1's repellents deleted user #2's jobs).
    No profile/repellents in cfg → the original full drop set (behavior-preserving default)."""
    reps = ((cfg or {}).get("profile") or {}).get("repellents")
    if reps is None:
        return {"hidden-german", "remote-only", "temp-agency"}
    return {str(r) for r in reps}


def apply_hard_filters(cfg: dict, con) -> dict[str, int]:
    """DESCRIBED → детерминированные отсечения (гейтятся на repellents пользователя)."""
    reps = active_repellents(cfg)
    rows = db.by_status(con, Status.DESCRIBED)
    counts = {"language_de": 0, "temp_agency": 0, "kept": 0}
    for row in rows:
        if "hidden-german" in reps and german_required(row["description"] or ""):
            db.set_status(con, row["id"], Status.FILTERED,
                          filter_reason=FilterReason.LANGUAGE_DE)
            counts["language_de"] += 1
        elif "temp-agency" in reps and row["is_temp_agency"] == 1:
            db.set_status(con, row["id"], Status.FILTERED,
                          filter_reason=FilterReason.TEMP_AGENCY)
            counts["temp_agency"] += 1
        else:
            counts["kept"] += 1
    db.log_funnel(con, "hardfilter", counts["language_de"] + counts["temp_agency"],
                  detail=f"language_de={counts['language_de']} temp_agency={counts['temp_agency']} "
                         f"kept={counts['kept']}")
    return counts
