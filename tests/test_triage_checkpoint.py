"""Checkpoint-the-previous-model registry (serve-start retrain) — isolated from training."""
from __future__ import annotations

import json

import pytest

from schabasch import triage


def _cfg(tmp_path):
    return {"paths": {"model_dir": str(tmp_path)}, "triage": {}}


def _fake_artifact(tmp_path, labels_sha: str):
    joblib = pytest.importorskip("joblib")
    payload = {"regressor": {"x": 1}, "calibrator": None, "n_features": 1053,
               "labels_sha": labels_sha, "trained_at": "2026-06-15T10:00:00+00:00",
               "metrics": {"ndcg_10": 0.6, "mae": 0.7}}
    art = tmp_path / "triage.joblib"
    joblib.dump(payload, art)
    art.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in payload.items() if k not in ("regressor", "calibrator")}))
    return art


def test_checkpoint_archives_and_registers_on_label_change(tmp_path):
    cfg = _cfg(tmp_path)
    art = _fake_artifact(tmp_path, "OLDSHA")
    dest = triage._checkpoint_previous(cfg, labels_sha="NEWSHA")
    assert dest is not None
    archived = list((tmp_path / "archive").glob("triage_*.joblib"))
    assert len(archived) == 1                      # previous model kept
    assert art.exists()                            # original COPIED, not moved
    reg = (tmp_path / "registry.jsonl").read_text().strip()
    line = json.loads(reg)
    assert line["labels_sha"] == "OLDSHA"          # the PREVIOUS model's identity
    assert line["metrics"]["ndcg_10"] == 0.6       # its metrics
    assert line["trained_at"].startswith("2026-06-15")   # and date


def test_checkpoint_noop_when_labels_unchanged(tmp_path):
    cfg = _cfg(tmp_path)
    _fake_artifact(tmp_path, "SAME")
    assert triage._checkpoint_previous(cfg, labels_sha="SAME") is None
    assert not list((tmp_path / "archive").glob("*.joblib"))   # nothing archived
    assert not (tmp_path / "registry.jsonl").exists()


def test_checkpoint_noop_without_prior_model(tmp_path):
    assert triage._checkpoint_previous(_cfg(tmp_path), labels_sha="X") is None
