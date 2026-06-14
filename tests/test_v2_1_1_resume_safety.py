from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl

import scripts.build_model_features_v2_1_1 as runner
from src.features.feature_sets_v2_1_1 import load_feature_set_yaml, validate_feature_sets_from_file, write_feature_set_yaml
from src.features.history_builder_v2_1_1 import STATE_VERSION, new_state


def make_complete(tmp_path: Path, monkeypatch, years: list[int]) -> dict:
    base = tmp_path / "base"
    out = tmp_path / "out"
    cpdir = tmp_path / "cp"
    cfg = tmp_path / "feature_sets_v2_1_1.yaml"
    monkeypatch.setattr(runner, "BASE_DIR", base)
    monkeypatch.setattr(runner, "OUT_DIR", out)
    monkeypatch.setattr(runner, "CHECKPOINT_DIR", cpdir)
    monkeypatch.setattr(runner, "FEATURE_SET_YAML", cfg)
    write_feature_set_yaml(cfg)
    checkpoint = {"years": {}}
    for year in years:
        bpath = base / f"year={year}" / "data.parquet"
        bpath.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"Year": [year], "entry_id": [f"e{year}"]}).write_parquet(bpath)
        opath = out / f"year={year}" / "data.parquet"
        opath.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"Year": [year], "entry_id": [f"e{year}"]}).write_parquet(opath)
        spath = cpdir / f"history_state_after_{year}.pkl"
        spath.parent.mkdir(parents=True, exist_ok=True)
        state = new_state()
        state["completed_year"] = year
        with spath.open("wb") as f:
            pickle.dump(state, f)
        checkpoint["years"][str(year)] = {
            "status": "complete",
            "input": runner.file_fingerprint(bpath),
            "feature_set_hash": runner.feature_set_hash(),
            "code_bundle_hash": runner.code_bundle_hash(),
            "state_version": STATE_VERSION,
            "rows": 1,
        }
    return checkpoint


def test_feature_validation_always_available_and_dedicated_yaml(tmp_path) -> None:
    cfg = tmp_path / "feature_sets_v2_1_1.yaml"
    write_feature_set_yaml(cfg)
    assert load_feature_set_yaml(cfg)
    assert validate_feature_sets_from_file(cfg)[0]["status"] == "pass"
    assert cfg.name == "feature_sets_v2_1_1.yaml"


def test_contiguous_complete_years_stop_at_gap() -> None:
    checkpoint = {"years": {"2016": {"status": "complete"}, "2018": {"status": "complete"}}}
    contiguous, issues = runner.complete_years_contiguous(checkpoint)
    assert contiguous == [2016]
    assert issues and issues[0]["field"] == "non_contiguous_complete_year"


def test_validate_resume_detects_state_output_rows_and_versions(tmp_path, monkeypatch) -> None:
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    fhash = runner.feature_set_hash()
    chash = runner.code_bundle_hash()
    assert runner.validate_completed_year(checkpoint, 2016, fhash, chash) == []
    runner.state_path(2016).unlink()
    assert any(i["field"] == "state_exists" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    runner.out_path(2016).unlink()
    assert any(i["field"] == "output_exists" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    pl.DataFrame({"Year": [2016, 2016]}).write_parquet(runner.out_path(2016))
    assert any(i["field"] == "output_rows" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    state = new_state()
    state["history_state_version"] = "bad"
    state["completed_year"] = 2016
    with runner.state_path(2016).open("wb") as f:
        pickle.dump(state, f)
    assert any(i["field"] == "state_version" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))


def test_validate_resume_detects_input_feature_and_code_hash(tmp_path, monkeypatch) -> None:
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    fhash = runner.feature_set_hash()
    chash = runner.code_bundle_hash()
    checkpoint["years"]["2016"]["input"]["sha256"] = "bad"
    assert any(i["field"] == "input_sha256" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    checkpoint["years"]["2016"]["feature_set_hash"] = "bad"
    assert any(i["field"] == "feature_set_hash" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    checkpoint["years"]["2016"]["code_bundle_hash"] = "bad"
    assert any(i["field"] == "code_bundle_hash" for i in runner.validate_completed_year(checkpoint, 2016, fhash, chash))


def test_git_sha_only_change_does_not_fail_resume(tmp_path, monkeypatch) -> None:
    checkpoint = make_complete(tmp_path, monkeypatch, [2016])
    checkpoint["years"]["2016"]["git_commit_sha"] = "changed"
    assert runner.validate_completed_year(checkpoint, 2016, runner.feature_set_hash(), runner.code_bundle_hash()) == []


def test_rebuild_loads_previous_year_state_and_requires_it(tmp_path, monkeypatch) -> None:
    make_complete(tmp_path, monkeypatch, [2018])
    state = runner.load_rebuild_state(2019)
    assert state["completed_year"] == 2018
    runner.state_path(2018).unlink()
    try:
        runner.load_rebuild_state(2019)
    except runner.ResumeValidationError as exc:
        assert exc.issues[0]["field"] == "previous_state_exists"
    else:
        raise AssertionError("expected ResumeValidationError")


def test_invalidate_from_year_only_touches_target_and_later(tmp_path, monkeypatch) -> None:
    checkpoint = make_complete(tmp_path, monkeypatch, [2016, 2017, 2018])
    runner.invalidate_from_year(checkpoint, 2017, runner.setup_logging())
    assert checkpoint["years"]["2016"]["status"] == "complete"
    assert checkpoint["years"]["2017"]["status"] == "invalidated"
    assert not runner.out_path(2017).exists()
    assert runner.out_path(2016).exists()
