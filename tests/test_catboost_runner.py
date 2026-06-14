from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.datasets import make_classification


def test_model_save_reload_prediction_match(tmp_path) -> None:
    x, y = make_classification(n_samples=100, n_features=6, random_state=42)
    model = CatBoostClassifier(iterations=5, task_type="CPU", verbose=False, allow_writing_files=False, random_seed=42)
    model.fit(x, y)
    path = tmp_path / "model.cbm"
    model.save_model(path)
    loaded = CatBoostClassifier()
    loaded.load_model(path)
    assert (model.predict_proba(x[:5]) == loaded.predict_proba(x[:5])).all()
