import json
import os

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _needs(p):
    full = os.path.join(ROOT, p)
    if not os.path.exists(full):
        pytest.skip(f"missing {p} (run `make all`)")
    return full


def test_games_parquet():
    df = pd.read_parquet(_needs("data/games.parquet"))
    assert len(df) > 10_000
    assert df["a_won"].isin([0, 1]).all()
    # team-A perspective should average close to 50% wins
    assert 0.45 < df["a_won"].mean() < 0.55


def test_teams_parquet():
    df = pd.read_parquet(_needs("data/games_teams.parquet"))
    assert len(df) > 1000
    # plausible offensive efficiency range
    assert df["off_eff"].between(60, 140).all()


def test_xgb_predicts_in_range():
    clf = xgb.XGBClassifier()
    clf.load_model(_needs("models/xgboost.json"))
    # 11 features in the same order as train.py
    # big edge at home should be a heavy favorite
    row = np.array([[15, 10, -5, 0, 0, 0, 0, 0.2, 10, 0, 1]], dtype=float)
    p = clf.predict_proba(row)[0, 1]
    assert p > 0.7
    # big edge against, away game, should be a heavy underdog
    row = np.array([[-15, -10, 5, 0, 0, 0, 0, -0.2, -10, 0, 0]], dtype=float)
    p = clf.predict_proba(row)[0, 1]
    assert p < 0.3


def test_metrics():
    with open(_needs("models/report.json")) as f:
        rep = json.load(f)
    xgb_row = next(m for m in rep["metrics"] if m["model"] == "xgboost")
    assert xgb_row["auc"] > 0.60
