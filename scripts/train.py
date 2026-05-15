"""Train a head-to-head game prediction model.

Each row in the training table is one team's view of one game (so every game
shows up twice, once as A-versus-B and once as B-versus-A). We deduplicate to
one row per game to keep labels balanced, and split by season: train on
2018-2022 (regular season + postseason), test on 2023-2024 postseason.

The features are the difference between the two teams' regular-season stats
(net_eff diff, pace diff, three_par diff, etc.). Difference framing makes the
model invariant to which team is "A" vs "B".
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DIFF_FEATS = [
    "net_eff", "off_eff", "def_eff", "three_par",
    "ft_rate", "tov_pct", "oreb_pct", "win_pct", "avg_margin",
]


def featurize(df):
    out = pd.DataFrame()
    for f in DIFF_FEATS:
        out[f"d_{f}"] = df[f"a_{f}"] - df[f"b_{f}"]
    out["d_pace"] = df["a_poss"] - df["b_poss"]
    out["a_home"] = df["a_home"]
    return out


def evaluate(name, y, p):
    return {
        "model": name,
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/games.parquet")
    ap.add_argument("--out-dir", default="models")
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    # deduplicate: pick the row where a_team_id < b_team_id so each game has one row
    df = df[df["team_id"] < df["opponent_team_id"]].reset_index(drop=True)
    df = df.dropna()
    print(f"unique games: {len(df):,}")

    # train: regular season + tournament 2018-2022; test: tournament 2023+2024
    train_mask = df["season"] <= 2022
    test_mask = (df["season"] >= 2023) & (df["season_type"] == 3)
    train = df[train_mask].reset_index(drop=True)
    test = df[test_mask].reset_index(drop=True)
    print(f"  train: {len(train):,}  test (tournament 2023-2024): {len(test):,}")

    Xtr = featurize(train)
    Xte = featurize(test)
    ytr = train["a_won"].values
    yte = test["a_won"].values

    feats = Xtr.columns.tolist()
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    # naive: home team always wins (about 60% of regular-season games)
    results.append(evaluate("constant_0.5", yte, np.full_like(yte, 0.5, dtype=float)))
    results.append(evaluate("home_advantage", yte,
                            np.where(Xte["a_home"] == 1, 0.62,
                                     np.where(Xte["a_home"] == 0, 0.38, 0.5))))

    # net_eff differential alone (the kenpom-lite baseline)
    p_net = 1 / (1 + np.exp(-0.07 * Xte["d_net_eff"].values))
    results.append(evaluate("net_eff_only_logit", yte, p_net))

    # full logistic
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr.values)
    Xte_s = scaler.transform(Xte.values)
    logit = LogisticRegression(max_iter=2000)
    logit.fit(Xtr_s, ytr)
    p_logit = logit.predict_proba(Xte_s)[:, 1]
    results.append(evaluate("logistic", yte, p_logit))

    # xgboost
    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=-1,
    )
    xgb.fit(Xtr.values, ytr, eval_set=[(Xte.values, yte)], verbose=False)
    p_xgb = xgb.predict_proba(Xte.values)[:, 1]
    results.append(evaluate("xgboost", yte, p_xgb))

    print()
    print(f"{'model':<22} {'log_loss':>10} {'brier':>10} {'auc':>8}")
    for r in results:
        print(f"{r['model']:<22} {r['log_loss']:>10.4f} {r['brier']:>10.4f} {r['auc']:>8.4f}")

    imp = sorted(zip(feats, xgb.feature_importances_), key=lambda kv: -kv[1])
    print("\nxgb feature importance:")
    for n, v in imp:
        print(f"  {n:<14} {v:.3f}")

    cal = {}
    for label, p in [("xgb", p_xgb), ("logit", p_logit), ("net_eff", p_net)]:
        pt, pp = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        cal[label] = {"pred": pp.tolist(), "true": pt.tolist()}

    xgb.save_model(os.path.join(args.out_dir, "xgboost.json"))

    test_out = test.copy()
    test_out["p_xgb"] = p_xgb
    test_out["p_logit"] = p_logit
    test_out.to_parquet(os.path.join(args.out_dir, "test_predictions.parquet"))

    # also score every game for the simulator
    full_X = featurize(df)
    df_out = df.copy()
    df_out["p_xgb"] = xgb.predict_proba(full_X.values)[:, 1]
    df_out.to_parquet(os.path.join(args.out_dir, "all_predictions.parquet"))

    report = {
        "metrics": results,
        "feature_importance": [{"feature": k, "importance": float(v)} for k, v in imp],
        "calibration": cal,
        "n_train": len(train),
        "n_test": len(test),
    }
    with open(os.path.join(args.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
