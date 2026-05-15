"""Charts for the bracketology writeup.

  - calibration.png       calibration curve
  - importance.png        xgb feature importance
  - upset_distribution.png distribution of predicted P(favorite wins)
  - rating_top.png        top 25 teams by model's net efficiency rating
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def calibration_chart(report, out_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect")
    for k, color, label in [
        ("xgb", "#0a84ff", "xgboost"),
        ("logit", "#ff8c00", "logistic"),
        ("net_eff", "#888", "net eff diff only"),
    ]:
        c = report["calibration"][k]
        ax.plot(c["pred"], c["true"], marker="o", ms=4, color=color, label=label)
    ax.set_xlabel("predicted P(team A wins)")
    ax.set_ylabel("empirical win rate")
    ax.set_title("calibration on 2023-2024 tournament games (n=233)")
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def importance_chart(report, out_path):
    fi = report["feature_importance"]
    names = [f["feature"] for f in fi][::-1]
    vals = [f["importance"] for f in fi][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals, color="#0a84ff", alpha=0.85)
    ax.set_xlabel("importance")
    ax.set_title("xgboost feature importance")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#444")
    ax.set_xlim(0, max(vals) * 1.18)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def upset_distribution(out_path):
    df = pd.read_parquet("models/test_predictions.parquet")
    df["fav_prob"] = np.maximum(df["p_xgb"], 1 - df["p_xgb"])
    df["fav_won"] = np.where(df["p_xgb"] >= 0.5, df["a_won"], 1 - df["a_won"])
    # bucket by predicted favorite prob
    bins = pd.cut(df["fav_prob"], bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                  labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
    agg = df.groupby(bins, observed=True).agg(
        games=("fav_won", "size"),
        actual_win_rate=("fav_won", "mean"),
        predicted_win_rate=("fav_prob", "mean"),
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    w = 0.35
    ax.bar(x - w/2, agg["predicted_win_rate"], w, label="model predicted",
           color="#0a84ff", alpha=0.85)
    ax.bar(x + w/2, agg["actual_win_rate"], w, label="actual",
           color="#34c759", alpha=0.85)
    for i, n in enumerate(agg["games"]):
        ax.text(i, max(agg["predicted_win_rate"].iloc[i],
                       agg["actual_win_rate"].iloc[i]) + 0.02,
                f"n={n}", ha="center", fontsize=9, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index)
    ax.set_xlabel("model's confidence in the favorite")
    ax.set_ylabel("win rate")
    ax.set_title("favorite win rate vs model confidence (2023-24 tournament)")
    ax.set_ylim(0, 1.1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def ratings_chart(out_path):
    teams = pd.read_parquet("data/games_teams.parquet")
    latest = teams[teams["season"] == teams["season"].max()].copy()
    latest = latest.nlargest(25, "net_eff")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(latest["team_name"][::-1], latest["net_eff"][::-1],
            color="#0a84ff", alpha=0.85)
    ax.set_xlabel("net efficiency (pts per 100 possessions)")
    ax.set_title(f"top 25 by net efficiency, {int(latest['season'].iloc[0])} regular season")
    for i, (_, r) in enumerate(latest.iloc[::-1].iterrows()):
        ax.text(r["net_eff"] + 0.3, i, f"{r['net_eff']:.1f}",
                va="center", fontsize=9, color="#444")
    ax.set_xlim(0, latest["net_eff"].max() * 1.1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f"  wrote {out_path}")


def main():
    os.makedirs("charts", exist_ok=True)
    with open("models/report.json") as f:
        report = json.load(f)
    calibration_chart(report, "charts/calibration.png")
    importance_chart(report, "charts/importance.png")
    upset_distribution("charts/upset_distribution.png")
    ratings_chart("charts/rating_top.png")


if __name__ == "__main__":
    main()
