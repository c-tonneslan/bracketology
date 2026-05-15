"""Monte Carlo head-to-head simulator.

Given a list of teams (one per line, matched against ESPN's display names),
the simulator pairs them in order and runs `n_sims` random tournaments where
each game is decided by a Bernoulli draw from the model's predicted P(A wins).
Reports per-team probabilities of advancing to each round and winning it all.

Usage:
  python scripts/simulate.py --teams sample_bracket.txt --n 10000
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

FEATS = [
    "d_net_eff", "d_off_eff", "d_def_eff", "d_three_par",
    "d_ft_rate", "d_tov_pct", "d_oreb_pct", "d_win_pct", "d_avg_margin",
    "d_pace", "a_home",
]


def predict(model, a, b, neutral=True):
    """Score A vs B from team-season feature rows. Returns P(A wins)."""
    diff = {
        "d_net_eff": a["net_eff"] - b["net_eff"],
        "d_off_eff": a["off_eff"] - b["off_eff"],
        "d_def_eff": a["def_eff"] - b["def_eff"],
        "d_three_par": a["three_par"] - b["three_par"],
        "d_ft_rate": a["ft_rate"] - b["ft_rate"],
        "d_tov_pct": a["tov_pct"] - b["tov_pct"],
        "d_oreb_pct": a["oreb_pct"] - b["oreb_pct"],
        "d_win_pct": a["win_pct"] - b["win_pct"],
        "d_avg_margin": a["avg_margin"] - b["avg_margin"],
        "d_pace": a["poss"] - b["poss"],
        "a_home": 0 if neutral else 1,
    }
    row = np.array([[diff[f] for f in FEATS]], dtype=float)
    return float(model.predict_proba(row)[0, 1])


def simulate(teams_df, model, n_sims, rng):
    n = len(teams_df)
    rounds = int(np.log2(n))
    if 2**rounds != n:
        raise SystemExit(f"need a power-of-two bracket, got {n} teams")
    team_names = teams_df["team_name"].tolist()
    advance_counts = np.zeros((rounds + 1, n), dtype=int)
    advance_counts[0] = n_sims  # everyone "advances" past round 0

    # precompute every possible matchup probability (n^2 cells, fine for 64)
    probs = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            probs[i][j] = predict(model, teams_df.iloc[i], teams_df.iloc[j], neutral=True)

    for s in range(n_sims):
        alive = list(range(n))
        for r in range(rounds):
            next_alive = []
            for k in range(0, len(alive), 2):
                a, b = alive[k], alive[k + 1]
                p = probs[a][b]
                winner = a if rng.random() < p else b
                next_alive.append(winner)
            alive = next_alive
            for w in alive:
                advance_counts[r + 1][w] += 1

    rows = []
    for i in range(n):
        rows.append({
            "team": team_names[i],
            "first_round": 1.0,
            **{f"r_{r+1}": advance_counts[r + 1][i] / n_sims for r in range(rounds)},
        })
    return pd.DataFrame(rows).sort_values(f"r_{rounds}", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teams", required=True,
                    help="text file with one team per line (16, 32, or 64)")
    ap.add_argument("--season", type=int, default=2024,
                    help="which season's ratings to use")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not os.path.exists(args.teams):
        sys.exit(f"missing {args.teams}")
    with open(args.teams) as f:
        names = [ln.strip() for ln in f if ln.strip()]

    teams = pd.read_parquet("data/games_teams.parquet")
    teams = teams[teams["season"] == args.season]
    # match team_display_name from the box rows
    teams = teams.set_index("team_name")
    missing = [n for n in names if n not in teams.index]
    if missing:
        sys.exit(f"team(s) not found in season {args.season}: {missing}")
    bracket = teams.loc[names].reset_index()

    model = xgb.XGBClassifier()
    model.load_model("models/xgboost.json")

    rng = np.random.default_rng(args.seed)
    result = simulate(bracket, model, args.n, rng)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
