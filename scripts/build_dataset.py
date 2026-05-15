"""Build team-season stats and game-level training data.

For each (team, season) we compute regular-season averages:
  off_eff, def_eff, pace, ortg_oreb%, ortg_3par, tov%, ft_rate.

Then for each game we attach the offense team's and defense team's preseason
to-date stats and label the row 1 if the offense won, else 0.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


REG_TYPE = 2  # season_type 2 = regular season


def estimate_possessions(row):
    """Dean Oliver-flavored possession estimate from box-score columns."""
    return (
        row["field_goals_attempted"]
        + 0.44 * row["free_throws_attempted"]
        - row["offensive_rebounds"]
        + row["turnovers"]
    )


def build(data_dir, out_path):
    files = sorted(glob.glob(os.path.join(data_dir, "team_box_*.parquet")))
    if not files:
        raise SystemExit(f"no team_box files in {data_dir}")
    print(f"reading {len(files)} season files")

    cols = [
        "game_id", "season", "season_type", "game_date",
        "team_id", "team_display_name",
        "team_score", "team_winner", "team_home_away",
        "field_goals_attempted", "field_goals_made",
        "three_point_field_goals_attempted", "three_point_field_goals_made",
        "free_throws_attempted", "free_throws_made",
        "offensive_rebounds", "defensive_rebounds",
        "turnovers", "total_rebounds",
        "opponent_team_id", "opponent_team_display_name", "opponent_team_score",
    ]
    frames = [pd.read_parquet(f, columns=cols) for f in files]
    raw = pd.concat(frames, ignore_index=True)
    print(f"  rows: {len(raw):,}")

    raw["poss"] = raw.apply(estimate_possessions, axis=1)
    raw["off_eff"] = 100 * raw["team_score"] / raw["poss"]
    raw["def_eff"] = 100 * raw["opponent_team_score"] / raw["poss"]
    raw["margin"] = raw["team_score"] - raw["opponent_team_score"]
    raw["three_par"] = raw["three_point_field_goals_attempted"] / raw["field_goals_attempted"]
    raw["ft_rate"] = raw["free_throws_attempted"] / raw["field_goals_attempted"]
    raw["tov_pct"] = raw["turnovers"] / raw["poss"]
    raw["oreb_pct"] = raw["offensive_rebounds"] / (
        raw["offensive_rebounds"] + raw["defensive_rebounds"].rolling(1).mean()
    )
    # use opponent DREB for the actual oreb% calculation, but we only have
    # team-level rows; approximate from the same row's totals
    raw["oreb_pct"] = raw["offensive_rebounds"] / np.maximum(
        raw["offensive_rebounds"] + raw["defensive_rebounds"] - 0.5, 1
    )

    # season aggregates from regular-season games only
    reg = raw[raw["season_type"] == REG_TYPE]
    print(f"  regular season rows: {len(reg):,}")
    agg = reg.groupby(["season", "team_id"]).agg(
        team_name=("team_display_name", "first"),
        games=("game_id", "size"),
        wins=("team_winner", "sum"),
        off_eff=("off_eff", "mean"),
        def_eff=("def_eff", "mean"),
        poss=("poss", "mean"),
        three_par=("three_par", "mean"),
        ft_rate=("ft_rate", "mean"),
        tov_pct=("tov_pct", "mean"),
        oreb_pct=("oreb_pct", "mean"),
        avg_margin=("margin", "mean"),
    ).reset_index()
    agg["win_pct"] = agg["wins"] / agg["games"]
    agg["net_eff"] = agg["off_eff"] - agg["def_eff"]
    agg = agg[agg["games"] >= 15]  # drop teams with weird short schedules
    print(f"  team-seasons: {len(agg):,}")

    # game-level join: one row per game with both teams' season stats
    games = raw.merge(
        agg.rename(columns=lambda c: c if c in {"season", "team_id"} else f"a_{c}"),
        on=["season", "team_id"], how="inner",
    )
    games = games.merge(
        agg.rename(columns=lambda c: c if c == "season"
                   else "team_id" if c == "team_id"
                   else f"b_{c}").rename(columns={"team_id": "opponent_team_id"}),
        on=["season", "opponent_team_id"], how="inner",
    )
    games["a_won"] = games["team_winner"].astype(int)
    games["a_home"] = (games["team_home_away"] == "home").astype(int)

    out_cols = [
        "game_id", "season", "season_type", "game_date",
        "team_id", "opponent_team_id", "a_won", "a_home",
        "a_team_name", "b_team_name", "a_win_pct", "b_win_pct",
        "a_off_eff", "a_def_eff", "a_net_eff", "a_three_par",
        "a_ft_rate", "a_tov_pct", "a_oreb_pct", "a_poss", "a_avg_margin",
        "b_off_eff", "b_def_eff", "b_net_eff", "b_three_par",
        "b_ft_rate", "b_tov_pct", "b_oreb_pct", "b_poss", "b_avg_margin",
    ]
    games = games[out_cols]
    print(f"  joined game rows: {len(games):,}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    games.to_parquet(out_path, index=False)
    agg.to_parquet(out_path.replace(".parquet", "_teams.parquet"), index=False)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="data/games.parquet")
    args = ap.parse_args()
    build(args.data, args.out)


if __name__ == "__main__":
    main()
