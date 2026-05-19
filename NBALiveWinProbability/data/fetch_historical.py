import argparse
import os
import re
import time

import pandas as pd
from nba_api.stats.endpoints.leaguegamefinder import LeagueGameFinder
from nba_api.stats.endpoints.playbyplayv3 import PlayByPlayV3


def fetch_season_game_ids(season: str) -> list[str]:
    finder = LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    df = finder.get_data_frames()[0]
    return df["GAME_ID"].astype(str).drop_duplicates().tolist()


def fetch_pbp_for_game(game_id: str) -> pd.DataFrame | None:
    time.sleep(0.6)
    try:
        pbp = PlayByPlayV3(game_id=game_id)
        return pbp.get_data_frames()[0]
    except Exception:
        return None


def _parse_clock(clock_val) -> float:
    """Parse ISO 8601 game clock 'PT{M}M{S}.{ss}S' to seconds remaining."""
    if pd.isna(clock_val) or str(clock_val).strip() == "":
        return 0.0
    m = re.match(r"PT(\d+)M([\d.]+)S", str(clock_val))
    if not m:
        return 0.0
    try:
        return int(m.group(1)) * 60 + float(m.group(2))
    except ValueError:
        return 0.0


def _to_int_score(val) -> int:
    s = str(val).strip()
    if not s or s == "nan":
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def build_win_probability_rows(pbp_df: pd.DataFrame, game_id: str) -> list[dict]:
    df = pbp_df.copy()

    # Forward-fill scores: V3 only populates these on scoring plays
    for col in ("scoreHome", "scoreAway"):
        df[col] = df[col].replace("", pd.NA).ffill().fillna("0")

    # home_win from final score
    home_win = int(
        _to_int_score(df["scoreHome"].iloc[-1]) > _to_int_score(df["scoreAway"].iloc[-1])
    )

    rows = []
    home_fouls = 0
    away_fouls = 0
    last_possession = 0

    for _, row in df.iterrows():
        period = int(row["period"]) if pd.notna(row.get("period")) else 1
        seconds_in_period = _parse_clock(row.get("clock"))

        # OT periods clamp to 0
        if period > 4:
            seconds_remaining_in_game = 0.0
        else:
            seconds_remaining_in_game = float((4 - period) * 720 + seconds_in_period)

        score_diff = _to_int_score(row["scoreHome"]) - _to_int_score(row["scoreAway"])

        action = str(row.get("actionType", ""))
        location = str(row.get("location", ""))

        # Foul tracking: location tells us which team committed the foul
        if action == "Foul" and location in ("h", "v"):
            if location == "h":
                home_fouls += 1
            else:
                away_fouls += 1

        # Possession tracking: update on shots, FTs, rebounds, turnovers
        if action in {"Made Shot", "Missed Shot", "Free Throw", "Rebound", "Turnover"} and location in ("h", "v"):
            last_possession = 1 if location == "h" else 0

        rows.append({
            "game_id": game_id,
            "period": period,
            "seconds_remaining_in_period": float(seconds_in_period),
            "seconds_remaining_in_game": float(seconds_remaining_in_game),
            "score_diff": int(score_diff),
            "home_fouls": int(home_fouls),
            "away_fouls": int(away_fouls),
            "foul_diff": int(home_fouls - away_fouls),
            "possession": int(last_possession),
            "home_win": int(home_win),
        })

    return rows


def _flush(rows: list[dict], output_path: str):
    pd.DataFrame(rows).to_csv(
        output_path,
        mode="a",
        header=not os.path.exists(output_path),
        index=False,
    )


def fetch_and_save(season: str, output_path: str, max_games: int | None = None):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Fetching game list for {season}...")
    time.sleep(0.6)
    game_ids = fetch_season_game_ids(season)

    if max_games is not None:
        game_ids = game_ids[:max_games]

    total = len(game_ids)
    print(f"Processing {total} game(s)...")

    accumulated: list[dict] = []

    for idx, game_id in enumerate(game_ids, 1):
        if idx % 10 == 0:
            print(f"Game {idx}/{total} — {game_id}")

        pbp_df = fetch_pbp_for_game(game_id)
        if pbp_df is None:
            print(f"  Skipping {game_id}: fetch failed")
            continue

        accumulated.extend(build_win_probability_rows(pbp_df, game_id))

        if idx % 100 == 0 and accumulated:
            _flush(accumulated, output_path)
            accumulated = []
            print(f"  Checkpoint saved at game {idx}")

    if accumulated:
        _flush(accumulated, output_path)

    print(f"Done — saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical NBA play-by-play data")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Fetch only 5 games for quick verification",
    )
    args = parser.parse_args()

    fetch_and_save(
        season="2025-26",
        output_path="data/historical/pbp_2025_26.csv",
        max_games=5 if args.test else None,
    )
