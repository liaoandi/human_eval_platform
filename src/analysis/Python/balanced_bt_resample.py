#!/usr/bin/env python3
"""
Simulate balanced game sampling (1:1:1) for Bradley-Terry rankings.

Usage example:
    python balanced_bt_resample.py \
        --matrix /path/to/winrate_matrix.csv \
        --output-dir outputs/balanced_bt_1to1 \
        --strategy upsample_to_max \
        --game-ids game_id_1,game_id_2,game_id_3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from run_bt_from_matrix import (
	calculate_bt_from_aggregated_data,
	reconstruct_matches_from_matrix,
)


DEFAULT_GAME_IDS = ["game_id_1", "game_id_2", "game_id_3"]  # game_a, game_b, game_c
GAME_NAME_MAP = {
	"game_id_1": "game_a",
	"game_id_2": "game_b",
	"game_id_3": "game_c",
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Balance game samples (1:1:1) and recompute BT rankings"
	)
	parser.add_argument(
		"--matrix",
		type=Path,
		required=True,
		help="Path to winrate matrix CSV (output of winrate_matrix_stratified.sql)",
	)
	parser.add_argument(
		"--game-ids",
		type=str,
		default=",".join(DEFAULT_GAME_IDS),
		help="Comma separated game_id list to balance (default: game_a,game_b,game_c)",
	)
	parser.add_argument(
		"--strategy",
		type=str,
		choices=("upsample_to_max", "downsample_to_min"),
		default="upsample_to_max",
		help="Resampling strategy. upsample_to_max boosts smaller games to match the largest.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("outputs/balanced_bt_simulation"),
		help="Directory to store CSV outputs",
	)
	return parser.parse_args()


def load_matrix(matrix_path: Path, target_games: List[str]) -> pd.DataFrame:
	df = pd.read_csv(matrix_path)
	df["game_id"] = df["game_id"].astype(str)
	return df[df["game_id"].isin(target_games)].reset_index(drop=True)


def balance_games(
	matches_agg: pd.DataFrame,
	target_games: List[str],
	strategy: str,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
	matches = matches_agg.copy()
	matches["game_id"] = matches["game_id"].astype(str)
	for col in ("left_wins", "right_wins", "total_matches"):
		matches[col] = matches[col].astype(float)
	game_totals = matches.groupby("game_id")["total_matches"].sum()
	game_totals = game_totals[target_games]
	if strategy == "upsample_to_max":
		target_total = game_totals.max()
	else:
		target_total = game_totals.min()
	scale_map: Dict[str, float] = {}
	for game_id in target_games:
		source_total = game_totals.loc[game_id]
		scale = target_total / source_total if source_total > 0 else 0.0
		scale_map[game_id] = scale
		mask = matches["game_id"] == game_id
		for col in ("left_wins", "right_wins", "total_matches"):
			matches.loc[mask, col] = matches.loc[mask, col] * scale
	summary = pd.DataFrame({
		"game_id": target_games,
		"game_name": [GAME_NAME_MAP.get(gid, gid) for gid in target_games],
		"original_matches": [game_totals.loc[gid] for gid in target_games],
		"target_matches": target_total,
		"scale_factor": [scale_map[gid] for gid in target_games],
	})
	return matches, scale_map, summary


def run_bt_rankings(matches: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	global_rank = calculate_bt_from_aggregated_data(matches)
	by_game = calculate_bt_from_aggregated_data(matches, group_by="game_id")
	by_dim = calculate_bt_from_aggregated_data(matches, group_by="eval_dim_key")
	return global_rank, by_game, by_dim


def main():
	args = parse_args()
	target_games = [gid.strip() for gid in args.game_ids.split(",") if gid.strip()]
	matrix_df = load_matrix(args.matrix, target_games)
	if matrix_df.empty:
		raise ValueError("No rows left after filtering to specified game_ids.")
	matches_agg = reconstruct_matches_from_matrix(matrix_df, use_stratified=True)
	balanced_matches, scale_map, balance_summary = balance_games(
		matches_agg, target_games, args.strategy
	)
	global_rank, by_game_rank, by_dim_rank = run_bt_rankings(balanced_matches)
	args.output_dir.mkdir(parents=True, exist_ok=True)
	global_rank.to_csv(args.output_dir / "bt_rankings_balanced_global.csv", index=False)
	by_game_rank.to_csv(args.output_dir / "bt_rankings_balanced_by_game.csv", index=False)
	by_dim_rank.to_csv(args.output_dir / "bt_rankings_balanced_by_dim.csv", index=False)
	balance_summary.to_csv(args.output_dir / "balance_summary.csv", index=False)
	print("Resampling factors:")
	print(balance_summary.to_string(index=False))
	print("\nTop-5 global rankings after balancing:")
	print(global_rank.head(5).to_string(index=False))


if __name__ == "__main__":
	main()

