#!/usr/bin/env python3
"""
Render tables/plots based on outputs/dim_contributions_by_game.csv.

Generates:
1. CSV tables summarizing win-rate/ELO contribution shares per opponent.
2. Stacked bar charts (PNG) showing per-game contribution composition.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize dimension contributions.")
	parser.add_argument(
		"--input",
		default=Path("outputs/dim_contributions_by_game.csv"),
		type=Path,
		help="CSV produced by dimension_contribution_by_game.py",
	)
	parser.add_argument(
		"--output_dir",
		default=Path("outputs/dim_contribution_plots"),
		type=Path,
		help="Directory for derived tables/plots.",
	)
	parser.add_argument(
		"--metric",
		choices=["winrate_share", "elo_share"],
		default="winrate_share",
		help="Primary share metric to visualize (second figure uses the other).",
	)
	return parser.parse_args()


DIMENSION_ORDER = ["model_style", "result_relevance", "result_usefulness"]


def load_data(path: Path) -> pd.DataFrame:
	df = pd.read_csv(path)
	required_cols = {
		"game_id",
		"dimension",
		"baseline_model",
		"winrate_share",
		"elo_share",
		"total_winrate_gap",
		"total_elo_gap",
	}
	missing = required_cols - set(df.columns)
	if missing:
		raise ValueError(f"Missing columns in {path}: {missing}")
	return df


def build_share_table(df: pd.DataFrame, metric: str) -> pd.DataFrame:
	table = (
		df.pivot_table(
			index=["game_id", "baseline_model"],
			columns="dimension",
			values=metric,
			aggfunc="sum",
		)
		.fillna(0.0)
	)
	table = table.reindex(columns=DIMENSION_ORDER, fill_value=0.0)
	table = (table * 100).round(2)
	table = table.reset_index()
	table.columns.name = None
	return table


def plot_by_game(
	df: pd.DataFrame,
	metric_column: str,
	metric_name: str,
	output_path: Path,
):
	games = sorted(df["game_id"].unique())
	if not games:
		return

	color_map = {
		"model_style": "#5C6AC4",      # muted indigo
		"result_relevance": "#F28F6B",  # coral orange
		"result_usefulness": "#46B29D", # teal green
	}

	n_cols = len(games)
	fig, axes = plt.subplots(
		1, n_cols, figsize=(6 * n_cols, 5), sharey=True
	)
	if n_cols == 1:
		axes = [axes]

	zero_col = (
		"winrate_share_is_zero_gap"
		if metric_column == "winrate_share"
		else "elo_share_is_zero_gap"
	)
	sign_col = (
		"winrate_share_sign" if metric_column == "winrate_share" else "elo_share_sign"
	)
	total_col = (
		"total_winrate_gap"
		if metric_column == "winrate_share"
		else "total_elo_gap"
	)

	for ax, game in zip(axes, games):
		game_df = df[df["game_id"] == game]
		zero_flags = (
			game_df.groupby("baseline_model")[zero_col].max()
			if zero_col in game_df.columns
			else pd.Series(dtype=bool)
		)
		total_gap_map = (
			game_df.groupby("baseline_model")[total_col].max()
			if total_col in game_df.columns
			else pd.Series(dtype=float)
		)
		table = (
			game_df.pivot_table(
				index="baseline_model",
				columns="dimension",
				values=metric_column,
				aggfunc="sum",
			)
			.reindex(columns=DIMENSION_ORDER, fill_value=0.0)
			.fillna(0.0)
			* 100
		).round(1)
		sign_table = (
			game_df.pivot_table(
				index="baseline_model",
				columns="dimension",
				values=sign_col,
				aggfunc="max",
			)
			.reindex(columns=DIMENSION_ORDER, fill_value=0.0)
			.reindex(index=table.index, fill_value=0.0)
			.fillna(0.0)
		)

		x = range(len(table))
		bottom_pos = [0.0] * len(table)
		bottom_neg = [0.0] * len(table)
		for dim in DIMENSION_ORDER:
			values = table[dim].values if dim in table.columns else [0.0] * len(table)
			dim_label_used = False
			for idx, baseline in enumerate(table.index):
				is_zero_gap = bool(zero_flags.get(baseline, False))
				if is_zero_gap:
					continue

				val = values[idx]
				if val <= 0:
					continue
				sign = sign_table.loc[baseline, dim] if baseline in sign_table.index else 0
				color = color_map.get(dim)
				label = dim if not dim_label_used else None

				hatch = "//" if sign < 0 else None
				ax.bar(
					[idx],
					val,
					bottom=bottom_pos[idx],
					label=label,
					color=color,
					hatch=hatch,
					edgecolor="#333333" if hatch else None,
				)
				text = f"-{val:.1f}%" if sign < 0 else f"{val:.1f}%"
				ax.text(
					idx,
					bottom_pos[idx] + val / 2,
					text,
					ha="center",
					va="center",
					color="white",
					fontsize=9,
				)
				bottom_pos[idx] += val

				if label is not None:
					dim_label_used = True

		for idx, baseline in enumerate(table.index):
			if bool(zero_flags.get(baseline, False)):
				total_gap = float(total_gap_map.get(baseline, 0.0))
				ax.add_patch(
					Rectangle(
						(idx - 0.35, -5),
						0.7,
						10,
						facecolor="#f4f4f4",
						edgecolor="#d0d0d0",
						hatch="//",
						alpha=0.8,
					)
				)
				ax.text(
					idx,
					5,
				f"≈0 gap\nΔ={total_gap:.3f}",
					ha="center",
					va="center",
					color="#666666",
					fontsize=9,
				)

		ax.set_title(f"{game}")
		ax.set_ylabel(f"{metric_name} share (%)")
		ax.set_xticks(list(x))
		ax.set_xticklabels(table.index, rotation=20, ha="right")
		ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")

	handles, labels = axes[0].get_legend_handles_labels()
	title = fig.suptitle(f"target-model vs baselines - {metric_name} composition", y=0.95)
	fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.90))
	plt.tight_layout(rect=(0, 0, 1, 0.85))
	plt.savefig(output_path, dpi=200)
	plt.close(fig)


def main():
	args = parse_args()
	df = load_data(args.input)
	args.output_dir.mkdir(parents=True, exist_ok=True)

	win_table = build_share_table(df, "winrate_share")
	elo_table = build_share_table(df, "elo_share")

	win_table.to_csv(args.output_dir / "winrate_share_by_game.csv", index=False)
	elo_table.to_csv(args.output_dir / "elo_share_by_game.csv", index=False)

	plot_by_game(df, "winrate_share", "Win-rate", args.output_dir / "winrate_share_stacked.png")
	plot_by_game(df, "elo_share", "ELO", args.output_dir / "elo_share_stacked.png")

	print(f"Saved tables/plots under {args.output_dir}")


if __name__ == "__main__":
	main()

