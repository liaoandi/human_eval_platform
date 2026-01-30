#!/usr/bin/env python3
"""
生成 Bradley-Terry 评分的 Markdown 报告：
- 全局排名
- 按 game_id 的排名
- 按评估维度的排名
- game × 评估维度 组合排名
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from run_bt_from_matrix import (
	reconstruct_matches_from_matrix,
	BradleyTerryModelAggregated
)


DEFAULT_MATRIX_PATH = "/Users/antonio/Downloads/DataWorks_数据开发_20251112161150_0.csv"
REPORT_PATH_MD = Path("outputs/BT_ANALYSIS_REPORT.md")
REPORT_PATH_CSV = Path("outputs/BT_ANALYSIS_REPORT.csv")


def df_to_markdown_table(
	df: pd.DataFrame,
	round_map: Optional[Dict[str, int]] = None,
	default_decimals: int = 3
) -> str:
	"""
	将 DataFrame 渲染为 Markdown 表格，支持为不同列指定保留小数位。
	"""
	display_df = df.copy()
	for col in display_df.columns:
		if pd.api.types.is_float_dtype(display_df[col]):
			decimals = round_map.get(col, default_decimals) if round_map else default_decimals
			display_df[col] = display_df[col].map(lambda x: f"{x:.{decimals}f}")
	
	# 生成 Markdown 表格
	lines = []
	# 表头
	header = "| " + " | ".join(str(col) for col in display_df.columns) + " |"
	lines.append(header)
	# 分隔线
	separator = "|" + "|".join([" --- " for _ in display_df.columns]) + "|"
	lines.append(separator)
	# 数据行
	for _, row in display_df.iterrows():
		row_line = "| " + " | ".join(str(val) for val in row) + " |"
		lines.append(row_line)
	
	return "\n" + "\n".join(lines) + "\n"


def compute_game_dimension_rankings(matches_agg: pd.DataFrame) -> pd.DataFrame:
	"""
	对每个 game × eval_dim_key 组合拟合 BT，并返回拼接后的排名结果。
	"""
	records = []
	group_cols = ['game_id', 'eval_dim_key']
	for (game_id, dim_key), group in matches_agg.groupby(group_cols):
		if game_id == 'all' or group.empty:
			continue
		model = BradleyTerryModelAggregated(max_iter=200, tol=1e-6)
		model.fit_from_aggregated(group)
		rankings = model.get_rankings()
		rankings['game_id'] = game_id
		rankings['eval_dim_key'] = dim_key
		records.append(rankings)
	
	if records:
		return pd.concat(records, ignore_index=True)
	
	return pd.DataFrame(
		columns=[
			'rank', 'model', 'bt_strength', 'elo_equivalent',
			'total_matches', 'total_wins', 'raw_winrate',
			'game_id', 'eval_dim_key'
		]
	)


def build_markdown_report(
	bt_global: pd.DataFrame,
	bt_by_game: pd.DataFrame,
	bt_by_dim: pd.DataFrame,
	game_dim_rankings: pd.DataFrame,
	matrix_path: Path
) -> str:
	timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	md_lines = [
		"# Bradley-Terry 分析报告",
		"",
		f"- 生成时间：{timestamp}",
		f"- 胜率矩阵来源：{matrix_path}",
		""
	]
	
	# 全局排名
	md_lines.append("## 全局排名")
	global_table = bt_global[
		['rank', 'model', 'elo_equivalent', 'bt_strength',
		 'total_matches', 'total_wins', 'raw_winrate']
	].copy()
	global_table['total_matches'] = global_table['total_matches'].round().astype(int)
	global_table['total_wins'] = global_table['total_wins'].round(1)
	md_lines.append(df_to_markdown_table(
		global_table,
		round_map={
			'elo_equivalent': 1,
			'bt_strength': 3,
			'raw_winrate': 3,
			'total_wins': 1
		}
	))
	
	# 按 game_id 排名
	md_lines.append("## 按 game_id 的排名")
	for game_id in sorted(bt_by_game['game_id'].unique()):
		md_lines.append(f"### game_id = {game_id}")
		game_table = bt_by_game[bt_by_game['game_id'] == game_id][[
			'rank', 'model', 'elo_equivalent', 'bt_strength',
			'total_matches', 'total_wins', 'raw_winrate'
		]].copy()
		game_table['total_matches'] = game_table['total_matches'].round().astype(int)
		game_table['total_wins'] = game_table['total_wins'].round(1)
		md_lines.append(df_to_markdown_table(
			game_table,
			round_map={
				'elo_equivalent': 1,
				'bt_strength': 3,
				'raw_winrate': 3,
				'total_wins': 1
			}
		))
	
	# 按评估维度排名 (game_id='all')
	md_lines.append("## 按评估维度的全局排名（game_id = 'all'）")
	for dim_key in sorted(bt_by_dim['eval_dim_key'].unique()):
		md_lines.append(f"### 评估维度：{dim_key}")
		dim_table = bt_by_dim[bt_by_dim['eval_dim_key'] == dim_key][[
			'rank', 'model', 'elo_equivalent', 'bt_strength',
			'total_matches', 'total_wins', 'raw_winrate'
		]].copy()
		dim_table['total_matches'] = dim_table['total_matches'].round().astype(int)
		dim_table['total_wins'] = dim_table['total_wins'].round(1)
		md_lines.append(df_to_markdown_table(
			dim_table,
			round_map={
				'elo_equivalent': 1,
				'bt_strength': 3,
				'raw_winrate': 3,
				'total_wins': 1
			}
		))
	
	# game × 评估维度
	if not game_dim_rankings.empty:
		md_lines.append("## game × 评估维度 组合排名")
		for game_id in sorted(game_dim_rankings['game_id'].unique()):
			md_lines.append(f"### game_id = {game_id}")
			for dim_key in sorted(game_dim_rankings[game_dim_rankings['game_id'] == game_id]['eval_dim_key'].unique()):
				combo_table = game_dim_rankings[
					(game_dim_rankings['game_id'] == game_id) &
					(game_dim_rankings['eval_dim_key'] == dim_key)
				][[
					'rank', 'model', 'elo_equivalent', 'bt_strength',
					'total_matches', 'total_wins', 'raw_winrate'
				]].copy()
				combo_table['total_matches'] = combo_table['total_matches'].round().astype(int)
				combo_table['total_wins'] = combo_table['total_wins'].round(1)
				md_lines.append(f"#### 维度：{dim_key}")
				md_lines.append(df_to_markdown_table(
					combo_table,
					round_map={
						'elo_equivalent': 1,
						'bt_strength': 3,
						'raw_winrate': 3,
						'total_wins': 1
					}
				))
		md_lines.append("## game × 评估维度 组合排名")
		md_lines.append("_当前数据集中缺少对应组合，未生成结果。_")
	
	return "\n".join(md_lines)


def load_dataframes(matrix_path: Path):
	bt_global = pd.read_csv("outputs/bt_rankings_from_matrix_global.csv")
	bt_by_game = pd.read_csv("outputs/bt_rankings_from_matrix_by_game.csv")
	bt_by_dim = pd.read_csv("outputs/bt_rankings_from_matrix_by_dim.csv")
	matrix = pd.read_csv(matrix_path)
	return bt_global, bt_by_game, bt_by_dim, matrix


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="生成 Bradley-Terry Markdown 报告"
	)
	parser.add_argument(
		"--matrix",
		type=Path,
		default=DEFAULT_MATRIX_PATH,
		help="胜率矩阵 CSV 路径"
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=REPORT_PATH_MD,
		help="输出报告路径"
	)
	return parser.parse_args()


def main():
	args = parse_args()
	
	# 加载数据
	bt_global, bt_by_game, bt_by_dim, matrix = load_dataframes(args.matrix)
	
	# 重构对局
	matches_agg = reconstruct_matches_from_matrix(matrix, use_stratified=True)
	
	# 计算 game × 维度组合
	game_dim_rankings = compute_game_dimension_rankings(matches_agg)
	
	# 生成 Markdown 报告
	report_md = build_markdown_report(
		bt_global,
		bt_by_game,
		bt_by_dim,
		game_dim_rankings,
		args.matrix
	)
	
	# 写 Markdown 文件
	args.output.parent.mkdir(parents=True, exist_ok=True)
	args.output.write_text(report_md, encoding='utf-8')
	print(f"Markdown 报告已生成：{args.output.absolute()}")
	
	# 生成 CSV 汇总
	csv_output = args.output.parent / REPORT_PATH_CSV.name
	csv_records = []
	
	# 全局排名
	for _, row in bt_global.iterrows():
		csv_records.append({
			'category': 'global',
			'game_id': 'all',
			'eval_dim_key': 'all',
			'rank': int(row['rank']),
			'model': row['model'],
			'elo_equivalent': round(row['elo_equivalent'], 1),
			'bt_strength': round(row['bt_strength'], 3),
			'total_matches': int(row['total_matches']),
			'total_wins': round(row['total_wins'], 1),
			'raw_winrate': round(row['raw_winrate'], 3)
		})
	
	# 按 game_id 排名
	for _, row in bt_by_game.iterrows():
		csv_records.append({
			'category': 'by_game',
			'game_id': row['game_id'],
			'eval_dim_key': 'all',
			'rank': int(row['rank']),
			'model': row['model'],
			'elo_equivalent': round(row['elo_equivalent'], 1),
			'bt_strength': round(row['bt_strength'], 3),
			'total_matches': int(row['total_matches']),
			'total_wins': round(row['total_wins'], 1),
			'raw_winrate': round(row['raw_winrate'], 3)
		})
	
	# 按评估维度排名
	for _, row in bt_by_dim.iterrows():
		csv_records.append({
			'category': 'by_dimension',
			'game_id': 'all',
			'eval_dim_key': row['eval_dim_key'],
			'rank': int(row['rank']),
			'model': row['model'],
			'elo_equivalent': round(row['elo_equivalent'], 1),
			'bt_strength': round(row['bt_strength'], 3),
			'total_matches': int(row['total_matches']),
			'total_wins': round(row['total_wins'], 1),
			'raw_winrate': round(row['raw_winrate'], 3)
		})
	
	# game × 维度组合
	for _, row in game_dim_rankings.iterrows():
		csv_records.append({
			'category': 'game_x_dimension',
			'game_id': row['game_id'],
			'eval_dim_key': row['eval_dim_key'],
			'rank': int(row['rank']),
			'model': row['model'],
			'elo_equivalent': round(row['elo_equivalent'], 1),
			'bt_strength': round(row['bt_strength'], 3),
			'total_matches': int(row['total_matches']),
			'total_wins': round(row['total_wins'], 1),
			'raw_winrate': round(row['raw_winrate'], 3)
		})
	
	csv_df = pd.DataFrame(csv_records)
	csv_df.to_csv(csv_output, index=False, encoding='utf-8-sig')
	print(f"CSV 汇总已生成：{csv_output.absolute()}")


if __name__ == "__main__":
	main()
