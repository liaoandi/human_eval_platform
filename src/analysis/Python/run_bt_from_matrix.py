#!/usr/bin/env python3
"""
从 winrate matrix CSV 运行 Bradley-Terry 模型

功能：
1. 读取 winrate_matrix_stratified.sql 的输出 CSV
2. 重构对局数据（基于胜率和对局数）
3. 计算 Bradley-Terry 评分
4. 输出全局排名和分组排名
"""

import sys
import os
import pandas as pd
import numpy as np


def load_winrate_matrix(csv_path: str) -> pd.DataFrame:
	"""
	加载 winrate matrix CSV
	
	Args:
		csv_path: CSV 文件路径
	
	Returns:
		DataFrame
	"""
	print(f"读取文件: {csv_path}")
	df = pd.read_csv(csv_path)
	print(f"加载完成，共 {len(df)} 条记录")
	print(f"\n数据概览：")
	print(f"  - game_id 数量: {df['game_id'].nunique()}")
	print(f"  - eval_dim_key 数量: {df['eval_dim_key'].nunique()}")
	print(f"  - 模型数量: {len(set(df['row_model'].unique()) | set(df['col_model'].unique()))}")
	return df


def reconstruct_matches_from_matrix(matrix_df: pd.DataFrame, use_stratified: bool = True) -> pd.DataFrame:
	"""
	从胜率矩阵重构对局数据
	
	Args:
		matrix_df: 胜率矩阵 DataFrame
		use_stratified: 是否使用 stratified_winrate（否则用 raw_winrate）
	
	Returns:
		对局数据 DataFrame，包含列：
		- game_id
		- eval_dim_key
		- left_model_name
		- right_model_name
		- left_wins: 左侧模型赢的场数
		- right_wins: 右侧模型赢的场数
		- draws: 平局场数（假设为0）
		- total_matches
	"""
	winrate_col = 'stratified_winrate' if use_stratified else 'raw_winrate'
	
	matches = []
	for _, row in matrix_df.iterrows():
		# row_model 的胜率
		row_winrate = row[winrate_col]
		total = row['total_matches']
		
		# 计算胜负场数（假设没有平局，简化处理）
		row_wins = row_winrate * total
		col_wins = (1 - row_winrate) * total
		
		matches.append({
			'game_id': row['game_id'],
			'eval_dim_key': row['eval_dim_key'],
			'left_model_name': row['row_model'],
			'right_model_name': row['col_model'],
			'left_wins': row_wins,
			'right_wins': col_wins,
			'total_matches': total
		})
	
	return pd.DataFrame(matches)


def expand_matches_to_individual_games(matches_agg: pd.DataFrame) -> pd.DataFrame:
	"""
	将聚合的对局数据展开为单场对局（用于 BT 拟合）
	
	注意：这里做了简化，假设没有平局
	
	Args:
		matches_agg: 聚合的对局数据
	
	Returns:
		展开的对局数据，每一行代表一场对局
	"""
	individual_games = []
	
	for _, row in matches_agg.iterrows():
		game_id = row['game_id']
		eval_dim_key = row['eval_dim_key']
		left_model = row['left_model_name']
		right_model = row['right_model_name']
		left_wins = int(round(row['left_wins']))
		right_wins = int(round(row['right_wins']))
		
		# 添加左侧胜利的对局
		for _ in range(left_wins):
			individual_games.append({
				'game_id': game_id,
				'eval_dim_key': eval_dim_key,
				'left_model_name': left_model,
				'right_model_name': right_model,
				'winner': 'left'
			})
		
		# 添加右侧胜利的对局
		for _ in range(right_wins):
			individual_games.append({
				'game_id': game_id,
				'eval_dim_key': eval_dim_key,
				'left_model_name': left_model,
				'right_model_name': right_model,
				'winner': 'right'
			})
	
	return pd.DataFrame(individual_games)


def calculate_bt_from_aggregated_data(matches_agg: pd.DataFrame, group_by: str = None) -> pd.DataFrame:
	"""
	从聚合数据直接计算 BT（不展开为单场对局）
	
	使用修改版的 BT 模型，直接接受胜负统计
	
	Args:
		matches_agg: 聚合的对局数据
		group_by: 分组字段
	
	Returns:
		排名 DataFrame
	"""
	if group_by:
		all_rankings = []
		
		for group_value, group_df in matches_agg.groupby(group_by):
			print(f"\n{'='*60}")
			print(f"计算 {group_by}={group_value} 的 Bradley-Terry 评分")
			print(f"{'='*60}")
			
			model = BradleyTerryModelAggregated(max_iter=100, tol=1e-6)
			model.fit_from_aggregated(group_df)
			
			rankings = model.get_rankings()
			rankings[group_by] = group_value
			all_rankings.append(rankings)
		
		return pd.concat(all_rankings, ignore_index=True)
	else:
		print(f"\n{'='*60}")
		print("计算全局 Bradley-Terry 评分")
		print(f"{'='*60}")
		
		model = BradleyTerryModelAggregated(max_iter=100, tol=1e-6)
		model.fit_from_aggregated(matches_agg)
		
		return model.get_rankings()


class BradleyTerryModelAggregated:
	"""
	Bradley-Terry 模型（聚合数据版本）
	
	直接使用胜负统计，无需展开为单场对局
	"""
	
	def __init__(self, max_iter: int = 100, tol: float = 1e-6):
		self.max_iter = max_iter
		self.tol = tol
		self.strengths = {}
		self.models = []
		self.win_matrix = None
		self.match_matrix = None
		self.converged = False
		self.iterations = 0
	
	def fit_from_aggregated(self, matches_agg: pd.DataFrame):
		"""
		从聚合数据拟合
		
		Args:
			matches_agg: 聚合对局数据，需要包含：
				- left_model_name
				- right_model_name
				- left_wins
				- right_wins
		"""
		# 获取所有模型
		models_set = set(matches_agg['left_model_name'].unique()) | \
					 set(matches_agg['right_model_name'].unique())
		models_set.discard(None)
		self.models = sorted(list(models_set))
		n = len(self.models)
		
		if n == 0:
			print("警告：没有有效的模型数据")
			return
		
		print(f"模型数量: {n}")
		print(f"配对数量: {len(matches_agg)}")
		
		model_to_idx = {model: i for i, model in enumerate(self.models)}
		
		# 构建胜负矩阵
		self.win_matrix = np.zeros((n, n))
		self.match_matrix = np.zeros((n, n))
		
		for _, row in matches_agg.iterrows():
			left_model = row['left_model_name']
			right_model = row['right_model_name']
			
			if left_model not in model_to_idx or right_model not in model_to_idx:
				continue
			
			i = model_to_idx[left_model]
			j = model_to_idx[right_model]
			
			# 累加胜场和对局数
			self.win_matrix[i, j] += row['left_wins']
			self.win_matrix[j, i] += row['right_wins']
			self.match_matrix[i, j] += row['total_matches']
			self.match_matrix[j, i] += row['total_matches']
		
		# 迭代求解
		self._iterative_solve()
	
	def _iterative_solve(self):
		"""使用 MM 算法迭代求解"""
		n = len(self.models)
		strengths = np.ones(n)
		total_wins = self.win_matrix.sum(axis=1)
		
		print(f"\n开始迭代求解...")
		
		for iteration in range(self.max_iter):
			old_strengths = strengths.copy()
			
			for i in range(n):
				denominator = 0.0
				for j in range(n):
					if i != j and self.match_matrix[i, j] > 0:
						denominator += self.match_matrix[i, j] / (strengths[i] + strengths[j])
				
				if denominator > 0:
					strengths[i] = total_wins[i] / denominator
			
			# 归一化
			strengths = strengths / strengths.mean()
			
			# 检查收敛
			diff = np.abs(strengths - old_strengths).max()
			
			if (iteration + 1) % 10 == 0:
				print(f"  迭代 {iteration + 1}: 最大变化 = {diff:.6f}")
			
			if diff < self.tol:
				self.converged = True
				self.iterations = iteration + 1
				break
		
		self.strengths = {model: strengths[i] for i, model in enumerate(self.models)}
		
		if not self.converged:
			print(f"\n警告：未在 {self.max_iter} 次迭代内收敛")
		else:
			print(f"\n✓ 已收敛，迭代次数: {self.iterations}")
	
	def get_rankings(self) -> pd.DataFrame:
		"""获取排名"""
		rankings = []
		
		for model, strength in self.strengths.items():
			model_idx = self.models.index(model)
			total_matches = self.match_matrix[model_idx, :].sum()
			total_wins = self.win_matrix[model_idx, :].sum()
			
			# ELO 等效评分
			elo_rating = 1500 + 400 * np.log10(strength)
			
			rankings.append({
				'model': model,
				'bt_strength': strength,
				'elo_equivalent': elo_rating,
				'total_matches': int(total_matches),
				'total_wins': total_wins,
				'raw_winrate': total_wins / total_matches if total_matches > 0 else 0
			})
		
		df = pd.DataFrame(rankings)
		df = df.sort_values('bt_strength', ascending=False).reset_index(drop=True)
		df['rank'] = df.index + 1
		
		return df[['rank', 'model', 'bt_strength', 'elo_equivalent', 
				   'total_matches', 'total_wins', 'raw_winrate']]


def main():
	"""主函数"""
	import sys
	
	if len(sys.argv) < 2:
		print("用法: python run_bt_from_matrix.py <csv_file>")
		print("示例: python run_bt_from_matrix.py DataWorks_数据开发_20251111200257_0.csv")
		sys.exit(1)
	
	csv_path = sys.argv[1]
	
	# 加载数据
	matrix_df = load_winrate_matrix(csv_path)
	
	# 重构对局数据
	print(f"\n重构对局数据（使用 stratified_winrate）...")
	matches_agg = reconstruct_matches_from_matrix(matrix_df, use_stratified=True)
	print(f"重构完成，共 {len(matches_agg)} 个配对")
	
	# 全局评分
	print("\n" + "="*80)
	print("全局 Bradley-Terry 评分")
	print("="*80)
	
	all_matches = matches_agg.copy()
	rankings_global = calculate_bt_from_aggregated_data(all_matches)
	print("\n全局排名：")
	print(rankings_global.to_string(index=False))
	
	# 按 game_id 分组
	print("\n" + "="*80)
	print("按 game_id 分组的 Bradley-Terry 评分")
	print("="*80)
	
	rankings_by_game = calculate_bt_from_aggregated_data(
		matches_agg[matches_agg['game_id'] != 'all'],
		group_by='game_id'
	)
	
	for game_id in sorted(rankings_by_game['game_id'].unique()):
		print(f"\n{game_id}:")
		game_rankings = rankings_by_game[rankings_by_game['game_id'] == game_id]
		print(game_rankings.to_string(index=False))
	
	# 按 eval_dim_key 分组
	print("\n" + "="*80)
	print("按 eval_dim_key 分组的 Bradley-Terry 评分")
	print("="*80)
	
	rankings_by_dim = calculate_bt_from_aggregated_data(
		matches_agg[matches_agg['game_id'] == 'all'],
		group_by='eval_dim_key'
	)
	
	for dim_key in sorted(rankings_by_dim['eval_dim_key'].unique()):
		print(f"\n{dim_key}:")
		dim_rankings = rankings_by_dim[rankings_by_dim['eval_dim_key'] == dim_key]
		print(dim_rankings.to_string(index=False))
	
	# 保存结果
	output_dir = "outputs"
	os.makedirs(output_dir, exist_ok=True)
	
	rankings_global.to_csv(f"{output_dir}/bt_rankings_from_matrix_global.csv", index=False)
	rankings_by_game.to_csv(f"{output_dir}/bt_rankings_from_matrix_by_game.csv", index=False)
	rankings_by_dim.to_csv(f"{output_dir}/bt_rankings_from_matrix_by_dim.csv", index=False)
	
	print(f"\n结果已保存到 {output_dir}/")
	print("  - bt_rankings_from_matrix_global.csv")
	print("  - bt_rankings_from_matrix_by_game.csv")
	print("  - bt_rankings_from_matrix_by_dim.csv")


if __name__ == "__main__":
	main()

