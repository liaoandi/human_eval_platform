#!/usr/bin/env python3
"""
胜率差异分析工具

功能：
1. 比较 raw_winrate 和 stratified_winrate 的差异
2. 统计差异分布
3. 找出偏差最大的模型对
4. 绘制差异热力图和散点图
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def analyze_differences(df, output_dir=None):
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 计算差异
    df['diff'] = df['raw_winrate'] - df['stratified_winrate']
    df['abs_diff'] = df['diff'].abs()
    
    # 总体统计
    print("\n=== 总体差异统计 (Raw - Stratified) ===")
    print(df['diff'].describe())
    
    # 找出差异最大的前10个记录
    print("\n=== 差异最大的前10个记录 (按绝对值) ===")
    top_diff = df.sort_values('abs_diff', ascending=False).head(10)
    
    cols_to_show = ['game_id', 'eval_dim_key', 'row_model', 'col_model', 'total_matches', 'raw_winrate', 'stratified_winrate', 'diff']
    print(top_diff[cols_to_show].to_string(index=False))
    
    # 散点图 comparison
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='raw_winrate', y='stratified_winrate', hue='game_id', alpha=0.6)
    
    # 添加对角线
    min_val = min(df['raw_winrate'].min(), df['stratified_winrate'].min())
    max_val = max(df['raw_winrate'].max(), df['stratified_winrate'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    
    plt.title('Raw Winrate vs Stratified Winrate')
    plt.grid(True, alpha=0.3)
    
    if output_dir:
        plt.savefig(output_dir / 'winrate_scatter.png', dpi=300)
        print(f"\n保存散点图: {output_dir / 'winrate_scatter.png'}")
    
    # 差异热力图 (针对 game_id='all')
    df_all = df[df['game_id'] == 'all'].copy()
    if len(df_all) > 0:
        dims = df_all['eval_dim_key'].unique()
        
        fig, axes = plt.subplots(1, len(dims), figsize=(6 * len(dims), 5))
        if len(dims) == 1:
            axes = [axes]
            
        for idx, dim in enumerate(dims):
            subset = df_all[df_all['eval_dim_key'] == dim]
            matrix = subset.pivot(index='row_model', columns='col_model', values='diff')
            
            sns.heatmap(matrix, annot=True, fmt='.2%', cmap='coolwarm', center=0, ax=axes[idx])
            axes[idx].set_title(f'Difference (Raw - Stratified)\n{dim}')
            
        plt.tight_layout()
        if output_dir:
            plt.savefig(output_dir / 'winrate_diff_heatmap_all.png', dpi=300)
            print(f"保存差异热力图: {output_dir / 'winrate_diff_heatmap_all.png'}")

def main():
    parser = argparse.ArgumentParser(description='分析胜率差异')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出目录')
    
    args = parser.parse_args()
    
    df = load_data(args.input)
    analyze_differences(df, args.output)

if __name__ == '__main__':
    main()
