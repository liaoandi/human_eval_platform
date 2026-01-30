#!/usr/bin/env python3
"""
胜率矩阵可视化工具

功能：
1. 读取长格式的胜率矩阵数据
2. 转换为热力图矩阵格式
3. 为每个game_id和eval_dim_key组合生成热力图
4. 支持批量输出和交互式查看

使用方法：
    python plot_winrate_matrix.py --input winrate_matrix.csv --output outputs/
    python plot_winrate_matrix.py --input winrate_matrix.csv --game_id all --dimension model_style
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from pathlib import Path
import numpy as np


def load_data(file_path):
    """加载胜率矩阵数据"""
    # 自动检测分隔符
    df = pd.read_csv(file_path)
    print(f"加载数据: {len(df)} 行")
    print(f"游戏场次: {df['game_id'].unique().tolist()}")
    print(f"评测维度: {df['eval_dim_key'].unique().tolist()}")
    return df


def pivot_to_matrix(df, value_col='stratified_winrate'):
    """
    将长格式数据转换为矩阵格式
    
    Args:
        df: 包含row_model, col_model和胜率的DataFrame
        value_col: 用于填充矩阵的值列名
    
    Returns:
        pivot后的矩阵DataFrame
    """
    matrix = df.pivot(index='row_model', columns='col_model', values=value_col)
    return matrix


def plot_winrate_heatmap(matrix, title, matches_matrix=None, output_path=None, 
                          figsize=(10, 8), cmap='RdYlGn', vmin=0, vmax=1):
    """
    绘制胜率矩阵热力图
    
    Args:
        matrix: 胜率矩阵（行=row_model，列=col_model）
        title: 图表标题
        matches_matrix: 对局数矩阵（用于标注）
        output_path: 输出文件路径（None则显示）
        figsize: 图表大小
        cmap: 颜色映射
        vmin, vmax: 颜色值范围
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建标注文本：胜率 + 对局数
    if matches_matrix is not None:
        annot_text = pd.DataFrame('', index=matrix.index, columns=matrix.columns, dtype=str)
        for i in range(len(matrix.index)):
            for j in range(len(matrix.columns)):
                row_model = matrix.index[i]
                col_model = matrix.columns[j]
                if row_model in matches_matrix.index and col_model in matches_matrix.columns:
                    winrate_val = matrix.iloc[i, j]
                    matches_val = matches_matrix.loc[row_model, col_model]
                    if not pd.isna(winrate_val) and not pd.isna(matches_val):
                        annot_text.iloc[i, j] = f'{winrate_val:.2%}\n(n={int(matches_val)})'
    else:
        annot_text = matrix.applymap(lambda x: f'{x:.2%}' if not pd.isna(x) else '')
    
    # 绘制热力图
    sns.heatmap(
        matrix,
        annot=annot_text,
        fmt='',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Row Model Winrate'},
        linewidths=0.5,
        linecolor='gray',
        square=True,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Column Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Row Model', fontsize=12, fontweight='bold')
    
    # 旋转标签
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {output_path}")
        plt.close()
    else:
        plt.show()


def plot_all_combinations(df, output_dir=None, show_plots=False, value_col='stratified_winrate'):
    """
    为所有game_id和eval_dim_key组合生成热力图
    
    Args:
        df: 胜率矩阵数据
        output_dir: 输出目录
        show_plots: 是否显示图表
        value_col: 用于绘图的胜率列名
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    game_ids = sorted(df['game_id'].unique(), key=lambda x: (x != 'all', x))
    dimensions = sorted(df['eval_dim_key'].unique())
    
    print(f"\n开始生成图表...")
    print(f"游戏场次: {len(game_ids)} 个")
    print(f"评测维度: {len(dimensions)} 个")
    print(f"总计: {len(game_ids) * len(dimensions)} 张图表\n")
    
    count = 0
    for game_id in game_ids:
        for dimension in dimensions:
            # 筛选数据
            subset = df[
                (df['game_id'] == game_id) & 
                (df['eval_dim_key'] == dimension)
            ].copy()
            
            if len(subset) == 0:
                continue
            
            # 转换为矩阵
            winrate_matrix = pivot_to_matrix(subset, value_col)
            matches_matrix = pivot_to_matrix(subset, 'total_matches')
            
            # 生成标题
            title = f'Winrate Matrix ({value_col}): {dimension}\n(game_id={game_id})'
            
            # 生成文件名
            if output_dir:
                filename = f'winrate_{game_id}_{dimension}.png'
                output_path = output_dir / filename
            else:
                output_path = None
            
            # 绘图
            plot_winrate_heatmap(
                winrate_matrix,
                title=title,
                matches_matrix=matches_matrix,
                output_path=output_path,
                figsize=(12, 10)
            )
            
            count += 1
            
            # 如果需要显示，只显示前几张避免卡顿
            if show_plots and count <= 3:
                plt.show()
    
    print(f"\n完成! 共生成 {count} 张图表")


def plot_single_comparison(df, game_id='all', dimension=None, output_path=None, value_col='stratified_winrate'):
    """
    绘制单个game_id的对比（可选择特定维度或全部维度）
    
    Args:
        df: 胜率矩阵数据
        game_id: 游戏场次ID
        dimension: 评测维度（None则绘制所有维度）
        output_path: 输出路径
        value_col: 用于绘图的胜率列名
    """
    subset = df[df['game_id'] == game_id].copy()
    
    if len(subset) == 0:
        print(f"警告: 没有找到game_id={game_id}的数据")
        return
    
    if dimension:
        subset = subset[subset['eval_dim_key'] == dimension]
        if len(subset) == 0:
            print(f"警告: 没有找到dimension={dimension}的数据")
            return
        dimensions = [dimension]
    else:
        dimensions = sorted(subset['eval_dim_key'].unique())
    
    # 创建子图
    n_dims = len(dimensions)
    fig, axes = plt.subplots(1, n_dims, figsize=(12 * n_dims, 10))
    if n_dims == 1:
        axes = [axes]
    
    for idx, dim in enumerate(dimensions):
        dim_data = subset[subset['eval_dim_key'] == dim]
        
        # 转换为矩阵
        winrate_matrix = pivot_to_matrix(dim_data, value_col)
        matches_matrix = pivot_to_matrix(dim_data, 'total_matches')
        
        # 创建标注
        annot_text = pd.DataFrame('', index=winrate_matrix.index, columns=winrate_matrix.columns, dtype=str)
        for i in range(len(winrate_matrix.index)):
            for j in range(len(winrate_matrix.columns)):
                row_model = winrate_matrix.index[i]
                col_model = winrate_matrix.columns[j]
                winrate_val = winrate_matrix.iloc[i, j]
                matches_val = matches_matrix.loc[row_model, col_model]
                if not pd.isna(winrate_val) and not pd.isna(matches_val):
                    annot_text.iloc[i, j] = f'{winrate_val:.2%}\n(n={int(matches_val)})'
        
        # 绘制热力图
        sns.heatmap(
            winrate_matrix,
            annot=annot_text,
            fmt='',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            cbar_kws={'label': f'Row Winrate ({value_col})'},
            linewidths=0.5,
            linecolor='gray',
            square=True,
            ax=axes[idx]
        )
        
        axes[idx].set_title(f'{dim}', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Column Model', fontsize=12)
        axes[idx].set_ylabel('Row Model', fontsize=12)
        
        # 旋转标签
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.suptitle(f'Winrate Matrix Comparison (game_id={game_id}) - {value_col}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"保存图表: {output_path}")
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='绘制胜率矩阵热力图')
    parser.add_argument('--input', '-i', required=True, help='输入CSV文件路径')
    parser.add_argument('--output', '-o', help='输出目录（不指定则显示图表）')
    parser.add_argument('--game_id', '-g', help='指定game_id（不指定则生成全部）')
    parser.add_argument('--dimension', '-d', help='指定评测维度（不指定则生成全部）')
    parser.add_argument('--mode', '-m', choices=['all', 'single', 'batch', 'batch_comparison'], 
                        default='batch', help='模式：all/batch=所有组合(单图)，single=单个对比(多图)，batch_comparison=所有场次对比(多图)')
    parser.add_argument('--show', action='store_true', help='是否显示图表（默认仅保存）')
    parser.add_argument('--value_col', '-v', default='stratified_winrate', 
                        help='用于绘图的胜率列名（默认：stratified_winrate）')
    
    args = parser.parse_args()
    
    # 加载数据
    df = load_data(args.input)
    
    if args.mode == 'batch_comparison':
        # 为所有game_id生成对比图
        game_ids = sorted(df['game_id'].unique(), key=lambda x: (x != 'all', x))
        print(f"将为以下game_id生成对比图: {game_ids}")
        
        for gid in game_ids:
            output_path = None
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                filename = f'winrate_comparison_{gid}'
                if args.dimension:
                    filename += f'_{args.dimension}'
                filename += '.png'
                output_path = output_dir / filename
            
            print(f"正在生成 game_id={gid} 的图表...")
            plot_single_comparison(df, game_id=gid, dimension=args.dimension, 
                                   output_path=output_path, value_col=args.value_col)
            
    elif args.mode == 'all' or (args.mode == 'batch' and not args.game_id):
        # 生成所有组合
        plot_all_combinations(df, output_dir=args.output, show_plots=args.show, value_col=args.value_col)
    
    elif args.mode == 'single' or args.game_id:
        # 单个game_id的对比
        game_id = args.game_id or 'all'
        output_path = None
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            filename = f'winrate_comparison_{game_id}'
            if args.dimension:
                filename += f'_{args.dimension}'
            filename += '.png'
            output_path = output_dir / filename
        
        plot_single_comparison(df, game_id=game_id, dimension=args.dimension, 
                               output_path=output_path, value_col=args.value_col)


if __name__ == '__main__':
    main()
