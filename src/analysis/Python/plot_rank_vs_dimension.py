#!/usr/bin/env python3
"""
绘制模型排名 vs 维度 的热力图
颜色表示 BT Strength
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

def shorten_name(name):
    """缩短模型名称用于显示"""
    name_lower = str(name).lower()
    if 'gemini' in name_lower: return 'GEM'
    if 'target' in name_lower: return 'TGT'
    if 'gpt' in name_lower: return 'GPT'
    if 'doubao' in name_lower: return 'DB'
    if 'perplexity' in name_lower or 'pplx' in name_lower: return 'PPLX'
    return name[:4].upper()

def main():
    parser = argparse.ArgumentParser(description="Plot Rank vs Dimension heatmap")
    parser.add_argument("--input", default="outputs/BT_ANALYSIS_REPORT.csv", help="Input CSV file")
    parser.add_argument("--output", default="outputs/rank_vs_dimension.png", help="Output PNG file")
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.")
        return

    # Filter for game_x_dimension
    df_plot = df[df['category'] == 'game_x_dimension'].copy()
    
    if df_plot.empty:
        print("No data found for category 'game_x_dimension'.")
        return

    # Get unique games and dimensions
    games = sorted(df_plot['game_id'].unique())
    dims = sorted(df_plot['eval_dim_key'].unique())
    
    print(f"Games: {games}")
    print(f"Dimensions: {dims}")

    # Determine global min/max strength for consistent color scaling
    vmin = df_plot['bt_strength'].min()
    vmax = df_plot['bt_strength'].max()

    # Setup plot
    n_games = len(games)
    fig, axes = plt.subplots(1, n_games, figsize=(5 * n_games, 6), sharey=True)
    if n_games == 1:
        axes = [axes]

    for i, game in enumerate(games):
        ax = axes[i]
        game_df = df_plot[df_plot['game_id'] == game]
        
        # Pivot tables
        # Rows: Dimension, Cols: Rank
        pivot_strength = game_df.pivot(index='eval_dim_key', columns='rank', values='bt_strength')
        pivot_model = game_df.pivot(index='eval_dim_key', columns='rank', values='model')
        
        # Ensure all ranks and dimensions are present
        # Ranks should be 1, 2, 3, 4 (assuming 4 models)
        max_rank = int(df_plot['rank'].max())
        all_ranks = range(1, max_rank + 1)
        pivot_strength = pivot_strength.reindex(columns=all_ranks, index=dims)
        pivot_model = pivot_model.reindex(columns=all_ranks, index=dims)
        
        # Shorten names
        pivot_model_short = pivot_model.applymap(lambda x: shorten_name(x) if pd.notnull(x) else "")
        
        # Plot heatmap
        sns.heatmap(pivot_strength, ax=ax, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                    annot=pivot_model_short, fmt='', cbar=False,
                    linewidths=0.5, linecolor='gray', annot_kws={"size": 12, "weight": "bold"})
        
        ax.set_title(f"Game {game}", fontsize=14, pad=10)
        ax.set_xlabel("Rank", fontsize=12)
        if i == 0:
            ax.set_ylabel("Dimension", fontsize=12)
        else:
            ax.set_ylabel("")
            
        # Rotate y-labels for better readability
        if i == 0:
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Add common colorbar
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4]) # [left, bottom, width, height]
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label='BT Strength')

    plt.suptitle("Model Rank vs Dimension (color = BT strength)", fontsize=16, y=1.05)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    main()
