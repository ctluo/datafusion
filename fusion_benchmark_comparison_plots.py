# -*- coding: utf-8 -*-
"""
为4个LTV_CN融合基准结果文件生成简化对比图：
  - 每个文件生成2x2子图（RMSE, MAE第一排；R2, MeanRelErr(%)第二排）
  - 输出PNG和EPS两种格式
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ---------- Global style ----------
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# Algorithm colors (academic style)
ALGO_COLORS = {
    'MF-IDW':          '#4C72B0',  # blue
    'ConvexHull-GP':   '#DD8452',  # orange
    'GPy-CoKriging':   '#55A868',  # green
    'GPy-InvDistMean': '#C44E52',  # red
}

ALGO_ORDER = ['MF-IDW', 'ConvexHull-GP', 'GPy-CoKriging', 'GPy-InvDistMean']

# File config: (filename, chart title, output prefix)
FILES = [
    ('fusion_benchmark_results.xlsx',       'DataFusion Algorithm Benchmark (LF: 1000 samples | HF Train: 10% | HF Test: 90%)', 'fusion_benchmark'),
    ('fusion_benchmark_20pct_results.xlsx', 'DataFusion Algorithm Benchmark (LF: 1000 samples | HF Train: 20% | HF Test: 80%)', 'fusion_benchmark_20pct'),
    ('fusion_benchmark_50pct_results.xlsx', 'DataFusion Algorithm Benchmark (LF: 1000 samples | HF Train: 50% | HF Test: 50%)', 'fusion_benchmark_50pct'),
    ('fusion_benchmark_80pct_results.xlsx', 'DataFusion Algorithm Benchmark (LF: 1000 samples | HF Train: 80% | HF Test: 20%)', 'fusion_benchmark_80pct'),
]

METRICS = [
    ('RMSE',           'RMSE',          'lower_better'),
    ('MAE',            'MAE',           'lower_better'),
    ('R2',             r'$R^2$',        'higher_better'),
    ('MeanRelErr(%)',  'Avg Rel Err (%)', 'lower_better'),
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def draw_bar(ax, df, metric_col, metric_label, direction):
    """在指定ax上绘制单指标柱状图"""
    algos = ALGO_ORDER
    values = [df.loc[df['Algorithm'] == a, metric_col].values[0] for a in algos]
    colors = [ALGO_COLORS[a] for a in algos]

    x = np.arange(len(algos))
    bars = ax.bar(x, values, color=colors, width=0.6, edgecolor='white', linewidth=0.8, zorder=3)

    # 数值标注 — offset based on data range, not max value
    ymin, ymax = min(values), max(values)
    data_range = ymax - ymin if ymax != ymin else ymax * 0.1
    offset = data_range * 0.04
    for bar, val in zip(bars, values):
        if metric_col == 'R2':
            text = f'{val:.4f}'
        elif metric_col == 'MeanRelErr(%)':
            text = f'{val:.2f}'
        else:
            text = f'{val:.4f}'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                text, ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize=14, rotation=20, ha='right')
    ax.set_ylabel(metric_label, fontsize=14)
    ax.set_title(metric_label, fontsize=16, fontweight='bold', pad=8)

    # Y轴范围留白
    if direction == 'higher_better':
        ax.set_ylim(ymin - data_range * 0.15, ymax + data_range * 0.40)
    else:
        ax.set_ylim(0, ymax + data_range * 0.25 if ymax > 0 else 1)

    ax.grid(axis='y', linestyle='--', color='#CCCCCC', linewidth=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def generate_chart(xlsx_path, title, out_prefix):
    df = pd.read_excel(xlsx_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 13))
    fig.suptitle(title,
                 fontsize=16, fontweight='bold', y=0.97)

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (row, col), (metric_col, metric_label, direction) in zip(positions, METRICS):
        draw_bar(axes[row][col], df, metric_col, metric_label, direction)

    plt.subplots_adjust(top=0.91, bottom=0.07, left=0.08, right=0.95, hspace=0.45, wspace=0.3)

    png_path = os.path.join(BASE_DIR, f'{out_prefix}_comparison.png')
    eps_path = os.path.join(BASE_DIR, f'{out_prefix}_comparison.eps')

    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(eps_path, format='eps', bbox_inches='tight')
    plt.close(fig)

    print(f'  [OK] {png_path}')
    print(f'  [OK] {eps_path}')
    return png_path, eps_path


def main():
    print('Generating comparison charts...')
    all_outputs = []
    for fname, title, out_prefix in FILES:
        fpath = os.path.join(BASE_DIR, fname)
        if not os.path.exists(fpath):
            print(f'  [SKIP] File not found: {fpath}')
            continue
        print(f'\nProcessing: {fname}')
        png, eps = generate_chart(fpath, title, out_prefix)
        all_outputs.extend([png, eps])

    print(f'\nGenerated {len(all_outputs)} files:')
    for p in all_outputs:
        print(f'  {p}')


if __name__ == '__main__':
    main()
