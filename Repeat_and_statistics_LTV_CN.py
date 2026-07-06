"""
Repeat_and_statistics_LTV.py
对4种数据融合算法进行10次重复测试（不同随机种子），统计性能指标的均值和方差。

- 低置信度数据: LTV_Low-Fidelity.xlsx  (全量, 1000条)
- 高置信度数据: LTV_High-Fidelity.xlsx (每次不同随机种子 50% 训练 / 50% 测试)
- 特征列: Mach, Alpha
- 目标列: CN
- 评估指标: RMSE / MAE / R2
- 重复次数: 10次 (seed=0~9)
"""

import sys
import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ── 路径设置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from DataFusionAlgorithms import (
    mf_idw_interpolate,
    mf_ConvexHull,
    mf_GPy_CoKriging,
    mf_GPy_inverseDistanceMean,
)

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  DataFusion 10-Repeat Statistics Benchmark  [LTV Dataset]")
print("=" * 65)

df_lf = pd.read_excel(os.path.join(SCRIPT_DIR, 'LTV_Low-Fidelity.xlsx'))
df_hf = pd.read_excel(os.path.join(SCRIPT_DIR, 'LTV_High-Fidelity.xlsx'))

FEATURES = ['Mach', 'Alpha']
TARGET   = 'CN'

X_lf_all = df_lf[FEATURES].values.astype(float)
Y_lf_all = df_lf[TARGET].values.astype(float)   # 一维，供 MF-IDW 使用
Y_lf_all_2d = Y_lf_all.reshape(-1, 1)            # 二维，供其他算法使用

X_hf_all = df_hf[FEATURES].values.astype(float)
Y_hf_all = df_hf[TARGET].values.astype(float)
Y_hf_all_2d = Y_hf_all.reshape(-1, 1)

print(f"\n数据概览:")
print(f"  低置信度 (LF) : {X_lf_all.shape[0]} 条  (每次全量使用)")
print(f"  高置信度 (HF) : {X_hf_all.shape[0]} 条  (每次 50% 训练 / 50% 测试)")
print(f"  输入特征       : {FEATURES}")
print(f"  目标变量       : {TARGET}")
print(f"  重复次数       : 10  (seed=0~9)")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 定义评估函数
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred):
    """返回 (RMSE, MAE, R2)"""
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 10次重复测试
# ═══════════════════════════════════════════════════════════════════════════════
ALGO_NAMES = ['MF-IDW', 'ConvexHull-GP', 'GPy-CoKriging', 'GPy-InvDistMean']

# 结果存储: {algo_name: {'RMSE': [...], 'MAE': [...], 'R2': [...]}}
results = {name: {'RMSE': [], 'MAE': [], 'R2': []} for name in ALGO_NAMES}

total_start = time.time()

for seed in range(10):
    print(f"\n{'─'*60}")
    print(f"  Round {seed+1}/10  (random_state={seed})")
    print(f"{'─'*60}")

    # 每次用不同种子划分 HF 数据
    X_hf_train, X_hf_test, Y_hf_train, Y_hf_test = train_test_split(
        X_hf_all, Y_hf_all, test_size=0.50, random_state=seed
    )
    Y_hf_train_2d = Y_hf_train.reshape(-1, 1)
    Y_hf_test_2d = Y_hf_test.reshape(-1, 1)

    print(f"  HF 训练集: {X_hf_train.shape[0]} 条 | HF 测试集: {X_hf_test.shape[0]} 条")

    # ── MF-IDW ───────────────────────────────────────────────────────────────
    name = 'MF-IDW'
    print(f"  [{name}] 运行中...", end=' ', flush=True)
    t0 = time.time()
    try:
        y_pred = mf_idw_interpolate(X_lf_all, Y_lf_all, X_hf_train, Y_hf_train, X_hf_test)
        elapsed = time.time() - t0
        rmse, mae, r2 = evaluate(Y_hf_test, y_pred)
        results[name]['RMSE'].append(rmse)
        results[name]['MAE'].append(mae)
        results[name]['R2'].append(r2)
        print(f"完成  [{elapsed:.1f}s]  R2={r2:.4f}  RMSE={rmse:.5f}")
    except Exception as e:
        print(f"失败: {e}")
        results[name]['RMSE'].append(np.nan)
        results[name]['MAE'].append(np.nan)
        results[name]['R2'].append(np.nan)

    # ── ConvexHull-GP ─────────────────────────────────────────────────────────
    name = 'ConvexHull-GP'
    print(f"  [{name}] 运行中...", end=' ', flush=True)
    t0 = time.time()
    try:
        y_pred = mf_ConvexHull(X_lf_all, Y_lf_all_2d, X_hf_train, Y_hf_train_2d, X_hf_test)
        elapsed = time.time() - t0
        rmse, mae, r2 = evaluate(Y_hf_test, y_pred)
        results[name]['RMSE'].append(rmse)
        results[name]['MAE'].append(mae)
        results[name]['R2'].append(r2)
        print(f"完成  [{elapsed:.1f}s]  R2={r2:.4f}  RMSE={rmse:.5f}")
    except Exception as e:
        print(f"失败: {e}")
        results[name]['RMSE'].append(np.nan)
        results[name]['MAE'].append(np.nan)
        results[name]['R2'].append(np.nan)

    # ── GPy-CoKriging ────────────────────────────────────────────────────────
    name = 'GPy-CoKriging'
    print(f"  [{name}] 运行中...", end=' ', flush=True)
    t0 = time.time()
    try:
        y_pred = mf_GPy_CoKriging(X_lf_all, Y_lf_all_2d, X_hf_train, Y_hf_train_2d, X_hf_test)
        elapsed = time.time() - t0
        rmse, mae, r2 = evaluate(Y_hf_test, y_pred)
        results[name]['RMSE'].append(rmse)
        results[name]['MAE'].append(mae)
        results[name]['R2'].append(r2)
        print(f"完成  [{elapsed:.1f}s]  R2={r2:.4f}  RMSE={rmse:.5f}")
    except Exception as e:
        print(f"失败: {e}")
        results[name]['RMSE'].append(np.nan)
        results[name]['MAE'].append(np.nan)
        results[name]['R2'].append(np.nan)

    # ── GPy-InvDistMean ──────────────────────────────────────────────────────
    name = 'GPy-InvDistMean'
    print(f"  [{name}] 运行中...", end=' ', flush=True)
    t0 = time.time()
    try:
        y_pred = mf_GPy_inverseDistanceMean(X_lf_all, Y_lf_all_2d, X_hf_train, Y_hf_train_2d, X_hf_test)
        elapsed = time.time() - t0
        rmse, mae, r2 = evaluate(Y_hf_test, y_pred)
        results[name]['RMSE'].append(rmse)
        results[name]['MAE'].append(mae)
        results[name]['R2'].append(r2)
        print(f"完成  [{elapsed:.1f}s]  R2={r2:.4f}  RMSE={rmse:.5f}")
    except Exception as e:
        print(f"失败: {e}")
        results[name]['RMSE'].append(np.nan)
        results[name]['MAE'].append(np.nan)
        results[name]['R2'].append(np.nan)

total_elapsed = time.time() - total_start

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 统计汇总（均值 ± 标准差）
# ═══════════════════════════════════════════════════════════════════════════════
print("\n")
print("=" * 70)
print("  10次重复测试统计汇总  (均值 ± 标准差)")
print("=" * 70)

summary_rows = []
for name in ALGO_NAMES:
    rmse_arr = np.array(results[name]['RMSE'])
    mae_arr  = np.array(results[name]['MAE'])
    r2_arr   = np.array(results[name]['R2'])
    summary_rows.append({
        'Algorithm': name,
        'RMSE_mean': np.nanmean(rmse_arr),
        'RMSE_std':  np.nanstd(rmse_arr),
        'MAE_mean':  np.nanmean(mae_arr),
        'MAE_std':   np.nanstd(mae_arr),
        'R2_mean':   np.nanmean(r2_arr),
        'R2_std':    np.nanstd(r2_arr),
    })

df_summary = pd.DataFrame(summary_rows).set_index('Algorithm')

# 格式化打印
print(f"\n{'Algorithm':<18s} {'RMSE (mean±std)':<25s} {'MAE (mean±std)':<25s} {'R2 (mean±std)':<20s}")
print("─" * 90)
for name in ALGO_NAMES:
    row = df_summary.loc[name]
    rmse_str = f"{row['RMSE_mean']:.5f} ± {row['RMSE_std']:.5f}"
    mae_str  = f"{row['MAE_mean']:.5f} ± {row['MAE_std']:.5f}"
    r2_str   = f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}"
    print(f"{name:<18s} {rmse_str:<25s} {mae_str:<25s} {r2_str:<20s}")

print(f"\n总耗时: {total_elapsed:.1f}s")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 保存结果到 Excel
# ═══════════════════════════════════════════════════════════════════════════════
out_xlsx = os.path.join(SCRIPT_DIR, 'Repeat_statistics_LTV_results.xlsx')

# Sheet 1: Summary (均值 ± 标准差)
df_summary_out = df_summary.reset_index().copy()
# 格式化：把 mean 和 std 合并为一列显示
for metric in ['RMSE', 'MAE', 'R2']:
    df_summary_out[f'{metric} (mean±std)'] = df_summary_out.apply(
        lambda r: f"{r[f'{metric}_mean']:.5f} ± {r[f'{metric}_std']:.5f}", axis=1
    )
df_summary_out = df_summary_out[['Algorithm', 'RMSE (mean±std)', 'MAE (mean±std)', 'R2 (mean±std)',
                                  'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std']]

# Sheet 2: PerRun (每次运行的详细值)
per_run_rows = []
for seed in range(10):
    for name in ALGO_NAMES:
        per_run_rows.append({
            'Seed': seed,
            'Algorithm': name,
            'RMSE': results[name]['RMSE'][seed],
            'MAE':  results[name]['MAE'][seed],
            'R2':   results[name]['R2'][seed],
        })
df_perrun = pd.DataFrame(per_run_rows)

with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    df_summary.reset_index().to_excel(writer, sheet_name='Summary', index=False)
    df_perrun.to_excel(writer, sheet_name='PerRun', index=False)

print(f"\n结果已保存至: {out_xlsx}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 可视化
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DataFusion Algorithm 10-Repeat Benchmark (LTV)\n'
             'Mean ± Std of RMSE / MAE / R²  (+ RMSE trend over 10 seeds)',
             fontsize=13, fontweight='bold', y=1.02)

colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
x = np.arange(len(ALGO_NAMES))
width = 0.35

# ── 子图1: RMSE 均值±std 柱状图 ──────────────────────────────────────────────
ax1 = axes[0, 0]
rmse_means = [df_summary.loc[n, 'RMSE_mean'] for n in ALGO_NAMES]
rmse_stds  = [df_summary.loc[n, 'RMSE_std']  for n in ALGO_NAMES]
bars1 = ax1.bar(x, rmse_means, yerr=rmse_stds, capsize=8, color=colors,
                alpha=0.85, edgecolor='white', error_kw={'elinewidth': 2, 'capthick': 2})
ax1.set_xticks(x)
ax1.set_xticklabels(ALGO_NAMES, fontsize=8)
ax1.set_ylabel('RMSE', fontsize=10, fontweight='bold')
ax1.set_title('RMSE  (mean ± std, lower is better)', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar, v, std in zip(bars1, rmse_means, rmse_stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0005,
             f'{v:.5f}', ha='center', va='bottom', fontsize=7.5)

# ── 子图2: MAE 均值±std 柱状图 ───────────────────────────────────────────────
ax2 = axes[0, 1]
mae_means = [df_summary.loc[n, 'MAE_mean'] for n in ALGO_NAMES]
mae_stds  = [df_summary.loc[n, 'MAE_std']  for n in ALGO_NAMES]
bars2 = ax2.bar(x, mae_means, yerr=mae_stds, capsize=8, color=colors,
                alpha=0.85, edgecolor='white', error_kw={'elinewidth': 2, 'capthick': 2})
ax2.set_xticks(x)
ax2.set_xticklabels(ALGO_NAMES, fontsize=8)
ax2.set_ylabel('MAE', fontsize=10, fontweight='bold')
ax2.set_title('MAE  (mean ± std, lower is better)', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, v, std in zip(bars2, mae_means, mae_stds):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0003,
             f'{v:.5f}', ha='center', va='bottom', fontsize=7.5)

# ── 子图3: R2 均值±std 柱状图 ────────────────────────────────────────────────
ax3 = axes[1, 0]
r2_means = [df_summary.loc[n, 'R2_mean'] for n in ALGO_NAMES]
r2_stds  = [df_summary.loc[n, 'R2_std']  for n in ALGO_NAMES]
bars3 = ax3.bar(x, r2_means, yerr=r2_stds, capsize=8, color=colors,
                alpha=0.85, edgecolor='white', error_kw={'elinewidth': 2, 'capthick': 2})
ax3.set_xticks(x)
ax3.set_xticklabels(ALGO_NAMES, fontsize=8)
ax3.set_ylabel('R²', fontsize=10, fontweight='bold')
ax3.set_title('R²  (mean ± std, higher is better)', fontsize=10, fontweight='bold')
ax3.set_ylim(bottom=0.90)
ax3.grid(axis='y', alpha=0.3)
for bar, v, std in zip(bars3, r2_means, r2_stds):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
             f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

# ── 子图4: 10次运行 RMSE 折线趋势图 ──────────────────────────────────────────
ax4 = axes[1, 1]
for i, name in enumerate(ALGO_NAMES):
    rmse_vals = results[name]['RMSE']
    ax4.plot(range(10), rmse_vals, 'o-', color=colors[i], label=name,
             linewidth=1.5, markersize=5, alpha=0.85)
ax4.set_xticks(range(10))
ax4.set_xlabel('Seed Index (0~9)', fontsize=9)
ax4.set_ylabel('RMSE', fontsize=10, fontweight='bold')
ax4.set_title('RMSE Stability over 10 Seeds', fontsize=10, fontweight='bold')
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
out_fig = os.path.join(SCRIPT_DIR, 'Repeat_statistics_LTV.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"对比图已保存至: {out_fig}")

print("\nDone.")
