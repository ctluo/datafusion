"""
DataFusion 算法性能对比评测脚本
- 低置信度数据: LTV_Low-Fidelity.xlsx  (全量, 1000条)
- 高置信度数据: LTV_High-Fidelity.xlsx (10% 训练 / 90% 测试)
- 特征列: Mach, Alpha
- 目标列: CN
- 评估指标: RMSE / MAE / R2 / MaxError / MeanRelErr
"""

import sys
import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # 非交互后端，避免GUI弹窗
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  DataFusion 4-Algorithm Benchmark")
print("=" * 65)

df_lf = pd.read_excel(os.path.join(SCRIPT_DIR, 'LTV_Low-Fidelity.xlsx'))
df_hf = pd.read_excel(os.path.join(SCRIPT_DIR, 'LTV_High-Fidelity.xlsx'))

FEATURES = ['Mach', 'Alpha']
TARGET   = 'CN'

# 低置信度：全量
X_lf = df_lf[FEATURES].values.astype(float)
Y_lf = df_lf[TARGET].values.astype(float).reshape(-1, 1)

# 高置信度：10% 训练 / 90% 测试（按行随机拆分）
X_hf_all = df_hf[FEATURES].values.astype(float)
Y_hf_all = df_hf[TARGET].values.astype(float).reshape(-1, 1)

X_hf_train, X_hf_test, Y_hf_train, Y_hf_test = train_test_split(
    X_hf_all, Y_hf_all, test_size=0.90, random_state=42
)

print(f"\n数据概览:")
print(f"  低置信度训练集   : {X_lf.shape[0]} 条  (全量)")
print(f"  高置信度训练集   : {X_hf_train.shape[0]} 条  (10%)")
print(f"  高置信度测试集   : {X_hf_test.shape[0]} 条  (90%)")
print(f"  输入特征         : {FEATURES}")
print(f"  目标变量         : {TARGET}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 定义评估函数
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred, name):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    max_e  = np.max(np.abs(y_true - y_pred))
    # 相对误差（跳过真值接近0的点）
    mask   = np.abs(y_true) > 0.05
    mre    = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else np.nan
    return {'Algorithm': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'MaxError': max_e, 'MeanRelErr(%)': mre}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 运行4个算法
# ═══════════════════════════════════════════════════════════════════════════════
results   = []
preds_all = {}

algos = [
    ('MF-IDW',              mf_idw_interpolate),
    ('ConvexHull-GP',       mf_ConvexHull),
    ('GPy-CoKriging',       mf_GPy_CoKriging),
    ('GPy-InvDistMean',     mf_GPy_inverseDistanceMean),
]

for name, func in algos:
    print(f"\n[{name}] 运行中...", end=' ', flush=True)
    t0 = time.time()
    try:
        if name == 'MF-IDW':
            y_pred = func(X_lf, Y_lf.ravel(), X_hf_train, Y_hf_train.ravel(), X_hf_test)
        else:
            y_pred = func(X_lf, Y_lf, X_hf_train, Y_hf_train, X_hf_test)
        elapsed = time.time() - t0
        metrics = evaluate(Y_hf_test.ravel(), y_pred, name)
        metrics['Time(s)'] = round(elapsed, 2)
        results.append(metrics)
        preds_all[name] = y_pred
        print(f"完成  [{elapsed:.1f}s]  R2={metrics['R2']:.4f}  RMSE={metrics['RMSE']:.5f}")
    except Exception as e:
        print(f"失败: {e}")
        results.append({'Algorithm': name, 'RMSE': np.nan, 'MAE': np.nan,
                        'R2': np.nan, 'MaxError': np.nan,
                        'MeanRelErr(%)': np.nan, 'Time(s)': np.nan})

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 打印汇总表
# ═══════════════════════════════════════════════════════════════════════════════
df_res = pd.DataFrame(results).set_index('Algorithm')
print("\n")
print("=" * 65)
print("  算法性能对比汇总（测试集: HF 90%)")
print("=" * 65)
print(df_res.to_string(float_format=lambda x: f"{x:.5f}"))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 可视化：综合对比图
# ═══════════════════════════════════════════════════════════════════════════════
n_algos = len(preds_all)
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, n_algos, figure=fig, hspace=0.45, wspace=0.35)

y_true = Y_hf_test.ravel()
colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

# ── 行1: 预测值 vs 真值散点图 ────────────────────────────────────────────────
for idx, (name, y_pred) in enumerate(preds_all.items()):
    ax = fig.add_subplot(gs[0, idx])
    r2 = r2_score(y_true, y_pred)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, s=30, alpha=0.75, color=colors[idx], edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.2, label='Ideal')
    ax.set_xlabel('True CN', fontsize=9)
    ax.set_ylabel('Predicted CN', fontsize=9)
    ax.set_title(f'{name}\n$R^2$={r2:.4f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── 行2: 残差图 ──────────────────────────────────────────────────────────────
for idx, (name, y_pred) in enumerate(preds_all.items()):
    ax   = fig.add_subplot(gs[1, idx])
    resid = y_pred - y_true
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.scatter(y_true, resid, s=30, alpha=0.75, color=colors[idx], edgecolors='none')
    ax.axhline(0,  color='k', lw=1.2, linestyle='--')
    ax.axhline( rmse, color='gray', lw=0.8, linestyle=':')
    ax.axhline(-rmse, color='gray', lw=0.8, linestyle=':', label=f'±RMSE={rmse:.4f}')
    ax.set_xlabel('True CN', fontsize=9)
    ax.set_ylabel('Residual', fontsize=9)
    ax.set_title(f'{name} — Residuals', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# ── 行3: 指标柱状图对比 ──────────────────────────────────────────────────────
metric_names = ['RMSE', 'MAE', 'R2', 'MeanRelErr(%)']
metric_label = ['RMSE', 'MAE', 'R²', 'Mean Rel. Err (%)']

for m_idx, (metric, label) in enumerate(zip(metric_names, metric_label)):
    ax = fig.add_subplot(gs[2, m_idx] if m_idx < n_algos else gs[2, -1])
    vals  = [df_res.loc[n, metric] for n in preds_all.keys()]
    names = list(preds_all.keys())
    bars  = ax.bar(names, vals, color=colors[:len(names)], alpha=0.85, edgecolor='white')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('-', '\n') for n in names], fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

fig.suptitle('DataFusion Algorithm Benchmark\n'
             '(LF: 1000 samples | HF Train: 10% | HF Test: 90%)',
             fontsize=13, fontweight='bold', y=0.98)

out_fig = os.path.join(SCRIPT_DIR, 'fusion_benchmark.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
print(f"\n对比图已保存至: {out_fig}")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 保存结果到 Excel
# ═══════════════════════════════════════════════════════════════════════════════
out_xlsx = os.path.join(SCRIPT_DIR, 'fusion_benchmark_results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    # 性能汇总
    df_res.reset_index().to_excel(writer, sheet_name='Performance', index=False)
    # 各算法预测值对比
    df_detail = pd.DataFrame({'Mach': X_hf_test[:,0], 'Alpha': X_hf_test[:,1],
                              'CN_True': y_true})
    for name, y_pred in preds_all.items():
        df_detail[f'CN_{name}'] = y_pred
        df_detail[f'Resid_{name}'] = y_pred - y_true
    df_detail.to_excel(writer, sheet_name='Predictions', index=False)

print(f"结果已保存至: {out_xlsx}")
print("\nDone.")
