"""
DataFusion 算法性能对比评测脚本 —— LTV CN | 按特征范围划分训练/测试集（Alpha中段）
- 低置信度数据 : LTV_Low-Fidelity.xlsx   (全量, 1000条)
- 高置信度数据 : LTV_High-Fidelity.xlsx  (按特征范围拆分)
- 训练集筛选   : 1.2 < Mach < 3.5 且 7.0 < Alpha < 15.0
- 测试集       : 其余数据
- 特征列       : Mach, Alpha
- 目标列       : CN
- 评估指标     : RMSE / MAE / R2 / MaxError / MeanRelErr
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
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
# 配置
# ═══════════════════════════════════════════════════════════════════════════════
FEATURES  = ['Mach', 'Alpha']
TARGET    = 'CN'
ALGO_COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
ALGO_NAMES  = ['MF-IDW', 'ConvexHull-GP', 'GPy-CoKriging', 'GPy-InvDistMean']
TAG         = 'LTV_CN_region_mid'  # 输出文件前缀

# 训练集特征范围（Alpha中段）
TRAIN_MACH_MIN = 1.2
TRAIN_MACH_MAX = 3.5
TRAIN_ALPHA_MIN = 7.0
TRAIN_ALPHA_MAX = 15.0

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════════════════════
df_lf = pd.read_excel(os.path.join(SCRIPT_DIR, 'LTV_Low-Fidelity.xlsx'))
df_hf = pd.read_excel(os.path.join(SCRIPT_DIR, 'LTV_High-Fidelity.xlsx'))

# 按特征范围划分 HF 训练/测试集
mask_train = (df_hf['Mach'] > TRAIN_MACH_MIN) & (df_hf['Mach'] < TRAIN_MACH_MAX) & \
             (df_hf['Alpha'] > TRAIN_ALPHA_MIN) & (df_hf['Alpha'] < TRAIN_ALPHA_MAX)
df_hf_train = df_hf[mask_train].copy()
df_hf_test = df_hf[~mask_train].copy()

X_lf      = df_lf[FEATURES].values.astype(float)
Y_lf      = df_lf[TARGET].values.astype(float).reshape(-1, 1)
X_hf_train = df_hf_train[FEATURES].values.astype(float)
Y_hf_train = df_hf_train[TARGET].values.astype(float).reshape(-1, 1)
X_hf_test  = df_hf_test[FEATURES].values.astype(float)
Y_hf_test  = df_hf_test[TARGET].values.astype(float).reshape(-1, 1)

print("=" * 65)
print(f"  DataFusion 4-Algorithm Benchmark  [LTV CN | Region Split (Alpha Mid)]")
print("=" * 65)
print(f"  低置信度数据 : {X_lf.shape[0]} 条 (全量)")
print(f"  高置信度数据 : {X_hf_train.shape[0]} 条训练 | {X_hf_test.shape[0]} 条测试")
print(f"  特征列       : {FEATURES}")
print(f"  目标列       : {TARGET}")
print(f"\n  训练集特征范围:")
print(f"    Mach:  ({TRAIN_MACH_MIN}, {TRAIN_MACH_MAX})")
print(f"    Alpha: ({TRAIN_ALPHA_MIN}, {TRAIN_ALPHA_MAX})")
print(f"    实际: Mach=[{X_hf_train[:,0].min():.1f}, {X_hf_train[:,0].max():.1f}], "
      f"Alpha=[{X_hf_train[:,1].min():.1f}, {X_hf_train[:,1].max():.1f}]")
print(f"\n  测试集特征范围:")
print(f"    实际: Mach=[{X_hf_test[:,0].min():.1f}, {X_hf_test[:,0].max():.1f}], "
      f"Alpha=[{X_hf_test[:,1].min():.1f}, {X_hf_test[:,1].max():.1f}]")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 评估函数
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate(y_true, y_pred, name):
    y_true = np.array(y_true).ravel()
    y_pred = np.array(y_pred).ravel()
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    max_e = np.max(np.abs(y_true - y_pred))
    # CN 量级约 -0.04~7.8，取绝对值 > 0.1 的点计算相对误差
    mask  = np.abs(y_true) > 0.1
    mre   = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
             if mask.sum() > 0 else np.nan)
    return {'Algorithm': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'MaxError': max_e, 'MeanRelErr(%)': mre}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 算法评测
# ═══════════════════════════════════════════════════════════════════════════════
results = []
preds   = {}
y_true  = Y_hf_test.ravel()

algo_funcs = [
    ('MF-IDW',         mf_idw_interpolate),
    ('ConvexHull-GP',  mf_ConvexHull),
    ('GPy-CoKriging',  mf_GPy_CoKriging),
    ('GPy-InvDistMean',mf_GPy_inverseDistanceMean),
]

print(f"\n{'='*65}")
print("  算法性能对比（区域内训练 -> 区域外泛化测试）")
print(f"{'='*65}")

for name, func in algo_funcs:
    print(f"  [{name}] 运行中...", end=' ', flush=True)
    t0 = time.time()
    try:
        if name == 'MF-IDW':
            y_pred = func(X_lf, Y_lf.ravel(), X_hf_train, Y_hf_train.ravel(), X_hf_test)
        else:
            y_pred = func(X_lf, Y_lf, X_hf_train, Y_hf_train, X_hf_test)
        elapsed = time.time() - t0
        m = evaluate(y_true, y_pred, name)
        m['Time(s)'] = round(elapsed, 2)
        results.append(m)
        preds[name] = y_pred
        print(f"完成 [{elapsed:.1f}s]  R2={m['R2']:.4f}  RMSE={m['RMSE']:.5f}")
    except Exception as e:
        print(f"失败: {e}")
        results.append({'Algorithm': name, 'RMSE': np.nan, 'MAE': np.nan,
                        'R2': np.nan, 'MaxError': np.nan,
                        'MeanRelErr(%)': np.nan, 'Time(s)': np.nan})

df_res = pd.DataFrame(results).set_index('Algorithm')
print(f"\n  算法性能对比汇总")
print("  " + "-" * 60)
print(df_res.to_string(float_format=lambda x: f"{x:.5f}").replace('\n', '\n  '))

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 生成可视化图
# ═══════════════════════════════════════════════════════════════════════════════
n_algos = len(preds)
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, n_algos, figure=fig, hspace=0.45, wspace=0.35)

# 行1: 预测 vs 真值散点
for idx, (name, y_pred) in enumerate(preds.items()):
    ax   = fig.add_subplot(gs[0, idx])
    r2   = r2_score(y_true, y_pred)
    vmin = min(y_true.min(), y_pred.min())
    vmax = max(y_true.max(), y_pred.max())
    ax.scatter(y_true, y_pred, s=45, alpha=0.70,
               color=ALGO_COLORS[idx], edgecolors='none')
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.2, label='Ideal')
    ax.set_xlabel(f'True {TARGET}', fontsize=9)
    ax.set_ylabel(f'Predicted {TARGET}', fontsize=9)
    ax.set_title(f'{name}\n$R^2$={r2:.4f}', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 行2: 残差图
for idx, (name, y_pred) in enumerate(preds.items()):
    ax    = fig.add_subplot(gs[1, idx])
    resid = y_pred - y_true
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    ax.scatter(y_true, resid, s=45, alpha=0.70,
               color=ALGO_COLORS[idx], edgecolors='none')
    ax.axhline(0,      color='k',    lw=1.2, linestyle='--')
    ax.axhline( rmse,  color='gray', lw=0.8, linestyle=':')
    ax.axhline(-rmse,  color='gray', lw=0.8, linestyle=':',
               label=f'±RMSE={rmse:.4f}')
    ax.set_xlabel(f'True {TARGET}', fontsize=9)
    ax.set_ylabel('Residual', fontsize=9)
    ax.set_title(f'{name} — Residuals', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# 行3: 指标柱状图
metric_pairs = [('RMSE','RMSE'), ('MAE','MAE'),
                ('R2','R²'), ('MeanRelErr(%)','Mean Rel. Err (%)')]
for m_idx, (metric, label) in enumerate(metric_pairs):
    ax   = fig.add_subplot(gs[2, m_idx])
    vals = [df_res.loc[n, metric] for n in preds.keys()]
    names= list(preds.keys())
    bars = ax.bar(names, vals, color=ALGO_COLORS[:len(names)],
                  alpha=0.85, edgecolor='white')
    ax.set_title(label, fontsize=10, fontweight='bold')
    ax.set_ylabel(label, fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace('-', '\n') for n in names], fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f'{v:.4f}', ha='center', va='bottom', fontsize=7.5)

fig.suptitle(
    'DataFusion Algorithm Benchmark  [LTV CN | Region Split (Alpha Mid)]\n'
    f'(LF: {X_lf.shape[0]} samples | '
    f'HF Train: {X_hf_train.shape[0]} pts [1.2<Mach<3.5, 7.0<Alpha<15.0] | '
    f'HF Test: {X_hf_test.shape[0]} pts [Region Outside])',
    fontsize=13, fontweight='bold', y=0.98
)

out_fig = os.path.join(SCRIPT_DIR, f'{TAG}_benchmark.png')
plt.savefig(out_fig, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\n  图表已保存: {out_fig}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 保存汇总 Excel
# ═══════════════════════════════════════════════════════════════════════════════
out_xlsx = os.path.join(SCRIPT_DIR, f'{TAG}_benchmark_results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    # 性能汇总 sheet
    df_res.reset_index().to_excel(writer, sheet_name='Performance', index=False)

    # 逐点预测 sheet
    df_detail = pd.DataFrame({
        'Mach': X_hf_test[:, 0],
        'Alpha': X_hf_test[:, 1],
        f'{TARGET}_True': y_true,
    })
    for name, y_pred in preds.items():
        df_detail[f'{TARGET}_{name}']    = y_pred
        df_detail[f'Resid_{name}'] = y_pred - y_true
    df_detail.to_excel(writer, sheet_name='Predictions', index=False)

print(f"汇总结果已保存: {out_xlsx}")
print("\nAll Done.")
