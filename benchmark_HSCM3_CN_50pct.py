"""
DataFusion 算法性能对比评测脚本 —— 外形: HSCM3 | 目标变量: CN
- 低置信度数据 : HSCM3_CN_Low-Fidelity.xlsx   (全量, 28条)
- 高置信度数据 : HSCM3_CN_High-Fidelity.xlsx  (50% 训练)
- 特征列       : H, Ma, sinA
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
from sklearn.model_selection import train_test_split

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
FEATURES     = ['H', 'Ma', 'sinA']
TARGET       = 'CN'
HF_RATIOS    = [0.50]                # HF 训练集比例
ALGO_COLORS  = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
ALGO_NAMES   = ['MF-IDW', 'ConvexHull-GP', 'GPy-CoKriging', 'GPy-InvDistMean']
TAG          = 'HSCM3_CN'           # 输出文件前缀

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 加载数据
# ═══════════════════════════════════════════════════════════════════════════════
df_lf = pd.read_excel(os.path.join(SCRIPT_DIR, 'HSCM3_CN_Low-Fidelity.xlsx'))
df_hf = pd.read_excel(os.path.join(SCRIPT_DIR, 'HSCM3_CN_High-Fidelity.xlsx'))

X_lf      = df_lf[FEATURES].values.astype(float)
Y_lf      = df_lf[TARGET].values.astype(float).reshape(-1, 1)
X_hf_all  = df_hf[FEATURES].values.astype(float)
Y_hf_all  = df_hf[TARGET].values.astype(float).reshape(-1, 1)

print("=" * 65)
print(f"  DataFusion 4-Algorithm Benchmark  [Shape: HSCM3 | Target: {TARGET}]")
print("=" * 65)
print(f"  低置信度数据 : {X_lf.shape[0]} 条 (全量)")
print(f"  高置信度数据 : {X_hf_all.shape[0]} 条 (按比例拆分)")
print(f"  特征列       : {FEATURES}")
print(f"  目标列       : {TARGET}")

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
    # CN 量级约 0~0.45，取绝对值 > 0.01 的点计算相对误差
    mask  = np.abs(y_true) > 0.01
    mre   = (np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
             if mask.sum() > 0 else np.nan)
    return {'Algorithm': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2,
            'MaxError': max_e, 'MeanRelErr(%)': mre}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 主循环：遍历2个比例
# ═══════════════════════════════════════════════════════════════════════════════
all_results = {}
all_preds   = {}
all_tests   = {}

for ratio in HF_RATIOS:
    test_size  = round(1.0 - ratio, 2)
    pct_label  = f"{int(ratio*100)}%"
    print(f"\n{'='*65}")
    print(f"  HF 训练集: {pct_label}  |  测试集: {int(test_size*100)}%")
    print(f"{'='*65}")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_hf_all, Y_hf_all, test_size=test_size, random_state=42
    )
    print(f"  HF 训练: {X_train.shape[0]} 条  |  HF 测试: {X_test.shape[0]} 条")

    results = []
    preds   = {}
    y_true  = Y_test.ravel()

    algo_funcs = [
        ('MF-IDW',         mf_idw_interpolate),
        ('ConvexHull-GP',  mf_ConvexHull),
        ('GPy-CoKriging',  mf_GPy_CoKriging),
        ('GPy-InvDistMean',mf_GPy_inverseDistanceMean),
    ]

    for name, func in algo_funcs:
        print(f"  [{name}] 运行中...", end=' ', flush=True)
        t0 = time.time()
        try:
            if name == 'MF-IDW':
                y_pred = func(X_lf, Y_lf.ravel(), X_train, Y_train.ravel(), X_test)
            else:
                y_pred = func(X_lf, Y_lf, X_train, Y_train, X_test)
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
    print(f"\n  算法性能对比汇总（HF {pct_label} 训练 / {int(test_size*100)}% 测试）")
    print("  " + "-" * 60)
    print(df_res.to_string(float_format=lambda x: f"{x:.5f}").replace('\n', '\n  '))

    all_results[ratio] = df_res
    all_preds[ratio]   = preds
    all_tests[ratio]   = (X_test, y_true)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 逐比例生成可视化图
# ═══════════════════════════════════════════════════════════════════════════════
for ratio in HF_RATIOS:
    pct_tag    = f"{int(ratio*100)}pct"
    test_label = f"{int(round(1 - ratio, 2)*100)}%"
    train_label= f"{int(ratio*100)}%"
    n_train    = all_results[ratio].shape[0]   # 仅用于标题，实际数量从数据获取
    df_res     = all_results[ratio]
    preds      = all_preds[ratio]
    _, y_true  = all_tests[ratio]

    # 实际训练/测试条数
    n_hf_train = int(round(X_hf_all.shape[0] * ratio))
    n_hf_test  = X_hf_all.shape[0] - n_hf_train

    n_algos = len(preds)
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, n_algos, figure=fig, hspace=0.45, wspace=0.35)

    # 行1: 预测 vs 真值散点
    for idx, (name, y_pred) in enumerate(preds.items()):
        ax   = fig.add_subplot(gs[0, idx])
        r2   = r2_score(y_true, y_pred)
        vmin = min(y_true.min(), y_pred.min())
        vmax = max(y_true.max(), y_pred.max())
        ax.scatter(y_true, y_pred, s=55, alpha=0.80,
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
        ax.scatter(y_true, resid, s=55, alpha=0.80,
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
        f'DataFusion Algorithm Benchmark  [HSCM3 | Target: {TARGET}]\n'
        f'(LF: {X_lf.shape[0]} samples | '
        f'HF Train: {train_label} ({n_hf_train} pts) | '
        f'HF Test: {test_label} ({n_hf_test} pts))',
        fontsize=13, fontweight='bold', y=0.98
    )

    out_fig = os.path.join(SCRIPT_DIR, f'{TAG}_benchmark_{pct_tag}.png')
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  图表已保存: {out_fig}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 保存汇总 Excel
# ═══════════════════════════════════════════════════════════════════════════════
out_xlsx = os.path.join(SCRIPT_DIR, f'{TAG}_benchmark_50pct_results.xlsx')
with pd.ExcelWriter(out_xlsx, engine='openpyxl') as writer:
    for ratio in HF_RATIOS:
        pct_tag   = f"{int(ratio*100)}pct"
        test_size = round(1.0 - ratio, 2)
        df_res    = all_results[ratio]
        preds     = all_preds[ratio]
        X_test, y_true = all_tests[ratio]

        # 性能汇总 sheet
        df_res.reset_index().to_excel(
            writer, sheet_name=f'Perf_{pct_tag}', index=False)

        # 逐点预测 sheet（含三个特征列）
        df_detail = pd.DataFrame({
            'H':    X_test[:, 0],
            'Ma':   X_test[:, 1],
            'sinA': X_test[:, 2],
            f'{TARGET}_True': y_true,
        })
        for name, y_pred in preds.items():
            df_detail[f'{TARGET}_{name}']    = y_pred
            df_detail[f'Resid_{name}'] = y_pred - y_true
        df_detail.to_excel(writer, sheet_name=f'Pred_{pct_tag}', index=False)

    # 纵向对比 sheet
    rows = []
    for ratio in HF_RATIOS:
        df_res = all_results[ratio]
        for algo in ALGO_NAMES:
            if algo in df_res.index:
                row = {'HF_Train': f"{int(ratio*100)}%", 'Algorithm': algo}
                row.update(df_res.loc[algo].to_dict())
                rows.append(row)
    pd.DataFrame(rows).to_excel(writer, sheet_name='Summary_All', index=False)

print(f"\n汇总结果已保存: {out_xlsx}")
print("\nAll Done.")
