# DataFusion

> **面向气动力系数预测的多保真度数据融合算法库**

一个 Python 研究框架，实现并系统评测了四种多保真度数据融合算法。核心思想是将大量低置信度（低保真度）仿真数据与少量高置信度（高保真度）试验/CFD 数据相融合，以低成本实现高精度的气动力系数预测。

---

## 目录

- [背景](#背景)
- [融合算法](#融合算法)
- [项目结构](#项目结构)
- [数据集](#数据集)
- [安装部署](#安装部署)
- [快速开始](#快速开始)
- [评测结果](#评测结果)
- [评估指标](#评估指标)
- [API 接口说明](#api-接口说明)

---

## 背景

在航空航天工程中，高保真度 CFD 数值模拟和风洞试验能够提供精确的气动力数据，但代价极高。低保真度方法（如工程估算代码）速度快、成本低，但精度有限。多保真度数据融合在二者之间搭建桥梁：以低保真度数据建立全局代理模型，再利用稀疏的高保真度观测点对其进行修正。

本项目实现了四种融合策略，并在四个气动外形数据集（LTV 导弹体、AGARD-B、HB2、HSCM3）上进行了系统性评测。

---

## 融合算法

全部四种算法均实现于 `DataFusionAlgorithms.py`，对外接口统一：

```python
y_pred = algorithm(X_LF, Y_LF, X_HF, Y_HF, X_pred)
```

### 算法一：MF-IDW — 多保真度反距离加权插值

一种轻量级的纯几何插值方法，无需尺度变换因子。

**算法流程：**
1. 以 IDW 插值构建全局低保真度代理面
2. 在高保真度训练点处评估 LF 模型：`Ŷ_LF(X_HF)`
3. 计算残差：`r = Y_HF − Ŷ_LF(X_HF)`
4. 以 IDW 将残差插值到预测点：`r̂(X_pred)`
5. 融合预测：`Ŷ_MF = Ŷ_LF + r̂`（加法修正）

**关键参数：** `p_LF=2`（LF 插值幂次）、`p_R=2`（残差插值幂次）

---

### 算法二：ConvexHull-GP — 凸包距离引导的高斯过程融合

自适应策略：根据预测点是否落在高保真度训练数据凸包内外，分别采用不同的预测规则。

**算法流程：**
1. 以 LF 数据为基准，通过 MinMaxScaler 将输入归一化至 [−1, 1]
2. 对每个测试点，以 SLSQP 优化计算其到 HF 数据凸包的距离及最近凸包点
3. 分别对 LF 和 HF 数据拟合 GP 模型（RBF 核）
4. **凸包内**（距离 ≤ 0.01）：直接用 HF GP 模型预测
5. **凸包外**（距离 > 0.01）：`Ŷ = GP_LF(x) + [GP_HF(x_最近凸包点) − GP_LF(x_最近凸包点)]`

---

### 算法三：GPy-CoKriging — 协同克里金（Kennedy & O'Hagan 框架）

具有完整贝叶斯理论基础的概率融合方法，可输出预测不确定性。

**算法流程：**
1. 以 LF 数据拟合 GP 模型
2. 在 HF 位置预测 LF 值：`Ŷ_l(X_h)`
3. 以 OLS 线性回归估计缩放因子 ρ：`Y_h ≈ ρ · Ŷ_l`
4. 计算残差：`δ = Y_h − ρ · Ŷ_l(X_h)`
5. 对残差拟合修正 GP 模型
6. 融合预测：`Ŷ_h = ρ · GP_LF(x) + GP_δ(x)`，方差：`σ²_h = ρ² σ²_l + σ²_δ`

*可选择同时返回预测方差，用于不确定性量化应用。*

---

### 算法四：GPy-InvDistMean — GP + 逆距离加权残差修正

结合 GP 建模低保真度趋势与 IDW 混合残差的融合方法。

**算法流程：**
1. 将输入归一化至 [−1, 1]
2. 以 LF 数据拟合 GP 模型
3. 在 HF 位置和预测位置分别评估 LF GP
4. 计算 HF 位置处的残差
5. 对每个测试点，计算其到所有 HF 训练点的 IDW 权重（幂次为 1）
6. 融合预测：`Ŷ_new = Ŷ_LF(x) + Σ w_i · r_i`

**注意：** IDW 幂次固定为 1（即 `w = 1/d`，而非 `1/d²`）。

---

### 算法横向对比

| 维度 | MF-IDW | ConvexHull-GP | GPy-CoKriging | GPy-InvDistMean |
|------|--------|---------------|---------------|-----------------|
| LF 建模方式 | IDW 插值 | GP（RBF 核） | GP（RBF 核） | GP（RBF 核） |
| 残差处理方式 | IDW 插值残差 | GP 点预测差值 | GP 拟合残差 | IDW 加权残差 |
| 尺度因子 ρ | 无 | 无 | OLS 自动估计 | 无 |
| 输入归一化 | 无 | MinMaxScaler [−1,1] | 无 | MinMaxScaler [−1,1] |
| 预测不确定性 | 无 | 无 | 有（可选） | 无 |
| 域内/外感知 | 无 | 有（凸包判断） | 无 | 无 |
| 计算开销 | 低 | 高（逐点 SLSQP） | 中等 | 中等 |

---

## 项目结构

```
DataFusion/
│
├── DataFusionAlgorithms.py          # 核心算法库（4 种融合算法）
│
├── benchmark_fusion.py              # LTV CN — 10% HF 训练
├── benchmark_fusion_20pct.py        # LTV CN — 20% HF 训练
├── benchmark_fusion_50pct.py        # LTV CN — 50% HF 训练
├── benchmark_fusion_80pct.py        # LTV CN — 80% HF 训练
├── benchmark_CM.py                  # LTV CM — 10%/20%/50%/80% HF 训练
├── benchmark_AGARD_B_CL.py          # AGARD-B CL — 20%/50%/80% HF 训练
├── benchmark_HB2_CN.py              # HB2 CN — 20%/50%/80% HF 训练
├── benchmark_HSCM3_CN.py            # HSCM3 CN — 30%/60% HF 训练
├── benchmark_HSCM3_CN_50pct.py      # HSCM3 CN — 50% HF 训练
├── benchmark_LTV_CN_region.py       # LTV CN — 按特征区域划分
├── benchmark_LTV_CN_region_mid.py   # LTV CN — 中等区域划分
├── benchmark_LTV_CN_region_narrow.py# LTV CN — 窄区域划分
│
├── LTV_Low-Fidelity.xlsx            # LTV 导弹体 LF 数据（1000 条）
├── LTV_High-Fidelity.xlsx           # LTV 导弹体 HF 数据
├── AGARD-B_CL_Low-Fidelity.xlsx     # AGARD-B LF 数据（23 条）
├── AGARD-B_CL_High-Fidelity.xlsx    # AGARD-B HF 数据
├── HB2_CN_Low-Fidelity.xlsx         # HB2 LF 数据（39 条）
├── HB2_CN_High-Fidelity.xlsx        # HB2 HF 数据
├── HSCM3_CN_Low-Fidelity.xlsx       # HSCM3 LF 数据（28 条）
├── HSCM3_CN_High-Fidelity.xlsx      # HSCM3 HF 数据
│
├── check_env.py                     # 环境依赖检查工具
├── inspect_data.py                  # LTV 数据集检查工具
├── _inspect_agard.py                # AGARD-B 数据集检查
├── _inspect_hb2.py                  # HB2 数据集检查
├── _inspect_hscm3.py                # HSCM3 数据集检查
│
├── requirements.txt                 # Python 依赖包列表
│
└── [输出 *.png / *.xlsx]            # 评测结果图表与汇总表格
```

---

## 数据集

| 数据集 | 外形 | 特征列 | 目标列 | LF 条数 | 特征维度 |
|--------|------|--------|--------|---------|---------|
| LTV | 导弹体 | Mach, Alpha | CN, CM | 1000 | 2D |
| AGARD-B | AGARD 标准模型 | Ma, sinA | CL | 23 | 2D |
| HB2 | HB2 标准模型 | H, Ma, sinA | CN | 39 | 3D |
| HSCM3 | HSCM3 飞行器 | H, Ma, sinA | CN | 28 | 3D |

**特征列说明：**
- `Mach` / `Ma` — 马赫数
- `Alpha` — 攻角（度）
- `sinA` — sin（攻角）
- `H` — 飞行高度
- `CN` — 法向力系数
- `CL` — 升力系数
- `CM` — 俯仰力矩系数

---

## 安装部署

### 环境要求

- Python 3.9+
- 推荐使用虚拟环境

### 安装步骤

```bash
# 进入 DataFusion 目录
cd DataFusion

# 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 安装依赖
pip install -r requirements.txt

# 验证安装
python check_env.py
```

预期输出：
```
numpy       : 1.26.4
pandas      : 3.0.1
scipy       : 1.12.0
matplotlib  : 3.10.8
GPy         : 1.13.2
scikit-learn: 1.8.0
openpyxl    : 3.1.5

ALL IMPORTS OK - DataFusion venv is ready!
```

---

## 快速开始

### 运行单个评测脚本

```bash
# LTV CN，10% HF 训练比例
python benchmark_fusion.py

# LTV CN，50% HF 训练比例
python benchmark_fusion_50pct.py

# AGARD-B CL，多比例（20% / 50% / 80%）
python benchmark_AGARD_B_CL.py
```

每个脚本运行完毕后生成：
- **PNG 图表**：预测值 vs 真值散点图、残差图、指标柱状图（每种算法一列）
- **Excel 文件**：逐点预测结果及性能汇总表

### 在自己的代码中调用算法

```python
import numpy as np
from DataFusionAlgorithms import (
    mf_idw_interpolate,
    mf_ConvexHull,
    mf_GPy_CoKriging,
    mf_GPy_inverseDistanceMean,
)

# 示例：2D 输入，列向量输出
X_lf = np.random.rand(100, 2)       # 100 条 LF 数据，2 个特征
Y_lf = np.random.rand(100, 1)

X_hf = np.random.rand(20, 2)        # 20 条 HF 数据
Y_hf = np.random.rand(20, 1)

X_pred = np.random.rand(50, 2)      # 50 个预测点

# --- MF-IDW（Y 输入必须是 1D） ---
y_pred_idw = mf_idw_interpolate(X_lf, Y_lf.ravel(), X_hf, Y_hf.ravel(), X_pred)

# --- ConvexHull-GP ---
y_pred_ch = mf_ConvexHull(X_lf, Y_lf, X_hf, Y_hf, X_pred)

# --- GPy-CoKriging ---
y_pred_cok = mf_GPy_CoKriging(X_lf, Y_lf, X_hf, Y_hf, X_pred)

# --- GPy-InvDistMean ---
y_pred_idm = mf_GPy_inverseDistanceMean(X_lf, Y_lf, X_hf, Y_hf, X_pred)
```

> **注意：** MF-IDW 要求 Y 输入为 1D（已展平）数组；其余三种算法要求 Y 输入为列向量 `(n, 1)`。

---

## 评测结果

每个评测脚本输出一张三行布局的可视化图表：

| 行 | 内容 |
|----|------|
| 第一行 | 预测值 vs 真值散点图（每种算法一个子图），标注 R² 值 |
| 第二行 | 残差图，显示预测误差 vs 真值，并标注 ±RMSE 带 |
| 第三行 | RMSE、MAE、R²、平均相对误差（%）各指标的柱状对比图 |

结果同时保存至 Excel 文件，包含两个 sheet：
- **Performance**（性能汇总）— 各算法评估指标汇总表
- **Predictions**（预测明细）— 逐点预测值及残差

---

## 评估指标

| 指标 | 说明 |
|------|------|
| `RMSE` | 均方根误差 |
| `MAE` | 平均绝对误差 |
| `R²` | 决定系数（1 为完美拟合） |
| `MaxError` | 最大绝对预测误差 |
| `MeanRelErr(%)` | 平均相对误差（%），仅对 `|y_true| > 阈值` 的点计算，避免除零 |
| `Time(s)` | 算法运行时间（秒） |

各数据集相对误差过滤阈值：
- CN（LTV）：`|y_true| > 0.05`
- CL（AGARD-B）：`|y_true| > 0.005`
- CM（LTV）：`|y_true| > 0.5`

---

## API 接口说明

### `idw_interpolate(X_known, y_known, X_pred, p=2, epsilon=1e-8)`

标准反距离加权插值。

| 参数 | 类型 | 说明 |
|------|------|------|
| `X_known` | `ndarray (N, D)` | 已知点坐标 |
| `y_known` | `ndarray (N,)` | 已知点值（1D） |
| `X_pred` | `ndarray (M, D)` | 预测点坐标 |
| `p` | `float` | 幂参数，越大越局部化（默认：2） |
| `epsilon` | `float` | 防除零极小量（默认：1e-8） |

**返回：** `ndarray (M,)`

---

### `mf_idw_interpolate(X_LF, y_LF, X_HF, y_HF, X_pred, p_LF=2, p_R=2)`

多保真度 IDW 融合。

| 参数 | 类型 | 说明 |
|------|------|------|
| `X_LF, y_LF` | `ndarray` | LF 坐标和值 **（必须为 1D 展平）** |
| `X_HF, y_HF` | `ndarray` | HF 坐标和值 **（必须为 1D 展平）** |
| `X_pred` | `ndarray` | 预测点坐标 |
| `p_LF` | `float` | LF IDW 幂次（默认：2） |
| `p_R` | `float` | 残差 IDW 幂次（默认：2） |

**返回：** `ndarray (M,)`

---

### `mf_ConvexHull(X_l, Y_l, X_h, Y_h, X_new)`

基于凸包距离的 GP 融合。

| 参数 | 类型 | 说明 |
|------|------|------|
| `X_l, Y_l` | `ndarray (n_l, D)`, `(n_l, 1)` | LF 数据 |
| `X_h, Y_h` | `ndarray (n_h, D)`, `(n_h, 1)` | HF 数据 |
| `X_new` | `ndarray (n_new, D)` | 待预测点 |

**返回：** `ndarray (n_new,)`

---

### `mf_GPy_CoKriging(X_l, Y_l, X_h, Y_h, X_new)`

协同克里金多保真度融合（可选返回预测方差）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `X_l, Y_l` | `ndarray (n_l, D)`, `(n_l, 1)` | LF 数据（内部不做归一化） |
| `X_h, Y_h` | `ndarray (n_h, D)`, `(n_h, 1)` | HF 数据 |
| `X_new` | `ndarray (n_new, D)` | 待预测点 |

**返回：** `ndarray (n_new,)` *（如需同时返回方差，可修改源代码最后两行）*

---

### `mf_GPy_inverseDistanceMean(X_l, Y_l, X_h, Y_h, X_new)`

GP + 逆距离加权残差融合。

| 参数 | 类型 | 说明 |
|------|------|------|
| `X_l, Y_l` | `ndarray (n_l, D)`, `(n_l, 1)` | LF 数据 |
| `X_h, Y_h` | `ndarray (n_h, D)`, `(n_h, 1)` | HF 数据 |
| `X_new` | `ndarray (n_new, D)` | 待预测点 |

**返回：** `ndarray (n_new,)`

---

## 依赖包

| 包名 | 版本 | 用途 |
|------|------|------|
| `numpy` | 1.26.4 | 数值计算 |
| `scipy` | 1.12.0 | 距离计算、SLSQP 优化 |
| `GPy` | 1.13.2 | 高斯过程回归 |
| `scikit-learn` | 1.8.0 | 数据划分、评估指标、归一化 |
| `pandas` | 3.0.1 | 数据加载（Excel） |
| `matplotlib` | 3.10.8 | 可视化 |
| `openpyxl` | 3.1.5 | Excel 文件读写 |

---

## 注意事项

- 所有评测脚本均设置 `np.random.seed(42)`，保证结果可复现。
- GP 超参数（RBF 核的 `variance` 和 `lengthscale`）初始值均设为 1.0，通过最大似然估计（MLE）自动优化（`model.optimize()`），未进行多起始点搜索或交叉验证。
- ConvexHull-GP 由于对每个测试点逐一执行 SLSQP 优化，在测试集较大时运行速度较慢。
