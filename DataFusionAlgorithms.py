import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import GPy
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

############################## 算法1：IDW #######################################
### IDW融合算法特点：更简洁（相比MFK等方法），没有尺度变换的过程，即不需要缩放因子rho 
def idw_interpolate(X_known, y_known, X_pred, p=2, epsilon=1e-8):
    """
    Performs standard Inverse Distance Weighting (IDW) interpolation.

    Args:
        X_known (np.ndarray): Coordinates of known data points (N_samples, N_features).
        y_known (np.ndarray): Values at known data points (N_samples,).
        X_pred (np.ndarray): Coordinates of points to predict (N_pred, N_features).
        p (float): Power parameter. Higher p means local influence dominates.
        epsilon (float): Small number to avoid division by zero if X_pred is
                         exactly on top of X_known.

    Returns:
        np.ndarray: Predicted values at X_pred.
    """
    # Calculate distances between all prediction points and all known points
    # distances shape: (N_pred, N_samples)
    distances = cdist(X_pred, X_known, metric='euclidean')

    # Handle the case where distance is zero (prediction point = known point)
    # If distance is 0, set it to a very small number to avoid division by zero
    # in the weight calculation.
    distances[distances < epsilon] = epsilon

    # Calculate weights: w_i = 1 / d_i^p
    weights = 1.0 / (distances**p)

    # Calculate the weighted average
    # We use np.einsum for efficient matrix multiplication/summation
    # Numerator: Sum(w_i * y_i) for each prediction point
    numerator = np.einsum('ij,j->i', weights, y_known)

    # Denominator: Sum(w_i) for each prediction point
    denominator = np.sum(weights, axis=1)

    y_pred = numerator / denominator

    return y_pred


def mf_idw_interpolate(X_LF, y_LF, X_HF, y_HF, X_pred, p_LF=2, p_R=2):
    """
    Performs Multi-Fidelity Inverse Distance Weighting (MF-IDW)
    using the Hierarchical (Residual Correction) approach.

    Args:
        X_LF, y_LF: Low-Fidelity locations and values.
        X_HF, y_HF: High-Fidelity locations and values.
        X_pred: Locations to predict.
        p_LF: Power parameter for the LF model.
        p_R: Power parameter for the Residual model.

    Returns:
        np.ndarray: Fused multi-fidelity predictions at X_pred.
    """
    # Ensure inputs are numpy arrays and have the correct dimensions
    X_LF = np.atleast_2d(X_LF)
    X_HF = np.atleast_2d(X_HF)
    X_pred = np.atleast_2d(X_pred)
    y_LF = np.asarray(y_LF).ravel()
    y_HF = np.asarray(y_HF).ravel()

    # Step 1: Model the LF Surface across the prediction space
    y_pred_LF = idw_interpolate(X_LF, y_LF, X_pred, p=p_LF)

    # Step 2: Evaluate the LF model at the HF locations
    y_LF_at_HF = idw_interpolate(X_LF, y_LF, X_HF, p=p_LF)

    # Step 3: Calculate the Residuals (Error) at HF locations
    # Residual = HF_actual - LF_prediction_at_that_point
    residuals = y_HF - y_LF_at_HF

    # Step 4: Model the Residual Surface across the prediction space
    # We interpolate the residuals using the HF locations as the known points
    y_pred_R = idw_interpolate(X_HF, residuals, X_pred, p=p_R)

    # Step 5: Fusion (Additive Correction)
    # MF_prediction = LF_prediction + Residual_prediction
    y_pred_MF = y_pred_LF + y_pred_R

    return y_pred_MF.ravel()

############################## 算法2：ConvexHull #######################################

def mf_ConvexHull(X_l, Y_l, X_h, Y_h, X_new):
    """
    基于凸包距离的多保真度数据融合预测函数
    参数:
        X_l: 低保真度输入，形状 (n_l, input_dim)
        Y_l: 低保真度输出，形状 (n_l, 1)
        X_h: 高保真度输入，形状 (n_h, input_dim)
        Y_h: 高保真度输出，形状 (n_h, 1)
        X_new: 待预测输入，形状 (n_new, input_dim)
    返回:
        Y_pred: 融合后的预测值，形状 (n_new, 1)
    """


    input_dim = X_l.shape[1]
    n_l = X_l.shape[0]
    n_new = X_new.shape[0]

    # 步骤1: 定义计算点到凸包的距离函数
    def dist2ConvexHull(point, convex_hull):
        l = convex_hull.shape[0]

        def obj(x):
            result = point - sum(np.dot(np.diag(x), convex_hull))
            return np.linalg.norm(result)

        ineq_cons = {"type": "ineq", "fun": lambda x: x}
        eq_cons = {"type": "eq", "fun": lambda x: sum(x) - 1}

        x0 = np.ones(l) / l
        res = minimize(obj, x0, method='SLSQP',
                       constraints=[eq_cons, ineq_cons],
                       options={'ftol': 1e-9, 'disp': False})
        nrpoint = sum(np.dot(np.diag(res.x), convex_hull))
        return [res.fun, res.x, nrpoint]

    # 步骤2: 数据尺度变换
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_l)
    X_l_scaled = scaler.transform(X_l)
    X_h_scaled = scaler.transform(X_h)
    X_new_scaled = scaler.transform(X_new)

    # 步骤3: 计算新数据点(测试数据)到高保真度数据凸包的距离
    xnd = np.array([]) #
    for i in range(X_new_scaled.shape[0]):
        pointi = X_new_scaled[i, :]
        dist, lbd, nrpoint = dist2ConvexHull(pointi, X_h_scaled)
        nrpoint = sum(np.dot(np.diag(lbd), X_h_scaled))
        pointdist = np.hstack((nrpoint, dist))
        xnd = np.append(xnd, pointdist)
    xnd = xnd.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1] + 1)

    # 步骤4: 对低保真度数据拟合GP模型
    kernel_l = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
    model_l = GPy.models.GPRegression(X_l_scaled, Y_l, kernel_l)
    model_l.optimize(messages=False)

    # 步骤5: 对高保真度数据拟合GP模型
    kernel_h = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
    model_h = GPy.models.GPRegression(X_h_scaled, Y_h, kernel_h)
    model_h.optimize(messages=False)

    # 步骤6: 新数据点修正/预测
    Y_pred = np.zeros(n_new)
    for i in range(n_new):
        if xnd[i, input_dim] > 0.01: # 如果新数据点在凸包外，使用低保真度数据点修正/预测
            Y_pred[i] = model_l.predict(X_new_scaled[i, :].reshape(1, -1))[0] + (
                model_h.predict(xnd[i, :-1].reshape(1, -1))[0] -
                model_l.predict(xnd[i, :-1].reshape(1, -1))[0]
            )
        else: # 如果新数据点在凸包内，直接使用高保真度模型预测
            Y_pred[i] = model_h.predict(xnd[i, :-1].reshape(1, -1))[0]

    return Y_pred.ravel()


############################## 算法3：GPy_CoKriging #######################################
def mf_GPy_CoKriging(X_l, Y_l, X_h, Y_h, X_new):
    """
    多保真度协同克里金预测函数
    参数:
        X_l: 低保真度输入，形状 (n_l, input_dim)
        Y_l: 低保真度输出，形状 (n_l, 1)
        X_h: 高保真度输入，形状 (n_h, input_dim)
        Y_h: 高保真度输出，形状 (n_h, 1)
        X_new: 待预测输入，形状 (n_new, input_dim)
    返回:
        Y_pred: 高保真度预测值，形状 (n_new, 1)
    """

    input_dim = X_l.shape[1]

    # 步骤1: 对低保真度数据拟合GP模型
    kernel_l = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
    model_l = GPy.models.GPRegression(X_l, Y_l, kernel_l)
    model_l.optimize(messages=False)

    # 步骤2: 在高保真度数据点上预测低保真度值
    Y_l_hat_at_X_h, _ = model_l.predict(X_h)

    # 步骤3: 估计缩放因子 rho
    reg = LinearRegression().fit(Y_l_hat_at_X_h, Y_h)
    rho = reg.coef_[0][0]

    # 步骤4: 计算残差
    residual = Y_h - rho * Y_l_hat_at_X_h

    # 步骤5: 对残差拟合校正GP模型
    kernel_d = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
    model_d = GPy.models.GPRegression(X_h, residual, kernel_d)
    model_d.optimize(messages=False)

    # 预测函数（含不确定性）
    def predict_co_kriging_with_variance(X_new):
        Y_l_new, var_l_new = model_l.predict(X_new)
        delta_new, var_d_new = model_d.predict(X_new)
        Y_h_new = rho * Y_l_new + delta_new
        var_h_new = (rho ** 2) * var_l_new + var_d_new
        return Y_h_new, var_h_new

    Y_pred, _ = predict_co_kriging_with_variance(X_new)
    return Y_pred.ravel()
    #  如果需要返回方差，可以将最后两行改为：
    # Y_pred, var_pred = predict_co_kriging_with_variance(X_new)
    # return Y_pred, var_pred

############################## 算法4：GPy_inverseDistanceMean #######################################

def mf_GPy_inverseDistanceMean(X_l, Y_l, X_h, Y_h, X_new):

    # 步骤1: 数据尺度变换
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_l)
    X_l = scaler.transform(X_l)
    X_h = scaler.transform(X_h)
    X_new = scaler.transform(X_new)

    # 步骤2: 对低保真度数据拟合GP模型
    input_dim = X_l.shape[1]
    kernel_l = GPy.kern.RBF(input_dim=input_dim, variance=1.0, lengthscale=1.0)
    model_l = GPy.models.GPRegression(X_l, Y_l, kernel_l)
    model_l.optimize(messages=False)

    # 步骤3.1: 低保真度模型在高保真度数据点处的预测值
    Y_l_hat_at_X_h, _ = model_l.predict(X_h)
     # 步骤3.2:低保真度模型在新数据点处的预测值
    Y_l_hat_at_X_new, _ = model_l.predict(X_new)

    # 步骤4: 计算残差 (n_h,) 便于与 weights (n_new, n_h) 广播
    residual = (Y_h - Y_l_hat_at_X_h).ravel()

    # 步骤5: 计算X_new中每个点到X_h中所有点的距离
    distances = cdist(X_new, X_h, metric='euclidean')
    # 步骤6: 计算X_new中每个点到X_h中所有点的距离的倒数
    weights = 1.0 / distances
    # 步骤7: 计算X_new中每个点的权重
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    # 步骤8: 计算X_new中每个点的预测值 (weights (n_new,n_h) * residual (n_h,) -> (n_new,n_h))
    Y_pred = np.asarray(Y_l_hat_at_X_new).ravel() + np.sum(weights * residual, axis=1)
    return Y_pred.ravel()



