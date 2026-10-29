import warnings
from typing import cast

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import umap
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from plotly.subplots import make_subplots

warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings(action='ignore', category=DataConversionWarning, module='sklearn')


def _as_int(value, default):
    """把 dcc.Input(type='number') 可能返回的 None/字符串统一转成 int。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _align_xy(x, y):
    """把 x 统一成 ndarray、y 统一成「索引与 x 对齐」的 Series。

    直接对 DataFrame 调 np.mean(..., axis=0) 会返回「按列名索引」的 Series，
    再用整数位置去取就会 KeyError；这里统一成 ndarray，保证全程按位置取值。
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.ravel(np.asarray(y))
    name = y.name if isinstance(y, pd.Series) else 'label'
    return x_values, pd.Series(y_values, name=name)


def _as_numeric_labels(y):
    """PLS 回归（UVE / CARS）要求 y 是数值型。

    本项目的数据标签默认是类别名字符串（如 'Soybean-mintill'），
    直接送进 PLSRegression 会抛 ValueError，这里按出现顺序编码成 0..K-1。
    """
    values = np.asarray(y)
    if values.dtype.kind in "biufc":
        return values.astype(float)
    codes, _ = pd.factorize(values)
    return codes.astype(float)


def _encode_labels(*label_arrays):
    """把若干组标签统一编码成 0..K-1 的整数，并返回 (原始类别, 编码后各组)。

    用途：sklearn 1.7 的 MLPClassifier 在 early_stopping=True 且标签为字符串、
    类别数 > 2 时，会在 _score_with_function 里对字符串数组执行 np.isnan，
    抛出「ufunc 'isnan' not supported」。编码成整数即可规避，
    预测结果再用 classes[...] 映射回原标签，下游指标计算完全不受影响。
    """
    flat = np.concatenate([np.ravel(np.asarray(a)) for a in label_arrays])
    classes, inverse = np.unique(flat, return_inverse=True)
    sizes = [np.ravel(np.asarray(a)).size for a in label_arrays]
    encoded, start = [], 0
    for size in sizes:
        encoded.append(inverse[start:start + size])
        start += size
    return classes, encoded


def reduce_dimensionality(x, y, method, n_components=2, random_state=42, n_comps=8, cv=10, iteration=50):
    """降维 / 波段选择统一入口。

    返回 (df, fig)：
      df  —— 第一列为标签 y，其后为降维结果（Dimension n）或选中的波段
      fig —— plotly Figure

    方法名不在支持列表内时抛 ValueError。绝不允许隐式返回 None，
    否则调用方 `reduced_df, fig = reduce_dimensionality(...)` 会解包崩溃。
    """
    x_values, y_aligned = _align_xy(x, y)
    feature_names = np.asarray(x.columns) if isinstance(x, pd.DataFrame) else np.arange(x_values.shape[1])
    n_samples = x_values.shape[0]
    if n_samples < 2:
        raise ValueError("样本数少于 2，无法进行降维或波段选择。")

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_values)

    if method in ('PCA', 't-SNE', 'UMAP', 'MDS', 'Sparse PCA'):
        n_comp = max(1, _as_int(n_components, 2))
        # 慢方法的样本量上限：t-SNE/MDS/Sparse PCA 在几万样本上要跑几十分钟到几小时
        # （MDS 是 O(n²) 内存，5 万样本 = 20+ GB 距离矩阵，机器直接假死），
        # 超限时做确定性均匀抽稀，图上注明实际参与计算的样本数。
        slow_caps = {'t-SNE': 5000, 'UMAP': 10000, 'MDS': 2000, 'Sparse PCA': 5000}
        cap = slow_caps.get(method)
        if cap is not None and n_samples > cap:
            idx = np.linspace(0, n_samples - 1, cap).astype(int)
            x_fit = x_scaled[idx]
            y_fit = y_aligned.iloc[idx].reset_index(drop=True)
            subsample_note = f'showing {cap} of {n_samples} samples (uniform subsampling for speed)'
        else:
            x_fit = x_scaled
            y_fit = y_aligned
            subsample_note = None
        if method == 'PCA':
            pca = PCA(n_components=n_comp, random_state=random_state)
            x_transformed = pca.fit_transform(x_fit)
        elif method == 't-SNE':
            tsne = TSNE(n_components=n_comp, random_state=random_state)
            x_transformed = tsne.fit_transform(x_fit)
        elif method == 'UMAP':
            umap_model = umap.UMAP(n_components=n_comp, random_state=random_state)
            x_transformed = umap_model.fit_transform(x_fit)
        elif method == 'MDS':
            mds = MDS(n_components=n_comp, random_state=random_state)
            x_transformed = mds.fit_transform(x_fit)
        else:
            spa = SparsePCA(n_components=n_comp, random_state=random_state)
            x_transformed = spa.fit_transform(x_fit)

        column_names = [f'Dimension {i}' for i in range(1, n_comp + 1)]
        reduced_df = pd.DataFrame(data=x_transformed, columns=column_names)
        # 旧版 pandas 的 stub 把 Series.reset_index 的返回值标成联合类型，cast 收敛
        df = pd.concat([cast(pd.DataFrame, y_fit.reset_index(drop=True)), reduced_df], axis=1)

        # 散点图最多画 5000 个点：54129 个三维散点的 JSON 有几十 MB，浏览器渲染会卡死。
        # Reduced Data 表格仍保留全量（由调用方决定预览行数），这里只影响画图。
        MAX_PLOT_POINTS = 5000
        if len(reduced_df) > MAX_PLOT_POINTS:
            plot_idx = np.linspace(0, len(reduced_df) - 1, MAX_PLOT_POINTS).astype(int)
            plot_df = reduced_df.iloc[plot_idx]
            plot_y = y_fit.iloc[plot_idx]
        else:
            plot_df = reduced_df
            plot_y = y_fit

        if n_comp == 1:
            fig = px.scatter(plot_df, x='Dimension 1', color=plot_y, title='1D Visualization')
        elif n_comp == 2:
            fig = px.scatter(plot_df, x='Dimension 1', y='Dimension 2', color=plot_y,
                             title='2D Visualization')
        else:
            # 三维以上无法直接可视化，统一展示前三个维度，标题里注明总维度数
            fig = px.scatter_3d(plot_df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
                                color=plot_y, opacity=0.8,
                                title=f'{n_comp}D Visualization (仅显示前 3 维)')
            fig.update_traces(marker=dict(size=3))
        if subsample_note:
            fig.add_annotation(text=subsample_note, showarrow=False,
                               xref='paper', yref='paper', x=0.5, y=1.06,
                               font=dict(size=12, color='gray'))
        elif len(reduced_df) > MAX_PLOT_POINTS:
            fig.add_annotation(text=f'showing {MAX_PLOT_POINTS} of {len(reduced_df)} points',
                               showarrow=False, xref='paper', yref='paper', x=0.5, y=1.06,
                               font=dict(size=12, color='gray'))
        return df, fig

    elif method in ('SPA', 'UVE', 'CARS'):
        max_vars = max(1, _as_int(n_components, 2))
        n_comps = max(1, _as_int(n_comps, 8))
        cv = max(2, min(_as_int(cv, 10), n_samples))
        iteration = max(2, _as_int(iteration, 50))  # CARS 递推式含 1/(iteration-1)
        seed = _as_int(random_state, 42)
        # UVE / CARS 内部用 PLSRegression，标签必须是数值型
        y_numeric = _as_numeric_labels(y_aligned)

        if method == 'SPA':
            selected_vars = SPA(x_values, max_vars, seed)
        elif method == 'UVE':
            selected_vars = UVE(x_values, y_numeric, n_comps=n_comps, cv=cv)
        else:
            selected_vars = CARS(x_values, y_numeric, n_comps=n_comps, cv=cv,
                                 iteration=iteration, random_seed=seed)

        selected_vars = np.asarray(selected_vars, dtype=int)
        if selected_vars.size == 0:
            raise ValueError(f"{method} 未选出任何波段，请调整参数后重试。")

        # 关键：必须先转成 ndarray 再按「整数位置」取值。
        # 若直接用 DataFrame 调 np.mean(..., axis=0)，得到的是按列名索引的 Series，
        # 用整数位置去取会抛 KeyError（SPA/UVE/CARS 原实现即因此完全不可用）。
        average_spectrum = x_values.mean(axis=0)
        fig = px.line(
            x=feature_names,
            y=average_spectrum,
            labels={"x": "Wavelength", "y": "Intensity"},
            title="Spectral Data with Selected Features",
        )
        fig.add_scatter(
            x=feature_names[selected_vars],
            y=average_spectrum[selected_vars],
            mode="markers",
            name="Selected Wavelengths",
            marker_color="red",
        )
        selected_columns = pd.DataFrame(x_values[:, selected_vars],
                                        columns=[str(feature_names[i]) for i in selected_vars])
        df = pd.concat([y_aligned, selected_columns], axis=1)
        return df, fig

    raise ValueError(f"不支持的降维/波段选择方法: {method!r}")


def get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred):
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    test_recall = recall_score(y_test, y_test_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    test_f1 = f1_score(y_test, y_test_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    train_recall = recall_score(y_train, y_train_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]
    train_f1 = f1_score(y_train, y_train_pred, average='macro', zero_division=0)  # pyright: ignore[reportArgumentType]

    metrics_data = [
        {'Metric': 'Accuracy', 'Train': train_accuracy, 'Test': test_accuracy},
        {'Metric': 'Precision', 'Train': train_precision, 'Test': test_precision},
        {'Metric': 'Recall', 'Train': train_recall, 'Test': test_recall},
        {'Metric': 'F1_score', 'Train': train_f1, 'Test': test_f1}
    ]
    return metrics_data


def fit_RF(X_train, X_test, y_train, y_test, n_estimators, random_state, class_weight=None,
           return_model=False):
    # class_weight='balanced'：按类别频率反加权，显著改善小类「准确率为 0」的问题
    # n_jobs=-1：用满全部 CPU 核心。全量数据（38 万行 x 270 波段）单核训练要半小时以上
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state,
                                           class_weight=class_weight, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    y_train_pred = rf_classifier.predict(X_train)
    y_test_pred = rf_classifier.predict(X_test)
    if return_model:
        # 回调复用同一个已训练模型做全图预测，避免从头再训一遍（相当于训练时间翻倍）
        return y_train_pred, y_test_pred, rf_classifier.feature_importances_, rf_classifier
    return y_train_pred, y_test_pred, rf_classifier.feature_importances_


def fit_SVM(X_train, X_test, y_train, y_test, c, kernel='linear', class_weight=None,
            return_model=False):
    # class_weight='balanced'：小类别获得更大惩罚权重，缓解小类全错的失衡问题
    svm_classifier = SVC(kernel=kernel, C=c, class_weight=class_weight)
    svm_classifier.fit(X_train, y_train)
    y_train_pred = svm_classifier.predict(X_train)
    y_test_pred = svm_classifier.predict(X_test)
    if return_model:
        return y_train_pred, y_test_pred, svm_classifier
    return y_train_pred, y_test_pred


def fit_LDA(X_train, X_test, y_train, y_test, return_model=False):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    y_train_pred = lda.predict(X_train)
    y_test_pred = lda.predict(X_test)
    if return_model:
        # 回调复用同一模型做全图预测，避免在回调里再 fit 一遍（冗余训练）
        return y_train_pred, y_test_pred, lda
    return y_train_pred, y_test_pred


def _build_bpnn(hidden_layer_sizes, activation, alpha, learning_rate_init, max_iter,
                random_state, early_stopping):
    """构造 BPNN（标准化 + MLP）流水线。"""
    return Pipeline([
        ("scaler", StandardScaler()),  # BPNN 对特征尺度敏感，务必标准化
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=early_stopping,
            n_iter_no_change=10
        ))
    ])


def fit_BPNN_model(X_train, y_train,
                   hidden_layer_sizes=(100,),
                   activation="relu",
                   alpha=1e-4,
                   learning_rate_init=1e-3,
                   max_iter=300,
                   random_state=42,
                   early_stopping=True):
    """训练 BPNN，返回 (model, decode)。

    decode 把模型输出的 0..K-1 整数编码还原成原始标签，供全量分类图使用。
    集中在这里构造模型，避免调用方各写一份参数导致漂移。
    """
    classes, (y_train_enc,) = _encode_labels(y_train)
    model = _build_bpnn(hidden_layer_sizes, activation, alpha, learning_rate_init,
                        max_iter, random_state, early_stopping)
    model.fit(X_train, y_train_enc)
    return model, (lambda pred: classes[np.asarray(pred, dtype=int)])


def fit_BPNN(
        X_train, X_test, y_train, y_test,
        hidden_layer_sizes=(100,),  # (节点数, …) 可堆多层
        activation="relu",  # 'relu'、'tanh'、'logistic' 均可
        alpha=1e-4,  # L2 正则
        learning_rate_init=1e-3,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        return_model=False,
):
    """
    返回：训练集预测、测试集预测
    """
    model, decode = fit_BPNN_model(
        X_train, y_train,
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=early_stopping,
    )
    # 预测结果映射回原始标签（类别名 / 原始编号），保证下游指标口径一致
    y_train_pred = decode(model.predict(X_train))
    y_test_pred = decode(model.predict(X_test))
    if return_model:
        # 回调复用同一个已训练模型做全图预测，避免从头再训一遍神经网络
        return y_train_pred, y_test_pred, model, decode
    return y_train_pred, y_test_pred


def fit_KNN(X_train, X_test, y_train, y_test, n_neighbors=5, return_model=False):
    # n_jobs=-1：KNN 的 predict 是暴力距离计算，全量数据下多核能明显提速
    knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
    knn_classifier.fit(X_train, y_train)
    y_train_pred = knn_classifier.predict(X_train)
    y_test_pred = knn_classifier.predict(X_test)
    if return_model:
        return y_train_pred, y_test_pred, knn_classifier
    return y_train_pred, y_test_pred


def CARS(X_data, y, iteration=50, n_comps=8, cv=10, random_seed=42):
    """
    Competitive Adaptive Reweighted Sampling (接口保持不变)
    -----------------------------------------------------
    Parameters
    ----------
    X_data : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    iteration : int, default 50
        Monte-Carlo 抽样轮数
    n_comps : int, default 8
        PLSR 最大潜在变量数
    cv : int, default 10
        交叉验证折数
    random_seed : int, default 42
        随机种子

    Returns
    -------
    W_best : ndarray, 选出的最佳特征列索引
    """
    N, D = X_data.shape
    prob = 0.8  # 采样样本比例（与旧版一致）
    rng = np.random.default_rng(random_seed)

    # 递减子集规模 r(i) = a·exp(-k i)·D
    a = (D / 2) ** (1 / (iteration - 1))
    k = np.log(D / 2) / (iteration - 1)
    r = [round(a * np.exp(-k * i) * D) for i in range(1, iteration + 1)]

    weights = np.ones(D) / D  # 初始等权
    rmsecv_per_iter = []
    subsets = []

    for i in range(iteration):
        # ---- Monte-Carlo 抽样 ----
        id_cal = rng.choice(N, size=int(prob * N), replace=False)
        id_w = rng.choice(D, size=r[i], replace=False,
                          p=weights / weights.sum())
        subsets.append(id_w)

        X_cal = X_data[np.ix_(id_cal, id_w)]
        y_cal = y[id_cal]

        # ---- 拟合 PLSR 并更新权重 ----
        comp = min(n_comps, len(id_w))
        pls = PLSRegression(n_components=comp).fit(X_cal, y_cal)

        abs_coef = np.abs(pls.coef_).ravel()
        if abs_coef.sum() > 0:  # 避免除零
            weights[id_w] = abs_coef / abs_coef.sum()

        # ---- 交叉验证误差 ----
        mse = -cross_val_score(pls, X_cal, y_cal,
                               cv=cv, scoring="neg_mean_squared_error")
        rmsecv_per_iter.append(np.sqrt(mse).mean())

    # 选误差最小的一轮
    best_idx = int(np.argmin(rmsecv_per_iter))
    W_best = np.array(subsets[best_idx], dtype=int)
    return W_best


def UVE(X_data, y, n_comps=8, cv=10):
    """
    无信息变量消除算法
    :param X_data: 输入的光谱数据，形状为 (样本数, 变量数)
    :param y: 目标变量
    :param n_comps: PLS模型中的成分数
    :param cv: 交叉验证的折数
    :return: 被认为是信息性的变量的索引
    """
    # 偏最小二乘回归（PLSR）模型中使用的成分数。这应根据数据的复杂性和需要在响应变量中解释的方差进行调整。
    # cv（交叉验证折数）:用于交叉验证的折数。更改此数值可以影响模型评估的稳定性和准确性。增加折数通常可以提高评估的稳健性，但同样会增加计算负担。
    # 其他也许也能调整，看自己理解
    N, D = X_data.shape  # 获取样本数和变量数
    # 初始化PLS模型
    pls = PLSRegression(n_components=n_comps)
    # 基于整个数据集拟合PLS模型
    pls.fit(X_data, y)
    # 计算模型的系数
    original_coefs = np.abs(pls.coef_).reshape(-1)

    # 通过交叉验证计算每个变量的稳定性
    stability_scores = np.zeros(D)
    for i in range(D):
        # 从变量集中排除当前变量
        X_reduced = np.delete(X_data, i, axis=1)
        # 重新拟合PLS模型
        pls_cv = PLSRegression(n_components=n_comps)
        # 计算交叉验证的MSE分数
        cv_scores = cross_val_score(pls_cv, X_reduced, y, cv=cv, scoring='neg_mean_squared_error')
        stability_scores[i] = np.mean(np.abs(cv_scores))

    # 稳定性分数与原始系数的乘积决定了变量的信息性
    informative_scores = stability_scores * original_coefs
    # 设置阈值，选择信息性变量
    threshold = np.median(informative_scores)
    informative_vars = np.where(informative_scores >= threshold)[0]

    return informative_vars


def SPA(X_data, max_vars, random_seed):
    # max_vars（最大变量数）为输出的特征数量
    # 初始化参数
    n_samples, n_features = X_data.shape
    selected_vars = []
    remaining_vars = list(range(n_features))
    np.random.seed(random_seed)
    # 随机选择第一个特征
    first_var = np.random.choice(remaining_vars)
    selected_vars.append(first_var)
    remaining_vars.remove(first_var)

    # 定义正交投影函数
    def orthogonal_projection(v, V_proj):
        if V_proj.shape[1] == 0:
            return np.zeros_like(v)
        projection = V_proj @ np.linalg.pinv(V_proj.T @ V_proj) @ V_proj.T @ v
        return projection

    # 开始迭代选择其他特征
    while len(selected_vars) < max_vars and remaining_vars:
        max_residual = -np.inf
        next_var = None
        V_proj = X_data[:, selected_vars]  # 更新投影矩阵

        # 在剩余的特征中找到最小化投影残差的特征
        for var in remaining_vars:
            v = X_data[:, var]
            proj = orthogonal_projection(v, V_proj)
            residual = np.linalg.norm(v - proj)
            if residual > max_residual:
                max_residual = residual
                next_var = var

        if next_var is not None:
            selected_vars.append(next_var)
            remaining_vars.remove(next_var)

    return selected_vars


# ═══════════════════════════════════════════════════════════════════════════════
# Hyperspectral classification utilities
# ═══════════════════════════════════════════════════════════════════════════════
# 类别离散色板（与 layout_data.py 的 CLASS_PALETTE 保持一致）
_CLASS_PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
    '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff',
    '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1',
    '#000075', '#808080', '#f0f0f0', '#333333',
]

# ── 类别显隐（交互筛选）支持 ─────────────────────────────────────────────────
# 被隐藏的类别用中性浅灰：与背景黑区分开，一眼能看出「这里本来有像元、被筛掉了」，
# 而不是像透明那样看起来像空洞。
HIDDEN_CLASS_COLOR = '#cccccc'

# 客户端（layout_model 的类别显隐 clientside 回调）用同一份调色板重建 colorscale，
# 该 JS 源码由 unit._CLASS_PALETTE 序列化生成（单一来源，改这里两边同时生效）。
CLASS_PALETTE = list(_CLASS_PALETTE)


def class_color(class_id):
    """类别 id（1 起）对应的固定颜色；id<=0（背景）返回黑色。

    类别数超过调色板长度时循环取色（原实现用切片，超出部分会没有颜色）。
    """
    cid = int(class_id)
    if cid <= 0:
        return '#111111'
    return _CLASS_PALETTE[(cid - 1) % len(_CLASS_PALETTE)]


def build_class_colorscale(n_classes, hidden=()):
    """类别 id 0..n 的「硬阶梯」颜色映射。

    每个类别占据 z = i ± 0.5 的色带（而不是像逐点渐变那样被相邻候选色插值），
    这样：(1) 类别显示的就是调色板里的纯色；(2) 「隐藏某类」只需把该色带整体
    换成灰色——精确、无副作用。前端 JS 用同一公式重建，保证前后端一致。
    """
    n = max(int(n_classes), 1)
    hidden_ids = set(int(h) for h in hidden) if hidden else set()
    bg = HIDDEN_CLASS_COLOR if 0 in hidden_ids else '#111111'
    scale = [[0.0, bg], [0.5 / n, bg]]
    for i in range(1, n + 1):
        color = HIDDEN_CLASS_COLOR if i in hidden_ids else class_color(i)
        lo = (i - 0.5) / n
        hi = min((i + 0.5) / n, 1.0)
        scale.append([lo, color])
        scale.append([hi, color])
    return scale


def spectrum_figure(band_index, spectrum, title='Pixel Spectrum', reference=None,
                    reference_name='class mean'):
    """单个像元的原始光谱曲线（可叠加一条参考曲线，如该像元 GT 类别的均值光谱）。"""
    fig = go.Figure()
    if reference is not None:
        fig.add_trace(go.Scatter(x=band_index, y=reference, mode='lines',
                                 name=reference_name,
                                 line={'color': '#adb5bd', 'width': 2, 'dash': 'dash'}))
    fig.add_trace(go.Scatter(x=band_index, y=spectrum, mode='lines', name='Pixel spectrum',
                             line={'color': '#e6194b', 'width': 2}))
    fig.update_layout(title=title, xaxis_title='Band Index', yaxis_title='Reflectance',
                      margin=dict(t=40, b=56, l=0, r=0),
                      # 图例放横轴下方：放顶部会和长标题（含行列号）挤在同一行互相遮挡
                      legend=dict(orientation='h', yanchor='top', y=-0.18, x=0))
    return fig


def split_by_class(X_data, y, test_ratio=0.2, random_state=42):
    """
    Per-class stratified train/test split.
    Each class is independently sampled at *test_ratio* so the train/test
    composition stays balanced across all classes.
    Parameters
    ----------
    X_data : array-like, shape (N, B)
    y : array-like, shape (N,)
    test_ratio : float
    random_state : int
    Returns
    -------
    X_train, X_test, y_train, y_test : ndarray
    """
    rng = np.random.RandomState(random_state)
    X_data = np.asarray(X_data)
    y = np.asarray(y)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_mask = np.where(y == cls)[0]
        if len(cls_mask) == 1:
            # 单样本类别：全部放入训练集，避免测试集出现模型未见过的类
            train_idx.append(cls_mask[0])
            continue
        rng.shuffle(cls_mask)
        n_test = max(1, int(len(cls_mask) * test_ratio))
        test_idx.extend(cls_mask[:n_test])
        train_idx.extend(cls_mask[n_test:])
    return (X_data[train_idx], X_data[test_idx], y[train_idx], y[test_idx])


def oa_aa_kappa(y_true, y_pred, n_classes=None):
    """
    Compute Overall Accuracy (OA), Average Accuracy (AA), and Kappa.
    Parameters
    ----------
    y_true, y_pred : array-like
    n_classes : int, optional
        Total number of classes (defaults to max label in y_true).
    Returns
    -------
    dict with keys: 'OA', 'AA', 'Kappa'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    oa = accuracy_score(y_true, y_pred)
    per_class_acc = []
    for cls in labels:
        mask = y_true == cls
        if mask.sum() > 0:
            per_class_acc.append(accuracy_score(y_true[mask], y_pred[mask]))
    aa = np.mean(per_class_acc) if per_class_acc else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return {"OA": oa, "AA": aa, "Kappa": kappa}


def _to_class_ids(values, class_names):
    """把类别名（str）或类别 id（int）统一转为 1 起的类别 id。

    注意：不要在外层 np.asarray(values)，否则混合类型会被 numpy 强转成字符串。
    """
    name_to_id = {name: i + 1 for i, name in enumerate(class_names)}
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(name_to_id.get(v, 0))
        else:
            out.append(int(v))
    return np.array(out, dtype=int)


def confusion_matrix_fig(y_true, y_pred, class_names=None, title="Confusion Matrix"):
    """
    Build a Plotly heatmap figure of the confusion matrix.
    Rows = actual, columns = predicted.

    兼容两类输入：
      - y 为类别名（str）：labels 直接用类别名；
      - y 为类别 id（int，1 起）：labels 映射到 class_names 显示。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    if labels and isinstance(labels[0], str):
        tick_labels = labels
    else:
        tick_labels = [class_names[i - 1] if class_names and 0 < i <= len(class_names) else f"C{i}"
                       for i in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=tick_labels,
        y=tick_labels,
        text_auto=True,
        aspect="auto",
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title=title,
    )
    fig.update_layout(margin=dict(t=40, b=80), xaxis_tickangle=-45)
    return fig


def per_class_accuracy_table(y_true, y_pred, class_names=None):
    """
    Build a list of dicts (for dash_table) with per-class accuracy.
    Each row: {'Class': name, 'Samples': count, 'Correct': n, 'Accuracy': float}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(np.unique(y_true).tolist())
    rows = []
    for cls in labels:
        mask = y_true == cls
        total = int(mask.sum())
        correct = int((y_pred[mask] == cls).sum())
        acc = correct / total if total > 0 else 0.0
        if isinstance(cls, str):
            name = cls  # 类别名直接使用
        else:
            name = (class_names[cls - 1] if class_names and 0 < cls <= len(class_names)
                    else f"Class {cls}")
        rows.append({
            "Class": name,
            "Samples": total,
            "Correct": correct,
            "Accuracy": round(acc, 4),
        })
    return rows


def spatial_train_mask(gt, mode, cell=8):
    """按空间区域生成「有标签像元」的训练掩膜（行主序，与 data_loader.flatten 一致）。

    mode:
      'left-right'   左半图训练 / 右半图测试
      'top-bottom'   上半图训练 / 下半图测试
      'checkerboard' 棋盘格交错划分（cell×cell 像元为一格）
    返回 bool ndarray，shape = (N_labeled,)，True 为训练样本。

    区域划分把训练/测试放在互不相邻（棋盘）或互不重叠（半图）的空间区域，
    用于检验模型的空间泛化能力——随机抽样会高估同图精度。
    """
    gt = np.asarray(gt)
    if gt.ndim != 2:
        raise ValueError("spatial_train_mask requires a 2-D ground-truth map.")
    h_idx, w_idx = np.nonzero(gt > 0)
    if mode == 'left-right':
        is_train = w_idx < (gt.shape[1] / 2.0)
    elif mode == 'top-bottom':
        is_train = h_idx < (gt.shape[0] / 2.0)
    elif mode == 'checkerboard':
        is_train = ((h_idx // cell) + (w_idx // cell)) % 2 == 0
    else:
        raise ValueError(f"Unknown spatial split mode: {mode!r}")
    if is_train.all() or (~is_train).all():
        raise ValueError("Spatial split produced an empty train or test side; "
                         "check the ground-truth map.")
    return is_train


def classification_map(y_pred, gt, class_names=None, title="Classification Map", hidden=()):
    """
    Reshape 1-D predictions back to 2-D (H, W) and build a figure.

    当 gt 是 2-D 数组时输出「Prediction | Ground Truth」并排双面板
    （预测错分位置与 GT 直观对照）；gt 只给 (H, W) 形状时退化为单面板。
    Background pixels (label 0) are rendered as black.

    参数
    ----
    y_pred : array-like, shape (N,)
        非背景像元的预测（类别 id，1 起；或与 class_names 对齐的类别名）。
    gt : ndarray (H, W) 或 tuple (H, W)
        标签图（提供形状、背景掩膜与对照面板）。
    class_names : list, optional
        类别名，用于色标图例。
    """
    y_pred = np.asarray(y_pred)
    has_gt_map = hasattr(gt, 'shape') and len(getattr(gt, 'shape', ())) == 2
    if has_gt_map:
        h, w = gt.shape
        mask = gt.flatten() > 0
    else:
        h, w = gt
        mask = None
    if mask is not None:
        if y_pred.size != int(mask.sum()):
            raise ValueError(
                f"y_pred 长度 {y_pred.size} 与背景掩膜像元数 {int(mask.sum())} 不一致")
        full_map_flat = np.zeros(h * w, dtype=int)
        full_map_flat[mask] = y_pred
    else:
        # 无标签图：顺序填充
        full_map_flat = np.zeros(h * w, dtype=int)
        full_map_flat[:len(y_pred)] = y_pred
    full_map = full_map_flat.reshape(h, w)

    n_classes = int(full_map.max())
    if has_gt_map:
        n_classes = max(n_classes, int(np.asarray(gt).max()))
    if class_names and n_classes > 0:
        # 0（背景）为黑色；1..C 按调色板映射；hidden 里的类别显示为浅灰（被筛掉）。
        # 用「硬阶梯」色带而非渐变色阶，前端 JS 可以用同一公式重建（见 build_class_colorscale）。
        colorscale = build_class_colorscale(n_classes, hidden=hidden)
        tickvals = list(range(1, n_classes + 1))
        ticktext = [(class_names[i - 1] if 0 < i <= len(class_names) else f"C{i}")
                    for i in tickvals]
        if has_gt_map:
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04,
                                subplot_titles=(title, "Ground Truth"))
            fig.add_trace(go.Heatmap(
                z=full_map, zmin=0, zmax=n_classes, colorscale=colorscale,
                colorbar=dict(tickvals=tickvals, ticktext=ticktext)), 1, 1)
            fig.add_trace(go.Heatmap(
                z=np.asarray(gt), zmin=0, zmax=n_classes, colorscale=colorscale,
                showscale=False), 1, 2)
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            # Heatmap 默认 aspect=auto 会把图拉伸成绘图区形状（如 450x940 的图
            # 被拉成正方形）；锁定 1:1 像元比例，超出部分在绘图区内留白。
            for col in (1, 2):
                fig.update_xaxes(constrain="domain", row=1, col=col)
                fig.update_yaxes(scaleanchor="x" if col == 1 else "x2", scaleratio=1,
                                 constrain="domain", row=1, col=col)
            return fig
        fig = go.Figure(go.Heatmap(
            z=full_map, zmin=0, zmax=n_classes, colorscale=colorscale,
            colorbar=dict(tickvals=tickvals, ticktext=ticktext)))
    else:
        if has_gt_map:
            fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.04,
                                subplot_titles=(title, "Ground Truth"))
            fig.add_trace(go.Heatmap(z=full_map, zmin=0, zmax=max(n_classes, 1)), 1, 1)
            fig.add_trace(go.Heatmap(z=np.asarray(gt), zmin=0, zmax=max(n_classes, 1),
                                     showscale=False), 1, 2)
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
            # 同上：锁定 1:1 像元比例
            for col in (1, 2):
                fig.update_xaxes(constrain="domain", row=1, col=col)
                fig.update_yaxes(scaleanchor="x" if col == 1 else "x2", scaleratio=1,
                                 constrain="domain", row=1, col=col)
            return fig
        fig = px.imshow(full_map)
    fig.update_layout(coloraxis_showscale=False,
                      margin=dict(t=40, b=0, l=0, r=0))
    # 单面板同样锁定 1:1 像元比例
    fig.update_xaxes(constrain="domain")
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_layout(title=title)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3D-CNN 深度学习分类器（空间-光谱联合分类）
#
# 与传统分类器的区别：输入不是单像素光谱，而是以目标像元为中心的
# patch_size × patch_size × bands 立体邻域块，用 3D 卷积同时提取空间纹理
# 与光谱特征——这是高光谱深度学习分类的主流做法。
#
# 工程约束：
# - torch 懒导入：未安装时给出人话提示，而不是让整个平台 import 失败；
# - 补丁惰性提取：Dataset 按 batch 从 (H, W, B) 特征立方体里现取现用，
#   不把几十万个 patch 一次性物化成数组（那需要数 GB 内存）；
# - 归一化参数（逐波段均值/标准差）在训练时拟合并存到模型上，
#   全图预测时复用同一套参数，保证口径一致。
# ═══════════════════════════════════════════════════════════════════════════

def _require_torch():
    try:
        import torch
        from torch import nn  # noqa: F401
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is not installed. Run: "
            'python -m pip install "torch==1.13.1" '
            "--index-url https://download.pytorch.org/whl/cpu "
            "(torch 1.13.1 CPU is the latest version that supports Python 3.7).")


def _standardize(cube, mean=None, std=None):
    """逐波段零均值/单位方差。返回 (标准化后的 cube, mean, std) 供预测复用。"""
    cube = np.asarray(cube, dtype=np.float32)
    valid = cube[cube != 0]  # 背景零值不参与统计
    if mean is None:
        mean = float(valid.mean()) if valid.size else 0.0
    if std is None:
        std = float(valid.std()) if valid.size else 1.0
    std = std if std > 1e-8 else 1.0
    return (cube - np.float32(mean)) / np.float32(std), float(mean), float(std)


def _pad_cube(cube, half):
    """空间维度零填充 half 圈（边界像元的邻域块越界部分补零），光谱维不填。"""
    return np.pad(np.asarray(cube, dtype=np.float32),
                  ((half, half), (half, half), (0, 0)), mode="constant")


def _patch_at(padded, row, col, patch_size):
    """从填充立方体中取出以 (row, col) 为中心的 patch_size×patch_size×B 邻域块。"""
    return padded[row:row + patch_size, col:col + patch_size, :]


def _build_3dcnn(patch_size, n_bands, n_classes, torch, nn):
    """轻量 3D-CNN（Chen et al. 2016 风格的简化版，CPU 可训）。

    输入 (N, 1, P, P, B)：两个 3D 卷积层在空间+光谱联合域提特征，
    自适应池化到固定尺寸后接全连接层输出类别 logits。
    光谱卷积核取 min(7, B 且 >=1)，兼容波段数很少的降维特征。
    """
    k_spec = 7 if n_bands >= 7 else (3 if n_bands >= 3 else 1)

    class CNN3D(nn.Module):
        def __init__(self):
            super(CNN3D, self).__init__()
            self.conv1 = nn.Conv3d(1, 8, kernel_size=(3, 3, k_spec),
                                   padding=(1, 1, 0))
            self.bn1 = nn.BatchNorm3d(8)
            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3),
                                   padding=(1, 1, 1))
            self.bn2 = nn.BatchNorm3d(16)
            self.pool = nn.AdaptiveAvgPool3d((2, 2, 4))
            self.fc1 = nn.Linear(16 * 2 * 2 * 4, 64)
            self.drop = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, n_classes)
            self.patch_size = patch_size

        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.max_pool3d(x, kernel_size=(2, 2, 1))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.flatten(start_dim=1)
            x = torch.relu(self.fc1(x))
            x = self.drop(x)
            return self.fc2(x)

    return CNN3D()


def _patch_dataset(torch, padded, coords, labels, patch_size):
    """惰性补丁数据集：__getitem__ 时才从填充立方体取邻域块，内存只占一个 batch。"""
    from torch.utils.data import Dataset

    class PatchDataset(Dataset):
        def __len__(self):
            return len(coords)

        def __getitem__(self, idx):
            row, col = int(coords[idx][0]), int(coords[idx][1])
            patch = _patch_at(padded, row, col, patch_size)
            x = torch.from_numpy(np.ascontiguousarray(patch))
            x = x.unsqueeze(0)  # (1, P, P, B)
            y = int(labels[idx]) if labels is not None else 0
            return x, y

    return PatchDataset()


def _predict_ids(torch, model, padded, coords, patch_size, batch_size=256):
    """批量预测，返回 0..K-1 类别 id 数组。"""
    from torch.utils.data import DataLoader
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            chunk = coords[start:start + batch_size]
            patches = np.stack([_patch_at(padded, int(r), int(c), patch_size)
                                for r, c in chunk])
            x = torch.from_numpy(np.ascontiguousarray(patches)).unsqueeze(1)
            logits = model(x)
            preds.extend(torch.argmax(logits, dim=1).tolist())
    return np.array(preds, dtype=int)


def fit_3DCNN(cube, coords_train, y_train_ids, coords_test, y_test_ids,
              patch_size=7, epochs=20, batch_size=64, learning_rate=0.001,
              random_state=18, max_train_samples=8000):
    """训练 3D-CNN 分类器。

    参数
    ----
    cube : ndarray (H, W, B) 特征立方体（背景像元为 0，float32）
    coords_train / coords_test : ndarray (N, 2) 训练/测试像元的 (row, col) 坐标
    y_train_ids / y_test_ids : 训练/测试标签（0..K-1 整数 id）
    patch_size : 邻域块边长（5/7/9/11），越大空间上下文越多、越慢
    epochs / batch_size / learning_rate : 常规训练超参
    max_train_samples : 训练样本上限（按类分层抽稀），0 表示不限制；
        3D-CNN 单样本计算量远大于传统分类器，不设上限会训练数小时

    返回
    ----
    (model, train_pred_ids, test_pred_ids, history, seconds)
    history 为每轮平均 loss 列表；model 可继续传给 predict_3dcnn 做全图预测。
    """
    torch = _require_torch()
    import time
    from torch import nn
    from torch.utils.data import DataLoader

    t0 = time.time()
    torch.manual_seed(int(random_state))
    np.random.seed(int(random_state))

    patch_size = int(patch_size)
    if patch_size < 3:
        patch_size = 3
    epochs = max(1, int(epochs))
    batch_size = max(8, int(batch_size))

    cube, mean, std = _standardize(cube)
    padded = _pad_cube(cube, patch_size // 2)

    # 训练样本上限：按类分层等比抽稀（确定性），防止 CPU 训练失控
    y_train_ids = np.asarray(y_train_ids, dtype=int)
    coords_train = np.asarray(coords_train, dtype=int)
    # 全量训练集坐标：训练可以抽稀，但训练预测必须覆盖全部训练样本，
    # 否则上层指标表/训练结果表与 y_train 行数不一致（曾致 500「点了没反应」）
    coords_train_full = coords_train
    if 0 < int(max_train_samples) < len(y_train_ids):
        from sklearn.model_selection import train_test_split as _split
        stratify = y_train_ids if len(np.unique(y_train_ids)) > 1 else None
        keep_idx, _drop = _split(
            np.arange(len(y_train_ids)), train_size=int(max_train_samples),
            random_state=int(random_state), stratify=stratify)
        coords_train = coords_train[keep_idx]
        y_train_ids = y_train_ids[keep_idx]

    n_classes = int(max(y_train_ids.max(), y_test_ids.max() if len(y_test_ids) else 0)) + 1
    model = _build_3dcnn(patch_size, cube.shape[2], n_classes, torch, nn)
    model._x_mean, model._x_std = mean, std  # 全图预测复用同一套归一化参数

    train_ds = _patch_dataset(torch, padded, coords_train, y_train_ids, patch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))

    history = []
    model.train()
    for epoch in range(epochs):
        total, count = 0.0, 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(y_batch)
            count += len(y_batch)
        history.append({"epoch": epoch + 1, "loss": round(total / max(count, 1), 4)})

    # 用训练好的模型直接出训练/测试预测（不再像 RF 那样重训一遍）
    test_coords = np.asarray(coords_test, dtype=int)
    train_pred = _predict_ids(torch, model, padded, coords_train_full, patch_size)
    test_pred = _predict_ids(torch, model, padded, test_coords, patch_size)
    return model, train_pred, test_pred, history, time.time() - t0


def predict_3dcnn(model, cube, coords, batch_size=256):
    """用训练好的 3D-CNN 对任意像元坐标批量预测（全图分类图用）。"""
    torch = _require_torch()
    cube = np.asarray(cube, dtype=np.float32)
    if getattr(model, "_x_mean", None) is not None:
        cube = (cube - np.float32(model._x_mean)) / np.float32(model._x_std if model._x_std > 1e-8 else 1.0)
    patch_size = int(getattr(model, "patch_size", 7))
    padded = _pad_cube(cube, patch_size // 2)
    coords = np.asarray(coords, dtype=int)
    return _predict_ids(torch, model, padded, coords, patch_size, batch_size)
