from __future__ import annotations  # Python 3.7 兼容：让 list[str] / tuple[...] 注解延迟求值

import json
import os

import dash_bootstrap_components as dbc
import data_loader
import global_var
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unit
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from myapp import app
from sklearn.model_selection import train_test_split
from typing import Any, cast


def _num(value, default=0.0):
    """dcc.Input(type='number') 回调返回字符串，统一转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    return int(_num(value, default))


def _with_error_alert(danger_builder):
    """回调级异常兜底装饰器：任何未捕获异常都转成红色人话提示。

    背景（2026-09-24）：3D-CNN 曾因「训练预测行数与 y_train 不一致」抛出
    ValueError，而回调只在训练段包了 try，异常逃逸成 HTTP 500 —— 浏览器端
    整个响应无法应用，用户看到的是「点了 Train 什么反应都没有」。
    这类静默失败极难自查，所以六个分类器回调统一套一层兜底：
    异常 → 面板内弹出具体原因，绝不再出现 500 无反馈。

    danger_builder(exc) 需返回该回调「失败时」的完整返回值元组
    （表格置空、图置空、alert 打开并显示原因）。
    """
    import functools

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except PreventUpdate:
                raise  # 防抖/未点击属正常控制流，必须原样透传
            except Exception as exc:  # noqa: BLE001 兜底就是要接住一切
                print("[callback error] %s: %s: %s"
                      % (fn.__name__, type(exc).__name__, exc), flush=True)
                return danger_builder(exc)

        return wrapper

    return deco


def _records(frame):
    """DataFrame -> list[dict]，**不**重置索引（与原实现保持一致）。

    旧版 pandas 的 stub 未声明 to_dict() 的 orient 参数，这里集中做一次类型收敛，
    避免每个调用点都写一遍 ignore。
    """
    return cast(list, cast(pd.DataFrame, frame).to_dict("records"))  # pyright: ignore[reportArgumentType]


# 训练/测试结果表格只送预览行：Salinas 训练集约 4.3 万行 × 206 列，
# 全量转 records + JSON 会让回调卡住几分钟、浏览器内存爆炸。
# CSV 保存用的是完整 train_df/test_df，不受预览影响。
TABLE_PREVIEW_ROWS = 100

# 划分方式：随机分层 = 类别按比例抽样；三种空间划分 = 训练/测试来自
# 互不重叠（半图）或互不相邻（棋盘格）的区域，检验空间泛化能力。
# 注意：必须定义在控件列表之前（rf_controls 等会引用它）。
SPLIT_MODES = {
    "Random (stratified)": "random",
    "Spatial: left | right": "left-right",
    "Spatial: top | bottom": "top-bottom",
    "Spatial: checkerboard": "checkerboard",
}


def _preview_records(frame):
    return _records(cast(pd.DataFrame, frame).head(TABLE_PREVIEW_ROWS))


def _spectrum_message_figure(message):
    """交互分析区光谱图的占位画面（未点击像元 / 取不到光谱时写明原因）。

    定义在布局之前：布局里要用它作为 dcc.Graph 的初始 figure，
    否则该区域一开始是一块空白，用户不知道这里能做什么。
    """
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[dict(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font={"size": 13, "color": "#888888"})],
        height=300,
    )
    return fig


def build_data_table(table_id):
    return dash_table.DataTable(
        id=table_id,
        virtualization=True,
        style_table={"height": "100%", "overflowY": "auto"},
        style_cell={"minWidth": 95, "maxWidth": 95, "width": 95, "textAlign": "left"},
        style_as_list_view=True,
        fixed_rows={"headers": True},
    )


def build_classifier_item(title, description, controls, metrics_id, train_graph_id, test_graph_id,
                          train_table_id, test_table_id, feature_graph_id=None,
                          eval_graph_id=None, eval_table_id=None, class_map_id=None, alert_id=None):
    # 显式标注为 list[Any]：否则会被推断成 list[P]（html.P 的泛型），
    # 后续 append dbc.Alert / dbc.Row / html.Hr 都会报类型不匹配。
    children: list[Any] = [
        html.P(description),
    ]
    if alert_id is not None:
        children.append(dbc.Alert(id=alert_id, is_open=False, color="info", dismissable=True,
                                  style={"marginTop": "8px"}))
    children.append(
        dbc.Row(
            [
                dbc.Col(controls, width=6),
                dbc.Col([build_data_table(metrics_id)], width=6),
            ]
        )
    )
    children.append(html.Hr())
    # 全图分类图放最前：这是用户最关心的结果（原来在面板最底部，要滚很久才看到）
    if class_map_id is not None:
        children.append(
            dbc.Row([dcc.Graph(id=class_map_id)], className="mb-2")
        )
    children.append(
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id=train_graph_id)], width=6),
                dbc.Col([dcc.Graph(id=test_graph_id)], width=6),
            ]
        )
    )
    if feature_graph_id is not None:
        children.append(dbc.Row([dcc.Graph(id=feature_graph_id)]))
    children.append(
        dbc.Row(
            [
                dbc.Col([build_data_table(train_table_id)], width=6),
                dbc.Col([build_data_table(test_table_id)], width=6),
            ],
            justify="between",
        )
    )
    # ── Evaluation zone: OA/AA/Kappa confusion matrix + per-class table ──
    if eval_graph_id is not None:
        children.append(html.Hr())
        children.append(html.H5("Hyperspectral Evaluation", className="mt-2"))
        children.append(
            dbc.Row(
                [
                    dbc.Col([dcc.Graph(id=eval_graph_id)], width=6),
                    dbc.Col([build_data_table(eval_table_id)], width=6),
                ]
            )
        )
    return dbc.AccordionItem(children, title=title)


rf_controls = [
    dbc.Row([dbc.Col([html.P("Split Mode:")], width=4),
             dbc.Col([dcc.Dropdown(id="rf_split_mode", options=list(SPLIT_MODES),
                                   value="Random (stratified)", clearable=False)], width=8)]),
    dbc.Row([dbc.Col([html.P("Validation Split Ratio:")], width=4),
             dbc.Col([dcc.Slider(id="rf_validation_ratio", min=0, max=1, step=0.1, value=0.2, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})], width=6)]),
    dbc.Row([dbc.Col([html.P("Random Seed for Splitting:")], width=4),
             dbc.Col([dbc.Input(id="rf_random_validation", type="number", min=0, value=18, max=10000)], width=6)]),
    dbc.Row([dbc.Col([html.P("Model Seed:")], width=4),
             dbc.Col([dbc.Input(id="rf_random", type="number", min=0, value=2026, max=5000)], width=6)]),
    dbc.Row([dbc.Col([html.P("Number of Trees:")], width=4),
             dbc.Col([dbc.Input(id="rf_n_estimators", type="number", min=1, value=200, max=100000)], width=6)]),
    dbc.Row([dbc.Col([html.P("Class Weight:")], width=4),
             dbc.Col([dcc.Dropdown(id="rf_class_weight", options=["balanced", "none"],
                                   value="balanced", clearable=False)], width=6)]),
    dbc.Col([
        dbc.InputGroup([
            html.Button("Train and Save Results", id="fit-rf-button", n_clicks=0, className="btn btn-primary", **{"data-zh": "训练并保存结果"}),
            dbc.Input(id="fit-rf-input", placeholder="Enter Folder Path for Saving"),
        ])
    ], width=10),
]
svm_controls = [
    dbc.Row([dbc.Col([html.P("Split Mode:")], width=4),
             dbc.Col([dcc.Dropdown(id="svm_split_mode", options=list(SPLIT_MODES),
                                   value="Random (stratified)", clearable=False)], width=8)]),
    dbc.Row([dbc.Col([html.P("Validation Split Ratio:")], width=4),
             dbc.Col([dcc.Slider(id="svm_validation_ratio", min=0, max=1, step=0.1, value=0.2, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})], width=6)]),
    dbc.Row([dbc.Col([html.P("Random Seed for Splitting:")], width=4),
             dbc.Col([dbc.Input(id="svm_random_validation", type="number", min=0, value=18, max=10000)], width=6)]),
    dbc.Row([dbc.Col([html.P("Kernel Function:")], width=4),
             # 默认 linear：高光谱数据（数百个波段）本身近似线性可分，实测 linear 比 rbf
             # 精度更高且快一个数量级；非线性核在数万行上核矩阵要十几 GB 内存，不可行。
             dbc.Col([dcc.Dropdown(id="svm_kernel", options=["linear", "poly", "rbf", "sigmoid"], value="linear")],
                     width=6)]),
    dbc.Row([dbc.Col([html.P("Regularization Parameter:")], width=4),
             dbc.Col([dbc.Input(id="svm_c", type="number", min=0.1, value=1, max=100000, step=0.1)], width=6)]),
    dbc.Row([dbc.Col([html.P("Class Weight:")], width=4),
             dbc.Col([dcc.Dropdown(id="svm_class_weight", options=["balanced", "none"],
                                   value="balanced", clearable=False)], width=6)]),
    dbc.Col([
        dbc.InputGroup([
            html.Button("Train and Save Results", id="fit-svm-button", n_clicks=0, className="btn btn-primary", **{"data-zh": "训练并保存结果"}),
            dbc.Input(id="fit-svm-input", placeholder="Enter Folder Path for Saving"),
        ])
    ], width=10),
]
bpnn_controls = [
    dbc.Row([dbc.Col([html.P("Split Mode:")], width=4),
             dbc.Col([dcc.Dropdown(id="bpnn_split_mode", options=list(SPLIT_MODES),
                                   value="Random (stratified)", clearable=False)], width=8)]),
    dbc.Row([dbc.Col([html.P("Validation Split Ratio:")], width=4),
             dbc.Col([dcc.Slider(id="bpnn_validation_ratio", min=0, max=1, step=0.1, value=0.2, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})], width=6)]),
    dbc.Row([dbc.Col([html.P("Random Seed for Splitting:")], width=4),
             dbc.Col([dbc.Input(id="bpnn_random_validation", type="number", min=0, value=18, max=10000)], width=6)]),
    dbc.Row([dbc.Col([html.P("Hidden Layer Sizes:")], width=4),
             dbc.Col([dbc.Input(id="bpnn_hidden_layers", type="text", value="100",
                                placeholder="e.g. 100 or 120,60")], width=6)]),
    dbc.Row([dbc.Col([html.P("Activation Function:")], width=4),
             dbc.Col([dcc.Dropdown(id="bpnn_activation", options=["relu", "tanh", "logistic"], value="relu")],
                     width=6)]),
    dbc.Row([dbc.Col([html.P("Learning Rate (init):")], width=4),
             dbc.Col([dbc.Input(id="bpnn_learning_rate", type="number", value=0.001, min=1e-5, max=1)], width=6)]),
    dbc.Row([dbc.Col([html.P("Alpha (L2 penalty):")], width=4),
             dbc.Col([dbc.Input(id="bpnn_alpha", type="number", value=1e-4, step=1e-4, min=0, max=10)], width=6)]),
    dbc.Row([dbc.Col([html.P("Max Iterations:")], width=4),
             dbc.Col([dbc.Input(id="bpnn_max_iter", type="number", value=300, min=50, max=2000)], width=6)]),
    dbc.Col([
        dbc.InputGroup([
            html.Button("Train and Save Results", id="fit-bpnn-button", n_clicks=0, className="btn btn-primary", **{"data-zh": "训练并保存结果"}),
            dbc.Input(id="fit-bpnn-input", placeholder="Enter Folder Path for Saving"),
        ])
    ], width=10),
]
lda_controls = [
    dbc.Row([dbc.Col([html.P("Split Mode:")], width=4),
             dbc.Col([dcc.Dropdown(id="lda_split_mode", options=list(SPLIT_MODES),
                                   value="Random (stratified)", clearable=False)], width=8)]),
    dbc.Row([dbc.Col([html.P("Validation Split Ratio:")], width=4),
             dbc.Col([dcc.Slider(id="lda_validation_ratio", min=0, max=1, step=0.1, value=0.2, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})], width=6)]),
    dbc.Row([dbc.Col([html.P("Random Seed for Splitting:")], width=4),
             dbc.Col([dbc.Input(id="lda_random_validation", type="number", min=0, value=18, max=10000)], width=6)]),
    dbc.Col([
        dbc.InputGroup([
            html.Button("Train and Save Results", id="fit-lda-button", n_clicks=0, className="btn btn-primary", **{"data-zh": "训练并保存结果"}),
            dbc.Input(id="fit-lda-input", placeholder="Enter Folder Path for Saving"),
        ])
    ], width=10),
]
knn_controls = [
    dbc.Row([dbc.Col([html.P("Split Mode:")], width=4),
             dbc.Col([dcc.Dropdown(id="knn_split_mode", options=list(SPLIT_MODES),
                                   value="Random (stratified)", clearable=False)], width=8)]),
    dbc.Row([dbc.Col([html.P("Validation Split Ratio:")], width=4),
             dbc.Col([dcc.Slider(id="knn_validation_ratio", min=0, max=1, step=0.1, value=0.2, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})], width=6)]),
    dbc.Row([dbc.Col([html.P("Random Seed for Splitting:")], width=4),
             dbc.Col([dbc.Input(id="knn_random_validation", type="number", min=0, value=18, max=10000)], width=6)]),
    dbc.Row([dbc.Col([html.P("K Value:")], width=4),
             dbc.Col([dbc.Input(id="knn_k", type="number", min=1, value=5, max=1000, step=1)], width=6)]),
    dbc.Col([
        dbc.InputGroup([
            html.Button("Train and Save Results", id="fit-knn-button", n_clicks=0, className="btn btn-primary", **{"data-zh": "训练并保存结果"}),
            dbc.Input(id="fit-knn-input", placeholder="Enter Folder Path for Saving"),
        ])
    ], width=10),
]
cnn_controls = [
    dbc.Row([dbc.Col([html.P(
        "3D-CNN trains on CPU: with the defaults below expect a few minutes after clicking "
        "Train (5 epochs underfit badly, so 20 epochs is the default).",
        className="text-muted mb-1", style={"fontSize": "12px", "lineHeight": "1.35"},
        **{"data-zh": "3D-CNN 在 CPU 上训练：按下方默认设置，点击「训练」后大约需要几分钟"
                      "（5 轮会明显欠拟合，因此默认 20 轮），请耐心等待。"})], width=12)]),
    dbc.Row([dbc.Col([html.P("Split Mode:")], width=4),
             dbc.Col([dcc.Dropdown(id="cnn_split_mode", options=list(SPLIT_MODES),
                                   value="Random (stratified)", clearable=False)], width=8)]),
    dbc.Row([dbc.Col([html.P("Validation Split Ratio:")], width=4),
             dbc.Col([dcc.Slider(id="cnn_validation_ratio", min=0, max=1, step=0.1, value=0.2, marks=None,
                                 tooltip={"placement": "bottom", "always_visible": True})], width=6)]),
    dbc.Row([dbc.Col([html.P("Random Seed:")], width=4),
             dbc.Col([dbc.Input(id="cnn_seed", type="number", min=0, value=18, max=10000)], width=6)]),
    dbc.Row([dbc.Col([html.P("Patch Size (spatial neighborhood):")], width=4),
             dbc.Col([dcc.Dropdown(id="cnn_patch_size", options=[5, 7, 9, 11], value=7, clearable=False)], width=6)]),
    dbc.Row([dbc.Col([html.P("Epochs:")], width=4),
             # 默认 20：实测 5 轮在全量数据上 OA 仅 0.55（多个小类全错，明显欠拟合），
             # 20 轮升到 0.76；再调大学习率反而更差，所以提轮数而不是提 lr。
             dbc.Col([dbc.Input(id="cnn_epochs", type="number", min=1, value=20, max=200)], width=6)]),
    dbc.Row([dbc.Col([html.P("Batch Size:")], width=4),
             dbc.Col([dbc.Input(id="cnn_batch_size", type="number", min=8, value=64, max=1024)], width=6)]),
    dbc.Row([dbc.Col([html.P("Learning Rate:")], width=4),
             dbc.Col([dbc.Input(id="cnn_lr", type="number", value=0.001, min=1e-5, max=1, step=0.0005)], width=6)]),
    dbc.Row([dbc.Col([html.P("Max Training Samples (0 = all):")], width=4),
             dbc.Col([dbc.Input(id="cnn_max_samples", type="number", min=0, value=8000)], width=6)]),
    dbc.Row([dbc.Col([dcc.Checklist(
        id="cnn_predict_map",
        options=[{"label": " Predict full-image map", "value": "yes"}],
        value=["yes"])], width=12)]),
    dbc.Row([dbc.Col([html.P("Max Full-Map Pixels:")], width=4),
             dbc.Col([dbc.Input(id="cnn_max_map", type="number", min=1000, value=100000)], width=6)]),
    dbc.Col([
        dbc.InputGroup([
            html.Button("Train and Save Results", id="fit-cnn-button", n_clicks=0, className="btn btn-primary", **{"data-zh": "训练并保存结果"}),
            dbc.Input(id="fit-cnn-input", placeholder="Enter Folder Path for Saving"),
        ])
    ], width=10),
]
accordion = dbc.Accordion(
    [
        build_classifier_item(
            "Random Forest Classification",
            "Random forest is a classic baseline for hyperspectral classification and also provides feature importance.",
            rf_controls,
            "rf_metrics",
            "graph_train_rf",
            "graph_test_rf",
            "table_train_rf",
            "table_test_rf",
            feature_graph_id="feature_rf",
            eval_graph_id="eval_cm_rf",
            eval_table_id="eval_table_rf",
            class_map_id="class_map_rf",
            alert_id="alert-rf",
        ),
        build_classifier_item(
            "Support Vector Machine Classification",
            "Support vector machine is a widely used nonlinear classifier for hyperspectral data.",
            svm_controls,
            "svm_metrics",
            "graph_train_svm",
            "graph_test_svm",
            "table_train_svm",
            "table_test_svm",
            eval_graph_id="eval_cm_svm",
            eval_table_id="eval_table_svm",
            class_map_id="class_map_svm",
            alert_id="alert-svm",
        ),
        build_classifier_item(
            "Back-Propagation Neural Network Classification",
            "BPNN (MLPClassifier) can model nonlinear class boundaries after spectral preprocessing or band selection.",
            bpnn_controls,
            "bpnn_metrics",
            "graph_train_bpnn",
            "graph_test_bpnn",
            "table_train_bpnn",
            "table_test_bpnn",
            eval_graph_id="eval_cm_bpnn",
            eval_table_id="eval_table_bpnn",
            class_map_id="class_map_bpnn",
            alert_id="alert-bpnn",
        ),
        build_classifier_item(
            "Linear Discriminant Analysis Classification",
            "LDA is a lightweight and interpretable baseline for multi-class spectral classification.",
            lda_controls,
            "lda_metrics",
            "graph_train_lda",
            "graph_test_lda",
            "table_train_lda",
            "table_test_lda",
            eval_graph_id="eval_cm_lda",
            eval_table_id="eval_table_lda",
            class_map_id="class_map_lda",
            alert_id="alert-lda",
        ),
        build_classifier_item(
            "K-Nearest Neighbors Classification",
            "KNN is a simple distance-based classifier commonly used for classical baseline comparison.",
            knn_controls,
            "knn_metrics",
            "graph_train_knn",
            "graph_test_knn",
            "table_train_knn",
            "table_test_knn",
            eval_graph_id="eval_cm_knn",
            eval_table_id="eval_table_knn",
            class_map_id="class_map_knn",
            alert_id="alert-knn",
        ),
        build_classifier_item(
            "3D-CNN Classification (Deep Learning)",
            "3D-CNN jointly extracts spatial texture and spectral features from a "
            "patch around each pixel (PyTorch, CPU). It needs the hyperspectral image "
            "loaded on the Data Loading page and is much slower than classical "
            "classifiers — the sample cap keeps CPU training manageable.",
            cnn_controls,
            "cnn_metrics",
            "graph_train_cnn",
            "graph_test_cnn",
            "table_train_cnn",
            "table_test_cnn",
            eval_graph_id="eval_cm_cnn",
            eval_table_id="eval_table_cnn",
            class_map_id="class_map_cnn",
            alert_id="alert-cnn",
        ),
    ],
    start_collapsed=True,
)

# ── 交互分析卡片：点击像元取谱 + 类别显隐筛选 ──
# 注意：class-legend / pixel-info 的内容由回调动态填充，因此这两个容器
# 不能带 data-zh 属性——否则中英文切换回调会用 textContent 覆盖掉动态结果。
interactive_card = dbc.Alert(
    [
        html.H4("Interactive Analysis", className="mb-2", **{"data-zh": "交互分析"}),
        html.P(
            "How it works: (1) Train any classifier below — its classification map appears "
            "at the bottom of that panel. (2) Click a pixel on that map: its raw spectrum, "
            "ground-truth class and the model's prediction (misclassified pixels are "
            "flagged) show up here. (3) Uncheck a class above to grey it out on every map. "
            "This card stays empty until a map exists.",
            className="mb-3", style={"fontSize": "13px"},
            **{"data-zh": "使用步骤：（1）先在下方训练任意一个分类器，分类图会出现在该面板底部；"
                          "（2）在分类图上单击某个像元，这里就会画出该像元的原始光谱，并标注"
                          "真实类别与预测类别（错分像元会标红）；（3）取消勾选上方某个类别，"
                          "可把所有分类图上的该类置灰。没有训练出分类图之前，这里一直是空的。"}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.P("Visible Classes:", className="mb-1",
                               **{"data-zh": "显示类别："}),
                        dcc.Dropdown(id="class-filter", options=[], value=[], multi=True,
                                     placeholder="Load data / train a model first"),
                        html.Div(id="class-legend", className="mt-2"),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        html.P("Selected Pixel:", className="mb-1",
                               **{"data-zh": "选中像元："}),
                        html.Div("Empty for now — train a classifier below, then click a pixel "
                                 "on its classification map; its coordinates, ground-truth "
                                 "class and prediction will show up here.",
                                 id="pixel-info", style={"fontSize": "13px"}),
                    ],
                    width=6,
                ),
            ]
        ),
        dcc.Graph(id="pixel-spectrum", figure=_spectrum_message_figure(
            "Empty for now — train a classifier below, then click a pixel on a "
            "classification map to draw its raw spectrum here.")),
    ],
    className="alert-light",
    style={"marginBottom": "16px"},
)

layout_model = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Alert(
                        [
                            dbc.Row(
                                [
                                    html.H4("File Input", **{"data-zh": "文件输入"}),
                                    dbc.Row(
                                        dbc.Col(
                                            html.Button("Retrieve Reduced Data", id="button-get-reduce", **{"data-zh": "载入降维数据"}, className="btn btn-primary",
                                                       style={"width": "195px"}),
                                            width="auto",
                                        ),
                                        justify="center",
                                        className="mb-3",
                                    ),
                                    dbc.Row(
                                        dbc.Col(
                                            dbc.InputGroup(
                                                [
                                                    html.Button("Read File", id="button-read5", className="btn btn-primary", **{"data-zh": "读取文件"}),
                                                    dbc.Input(id="input-path5", type="text",
                                                              placeholder="Enter CSV File Path"),
                                                ],
                                                style={"width": "195px"},
                                            ),
                                            width="auto",
                                        ),
                                        justify="center",
                                        className="mb-3",
                                    ),
                                    html.Hr(),
                                    html.H4("Loaded Data:", className="mb-3", **{"data-zh": "已载入数据："}),
                                    # 载入结果提示：成功显示行列数与来源，失败显示红色原因。
                                    # 之前 Read File 失败（路径错/文件读不了）是静默失败，
                                    # 用户不知道数据到底载入没有，容易拿旧数据继续训练。
                                    dbc.Alert(id="alert-model-load", is_open=False,
                                              dismissable=True, color="info",
                                              style={"fontSize": "12px", "padding": "8px"}),
                                    html.P("Label Column(Y):"),
                                    dcc.Dropdown(id="modle_y", clearable=False,
                                                 placeholder="Select Label Column(Y)"),
                                    html.P("Feature Columns(X):"),
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dcc.RangeSlider(
                                                        0, 10, 1, id="modle_x", marks=None,
                                                        tooltip={"placement": "bottom", "always_visible": True},
                                                    ),
                                                    dbc.Tooltip(id="tooltip5", target="modle_x",
                                                                placement="right", is_open=False),
                                                ]
                                            ),
                                        ]
                                    ),
                                    build_data_table("table_in5"),
                                ]
                            )
                        ],
                        className="alert-light",
                        style={"height": "calc(100vh - 80px)", "width": "15%", "position": "fixed"},
                    )
                ),
                dbc.Col([interactive_card, accordion], width=10),
            ]
        ),
    ],
    fluid=True,
)


def load_model_dataframe(triggered, path):
    if triggered == "button-read5":
        file_path = str(path).strip() if path else ""
        if not file_path:
            raise ValueError("Please enter the CSV file path first.")
        if os.path.isdir(file_path):
            csvs = sorted(f for f in os.listdir(file_path) if f.lower().endswith(".csv"))
            hint = (" Files inside: " + "; ".join(csvs[:5])) if csvs else ""
            raise ValueError(
                "'%s' is a folder, not a CSV file.%s "
                "Please copy the full path of the .csv file inside it." % (file_path, hint))
        if not os.path.exists(file_path):
            # 常见笔误：忘写 .csv 扩展名（如 .../20260921_101446），能补就补
            if os.path.exists(file_path + ".csv"):
                file_path = file_path + ".csv"
            else:
                raise ValueError(
                    "File not found: '%s'. Check the file name and extension (usually .csv)."
                    % file_path)
        # 旧版 pandas 的 stub 把 read_csv 标成「DataFrame | TextFileReader」，运行时必是 DataFrame
        return cast(pd.DataFrame, pd.read_csv(file_path))
    if triggered == "button-get-reduce":
        reduced_data = global_var.get_global_lower_table()
        # lower_table 可能是 list[dict]（旧）或 DataFrame（现在），len() 两者通用；
        # 不能写 `not reduced_data`（DataFrame 的真值判断会抛 ValueError）。
        if reduced_data is None or len(reduced_data) == 0:
            raise ValueError(
                "No reduced data is available yet. Run a dimensionality reduction method "
                "(PCA etc.) on the Dimensionality Reduction page first, or use Read File "
                "with a full-band CSV.")
        return pd.DataFrame(reduced_data)
    raise PreventUpdate


def prepare_dataset(modle_x, modle_y) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if modle_y is None or modle_x is None:
        raise ValueError("Please select the label column and feature columns first!")
    data = global_var.get_global_input_table()
    if data.empty:
        raise ValueError("No data is currently loaded.")
    column_name = global_var.get_global_column_name()
    # Y 失效自动回退：换过一次数据后，旧的 Y 选择可能已不存在
    # （如从原始波段切到降维数据后 Y 还停在 band_5），优先回退到 label 列。
    if modle_y not in column_name:
        if "label" in column_name:
            modle_y = "label"
        else:
            raise ValueError(
                "Label Column(Y) '%s' no longer exists in the loaded data. "
                "Please re-select it in the left panel." % str(modle_y))
    lo, hi = int(modle_x[0]), int(modle_x[1])  # RangeSlider 可能返回 float
    feature_columns = column_name[lo:hi + 1]
    feature_columns = [column for column in feature_columns if column != modle_y]
    # 自动剔除特征区间里的非数值列（如 label 混在滑杆范围内）：
    # sklearn 无法用字符串列训练，与其报错打断，不如直接排除并继续。
    def _is_numeric(column):
        kind = getattr(getattr(data[column], "dtype", None), "kind", "O")
        return kind in "iufcb"
    feature_columns = [column for column in feature_columns if _is_numeric(column)]
    if not feature_columns:
        raise ValueError(
            "The selected Feature Columns(X) range contains no numeric columns. "
            "Please widen the range in the left panel or re-select Label Column(Y).")
    # pandas 的 stub 把 data[list] / data[标量] 都标成 DataFrame | Series 的联合，
    # 实际前者必是 DataFrame、后者必是 Series，这里用 cast 收敛（运行时语义不变）。
    # 同时显式标注 DataFrame：不标注会让下面 isna().any() 被误判为「对 bool 取 .any()」。
    selected = cast(pd.DataFrame, data[feature_columns])
    data_x = cast(pd.DataFrame, selected.apply(pd.to_numeric, errors="coerce"))
    if data_x.isna().to_numpy().any():
        # 指名问题列（数值列里的缺失值），否则用户只看到一句笼统报错没法排查
        bad = [str(c) for c in data_x.columns
               if cast(pd.Series, data_x[c]).isna().any()]
        raise ValueError(
            "These feature columns contain missing values: "
            + ", ".join(bad)
            + ". Please narrow the Feature Columns(X) range or clean the data first.")
    data_y = cast(pd.Series, data[modle_y])
    # 分类目标必须是离散类别：Y 选成连续数值列（如降维数据的 Dimension 1）时，
    # sklearn 只会抛 "Unknown label type: 'continuous'"，用户无法理解，提前拦截。
    y_kind = getattr(data_y.dtype, "kind", "O")
    n_unique = cast(int, data_y.nunique())
    if y_kind == "f" or (y_kind in "iu" and n_unique > 100):
        raise ValueError(
            "Label Column(Y) '%s' is a continuous numeric column (%d unique values), "
            "not a class label. Classification needs a categorical column — "
            "select 'label' (class names) as Label Column(Y)." % (str(modle_y), n_unique))
    return data_x, data_y, feature_columns


def split_dataset(data_x: pd.DataFrame, data_y: pd.Series, validation_ratio, random_state,
                  split_mode="Random (stratified)"
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    mode = SPLIT_MODES.get(str(split_mode), "random")
    if mode != "random":
        # 空间划分：需要先在数据加载页加载高光谱图像（gt 提供像元坐标），
        # 且当前建模数据必须与展平像元一一对应（行主序、全量、无抽稀）。
        gt = global_var.get_global_gt()
        if gt is None or len(gt) == 0:
            raise ValueError(
                "Spatial split needs the hyperspectral image loaded on the Data Loading page first.")
        mask = unit.spatial_train_mask(gt, mode)
        if len(mask) != len(data_y):
            raise ValueError(
                "Spatial split only works with the full-band pixel data in image order "
                f"(got {len(data_y)} rows, image has {len(mask)} labeled pixels). "
                "Load it via 'Retrieve' buttons, not subsampled/reduced data.")
        return (cast(pd.DataFrame, data_x[mask]), cast(pd.DataFrame, data_x[~mask]),
                cast(pd.Series, data_y[mask]), cast(pd.Series, data_y[~mask]))
    validation_ratio = _num(validation_ratio, 0.2)
    # 滑块范围是 0~1，取 0 或 1 会让 train_test_split 直接抛错，这里夹到安全区间
    validation_ratio = min(max(validation_ratio, 0.05), 0.9)
    random_state = _int(random_state, 18)
    # 旧版 pandas 的 stub 把 value_counts().min() 标成 Dtype，取不出数值，这里 cast 成 float
    min_count = cast(float, data_y.value_counts().min())
    stratify = data_y if data_y.nunique() > 1 and min_count >= 2 else None
    try:
        parts = train_test_split(data_x, data_y, test_size=validation_ratio, random_state=random_state,
                                 stratify=stratify)
    except ValueError:
        parts = train_test_split(data_x, data_y, test_size=validation_ratio, random_state=random_state)
    # sklearn 的 train_test_split 类型标注声明返回 list[...]，直接解包会让
    # 类型检查器把每个元素推断成 Unknown | list[Unknown]，这里显式收敛成元组。
    x_train, x_test, y_train, y_test = parts
    return (cast(pd.DataFrame, x_train), cast(pd.DataFrame, x_test),
            cast(pd.Series, y_train), cast(pd.Series, y_test))


def _missing_train_classes(y_train, y_all) -> list:
    """训练集中没有出现的类别（如空间划分把小类整个划到了测试侧）。"""
    present = set(pd.unique(pd.Series(y_train)).tolist())
    total = set(pd.unique(pd.Series(y_all)).tolist())
    return sorted((c for c in total if c not in present), key=str)


def _missing_class_warning(missing) -> str:
    """缺类预警文案：提醒读者 AA 被拉低可能是划分造成的，不是模型差。"""
    if not missing:
        return ""
    shown = ", ".join(str(c) for c in missing[:8]) + ("..." if len(missing) > 8 else "")
    return (" Warning: " + shown + " has no training samples under this split, "
            "so its precision/recall are 0 and AA is dragged down by the split, "
            "not necessarily by the model. "
            "Try Random (stratified) or another spatial mode.")


def _train_alert(y_train, y_all, extra_note="", y_test=None, y_test_pred=None, elapsed=None):
    """成功提示（含缺类/零精度类别预警）：返回完整的 (children, is_open, color) 三元组。
    缺类、测试集存在 0 精度类别、或空间划分用在降维特征上时，绿色变黄色。
    elapsed 只是展示训练耗时，不影响颜色判定。"""
    missing = _missing_train_classes(y_train, y_all)
    zero = _zero_recall_classes(y_test, y_test_pred)
    elapsed_note = "" if elapsed is None else " Elapsed %.0f s for training." % elapsed
    text = ("Model Training Completed!" + _missing_class_warning(missing)
            + _zero_recall_warning(zero) + extra_note + elapsed_note)
    return (text, True,
            "warning" if (missing or zero or extra_note) else "success")


def _zero_recall_classes(y_test, y_test_pred) -> list:
    """测试集里一个都没预测对的类别（通常由类别失衡/样本太少导致）。"""
    if y_test is None or y_test_pred is None:
        return []
    y_true = pd.Series(list(y_test))
    y_pred = pd.Series(list(y_test_pred))
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return []
    zero = []
    for cls in pd.unique(y_true):
        mask = y_true == cls
        if int((y_pred[mask] == cls).sum()) == 0:
            zero.append(cls)
    return sorted(zero, key=str)


def _zero_recall_warning(zero) -> str:
    """0 精度类别预警文案：点名类别并给出可操作建议。"""
    if not zero:
        return ""
    shown = ", ".join(str(c) for c in zero[:8]) + ("..." if len(zero) > 8 else "")
    return (" Warning: every test sample of " + shown + " was misclassified (accuracy 0). "
            "This usually means class imbalance or too few samples for these classes — "
            "keep 'Class Weight: balanced' (RF/SVM), or retrain with more samples of them.")


def _reduced_feature_note(split_mode, feature_columns) -> str:
    """空间划分用在降维特征上时的提示：检验空间泛化是它的用途，
    但降维特征（如 t-SNE 维度）上空间泛化必然很差，提示读者别误以为模型差。"""
    if SPLIT_MODES.get(str(split_mode), "random") == "random":
        return ""
    hs_X = global_var.get_global_hs_X()
    if hs_X is None or len(hs_X) == 0 or hs_X.shape[1] == len(feature_columns):
        return ""
    return (" Note: the spatial split is applied to reduced features (%d dims) instead of "
            "image bands (%d). Spatial generalization on reduced features is expected to be "
            "low — use Random (stratified) split for reference accuracies."
            % (len(feature_columns), hs_X.shape[1]))


def _map_placeholder(reason: str):
    """全图分类图无法生成时，画一个带原因的占位图（原来是空坐标轴，用户不知道为什么）。

    Plotly 注释不会自动换行，长句超出面板宽度会被裁掉（用户只能看到半句话），
    这里手动按 ~78 字符折行并居中。
    """
    lines, cur = [], ""
    for word in reason.split(" "):
        if cur and len(cur) + 1 + len(word) > 78:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word) if cur else word
    if cur:
        lines.append(cur)
    fig = go.Figure()
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        annotations=[dict(text="<br>".join(lines), x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, align="center",
                          font={"size": 12, "color": "#888888"})],
        height=320,
    )
    return fig


def build_confusion_figure(y_true, y_pred, title):
    # 注意：pd.Index.union 不去重，必须用 set 去重后再排序；
    # px.imshow 传 DataFrame 在 plotly 7(narwhals) 下会因重名列报错，改传 numpy 值 + 显式坐标。
    labels = sorted(set(pd.Index(y_true)) | set(pd.Index(y_pred)))
    matrix = pd.crosstab(pd.Series(y_true, name="Actual"), pd.Series(y_pred, name="Predicted"))
    matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)
    return px.imshow(matrix.values, x=labels, y=labels, text_auto=True, aspect="auto",
                     labels={"x": "Predicted", "y": "Actual", "color": "Count"}, title=title)


def build_result_tables(x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series,
                        y_train_pred, y_test_pred) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = x_train.copy()
    train_df["y_train_true"] = y_train.to_numpy()
    train_df["y_train_pred"] = y_train_pred
    test_df = x_test.copy()
    test_df["y_test_true"] = y_test.to_numpy()
    test_df["y_test_pred"] = y_test_pred
    return train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════
# 交互分析（点击像元取谱 / 类别显隐筛选）
#
# 这两项把平台从「参数进去、结果出来」的单向流水线，变成可以交互探索、
# 诊断模型输出的分析界面：
#   - 在任意分类图上单击像元 → 画该像元的原始光谱曲线，并给出 GT 类别、
#     该分类器的预测类别、是否错分（人在回路诊断的入口）；
#   - 取消勾选类别 → 所有分类图上的该类置灰（观察剩余类别的空间分布）。
#
# 注意调用链必须单向（避免渲染函数互调造成循环）：
#   训练回调 → 写 _PRED_MAPS → 输出图
#   点击事件 → 读 _PRED_MAPS + 全局光谱 → 输出光谱图（不触发任何训练回调）
# ═══════════════════════════════════════════════════════════════════════════

# 分类图组件 id → 分类器显示名（用于光谱图的「预测类别」标注）
_CLASSIFIER_LABELS = {
    "class_map_rf": "Random Forest",
    "class_map_svm": "SVM",
    "class_map_bpnn": "BPNN",
    "class_map_lda": "LDA",
    "class_map_knn": "KNN",
    "class_map_cnn": "3D-CNN",
}
_CLASS_MAP_IDS = list(_CLASSIFIER_LABELS)

# 最近一次全图预测结果（H×W 类别 id 网格）。点击像元时反查预测类别与是否错分，
# 不必为了看一个点重新跑模型。键为分类图组件 id。
_PRED_MAPS: dict = {}

# 最近一次分类图的说明（成功且无异常时为空串）。用于把「图为什么出不来/为什么
# 自动换了数据集」这类原因抬到面板顶部的提示条——否则用户只看到一张空图或
# 一行灰色小字，很容易误判成「点了没反应」。
_MAP_NOTES: dict = {}

# 数据集短码 → 人话名称（提示文案用）
_DATASET_LABELS = {"IP": "Indian Pines", "SA": "Salinas",
                   "LK": "WHU-Hi LongKou", "HH": "WHU-Hi HongHu"}


def _map_note(graph_id) -> str:
    """取某张分类图的附加说明（拼进训练提示条）；没有则返回空串。"""
    note = str(_MAP_NOTES.get(graph_id) or "")
    return (" " + note) if note else ""


def _dataset_labels_used(labels):
    """训练数据里出现的类别名集合（只统计字符串标签）。"""
    used = set()
    if labels is None:
        return used
    for value in np.asarray(labels, dtype=object):
        if isinstance(value, str) and value:
            used.add(value)
    return used


def dataset_mismatch_reason(train_labels) -> str:
    """判断「内存里的高光谱图像」与「本次训练数据」是否互相矛盾。

    返回空串表示两者一致；否则返回原因（英文，直接进提示文案）。
    这是分类图出错最常见的根因：分类图始终画在内存中已载入的数据集上，与训练
    用的 CSV 无关；载入 A 数据集、却用 B 数据集的 CSV 训练时，A 的类别名与 B 的
    标签集根本对不上——图不是画错，而是被数据集的类别表判为「对不上」而不可用。

    判定只用「类别名集合」这一条：波段数不能当判据（用降维特征训练时波段数
    本来就不等于立方体波段数，那是合法用法）。
    """
    hs_X = global_var.get_global_hs_X()
    hs_gt = global_var.get_global_gt()
    if hs_X is None or len(hs_X) == 0 or hs_gt is None or len(hs_gt) == 0:
        return "no hyperspectral image is currently loaded"
    known = {str(c) for c in (global_var.get_global_class_names() or [])}
    used = _dataset_labels_used(train_labels)
    if known and used and not used.issubset(known):
        unknown = sorted(used - known)[:3]
        return ("the training labels (%s, ...) do not belong to the loaded dataset (%s)"
                % (", ".join(unknown), global_var.get_global_dataset_name() or "unknown"))
    return ""


def _try_recover_dataset(train_labels, data_x) -> str:
    """训练数据与内存图像对不上时，按训练数据自动载入正确的数据集。

    只用 CSV 的类别名 + 行数反查（比对 data/ 下的内置数据集），两条都命中才
    载入，因此不会误换数据集；也绝不需要为此重启服务。
    返回说明文案（无需恢复时为空串）。
    """
    if train_labels is None or data_x is None or len(data_x) == 0:
        return ""
    reason = dataset_mismatch_reason(train_labels)
    if not reason:
        return ""
    used = _dataset_labels_used(train_labels)
    if not used:
        return ""
    ds_name, mat_path = data_loader.match_dataset_by_training_data(used, len(data_x))
    if not ds_name or not mat_path:
        # 对不上任何内置数据集（如自定义数据）：保持原样，由调用方给出可操作提示
        return ""
    previous = global_var.get_global_dataset_name()
    previous = _DATASET_LABELS.get(previous, previous) or "no dataset"
    cube, gt, loaded_name = data_loader.load_mat(mat_path, ds_name)
    data_loader.apply_dataset_state(cube, gt, loaded_name)
    return ("Full-image map: automatically switched the loaded image to %s because your "
            "training data matches it, not the previously loaded %s (%s). No restart needed; "
            "re-select the dataset on the Data Loading page if you want its figures refreshed."
            % (_DATASET_LABELS.get(loaded_name, loaded_name), previous, reason))


def _csv_dataset_warning(data, label_column) -> str:
    """读取 CSV 时提前预警「这份 CSV 与内存里已载入的数据集不是同一套」。

    提前说比等到训练完发现分类图出不来友好得多；训练时还会自动载入匹配的
    数据集，所以这里只提示、不阻断。
    """
    known = {str(c) for c in (global_var.get_global_class_names() or [])}
    if not known or label_column is None:
        return ""
    if label_column not in getattr(data, "columns", []):
        return ""
    used = _dataset_labels_used(data[label_column].unique())
    if not used or used.issubset(known):
        return ""
    ds = global_var.get_global_dataset_name()
    return (" Note: this CSV's classes do not match the loaded dataset (%s). Training still "
            "runs, and the classification map will automatically load the matching dataset."
            % _DATASET_LABELS.get(ds, ds or "none"))


def _register_pred_map(graph_id, pred_ids, gt, class_names):
    """把全图预测结果按 H×W 网格登记下来（供点击像元时反查预测类别）。"""
    gt = np.asarray(gt)
    pred_ids = np.asarray(pred_ids)
    if gt.ndim != 2:
        return
    h_size, w_size = gt.shape
    grid = np.zeros((h_size, w_size), dtype=int)
    mask = gt.reshape(-1) > 0
    if pred_ids.shape == (h_size, w_size):
        grid = pred_ids.astype(int)
    elif pred_ids.size == int(mask.sum()):
        grid.reshape(-1)[mask] = pred_ids
    else:
        return  # 形状对不上就不登记：点击时只显示 GT 信息，不显示预测
    _PRED_MAPS[graph_id] = {
        "grid": grid,
        "class_names": [str(c) for c in (class_names or [])],
        "label": _CLASSIFIER_LABELS.get(graph_id, graph_id),
    }


def _class_label(class_id, class_names) -> str:
    """类别 id → 名字（0 为背景）。"""
    cid = int(class_id)
    if cid <= 0:
        return "background"
    if class_names and cid <= len(class_names):
        return str(class_names[cid - 1])
    return "Class %d" % cid


def _pixel_spectrum_at(row, col):
    """取 (row, col) 像元的原始光谱。返回 (spectrum, source)；取不到时 (None, None)。

    优先用全局立方体（任何像元都有光谱）；只载入过展平数据时退化为
    「有标签像元矩阵 + 行主序查找」（仅标签像元可取）。
    """
    gt = np.asarray(global_var.get_global_gt())
    if gt.ndim != 2 or not (0 <= row < gt.shape[0] and 0 <= col < gt.shape[1]):
        return None, None
    cube = global_var.get_global_cube()
    if cube is not None and len(cube) != 0:
        cube = np.asarray(cube)
        if cube.ndim == 3 and row < cube.shape[0] and col < cube.shape[1]:
            return np.asarray(cube[row, col, :], dtype=float), "image cube"
    hs_X = global_var.get_global_hs_X()
    if hs_X is None or len(hs_X) == 0:
        return None, None
    hs_X = np.asarray(hs_X)
    labeled_flat = np.flatnonzero(gt.reshape(-1) > 0)
    flat = row * gt.shape[1] + col
    pos = int(np.searchsorted(labeled_flat, flat))
    if pos < labeled_flat.size and labeled_flat[pos] == flat:
        return np.asarray(hs_X[pos, :], dtype=float), "labeled pixels"
    return None, None


def _class_mean_spectrum(class_id):
    """某一类的均值光谱（作为点击取谱时的参考曲线）。数据不足时返回 None。"""
    hs_X = global_var.get_global_hs_X()
    hs_y = global_var.get_global_hs_y()
    if hs_X is None or len(hs_X) == 0 or hs_y is None or len(hs_y) == 0:
        return None
    hs_X = np.asarray(hs_X)
    hs_y = np.asarray(hs_y)
    if hs_X.shape[0] != hs_y.shape[0]:
        return None
    selected = hs_X[hs_y == class_id]
    if selected.shape[0] == 0:
        return None
    return selected.mean(axis=0)


def _class_legend(class_names):
    """类别颜色图例：色块 + 序号 + 类名。"""
    chips = []
    for i, name in enumerate(class_names, start=1):
        chips.append(html.Span(
            [
                html.Span(style={
                    "display": "inline-block", "width": "11px", "height": "11px",
                    "backgroundColor": unit.class_color(i), "marginRight": "4px",
                    "border": "1px solid #ced4da", "verticalAlign": "middle"}),
                "%d. %s" % (i, name),
            ],
            style={"display": "inline-block", "marginRight": "12px",
                   "marginBottom": "4px", "fontSize": "12px"},
        ))
    return chips or ["- " + "No class list yet; load a dataset first."]


def hs_class_map(model, feature_columns, class_names, decode=None, data_x=None,
                 register_as=None, train_labels=None) -> Any:
    """在全局高光谱数据上预测全量非背景像元，生成 H×W 分类图。

    无法生成时不再返回空 dict（页面显示空坐标轴、用户不知道为什么），
    而是返回一张写明原因的占位图：
    (1) 未加载高光谱图像；或 (2) 特征维度与图像波段数不一致且训练数据
        行数与图像有标签像元数也对不上（无法按行主序还原空间位置）。

    降维特征也能出图：降维数据 / 导出 CSV 都来自同一套「行主序展平」，
    只要训练时用的特征数据行数 == 图像有标签像元数，行序即与图像一一对应，
    直接对训练数据逐行预测即可还原空间位置——不再强迫用户重训全波段。

    decode: BPNN 的模型输出是 0..K-1 整数编码，需要它还原成原始类别名。
            其余分类器直接输出原始标签，传 None 即可。
    data_x: 本次训练使用的全部特征数据（未划分），供降维特征出图用。
    register_as: 分类图组件 id（如 "class_map_rf"）。给了就把这次的全图预测
        结果登记下来，供「点击像元取谱」反查该像元的预测类别与是否错分。
    train_labels: 本次训练使用的标签（未划分）。给了就能在「内存里的数据集
        与训练数据对不上」时按类别名自动匹配并载入正确的数据集——分类图永远
        画在内存中已载入的数据集上，载入错数据集时图会直接出不来。
    """
    if register_as:
        _MAP_NOTES.pop(register_as, None)
    recovery_note = _try_recover_dataset(train_labels, data_x)
    if recovery_note:
        # 恢复后类别表已换成本次训练数据所属数据集，后续映射必须用新的
        class_names = global_var.get_global_class_names()
    hs_X = global_var.get_global_hs_X()
    hs_gt = global_var.get_global_gt()
    if hs_X is None or hs_gt is None or len(hs_X) == 0:
        message = ("Full-image classification map unavailable: no hyperspectral image is loaded. "
                   "Load a dataset on the Data Loading page, then retrain to generate the map.")
        if register_as:
            _MAP_NOTES[register_as] = message
        return _map_placeholder(message)
    hs_gt = np.asarray(hs_gt)
    fallback_note = None
    if hs_X.shape[1] == len(feature_columns):
        # 全波段训练：直接对图像全部像元预测（最快、口径最标准）。
        # 但先检查量纲：如果训练 CSV 是预处理后的（如逐波段标准化），而内存立方体
        # 是原始反射率，直接预测会让所有像元落进同一个分裂分支——整张图一个颜色。
        hs_arr = np.asarray(hs_X, dtype=np.float64)
        predict_X = hs_arr
        cols = [c for c in feature_columns if c in getattr(data_x, "columns", [])] \
            if data_x is not None else []
        if data_x is not None and len(cols) == len(feature_columns):
            tr = np.asarray(data_x[cols], dtype=np.float64)
            tr_mean = np.nanmean(tr, axis=0)
            tr_std = np.nanstd(tr, axis=0)
            hs_mean = np.nanmean(hs_arr, axis=0)
            hs_std = np.nanstd(hs_arr, axis=0)
            tr_std_med = float(np.median(tr_std))
            hs_std_med = float(np.median(hs_std))
            if hs_std_med > 1.0 and tr_std_med > 1e-9 and hs_std_med / tr_std_med > 3.0:
                # 训练数据量纲明显小于立方体 → 逐个识别预处理口径并在立方体上复现
                tr_row_mean = np.nanmean(tr, axis=1)
                tr_row_std = np.nanstd(tr, axis=1)
                row_mean_med = float(np.median(np.abs(tr_row_mean)))
                row_std_med = float(np.median(tr_row_std))
                tr_min_med = float(np.median(np.nanmin(tr, axis=0)))
                tr_max_med = float(np.median(np.nanmax(tr, axis=0)))
                col_abs_mean_med = float(np.median(np.abs(tr_mean)))
                if row_mean_med < 0.05 and 0.9 < row_std_med < 1.1:
                    # SNV（逐像元标准化，每行均值 0 / 标准差 ~1）：对立方体每行做同样变换。
                    # 注意必须先查 SNV——SNV 数据的逐列统计量没有规律，会被误判。
                    hs_row_mean = np.nanmean(hs_arr, axis=1, keepdims=True)
                    hs_row_std = np.nanstd(hs_arr, axis=1, keepdims=True)
                    denom = np.where(hs_row_std > 1e-9, hs_row_std, 1.0)
                    predict_X = (hs_arr - hs_row_mean) / denom
                    fallback_note = (
                        "Prediction applied per-pixel standardization (SNV) to the raw image cube "
                        "to match the standardized training data; without it every pixel would "
                        "fall into one class.")
                elif tr_min_med > -0.05 and 0.9 < tr_max_med < 1.05:
                    # 训练数据在 [0,1]（min-max）而立方体是原始量纲
                    hs_min = np.nanmin(hs_arr, axis=0)
                    hs_max = np.nanmax(hs_arr, axis=0)
                    span = np.where(hs_max - hs_min > 1e-9, hs_max - hs_min, 1.0)
                    predict_X = (hs_arr - hs_min) / span
                    fallback_note = (
                        "Prediction min-max scaled the raw image cube to [0, 1] to match the "
                        "normalized training data; without it every pixel would fall into one class.")
                elif col_abs_mean_med < 0.2:
                    # 训练数据是逐波段标准化（z-score）的：用立方体自身的统计量做同样的
                    # 标准化再预测（与导出 CSV 时的预处理口径一致）。
                    denom = np.where(hs_std > 1e-9, hs_std, 1.0)
                    predict_X = (hs_arr - hs_mean) / denom
                    fallback_note = (
                        "Prediction standardized the raw image cube per band to match the "
                        "standardized training data; without it every pixel would fall into "
                        "one class.")
                else:
                    fallback_note = (
                        "Warning: the training data and the loaded image are on different scales; "
                        "the map may be unreliable. Retrain on a CSV exported from the currently "
                        "loaded image.")
        all_pred = model.predict(predict_X)
    else:
        # 特征数与波段数不一致（如用降维特征训练）：退回用训练数据本身逐行预测
        labeled = int((hs_gt > 0).sum())
        if data_x is not None and len(data_x) == int(hs_gt.size):
            # 训练数据 = 整幅图全像元（含背景行，行主序，如导出的全图 CSV）：
            # 每一行都能按行主序找回空间位置，直接逐行预测再还原成 H×W。
            all_pred = model.predict(data_x)
            fallback_note = ("Predicted on the training rows themselves (all image pixels "
                             "including background rows, image order).")
        elif data_x is None or len(data_x) != labeled:
            message = (
                "Full-image classification map unavailable: the loaded image and your "
                "training data look like two different datasets. The image has %d bands "
                "and %d labeled pixels, but the training data has %d feature columns and "
                "%s rows — neither number matches. Fix: on the Data Loading page, Retrieve "
                "the same dataset your CSV was exported from (or Read a CSV exported from "
                "the currently loaded image), then retrain."
                % (hs_X.shape[1], labeled, len(feature_columns),
                   str(len(data_x)) if data_x is not None else "0"))
            if register_as:
                _MAP_NOTES[register_as] = message
            return _map_placeholder(message)
        else:
            all_pred = model.predict(data_x)
            fallback_note = (
                "Predicted on the training features themselves (%d dims, image order): the "
                "reduced-feature map still shows spatial context, but spectral detail is lower "
                "than a full-band map." % len(feature_columns))
    if decode is not None:
        all_pred = decode(all_pred)
    all_pred_ids = unit._to_class_ids(all_pred, class_names)
    if len(all_pred_ids) == int(hs_gt.size):
        # 全像元预测（含背景）：按行主序还原成网格后，只取有标签像元参与成图
        all_pred_ids = np.asarray(all_pred_ids).reshape(hs_gt.shape)[hs_gt > 0]
    if register_as:
        _register_pred_map(register_as, all_pred_ids, hs_gt, class_names)
        if recovery_note:
            _MAP_NOTES[register_as] = recovery_note
    fig = unit.classification_map(all_pred_ids, hs_gt, class_names)
    if fallback_note is not None:
        # 说明文字放图外底部：先加大下边距，再放到刻度行以下的留白区，
        # 否则会压在横轴刻度上、并被页面下一个面板裁掉
        fig.update_layout(margin=dict(t=40, b=78, l=0, r=0))
        fig.add_annotation(text=fallback_note, showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=-0.17, font={"size": 11, "color": "#888888"})
    return fig


@app.callback(
    Output("table_in5", "data"),
    Output("modle_x", "min"),
    Output("modle_x", "max"),
    Output("modle_x", "value"),
    Output("modle_y", "options"),
    Output("modle_y", "value"),
    Output("alert-model-load", "children"),
    Output("alert-model-load", "is_open"),
    Output("alert-model-load", "color"),
    # 交互分析的类别筛选：载入数据时同步填充类别下拉与颜色图例
    Output("class-filter", "options"),
    Output("class-filter", "value"),
    Output("class-legend", "children"),
    Input("button-read5", "n_clicks"),
    Input("button-get-reduce", "n_clicks"),
    State("input-path5", "value"),
)
def update_input_data5(read, reduce, path):
    triggered = ctx.triggered_id
    if not triggered:
        raise PreventUpdate
    try:
        data = load_model_dataframe(triggered, path)
    except PreventUpdate:
        raise
    except Exception as exc:  # 路径错 / 文件损坏等：之前静默失败，页面毫无反馈
        return (no_update, no_update, no_update, no_update, no_update, no_update,
                "Load failed: %s" % exc, True, "danger",
                no_update, no_update, no_update)
    column_names = data.columns.tolist()
    min_value = 0
    max_value = max(data.shape[1] - 1, 0)
    # Y 默认选 label：不设默认值时用户很容易选成数值列（如 Dimension 1），
    # 导致标签列混进特征范围、训练报 non-numeric 错误。
    default_y = "label" if "label" in column_names else column_names[0]
    # 载入成功提示：显式写出列数，用户能立刻发现载入的是降维数据（4 列）
    # 还是全波段数据（200+ 列）。
    if triggered == "button-read5":
        source = os.path.basename(str(path)) if path else "CSV file"
    else:
        source = "reduced data from the Dimensionality Reduction page"
    load_msg = "Loaded %d rows x %d columns from %s." % (data.shape[0], data.shape[1], source)
    load_msg += _csv_dataset_warning(data, default_y)
    global_var.set_global_column_name(column_names)
    global_var.set_global_input_table(data)
    # 类别筛选下拉：载入数据后立刻可用（默认全选 = 不筛选）
    filter_options, filter_names = _class_filter_options(data)
    filter_values = [opt["value"] for opt in filter_options]
    return (_preview_records(data), min_value, max_value, [min_value, max_value],
            column_names, default_y, load_msg, True, "success",
            filter_options, filter_values, _class_legend(filter_names))


@app.callback(
    Output("table_train_rf", "data"),
    Output("table_test_rf", "data"),
    Output("rf_metrics", "data"),
    Output("graph_train_rf", "figure"),
    Output("graph_test_rf", "figure"),
    Output("feature_rf", "figure"),
    Output("eval_cm_rf", "figure"),
    Output("eval_table_rf", "data"),
    Output("class_map_rf", "figure"),
    Output("alert-rf", "children"),
    Output("alert-rf", "is_open"),
    Output("alert-rf", "color"),
    Input("fit-rf-button", "n_clicks"),
    State("fit-rf-input", "value"),
    State("modle_x", "value"),
    State("modle_y", "value"),
    State("rf_split_mode", "value"),
    State("rf_validation_ratio", "value"),
    State("rf_random_validation", "value"),
    State("rf_random", "value"),
    State("rf_n_estimators", "value"),
    State("rf_class_weight", "value"),
)
@_with_error_alert(lambda exc: ([], [], [], {}, {}, {}, {}, [], {}, "Error: %s" % exc, True, "danger"))
def fit_rf(n_clicks, save_path, modle_x, modle_y, rf_split_mode, rf_validation_ratio, rf_random_validation, rf_random,
           rf_n_estimators, rf_class_weight):
    if not n_clicks:
        raise PreventUpdate
    try:
        data_x, data_y, feature_columns = prepare_dataset(modle_x, modle_y)
        x_train, x_test, y_train, y_test = split_dataset(data_x, data_y, rf_validation_ratio,
                                                         rf_random_validation, rf_split_mode)
        n_estimators = max(1, _int(rf_n_estimators, 200))
        random_state = _int(rf_random, 2026)
        class_weight = "balanced" if str(rf_class_weight) == "balanced" else None
        import time
        _t0 = time.time()
        print("[fit_rf] training on %d rows x %d features (n_estimators=%d)..."
              % (len(x_train), x_train.shape[1], n_estimators), flush=True)
        y_train_pred, y_test_pred, feature_importances, map_model = cast(
            Any, unit.fit_RF(
                x_train, x_test, y_train, y_test, n_estimators=n_estimators, random_state=random_state,
                class_weight=class_weight, return_model=True
            ))
        _fit_seconds = time.time() - _t0
        print("[fit_rf] training done in %.1f s, predicting full-image map..." % _fit_seconds, flush=True)
    except ValueError as exc:
        return [], [], [], {}, {}, {}, {}, [], {}, f"Error: {exc}", True, "danger"
    metrics_data = unit.get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred)
    metrics_data += [{'Metric': k, 'Train': '-', 'Test': round(v, 4)}
                     for k, v in unit.oa_aa_kappa(y_test, y_test_pred).items()]
    train_df, test_df = build_result_tables(x_train, x_test, y_train, y_test, y_train_pred, y_test_pred)
    importance_df = pd.DataFrame({"Feature Name": feature_columns, "Importance": feature_importances})
    importance_df = cast(pd.DataFrame, importance_df.sort_values("Importance", ascending=False))
    feature_fig = px.bar(importance_df, x="Feature Name", y="Importance", title="Feature Importance")
    plot_train = build_confusion_figure(y_train, y_train_pred, "Training Set Confusion Matrix")
    plot_test = build_confusion_figure(y_test, y_test_pred, "Test Set Confusion Matrix")
    # ── HS evaluation ──
    class_names = global_var.get_global_class_names()
    eval_cm_fig = unit.confusion_matrix_fig(y_test, y_test_pred, class_names,
                                            title="RF — Test Confusion Matrix")
    eval_table = unit.per_class_accuracy_table(y_test, y_test_pred, class_names)
    # 直接复用 fit_RF 返回的已训练模型做全图预测（旧代码在这里从头重训了一遍，训练时间翻倍）
    class_map_fig = hs_class_map(map_model, feature_columns, class_names, data_x=data_x,
                                 register_as="class_map_rf", train_labels=data_y)
    if save_path:
        train_df.to_csv(f"{save_path}/rf_train_table_data.csv", index=False)
        test_df.to_csv(f"{save_path}/rf_test_table_data.csv", index=False)
        pd.DataFrame(metrics_data).to_csv(f"{save_path}/rf_metrics_data.csv", index=False)
        importance_df.to_csv(f"{save_path}/rf_feature_importances.csv", index=False)
    return (
        _preview_records(train_df),
        _preview_records(test_df),
        metrics_data,
        plot_train,
        plot_test,
        feature_fig,
        eval_cm_fig,
        eval_table,
        class_map_fig,
        *_train_alert(y_train, data_y, _reduced_feature_note(rf_split_mode, feature_columns)
                      + _map_note("class_map_rf"),
                      y_test=y_test, y_test_pred=y_test_pred, elapsed=_fit_seconds),
    )


@app.callback(
    Output("table_train_cnn", "data"),
    Output("table_test_cnn", "data"),
    Output("cnn_metrics", "data"),
    Output("graph_train_cnn", "figure"),
    Output("graph_test_cnn", "figure"),
    Output("eval_cm_cnn", "figure"),
    Output("eval_table_cnn", "data"),
    Output("class_map_cnn", "figure"),
    Output("alert-cnn", "children"),
    Output("alert-cnn", "is_open"),
    Output("alert-cnn", "color"),
    Input("fit-cnn-button", "n_clicks"),
    State("fit-cnn-input", "value"),
    State("modle_x", "value"),
    State("modle_y", "value"),
    State("cnn_split_mode", "value"),
    State("cnn_validation_ratio", "value"),
    State("cnn_seed", "value"),
    State("cnn_patch_size", "value"),
    State("cnn_epochs", "value"),
    State("cnn_batch_size", "value"),
    State("cnn_lr", "value"),
    State("cnn_max_samples", "value"),
    State("cnn_predict_map", "value"),
    State("cnn_max_map", "value"),
)
@_with_error_alert(lambda exc: ([], [], [], {}, {}, {}, {}, {}, "Error: %s" % exc, True, "danger"))
def fit_cnn(n_clicks, save_path, modle_x, modle_y, cnn_split_mode, cnn_validation_ratio,
            cnn_seed, cnn_patch_size, cnn_epochs, cnn_batch_size, cnn_lr,
            cnn_max_samples, cnn_predict_map, cnn_max_map):
    if not n_clicks:
        raise PreventUpdate
    try:
        # 3D-CNN 需要像元的空间坐标来取邻域块：必须有 GT 标签图，
        # 且建模数据的行序与展平像元一一对应（行主序、全量）。
        # 先读训练数据，并在「内存里的数据集与本次 CSV 不是同一套」时按 CSV 的
        # 类别名自动载入匹配的数据集——否则会报「行序对不上」，用户很难想到是
        # 载错了数据集（同一 session 里换数据集不需要重启服务）。
        data_x, data_y, feature_columns = prepare_dataset(modle_x, modle_y)
        cnn_recovery = _try_recover_dataset(data_y, data_x)
        gt = global_var.get_global_gt()
        if gt is None or len(gt) == 0:
            raise ValueError(
                "3D-CNN needs spatial patches around each pixel, so the hyperspectral "
                "image must be loaded on the Data Loading page first (a CSV alone has "
                "no pixel coordinates).")
        gt = np.asarray(gt)
        h_size, w_size = gt.shape
        labeled = int((gt > 0).sum())
        y_names = pd.Series(np.ravel(np.asarray(data_y)))
        if len(y_names) == labeled:
            # 仅含标签像元：行主序与 GT 一一对应（Retrieve 按钮 / 导出的标注数据）
            cube = np.zeros((h_size, w_size, len(feature_columns)), dtype=np.float32)
            cube[gt > 0] = data_x.to_numpy(dtype=np.float32)
            feats = cast(pd.DataFrame, data_x.reset_index(drop=True))
        elif len(data_x) == h_size * w_size:
            # 全像元 CSV（含背景行）：整幅按行主序填回立方体，背景光谱保留为
            # 邻域上下文；标签改用 GT 标签图（背景行的 Y 列值无意义，不参与训练）
            cube = data_x.to_numpy(dtype=np.float32).reshape(h_size, w_size, -1)
            feats = cast(pd.DataFrame,
                         data_x.iloc[np.flatnonzero((gt > 0).ravel())].reset_index(drop=True))
            names = global_var.get_global_class_names()
            n_cls = int(gt.max())
            if names and len(names) >= n_cls:
                classes_all = [str(names[i]) for i in range(n_cls)]
            else:
                classes_all = ["Class %d" % (i + 1) for i in range(n_cls)]
            y_names = pd.Series([classes_all[i - 1] for i in gt[gt > 0]])
        else:
            raise ValueError(
                "3D-CNN needs the pixel data in image order: either labeled pixels only "
                "(%d rows, via the Retrieve buttons) or ALL pixels including background "
                "(%d rows, a full-image CSV). Got %d rows — check that the CSV was exported "
                "without subsampling." % (labeled, h_size * w_size, len(data_x)))
        # 用行号标记列复用 split_dataset（同一套分层/空间划分逻辑），
        # 训练后按行号找回每个样本的像元坐标。
        tagged = feats.assign(**{"__row_idx__": np.arange(len(feats))})
        x_train, x_test, y_train, y_test = split_dataset(
            tagged, y_names, cnn_validation_ratio, cnn_seed, cnn_split_mode)
        # pandas 存根把 x[...] 标成 DataFrame | Series | None 的联合；此处必是 Series，
        # 用 cast 收敛（运行时语义不变），否则 to_numpy 会被误报成「None 上没有该方法」。
        train_idx = cast(pd.Series, x_train["__row_idx__"]).to_numpy(dtype=int)
        test_idx = cast(pd.Series, x_test["__row_idx__"]).to_numpy(dtype=int)
        classes = sorted(set(pd.unique(y_names).tolist()), key=str)
        cls_to_id = {c: i for i, c in enumerate(classes)}
        y_train_ids = np.array([cls_to_id[v] for v in y_train.to_numpy()], dtype=int)
        y_test_ids = np.array([cls_to_id[v] for v in y_test.to_numpy()], dtype=int)

        coords_all = np.argwhere(gt > 0)  # 行主序，与 feats 行号一一对应

        model, y_tr_ids, y_te_ids, history, seconds = unit.fit_3DCNN(
            cube, coords_all[train_idx], y_train_ids, coords_all[test_idx], y_test_ids,
            patch_size=_int(cnn_patch_size, 7), epochs=_int(cnn_epochs, 5),
            batch_size=_int(cnn_batch_size, 64), learning_rate=_num(cnn_lr, 0.001),
            random_state=_int(cnn_seed, 18), max_train_samples=_int(cnn_max_samples, 8000))

        def _decode(ids):
            return np.array([classes[i] for i in ids], dtype=object)

        y_train_pred = _decode(y_tr_ids)
        y_test_pred = _decode(y_te_ids)
    except ValueError as exc:
        return [], [], [], {}, {}, {}, {}, {}, f"Error: {exc}", True, "danger"
    except ImportError as exc:  # 未安装 PyTorch 的人话提示
        return [], [], [], {}, {}, {}, {}, {}, f"Error: {exc}", True, "danger"
    except Exception as exc:  # 兜底：训练阶段任何异常都以红色提示呈现，避免 500 无反馈
        return [], [], [], {}, {}, {}, {}, {}, f"Error: {exc}", True, "danger"
    try:
        metrics_data = unit.get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred)
        metrics_data += [{'Metric': k, 'Train': '-', 'Test': round(v, 4)}
                         for k, v in unit.oa_aa_kappa(y_test, y_test_pred).items()]
        plain_x_train = cast(pd.DataFrame, x_train.drop(columns=["__row_idx__"]))
        plain_x_test = cast(pd.DataFrame, x_test.drop(columns=["__row_idx__"]))
        train_df, test_df = build_result_tables(
            plain_x_train, plain_x_test, y_train, y_test, y_train_pred, y_test_pred)
        plot_train = build_confusion_figure(y_train, y_train_pred, "Training Set Confusion Matrix")
        plot_test = build_confusion_figure(y_test, y_test_pred, "Test Set Confusion Matrix")
        # ── HS evaluation ──
        eval_cm_fig = unit.confusion_matrix_fig(y_test, y_test_pred, classes,
                                                title="3D-CNN — Test Confusion Matrix")
        eval_table = unit.per_class_accuracy_table(y_test, y_test_pred, classes)
        # ── 全图分类图：逐像元取 patch 批量预测 ──
        map_note = cnn_recovery  # 自动换数据集时把原因带进顶部提示条
        predict_map = "yes" in (cnn_predict_map or [])
        if predict_map and len(coords_all) <= _int(cnn_max_map, 100000):
            all_pred_ids = unit.predict_3dcnn(model, cube, coords_all, batch_size=256)
            all_pred = _decode(all_pred_ids)
            cnn_class_ids = unit._to_class_ids(all_pred, classes)
            _register_pred_map("class_map_cnn", cnn_class_ids, gt, classes)
            class_map_fig = unit.classification_map(cnn_class_ids, gt, classes)
        elif predict_map:
            class_map_fig = _map_placeholder(
                "Full-image 3D-CNN prediction skipped: the image has %d labeled pixels, "
                "above the limit of %d. Raise 'Max Full-Map Pixels' and retrain to "
                "generate the map (prediction is slow on CPU)."
                % (len(coords_all), _int(cnn_max_map, 100000)))
            map_note = " Full-image prediction was skipped (too many pixels)."
        else:
            class_map_fig = _map_placeholder(
                "Full-image prediction is turned off. Tick 'Predict full-image map' "
                "and retrain to generate the map.")
        base_text, _is_open, color = _train_alert(y_train, y_names, extra_note=map_note,
                                                  y_test=y_test, y_test_pred=y_test_pred)
        alert_text = base_text + (" Training took %.1fs for %d epochs on %d samples."
                                  % (seconds, len(history), len(y_train_ids)))
        if save_path:
            train_df.to_csv(f"{save_path}/cnn_train_table_data.csv", index=False)
            test_df.to_csv(f"{save_path}/cnn_test_table_data.csv", index=False)
            pd.DataFrame(metrics_data).to_csv(f"{save_path}/cnn_metrics_data.csv", index=False)
            pd.DataFrame(history).to_csv(f"{save_path}/cnn_loss_history.csv", index=False)
    except Exception as exc:  # 指标/绘图/保存阶段同样兜底，任何异常都人话提示而非 500
        return [], [], [], {}, {}, {}, {}, {}, f"Error: {exc}", True, "danger"
    return (
        _preview_records(train_df),
        _preview_records(test_df),
        metrics_data,
        plot_train,
        plot_test,
        eval_cm_fig,
        eval_table,
        class_map_fig,
        alert_text,
        True,
        color,
    )


@app.callback(
    Output("table_train_svm", "data"),
    Output("table_test_svm", "data"),
    Output("svm_metrics", "data"),
    Output("graph_train_svm", "figure"),
    Output("graph_test_svm", "figure"),
    Output("eval_cm_svm", "figure"),
    Output("eval_table_svm", "data"),
    Output("class_map_svm", "figure"),
    Output("alert-svm", "children"),
    Output("alert-svm", "is_open"),
    Output("alert-svm", "color"),
    Input("fit-svm-button", "n_clicks"),
    State("fit-svm-input", "value"),
    State("modle_x", "value"),
    State("modle_y", "value"),
    State("svm_split_mode", "value"),
    State("svm_validation_ratio", "value"),
    State("svm_random_validation", "value"),
    State("svm_kernel", "value"),
    State("svm_c", "value"),
    State("svm_class_weight", "value"),
)
@_with_error_alert(lambda exc: ([], [], [], {}, {}, {}, [], {}, "Error: %s" % exc, True, "danger"))
def fit_svm(n_clicks, save_path, modle_x, modle_y, svm_split_mode, svm_validation_ratio, svm_random_validation, svm_kernel, svm_c, svm_class_weight):
    if not n_clicks:
        raise PreventUpdate
    try:
        data_x, data_y, feature_columns = prepare_dataset(modle_x, modle_y)
        x_train, x_test, y_train, y_test = split_dataset(data_x, data_y, svm_validation_ratio,
                                                         svm_random_validation, svm_split_mode)
        c = max(1e-6, _num(svm_c, 1.0))
        kernel = svm_kernel or "linear"  # Dropdown 可清空，None 会让 SVC 报错；与控件默认保持一致
        class_weight = "balanced" if str(svm_class_weight) == "balanced" else None
        if kernel in ("rbf", "poly", "sigmoid") and len(x_train) > 30000:
            # 非线性核要把核矩阵整个装进内存，是 O(n^2)：3 万行约 7 GB、4.3 万行约 15 GB，
            # 训练量也是 O(n^2) 起步。线性核走 liblinear 的 O(n) 实现，数万行只要一两分钟，
            # 因此只拦非线性核，并明确指路到 linear。
            raise ValueError(
                "SVM with '%s' kernel on %d training rows is not feasible: the kernel matrix alone "
                "would need about %.1f GB. Switch 'Kernel Function' to 'linear' — on hyperspectral "
                "data it is both faster and usually more accurate — or use reduced features / a "
                "smaller training set." % (kernel, len(x_train), len(x_train) ** 2 * 8 / 1e9))
        import time
        _t0 = time.time()
        print("[fit_svm] training on %d rows x %d features..."
              % (len(x_train), x_train.shape[1]), flush=True)
        y_train_pred, y_test_pred, map_model = cast(
            Any, unit.fit_SVM(x_train, x_test, y_train, y_test, c, kernel,
                              class_weight=class_weight, return_model=True))
        print("[fit_svm] training done in %.1f s" % (time.time() - _t0), flush=True)
    except ValueError as exc:
        return [], [], [], {}, {}, {}, [], {}, f"Error: {exc}", True, "danger"
    metrics_data = unit.get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred)
    metrics_data += [{'Metric': k, 'Train': '-', 'Test': round(v, 4)}
                     for k, v in unit.oa_aa_kappa(y_test, y_test_pred).items()]
    train_df, test_df = build_result_tables(x_train, x_test, y_train, y_test, y_train_pred, y_test_pred)
    class_names = global_var.get_global_class_names()
    eval_cm_fig = unit.confusion_matrix_fig(y_test, y_test_pred, class_names,
                                            title="SVM — Test Confusion Matrix")
    eval_table = unit.per_class_accuracy_table(y_test, y_test_pred, class_names)
    # 复用 fit_SVM 返回的已训练模型做全图预测（旧代码在这里从头重训了一遍）
    class_map_fig = hs_class_map(map_model, feature_columns, class_names, data_x=data_x,
                                 register_as="class_map_svm", train_labels=data_y)
    if save_path:
        train_df.to_csv(f"{save_path}/svm_train_table_data.csv", index=False)
        test_df.to_csv(f"{save_path}/svm_test_table_data.csv", index=False)
        pd.DataFrame(metrics_data).to_csv(f"{save_path}/svm_metrics_data.csv", index=False)
    return (
        _preview_records(train_df),
        _preview_records(test_df),
        metrics_data,
        build_confusion_figure(y_train, y_train_pred, "Training Set Confusion Matrix"),
        build_confusion_figure(y_test, y_test_pred, "Test Set Confusion Matrix"),
        eval_cm_fig,
        eval_table,
        class_map_fig,
        *_train_alert(y_train, data_y, _reduced_feature_note(svm_split_mode, feature_columns)
                      + _map_note("class_map_svm"),
                      y_test=y_test, y_test_pred=y_test_pred,
                      elapsed=time.time() - _t0),
    )


@app.callback(
    Output("table_train_bpnn", "data"),
    Output("table_test_bpnn", "data"),
    Output("bpnn_metrics", "data"),
    Output("graph_train_bpnn", "figure"),
    Output("graph_test_bpnn", "figure"),
    Output("eval_cm_bpnn", "figure"),
    Output("eval_table_bpnn", "data"),
    Output("class_map_bpnn", "figure"),
    Output("alert-bpnn", "children"),
    Output("alert-bpnn", "is_open"),
    Output("alert-bpnn", "color"),
    Input("fit-bpnn-button", "n_clicks"),
    State("fit-bpnn-input", "value"),
    State("modle_x", "value"),
    State("modle_y", "value"),
    State("bpnn_split_mode", "value"),
    State("bpnn_validation_ratio", "value"),
    State("bpnn_random_validation", "value"),
    State("bpnn_hidden_layers", "value"),
    State("bpnn_activation", "value"),
    State("bpnn_learning_rate", "value"),
    State("bpnn_alpha", "value"),
    State("bpnn_max_iter", "value"),
)
@_with_error_alert(lambda exc: ([], [], [], {}, {}, {}, [], {}, "Error: %s" % exc, True, "danger"))
def fit_bpnn(n_clicks, save_path, modle_x, modle_y, bpnn_split_mode, val_ratio, seed_split, hidden_layers_text,
             activation, lr_init, alpha, max_iter):
    if not n_clicks:
        raise PreventUpdate
    try:
        data_x, data_y, feature_columns = prepare_dataset(modle_x, modle_y)
        x_train, x_test, y_train, y_test = split_dataset(data_x, data_y, val_ratio,
                                                         seed_split, bpnn_split_mode)
        hidden_text = hidden_layers_text or "100"
        hidden_layers = tuple(max(1, int(item.strip())) for item in hidden_text.split(",") if item.strip())
        hidden_layers = hidden_layers or (100,)
        activation = activation or "relu"  # Dropdown 可清空
        lr = max(1e-8, _num(lr_init, 0.001))
        alpha = max(0.0, _num(alpha, 1e-4))
        max_iter = max(10, _int(max_iter, 300))
        seed_split = _int(seed_split, 18)
        import time
        _t0 = time.time()
        print("[fit_bpnn] training on %d rows x %d features (max_iter=%d)..."
              % (len(x_train), x_train.shape[1], max_iter), flush=True)
        y_train_pred, y_test_pred, bpnn_model, bpnn_decode = cast(
            Any, unit.fit_BPNN(
                x_train, x_test, y_train, y_test,
                hidden_layer_sizes=hidden_layers,
                activation=activation,
                learning_rate_init=lr,
                alpha=alpha,
                max_iter=max_iter,
                random_state=seed_split,
                return_model=True,
            ))
        print("[fit_bpnn] training done in %.1f s" % (time.time() - _t0), flush=True)
    except ValueError as exc:
        return [], [], [], {}, {}, {}, [], {}, f"Error: {exc}", True, "danger"
    metrics_data = unit.get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred)
    metrics_data += [{'Metric': k, 'Train': '-', 'Test': round(v, 4)}
                     for k, v in unit.oa_aa_kappa(y_test, y_test_pred).items()]
    train_df, test_df = build_result_tables(x_train, x_test, y_train, y_test, y_train_pred, y_test_pred)
    class_names = global_var.get_global_class_names()
    eval_cm_fig = unit.confusion_matrix_fig(y_test, y_test_pred, class_names,
                                            title="BPNN — Test Confusion Matrix")
    eval_table = unit.per_class_accuracy_table(y_test, y_test_pred, class_names)
    # 复用 fit_BPNN 返回的已训练模型与 decode（旧代码在这里把神经网络从头重训了一遍）
    class_map_fig = hs_class_map(bpnn_model, feature_columns, class_names, decode=bpnn_decode,
                                 data_x=data_x, register_as="class_map_bpnn", train_labels=data_y)
    if save_path:
        train_df.to_csv(f"{save_path}/bpnn_train_table_data.csv", index=False)
        test_df.to_csv(f"{save_path}/bpnn_test_table_data.csv", index=False)
        pd.DataFrame(metrics_data).to_csv(f"{save_path}/bpnn_metrics_data.csv", index=False)
    return (
        _preview_records(train_df),
        _preview_records(test_df),
        metrics_data,
        build_confusion_figure(y_train, y_train_pred, "Training Set Confusion Matrix"),
        build_confusion_figure(y_test, y_test_pred, "Test Set Confusion Matrix"),
        eval_cm_fig,
        eval_table,
        class_map_fig,
        *_train_alert(y_train, data_y, _reduced_feature_note(bpnn_split_mode, feature_columns)
                      + _map_note("class_map_bpnn"),
                      y_test=y_test, y_test_pred=y_test_pred,
                      elapsed=time.time() - _t0),
    )


@app.callback(
    Output("table_train_lda", "data"),
    Output("table_test_lda", "data"),
    Output("lda_metrics", "data"),
    Output("graph_train_lda", "figure"),
    Output("graph_test_lda", "figure"),
    Output("eval_cm_lda", "figure"),
    Output("eval_table_lda", "data"),
    Output("class_map_lda", "figure"),
    Output("alert-lda", "children"),
    Output("alert-lda", "is_open"),
    Output("alert-lda", "color"),
    Input("fit-lda-button", "n_clicks"),
    State("fit-lda-input", "value"),
    State("modle_x", "value"),
    State("modle_y", "value"),
    State("lda_split_mode", "value"),
    State("lda_validation_ratio", "value"),
    State("lda_random_validation", "value"),
)
@_with_error_alert(lambda exc: ([], [], [], {}, {}, {}, [], {}, "Error: %s" % exc, True, "danger"))
def fit_lda(n_clicks, save_path, modle_x, modle_y, lda_split_mode, lda_validation_ratio, lda_random_validation):
    if not n_clicks:
        raise PreventUpdate
    try:
        data_x, data_y, feature_columns = prepare_dataset(modle_x, modle_y)
        x_train, x_test, y_train, y_test = split_dataset(data_x, data_y, lda_validation_ratio,
                                                         lda_random_validation, lda_split_mode)
        import time
        _t0 = time.time()
        y_train_pred, y_test_pred, _model = cast(
            Any, unit.fit_LDA(x_train, x_test, y_train, y_test, return_model=True))
    except ValueError as exc:
        return [], [], [], {}, {}, {}, [], {}, f"Error: {exc}", True, "danger"
    metrics_data = unit.get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred)
    metrics_data += [{'Metric': k, 'Train': '-', 'Test': round(v, 4)}
                     for k, v in unit.oa_aa_kappa(y_test, y_test_pred).items()]
    train_df, test_df = build_result_tables(x_train, x_test, y_train, y_test, y_train_pred, y_test_pred)
    class_names = global_var.get_global_class_names()
    eval_cm_fig = unit.confusion_matrix_fig(y_test, y_test_pred, class_names,
                                            title="LDA — Test Confusion Matrix")
    eval_table = unit.per_class_accuracy_table(y_test, y_test_pred, class_names)
    # 复用 fit_LDA 返回的已训练模型做全图预测（旧代码在这里新建 _LDA() 又 fit 了一遍）
    class_map_fig = hs_class_map(_model, feature_columns, class_names, data_x=data_x,
                                 register_as="class_map_lda", train_labels=data_y)
    if save_path:
        train_df.to_csv(f"{save_path}/lda_train_table_data.csv", index=False)
        test_df.to_csv(f"{save_path}/lda_test_table_data.csv", index=False)
        pd.DataFrame(metrics_data).to_csv(f"{save_path}/lda_metrics_data.csv", index=False)
    return (
        _preview_records(train_df),
        _preview_records(test_df),
        metrics_data,
        build_confusion_figure(y_train, y_train_pred, "Training Set Confusion Matrix"),
        build_confusion_figure(y_test, y_test_pred, "Test Set Confusion Matrix"),
        eval_cm_fig,
        eval_table,
        class_map_fig,
        *_train_alert(y_train, data_y, _reduced_feature_note(lda_split_mode, feature_columns)
                      + _map_note("class_map_lda"),
                      y_test=y_test, y_test_pred=y_test_pred,
                      elapsed=time.time() - _t0),
    )


@app.callback(
    Output("table_train_knn", "data"),
    Output("table_test_knn", "data"),
    Output("knn_metrics", "data"),
    Output("graph_train_knn", "figure"),
    Output("graph_test_knn", "figure"),
    Output("eval_cm_knn", "figure"),
    Output("eval_table_knn", "data"),
    Output("class_map_knn", "figure"),
    Output("alert-knn", "children"),
    Output("alert-knn", "is_open"),
    Output("alert-knn", "color"),
    Input("fit-knn-button", "n_clicks"),
    State("fit-knn-input", "value"),
    State("modle_x", "value"),
    State("modle_y", "value"),
    State("knn_split_mode", "value"),
    State("knn_validation_ratio", "value"),
    State("knn_random_validation", "value"),
    State("knn_k", "value"),
)
@_with_error_alert(lambda exc: ([], [], [], {}, {}, {}, [], {}, "Error: %s" % exc, True, "danger"))
def fit_knn(n_clicks, save_path, modle_x, modle_y, knn_split_mode, knn_validation_ratio, knn_random_validation, knn_k):
    if not n_clicks:
        raise PreventUpdate
    try:
        data_x, data_y, feature_columns = prepare_dataset(modle_x, modle_y)
        x_train, x_test, y_train, y_test = split_dataset(data_x, data_y, knn_validation_ratio,
                                                         knn_random_validation, knn_split_mode)
        k = max(1, _int(knn_k, 5))
        if len(x_train) > 100000:
            # KNN 预测要和每个训练样本算距离：10 万行以上全图预测要等很久，主动拦截给建议
            raise ValueError(
                "KNN training set is too large (%d rows) — prediction cost grows linearly with "
                "training size and the full-image map would take a very long time. Options: use "
                "reduced features, load the 10249-row labeled-only CSV, or switch to Random "
                "Forest." % len(x_train))
        import time
        _t0 = time.time()
        print("[fit_knn] fitting on %d rows x %d features..." % (len(x_train), x_train.shape[1]), flush=True)
        y_train_pred, y_test_pred, map_model = cast(
            Any, unit.fit_KNN(x_train, x_test, y_train, y_test, k, return_model=True))
        print("[fit_knn] fit done in %.1f s, predicting..." % (time.time() - _t0), flush=True)
    except ValueError as exc:
        return [], [], [], {}, {}, {}, [], {}, f"Error: {exc}", True, "danger"
    metrics_data = unit.get_classify_metrics(y_test, y_test_pred, y_train, y_train_pred)
    metrics_data += [{'Metric': k, 'Train': '-', 'Test': round(v, 4)}
                     for k, v in unit.oa_aa_kappa(y_test, y_test_pred).items()]
    train_df, test_df = build_result_tables(x_train, x_test, y_train, y_test, y_train_pred, y_test_pred)
    class_names = global_var.get_global_class_names()
    eval_cm_fig = unit.confusion_matrix_fig(y_test, y_test_pred, class_names,
                                            title="KNN — Test Confusion Matrix")
    eval_table = unit.per_class_accuracy_table(y_test, y_test_pred, class_names)
    # 复用 fit_KNN 返回的已训练模型做全图预测（旧代码在这里重新 fit 了一遍）
    class_map_fig = hs_class_map(map_model, feature_columns, class_names, data_x=data_x,
                                 register_as="class_map_knn", train_labels=data_y)
    if save_path:
        train_df.to_csv(f"{save_path}/knn_train_table_data.csv", index=False)
        test_df.to_csv(f"{save_path}/knn_test_table_data.csv", index=False)
        pd.DataFrame(metrics_data).to_csv(f"{save_path}/knn_metrics_data.csv", index=False)
    return (
        _preview_records(train_df),
        _preview_records(test_df),
        metrics_data,
        build_confusion_figure(y_train, y_train_pred, "Training Set Confusion Matrix"),
        build_confusion_figure(y_test, y_test_pred, "Test Set Confusion Matrix"),
        eval_cm_fig,
        eval_table,
        class_map_fig,
        *_train_alert(y_train, data_y, _reduced_feature_note(knn_split_mode, feature_columns)
                      + _map_note("class_map_knn"),
                      y_test=y_test, y_test_pred=y_test_pred,
                      elapsed=time.time() - _t0),
    )


@app.callback(
    Output("tooltip5", "children"),
    Output("tooltip5", "is_open"),
    Input("modle_x", "value"),
)
def update_tooltip(value):
    if value is None:
        raise PreventUpdate
    column_names = global_var.get_global_column_name()
    min_value, max_value = value
    tooltip_labels = ", ".join(column_names[int(min_value):int(max_value) + 1])
    return tooltip_labels, True


# ── 交互一：点击分类图上的像元 → 画该像元的原始光谱曲线 ──
@app.callback(
    Output("pixel-spectrum", "figure"),
    Output("pixel-info", "children"),
    [Input(graph_id, "clickData") for graph_id in _CLASS_MAP_IDS],
)
def show_pixel_spectrum(*click_data):
    """在任意分类图上单击像元，绘制其原始光谱并给出类别诊断。

    只读全局数据与 `_PRED_MAPS`（训练回调写入），不触发任何训练/渲染回调，
    调用链保持单向，不存在循环触发。
    """
    triggered = ctx.triggered_id
    if triggered not in _CLASS_MAP_IDS:
        raise PreventUpdate
    payload = click_data[_CLASS_MAP_IDS.index(triggered)]
    if not payload or not payload.get("points"):
        raise PreventUpdate
    point = payload["points"][0]
    row, col = int(point.get("y", -1)), int(point.get("x", -1))
    gt = np.asarray(global_var.get_global_gt())
    if gt.ndim != 2 or not (0 <= row < gt.shape[0] and 0 <= col < gt.shape[1]):
        raise PreventUpdate

    class_names = list(global_var.get_global_class_names() or [])
    gt_id = int(gt[row, col])
    registered = _PRED_MAPS.get(triggered)
    # 先取出显示名：pyright 无法从下面的 pred_id > 0 推断 registered 非空，显式收敛
    classifier_label = registered["label"] if registered is not None else "Classifier"
    pred_id = 0
    if registered is not None and registered["grid"].shape == gt.shape:
        pred_id = int(registered["grid"][row, col])

    muted = {"color": "#868e96"}
    info = [
        html.Div([html.Span("Pixel (row, col): ", style=muted),
                  html.B("%d, %d" % (row, col))]),
        html.Div([html.Span("Ground truth: ", style=muted),
                  html.B(_class_label(gt_id, class_names),
                         style={"color": unit.class_color(gt_id) if gt_id > 0 else "#868e96"})]),
    ]
    if pred_id > 0:
        is_correct = pred_id == gt_id
        info.append(html.Div([
            html.Span("%s prediction: " % classifier_label, style=muted),
            html.B(_class_label(pred_id, class_names),
                   style={"color": unit.class_color(pred_id)}),
            html.Span("  · correct" if is_correct else "  · MISCLASSIFIED",
                      style={"color": "#2b8a3e" if is_correct else "#c92a2a",
                             "fontWeight": "600"}),
        ]))
    else:
        info.append(html.Div("Train this classifier to also show its prediction here.",
                             style=muted))

    spectrum, source = _pixel_spectrum_at(row, col)
    if spectrum is None:
        info.append(html.Div(
            "No spectrum available for this pixel (background pixel and no image cube "
            "in memory — load the dataset on the Data Loading page).",
            style={"color": "#c92a2a"}))
        return _spectrum_message_figure(
            "No spectrum available for pixel (row %d, col %d)." % (row, col)), info

    reference = _class_mean_spectrum(gt_id) if gt_id > 0 else None
    if reference is not None and len(reference) != len(spectrum):
        reference = None
    reference_name = ("%s mean spectrum" % _class_label(gt_id, class_names)
                      if reference is not None else "class mean")
    label = registered["label"] if registered is not None else "Pixel"
    fig = unit.spectrum_figure(
        np.arange(len(spectrum)), spectrum,
        title="%s — raw spectrum at row %d, col %d" % (label, row, col),
        reference=reference, reference_name=reference_name)
    info.append(html.Div("Spectrum source: %s · %d bands" % (source, len(spectrum)),
                         style={"color": "#adb5bd"}))
    return fig, info


# ── 交互二：类别显隐筛选（纯前端改色带，不往返服务器）──
# 被取消勾选的类别在所有分类图上变灰：既保留「这里原本有像元」的信息，
# 又能让剩余类别的空间分布立刻凸显出来。
_CLASS_FILTER_JS = """
function (visible, options, figRf, figSvm, figBpnn, figLda, figKnn, figCnn) {
    /* 调色板与阶梯公式与 unit.build_class_colorscale 完全一致（同源生成）。 */
    var PALETTE = __PALETTE__;
    var HIDDEN = '#cccccc';
    var NO = window.dash_clientside.no_update;
    var all = (options || []).map(function (o) { return parseInt(o.value, 10); });
    if (!all.length) { return [NO, NO, NO, NO, NO, NO]; }
    var shown = {};
    (visible || []).forEach(function (v) { shown[parseInt(v, 10)] = true; });
    var hidden = all.filter(function (i) { return !shown[i]; });
    var sig = hidden.join(',');
    function buildScale() {
        var n = all.length;
        var bg = (hidden.indexOf(0) >= 0) ? HIDDEN : '#111111';
        var scale = [[0.0, bg], [0.5 / n, bg]];
        for (var i = 1; i <= n; i++) {
            var c = (hidden.indexOf(i) >= 0) ? HIDDEN : PALETTE[(i - 1) % PALETTE.length];
            var lo = (i - 0.5) / n;
            var hi = Math.min((i + 0.5) / n, 1.0);
            scale.push([lo, c]);
            scale.push([hi, c]);
        }
        return scale;
    }
    var scale = null;
    function patch(fig) {
        if (!fig || !fig.data || !fig.data.length) { return NO; }
        var meta = (fig.layout && fig.layout.meta) || {};
        /* 已按当前筛选渲染过就不再重复渲染：
           本回调的输入就是这些图的 figure，标记位避免自触发形成死循环。 */
        if (meta.wb_hidden === sig) { return NO; }
        if (scale === null) { scale = buildScale(); }
        var data = fig.data.map(function (tr) {
            var nt = Object.assign({}, tr);
            if (nt.colorscale !== undefined && nt.colorscale !== null) { nt.colorscale = scale; }
            return nt;
        });
        var layout = Object.assign({}, fig.layout);
        layout.meta = Object.assign({}, meta, {wb_hidden: sig});
        return Object.assign({}, fig, {data: data, layout: layout});
    }
    return [patch(figRf), patch(figSvm), patch(figBpnn), patch(figLda), patch(figKnn),
            patch(figCnn)];
}
""".replace("__PALETTE__", json.dumps(list(unit._CLASS_PALETTE)))

app.clientside_callback(
    _CLASS_FILTER_JS,
    [Output(graph_id, "figure", allow_duplicate=True) for graph_id in _CLASS_MAP_IDS],
    Input("class-filter", "value"),
    Input("class-filter", "options"),
    [State(graph_id, "figure") for graph_id in _CLASS_MAP_IDS],
    prevent_initial_call=True,
)


def _class_filter_options(data=None):
    """类别筛选下拉的选项：id ↔ 颜色/名称 与分类图一致。

    名称优先取 Data Loading 页写入的类别表；否则退回载入数据标签列的去重值。
    """
    class_names = [str(c) for c in (global_var.get_global_class_names() or [])]
    if not class_names and data is not None and "label" in getattr(data, "columns", []):
        class_names = sorted({str(v) for v in pd.unique(data["label"])})
    options = [{"label": "%d. %s" % (i, name), "value": i}
               for i, name in enumerate(class_names, start=1)]
    return options, class_names

