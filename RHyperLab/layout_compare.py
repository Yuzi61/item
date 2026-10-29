from __future__ import annotations  # Python 3.7 兼容：延迟求值类型注解

import os
import time
from typing import Any, cast

import dash_bootstrap_components as dbc
import global_var
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import unit
from dash import Input, Output, State, ctx, dash_table, dcc, html, no_update
from dash.exceptions import PreventUpdate
from myapp import app

# 复用分类建模页已验证的逻辑（同一套 prepare/split，保证口径一致）
from layout_model import (SPLIT_MODES, _missing_train_classes, _preview_records,
                          load_model_dataframe, prepare_dataset, split_dataset)

# 可参与对比的分类器；默认勾选快的三个，SVM/BPNN 大数据下慢，由用户显式开启
CLASSIFIERS = ["Random Forest", "SVM", "BPNN", "LDA", "KNN"]
DEFAULT_MODELS = ["Random Forest", "LDA", "KNN"]
MODEL_PARAM = {
    "Random Forest": {"zh": "随机森林"},
    "SVM": {"zh": "支持向量机"},
    "BPNN": {"zh": "神经网络"},
    "LDA": {"zh": "线性判别"},
    "KNN": {"zh": "K近邻"},
}


def _fit_one(name, x_train, x_test, y_train, y_test):
    """统一训练入口：返回 (train_pred, test_pred, 用时秒)。"""
    t0 = time.time()
    if name == "Random Forest":
        # 与建模页一致使用 balanced 类别权重，避免小类被大类淹没、逐类精度为 0
        preds = unit.fit_RF(x_train, x_test, y_train, y_test, 100, 18, class_weight="balanced")
        y_tr, y_te = preds[0], preds[1]
    elif name == "SVM":
        # 与 RF 分支同样的写法：fit_* 的返回类型是「2 元组 | 带模型的 3 元组」的联合，
        # 直接 `a, b = ...` 会让类型检查器按元组长度不匹配报错（运行时默认不带模型，
        # 行为不变）。显式按下标取，既过检查也防止日后改 return_model 时静默错位。
        svm_preds = unit.fit_SVM(x_train, x_test, y_train, y_test, 1.0, "linear",
                                 class_weight="balanced")
        y_tr, y_te = svm_preds[0], svm_preds[1]
    elif name == "BPNN":
        bpnn_preds = unit.fit_BPNN(x_train, x_test, y_train, y_test,
                                   hidden_layer_sizes=(64, 32), max_iter=200,
                                   random_state=18, early_stopping=True)
        y_tr, y_te = bpnn_preds[0], bpnn_preds[1]
    elif name == "LDA":
        lda_preds = unit.fit_LDA(x_train, x_test, y_train, y_test)
        y_tr, y_te = lda_preds[0], lda_preds[1]
    elif name == "KNN":
        knn_preds = unit.fit_KNN(x_train, x_test, y_train, y_test, 5)
        y_tr, y_te = knn_preds[0], knn_preds[1]
    else:
        raise ValueError("Unknown classifier: %s" % name)
    return y_tr, y_te, time.time() - t0


def _cap_samples(data_x, data_y, cap, seed):
    """训练样本上限：超限时按类分层等比抽稀（确定性，可复现）。"""
    if not cap or len(data_y) <= int(cap):
        return data_x, data_y, ""
    from sklearn.model_selection import train_test_split
    stratify = data_y if data_y.nunique() > 1 else None
    keep_x, _drop_x, keep_y, _drop_y = train_test_split(
        data_x, data_y, train_size=int(cap), random_state=seed, stratify=stratify)
    note = (" Note: %d rows were stratified-subsampled from %d for comparison speed."
            % (len(keep_y), len(data_y)))
    return cast(pd.DataFrame, keep_x), cast(pd.Series, keep_y), note


def _per_class_accuracy(y_test, y_pred, classes):
    """逐类精度（测试集）；某类精度无法评估时记 0。"""
    out = []
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    for cls in classes:
        mask = y_test == cls
        acc = float((y_pred[mask] == cls).mean()) if mask.sum() else 0.0
        out.append(acc)
    return out


def run_comparison(selected, modle_x, modle_y, split_mode, validation_ratio,
                   random_state, cap):
    """核心对比逻辑（与回调解耦，便于测试）：同一份数据、同一种划分训练多个分类器。

    返回 (rows, bar_fig, heat_fig, note, color)。
    """
    if not selected:
        raise ValueError("Please select at least one classifier to compare.")
    data_x, data_y, _feature_columns = prepare_dataset(modle_x, modle_y)
    data_x, data_y, cap_note = _cap_samples(data_x, data_y, cap, random_state)
    x_train, x_test, y_train, y_test = split_dataset(
        data_x, data_y, validation_ratio, random_state, split_mode=split_mode)

    classes = sorted(set(pd.unique(pd.Series(data_y)).tolist()), key=str)
    rows = []
    acc_matrix = []
    for name in selected:
        y_tr_pred, y_te_pred, seconds = _fit_one(name, x_train, x_test, y_train, y_test)
        metrics = unit.oa_aa_kappa(y_test, y_te_pred)
        rows.append({
            "Model": name,
            "OA": round(float(metrics["OA"]), 4),
            "AA": round(float(metrics["AA"]), 4),
            "Kappa": round(float(metrics["Kappa"]), 4),
            "Time (s)": round(seconds, 1),
        })
        acc_matrix.append(_per_class_accuracy(y_test, y_te_pred, classes))

    missing = _missing_train_classes(y_train, data_y)

    # 指标分组柱状图：柱顶标注精确数值——各模型分数挤在 0.85~0.95 窄带，
    # Y 轴从 0 起画时柱高差异肉眼不可辨（用户反馈「每个方法都一样」），
    # 数值标签让 2~4 个百分点的差距直接可读。
    bar = go.Figure()
    for key, color in (("OA", "#2c5f8a"), ("AA", "#4f9d69"), ("Kappa", "#c9a227")):
        bar.add_trace(go.Bar(
            name=key,
            x=[MODEL_PARAM.get(m, {}).get("zh", m) for m in selected],
            y=[r[key] for r in rows],
            marker_color=color,
            # plotly 存根把 texttemplate/textposition 标成不存在，cast 收敛
            text=cast(Any, ["%.3f" % r[key] for r in rows]),
            textposition=cast(Any, "auto"),
            insidetextfont={"size": 11, "color": "white"},
            cliponaxis=False,
        ))
    bar.update_layout(barmode="group", yaxis=dict(range=[0, 1], title="Score"),
                      margin=dict(t=10, b=0, l=0, r=0),
                      legend=dict(orientation="h", y=1.12))

    # 逐类精度热力图：行=模型，列=类别
    heat = px.imshow(
        np.array(acc_matrix) if acc_matrix else np.zeros((0, 0)),
        x=[str(c) for c in classes],
        y=[MODEL_PARAM.get(m, {}).get("zh", m) for m in selected],
        # plotly 存根把 text_auto 标成 bool，实际支持格式字符串（如 ".2f"），cast 收敛
        text_auto=cast(Any, ".2f"), aspect="auto", zmin=0, zmax=1,
        color_continuous_scale="RdYlGn",
        labels={"x": "Class", "y": "Model", "color": "Accuracy"},
    )
    heat.update_layout(margin=dict(t=10, b=0, l=0, r=0))

    note = cap_note
    if missing:
        shown = ", ".join(str(c) for c in missing[:8]) + ("..." if len(missing) > 8 else "")
        note += (" Warning: " + shown + " has no training samples under this split; "
                 "their per-class accuracy is 0 by construction.")
    return rows, bar, heat, note, ("warning" if missing or cap_note else "success")


# ---------------------------------------------------------------- layout
def build_data_table(table_id):
    return dash_table.DataTable(
        id=table_id,
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px", "fontSize": "13px",
                    "minWidth": "80px"},
        style_header={"fontWeight": "bold"},
    )


layout_compare = dbc.Container(
    # 与前四页统一的结构：左侧固定浅灰侧栏（alert-light, 15% 宽）+ 右侧内容卡片
    dbc.Row(
        [
            # ---------------- 左：固定侧栏（与其他页同款式样） ----------------
            dbc.Col(
                dbc.Alert(
                    [
                        html.H4("File Input", **{"data-zh": "文件输入"}),
                        dbc.InputGroup(
                            [
                                html.Button("Read File", id="cmp-read",
                                            className="btn btn-primary",
                                            **{"data-zh": "读取文件"}),
                                dbc.Input(id="cmp-path", type="text",
                                          placeholder="Enter CSV File Path"),
                            ],
                            style={"width": "230px"},
                        ),
                        # 载入结果提示：成功显示行列数，失败显示红色原因（与分类页一致）
                        dbc.Alert(id="cmp-load-alert", is_open=False,
                                  className="mt-2",
                                  style={"fontSize": "12px", "padding": "8px"}),
                        html.Hr(),
                        html.H4("Loaded Data:", className="mb-3",
                                **{"data-zh": "已载入数据："}),
                        # 载入预览表：Read File 回调的第一个 Output 就是它
                        # （此前回调输出指向不存在的 cmp-table，点击后整个响应
                        # 应用失败，表现为「点了没反应」——下拉框/提示条也不更新）
                        build_data_table("cmp-table"),
                        html.P("Label Column(Y):", **{"data-zh": "标签列(Y)："}),
                        dcc.Dropdown(id="cmp-y", clearable=False,
                                     placeholder="Select Label Column(Y)"),
                        html.P("Feature Columns(X):", **{"data-zh": "特征列范围(X)："}),
                        dcc.RangeSlider(id="cmp-x", min=0, max=1, value=[0, 1],
                                        marks=None,
                                        tooltip={"placement": "bottom",
                                                 "always_visible": True}),
                        html.Hr(),
                        html.H4("Classifiers", className="mb-2",
                                **{"data-zh": "分类器"}),
                        dcc.Checklist(
                            id="cmp-models",
                            options=[{"label": m, "value": m} for m in CLASSIFIERS],
                            value=list(DEFAULT_MODELS),
                        ),
                        html.P("SVM / BPNN are slow on large data.",
                               className="text-muted small mt-1",
                               **{"data-zh": "大数据集上 SVM / BPNN 较慢，请按需勾选。"}),
                        html.Hr(),
                        html.P("Split Mode:", **{"data-zh": "划分方式："}),
                        dcc.Dropdown(id="cmp-split",
                                     options=list(SPLIT_MODES.keys()),
                                     value="Random (stratified)", clearable=False),
                        html.P("Test Ratio:", className="mt-2",
                               **{"data-zh": "测试集比例："}),
                        dcc.Slider(id="cmp-ratio", min=0.05, max=0.9, step=0.05,
                                   value=0.3, marks=None,
                                   tooltip={"placement": "bottom",
                                            "always_visible": True}),
                        html.P("Random Seed:", className="mt-2",
                               **{"data-zh": "随机种子："}),
                        dbc.Input(id="cmp-seed", type="number", value=18),
                        html.P("Max Training Samples:", className="mt-2",
                               **{"data-zh": "最大训练样本数："}),
                        dbc.Input(id="cmp-cap", type="number", value=20000),
                        html.P("Cap keeps comparison fast; 0 = use all rows.",
                               className="text-muted small",
                               **{"data-zh": "上限保证对比速度；0 表示使用全部样本。"}),
                        html.Button("Run Comparison", id="cmp-run",
                                    className="btn btn-primary mt-3",
                                    style={"width": "230px"},
                                    **{"data-zh": "开始对比"}),
                        dbc.Alert("Comparison completed!", id="cmp-alert",
                                  is_open=False, className="mt-3",
                                  style={"fontSize": "12px", "padding": "8px"}),
                    ],
                    className="alert-light",
                    # 比其他页（15%）稍宽：对比页侧栏控件更多（勾选框/种子/上限）。
                    # 左列栅格 width=3（预留 25%），侧栏 24% 与之匹配，不会盖住右侧。
                    style={"height": "calc(100vh - 80px)", "width": "24%",
                           "position": "fixed", "overflowY": "auto"},
                ),
                width=3,
            ),
            # ---------------- 右：标题 + 内容卡片（与其他页同款 alert-light 卡片） ----------------
            dbc.Col(
                [
                    html.H2("Model Comparison", className="mt-3 mb-2",
                            **{"data-zh": "多模型对比"}),
                    html.P("Train multiple classifiers on the SAME data and the SAME split, "
                           "then compare OA / AA / Kappa and per-class accuracy side by side.",
                           className="text-muted", **{"data-zh": "在同一份数据、同一种划分上训练多个分类器，"
                                                              "横向对比 OA / AA / Kappa 与逐类精度。"}),
                    dbc.Alert(
                        [
                            html.H4("Metrics Comparison (Test Set)", className="mb-2",
                                    **{"data-zh": "指标对比（测试集）"}),
                            dcc.Graph(id="cmp-bar", style={"height": "340px"}),
                        ],
                        className="alert-light mb-3",
                    ),
                    dbc.Alert(
                        [
                            html.H4("Per-Class Accuracy", className="mb-2",
                                    **{"data-zh": "逐类精度"}),
                            dcc.Graph(id="cmp-heat", style={"height": "340px"}),
                        ],
                        className="alert-light mb-3",
                    ),
                    dbc.Alert(
                        [
                            html.H4("Summary Table", className="mb-2",
                                    **{"data-zh": "汇总表"}),
                            build_data_table("cmp-result-table"),
                            html.Hr(),
                            html.P("Save comparison CSV to:",
                                   **{"data-zh": "对比结果保存到："}),
                            dbc.Input(id="cmp-path-out", type="text",
                                      placeholder="D:\\data\\comparison.csv"),
                            html.Button("Export CSV", id="cmp-export",
                                        className="btn btn-secondary mt-2",
                                        **{"data-zh": "导出 CSV"}),
                            dbc.Alert("Export completed!", id="cmp-export-alert",
                                      is_open=False, className="mt-2"),
                        ],
                        className="alert-light mb-3",
                    ),
                ],
                width=9,
            ),
        ],
    ),
    fluid=True,
)

# ---------------------------------------------------------------- callbacks
@app.callback(
    Output("cmp-table", "data"),
    Output("cmp-x", "min"),
    Output("cmp-x", "max"),
    Output("cmp-x", "value"),
    Output("cmp-y", "options"),
    Output("cmp-y", "value"),
    Output("cmp-load-alert", "children"),
    Output("cmp-load-alert", "is_open"),
    Output("cmp-load-alert", "color"),
    Input("cmp-read", "n_clicks"),
    State("cmp-path", "value"),
)
def update_compare_data(read, path):
    if not read:
        raise PreventUpdate
    try:
        # 复用分类建模页的加载逻辑（含自动补 .csv、文件夹提示等容错）
        data = load_model_dataframe("button-read5", path)
    except Exception as exc:  # 路径错误等：红色人话提示，不再静默
        return (no_update, no_update, no_update, no_update,
                no_update, no_update, "Load failed: %s" % exc, True, "danger")
    column_names = data.columns.tolist()
    default_y = "label" if "label" in column_names else column_names[0]
    global_var.set_global_column_name(column_names)
    global_var.set_global_input_table(data)
    return (_preview_records(data), 0, max(data.shape[1] - 1, 0), [0, data.shape[1] - 1],
            column_names, default_y,
            "Loaded %d rows x %d columns." % (data.shape[0], data.shape[1]),
            True, "success")


@app.callback(
    Output("cmp-alert", "children"),
    Output("cmp-alert", "is_open"),
    Output("cmp-alert", "color"),
    Output("cmp-bar", "figure"),
    Output("cmp-heat", "figure"),
    Output("cmp-result-table", "data"),
    Input("cmp-run", "n_clicks"),
    State("cmp-models", "value"),
    State("cmp-x", "value"),
    State("cmp-y", "value"),
    State("cmp-split", "value"),
    State("cmp-ratio", "value"),
    State("cmp-seed", "value"),
    State("cmp-cap", "value"),
)
def run_compare_callback(n_clicks, selected, modle_x, modle_y, split_mode,
                         ratio, seed, cap):
    if not n_clicks:
        raise PreventUpdate
    try:
        rows, bar, heat, note, color = run_comparison(
            selected, modle_x, modle_y, split_mode, ratio, seed, cap)
    except Exception as exc:
        return ("Comparison failed: %s" % exc, True, "danger",
                no_update, no_update, no_update)
    global_var.set_global_lower_table(pd.DataFrame(rows))  # 供其他页面引用（可选）
    _LAST_ROWS.clear()
    _LAST_ROWS.extend(rows)
    text = ("Comparison completed! Trained %d model(s) on the same split (%s).%s"
            % (len(rows), str(split_mode), note))
    return (text, True, color, bar, heat, rows)


# 模块级结果缓存：导出 CSV 用
_LAST_ROWS: list = []


@app.callback(
    Output("cmp-export-alert", "children"),
    Output("cmp-export-alert", "is_open"),
    Output("cmp-export-alert", "color"),
    Input("cmp-export", "n_clicks"),
    State("cmp-path-out", "value"),
)
def export_compare_csv(n_clicks, path):
    if not n_clicks:
        raise PreventUpdate
    try:
        if not _LAST_ROWS:
            raise ValueError("Run the comparison first, then export.")
        file_path = str(path).strip() if path else ""
        if not file_path:
            raise ValueError("Please enter the output CSV path first.")
        if os.path.isdir(file_path):
            raise ValueError("'%s' is a folder, not a CSV file." % file_path)
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"
        pd.DataFrame(_LAST_ROWS).to_csv(file_path, index=False)
    except Exception as exc:
        return ("Export failed: %s" % exc, True, "danger")
    return ("Export completed: %s" % file_path, True, "success")
