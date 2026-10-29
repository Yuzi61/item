"""
Data visualization page: load .mat hyperspectral cube, display false-color
composite, ground-truth label map, per-class sample counts, per-class mean
spectra, and export flattened CSV for downstream pages.
"""

import os
from datetime import datetime

import dash_bootstrap_components as dbc
import data_loader
import global_var
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from myapp import app


# ── Helper functions ───────────────────────────────────────────────────────────

def _rgb_composite(cube, r_idx, g_idx, b_idx):
    """Build an RGB false-color composite from 3 band indices. Returns (H, W, 3) uint8."""
    bands = [r_idx, g_idx, b_idx]
    rgb = np.stack([cube[:, :, idx] for idx in bands], axis=-1).astype(np.float32)

    # Per-band 2%–98% stretch
    for c in range(3):
        lo, hi = np.percentile(rgb[:, :, c], [2, 98])
        if hi > lo:
            rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo), 0, 1)

    return (rgb * 255).astype(np.uint8)


def _gt_label_fig(gt, class_names):
    """Ground-truth label map as a discrete-color imshow."""
    fig = px.imshow(
        gt,
        color_continuous_scale=px.colors.qualitative.Light24,
        title="Ground Truth Label Map",
    )
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=40, b=0, l=0, r=0))
    return fig


def _class_count_fig(y, class_names):
    """Bar chart of sample count per class."""
    unique, counts = np.unique(y, return_counts=True)
    names = [class_names[i - 1] if 0 < i <= len(class_names) else f"Class {i}"
             for i in unique]
    df = pd.DataFrame({"Class": names, "Count": counts})
    fig = px.bar(df, x="Class", y="Count", title="Per-Class Sample Count")
    fig.update_layout(margin=dict(t=40, b=120))
    fig.update_xaxes(tickangle=-45)
    return fig


def _mean_spectra_fig(X, y, class_names):
    """Per-class mean spectral curves."""
    fig = go.Figure()
    unique = np.unique(y)
    for label in unique:
        spectra = X[y == label]
        mean_curve = np.mean(spectra, axis=0)
        name = class_names[label - 1] if 0 < label <= len(class_names) else f"Class {label}"
        fig.add_trace(go.Scatter(y=mean_curve, mode="lines", name=name))
    fig.update_layout(
        xaxis_title="Band Index",
        yaxis_title="Reflectance",
        title="Per-Class Mean Spectra",
        margin=dict(t=40, b=0),
    )
    return fig


def _load_error(message):
    """加载失败时只更新提示，不重置下拉框、波段滑块与数据集版本号。"""
    return (no_update, no_update, no_update, no_update, no_update, no_update, no_update,
            message, True, "danger", no_update)


def _default_mat_candidates():
    """列出项目 data/ 目录下的数据文件（排除 *_gt.mat 标签文件），按文件名排序。"""
    return data_loader.list_dataset_files()


# ── Layout ─────────────────────────────────────────────────────────────────────

layout_data = dbc.Container([
    # 数据集版本号：load_hyperspectral 写、update_data_visualization 读，
    # 用显式依赖替代「靠全局变量隐式传递」（Dash 对同一 Input 触发的多个回调
    # 是并发执行的，顺序不保证，直接读全局变量可能拿到上一份数据）。
    dcc.Store(id="store-dataset-rev", data=0),
    dbc.Row([
        # ── Left sidebar: file input ──
        dbc.Col(
            dbc.Alert([
                html.H4("Data Loading", **{"data-zh": "数据加载"}),
                dbc.InputGroup([
                    html.Button("Load .mat", id="button-load-mat", className="btn btn-primary", **{"data-zh": "加载 .mat"}),
                    dbc.Input(id="input-mat-path", type="text",
                              placeholder="留空 = 自动加载 data/ 里的数据集"),
                ], className="mb-3"),
                html.P("Dataset (auto-detect if not specified):", **{"data-zh": "数据集（留空自动识别）："}),
                dcc.Dropdown(
                    id="dropdown-dataset",
                    options=[
                        {"label": "Auto-detect", "value": ""},
                        {"label": "Indian Pines (IP)", "value": "IP"},
                        {"label": "Salinas (SA)", "value": "SA"},
                        {"label": "LongKou (LK)", "value": "LK"},
                        {"label": "HongHu (HH)", "value": "HH"},
                    ],
                    value="",
                    clearable=False,
                ),
                dbc.Alert(id="alert-data-load", is_open=False, color="info", dismissable=True,
                          style={"marginTop": "8px", "padding": "8px", "fontSize": "13px"}),
                html.Hr(),
                html.P("Band selection for false-color composite:", **{"data-zh": "伪彩合成波段选择："}),
                html.P("Red band:", className="mb-0", **{"data-zh": "红波段："}),
                dcc.Slider(0, 1, 1, id="band-r", value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.P("Green band:", className="mb-0 mt-2", **{"data-zh": "绿波段："}),
                dcc.Slider(0, 1, 1, id="band-g", value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.P("Blue band:", className="mb-0 mt-2", **{"data-zh": "蓝波段："}),
                dcc.Slider(0, 1, 1, id="band-b", value=0,
                           tooltip={"placement": "bottom", "always_visible": True}),
                html.Hr(),
                dbc.InputGroup([
                    html.Button("Export flattened CSV", id="button-export-csv", className="btn btn-primary", **{"data-zh": "导出展平 CSV"}),
                    dbc.Input(id="input-csv-path", type="text",
                              placeholder="Enter save folder path"),
                ]),
            ], className="alert-light",
               style={"height": "calc(100vh - 80px)", "width": "15%",
                      "position": "fixed", "overflowY": "auto"}),
        ),

        # ── Main content area ──
        dbc.Col([
            dbc.Alert([
                html.H4("False-Color Composite", className="mb-3", **{"data-zh": "伪彩合成"}),
                dcc.Graph(id="graph-false-color", style={"height": "350px"}),
                html.Hr(),
                html.H4("Ground Truth Label Map", className="mb-3", **{"data-zh": "GT 标签图"}),
                dcc.Graph(id="graph-gt-map", style={"height": "350px"}),
            ], className="alert-light mb-3"),
        ], width=5),
        dbc.Col([
            dbc.Alert([
                html.H4("Dataset Statistics", className="mb-3", **{"data-zh": "数据集统计"}),
                dcc.Graph(id="graph-class-count", style={"height": "350px"}),
                html.Hr(),
                html.H4("Per-Class Mean Spectra", className="mb-3", **{"data-zh": "逐类平均光谱"}),
                dcc.Graph(id="graph-mean-spectra", style={"height": "350px"}),
            ], className="alert-light"),
        ], width=5),
    ]),
], fluid=True)


# ── Callbacks ──────────────────────────────────────────────────────────────────

@app.callback(
    Output("dropdown-dataset", "value"),
    Output("band-r", "max"),
    Output("band-g", "max"),
    Output("band-b", "max"),
    Output("band-r", "value"),
    Output("band-g", "value"),
    Output("band-b", "value"),
    Output("alert-data-load", "children"),
    Output("alert-data-load", "is_open"),
    Output("alert-data-load", "color"),
    Output("store-dataset-rev", "data"),
    Input("button-load-mat", "n_clicks"),
    State("input-mat-path", "value"),
    State("dropdown-dataset", "value"),
)
def load_hyperspectral(n_clicks, mat_path, dataset_sel):
    if not n_clicks:
        raise PreventUpdate

    if not mat_path:
        # 路径留空时自动加载 data/ 目录下的数据集：
        # 选了具体数据集 → 挑该数据集的文件；Auto-detect → 取第一个。
        candidates = _default_mat_candidates()
        if not candidates:
            return _load_error(
                "Please enter a .mat file path first "
                "(no .mat file found in the project's data/ folder)."
            )
        if dataset_sel:
            matched = [p for p in candidates
                       if data_loader.guess_dataset_from_filename(p) == dataset_sel]
            if not matched:
                return _load_error(
                    f"No file for dataset {dataset_sel} was found in the project's "
                    f"data/ folder. Files available: "
                    f"{[os.path.basename(p) for p in candidates]}. "
                    f"Or enter a full .mat path above."
                )
            mat_path = matched[0]
        else:
            mat_path = candidates[0]

    try:
        dataset_name = dataset_sel if dataset_sel else None
        cube, gt, ds_name = data_loader.load_mat(mat_path, dataset_name)

        if cube.ndim != 3:
            raise ValueError(f"Expected a 3-D cube, got shape {cube.shape}.")
        if gt.shape[:2] != cube.shape[:2]:
            raise ValueError(f"Ground truth shape {gt.shape} does not match cube {cube.shape[:2]}.")

        H, W, B = cube.shape

        # 写入全局状态（与「按训练数据自动匹配数据集」共用同一套逻辑）
        X, y, class_names = data_loader.apply_dataset_state(cube, gt, ds_name)
    except (ValueError, OSError, TypeError, KeyError, IndexError, NotImplementedError) as exc:
        return _load_error(f"Load failed: {exc}")

    # Set default band selections spread across spectrum
    r_default = 0
    g_default = B // 2
    b_default = B - 1

    return (ds_name, B - 1, B - 1, B - 1, r_default, g_default, b_default,
            f"Loaded {ds_name}: {H}×{W}×{B}, {len(class_names)} classes, {X.shape[0]} labeled pixels.",
            True, "success", n_clicks)


@app.callback(
    Output("graph-false-color", "figure"),
    Input("band-r", "value"),
    Input("band-g", "value"),
    Input("band-b", "value"),
)
def update_false_color(r, g, b):
    cube = global_var.get_global_cube()
    if cube is None or len(cube) == 0:
        raise PreventUpdate
    if r is None or g is None or b is None:
        # Slider 尚未初始化时不要用 None 去索引波段
        raise PreventUpdate

    n_bands = cube.shape[-1]
    r = max(0, min(int(r), n_bands - 1))
    g = max(0, min(int(g), n_bands - 1))
    b = max(0, min(int(b), n_bands - 1))

    rgb = _rgb_composite(cube, r, g, b)
    fig = px.imshow(rgb, title=f"False-Color (R={r}, G={g}, B={b})")
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))
    return fig


@app.callback(
    Output("graph-gt-map", "figure"),
    Output("graph-class-count", "figure"),
    Output("graph-mean-spectra", "figure"),
    Input("store-dataset-rev", "data"),
)
def update_data_visualization(rev):
    # 改为依赖 store-dataset-rev：由 load_hyperspectral 写入，
    # 保证本回调一定在数据写入全局状态之后才执行。
    if not rev:
        raise PreventUpdate

    gt = global_var.get_global_gt()
    X = global_var.get_global_hs_X()
    y = global_var.get_global_hs_y()
    class_names = global_var.get_global_class_names()

    if len(gt) == 0 or len(X) == 0:
        raise PreventUpdate

    gt_fig = _gt_label_fig(gt, class_names)
    count_fig = _class_count_fig(y, class_names)
    spectra_fig = _mean_spectra_fig(X, y, class_names)

    return gt_fig, count_fig, spectra_fig


@app.callback(
    Output("alert-data-save", "children"),
    Output("alert-data-save", "is_open"),
    Output("alert-data-save", "color"),
    Input("button-export-csv", "n_clicks"),
    State("input-csv-path", "value"),
)
def export_flattened_csv(n_clicks, save_path):
    if not n_clicks:
        raise PreventUpdate

    if not save_path:
        return "Error: No save path entered", True, "danger"

    X = global_var.get_global_hs_X()
    y = global_var.get_global_hs_y()
    ds_name = global_var.get_global_dataset_name()

    if len(X) == 0:
        return "Error: No data loaded", True, "danger"

    df = data_loader.to_dataframe(X, y, ds_name)

    if not save_path.endswith(os.path.sep):
        save_path += os.path.sep
    file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
    try:
        df.to_csv(save_path + file_name, index=False)
    except OSError as exc:
        return f"Error: {exc}", True, "danger"
    return f"Saved to {save_path}{file_name}", True, "success"
