import os
from datetime import datetime
from typing import Any, cast

import dash_bootstrap_components as dbc
import global_var
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import pretreatment
import pywt
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from myapp import app

# pywt 的 Wavelet 类由 C 扩展 _pywt 注入到 pywt 命名空间，发行包里没有 .pyi，
# 静态检查器看不到 pywt.Wavelet（运行时一定存在）。用 getattr 取回，行为完全一致。
_Wavelet = getattr(pywt, "Wavelet")


def _num(value, default=0.0):
    """dcc.Input(type='number') / Slider 回调可能给出 None 或字符串，统一转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    return int(_num(value, default))

accordion = dbc.Accordion([
    dbc.AccordionItem(
        title="Standard Normal Variate",
        id="accordion-page-1",
        children=[
            html.P("Standard Normal Variate treats each spectrum as its own reference: the mean and standard deviation are computed along the spectral axis of every sample, and each reading is re-expressed as a deviation from that per-sample average. Every spectrum ends up centred at zero with unit spread, which cancels offset and gain differences caused by particle size, illumination geometry and instrument response. Because the correction is applied independently to each sample, broadband noise shared by all bands is not removed and may call for an additional filtering step.", style={"text-indent": "2em"}),
            html.Hr(),
            html.Button("Calculate", id="snv-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Multiple Scatter Correction",
        id="accordion-page-2",
        children=[
            html.P("Multiple Scatter Correction compensates the additive and multiplicative distortions that light scattering imprints on spectra. Each sample spectrum is linearly regressed against a chosen reference spectrum, usually the average of the whole set, and the fitted intercept and slope are then used to shift and rescale the sample onto the reference level. After this adjustment, differences from particle size, surface shape and optical path are largely removed, so the remaining variation reflects chemical rather than physical properties.", style={"text-indent": "2em"}),
            html.Hr(),
            html.Button("Calculate", id="msc-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Differentiation Processing",
        id="accordion-page-3",
        children=[
            html.P("Derivative processing turns broad, overlapping band shapes into sharper and better separated features. Integer-order derivatives are computed as successive differences along the spectral axis, while fractional orders use weighted Gregory coefficients so that the differentiation strength can be tuned continuously; moderate orders flatten slowly varying baselines, and stronger orders highlight fine structure at the cost of amplifying high-frequency noise, so the order is usually balanced against a smoothing step.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.InputGroup([
                html.Button("Calculate", id="diff-button", className="btn btn-primary", **{"data-zh": "计算"}),
                dbc.Input(id="order", type="number", min=0, max=5, step=0.1, placeholder="Please enter the order"),
            ], className="mb-3", )
        ]
    ),
    dbc.AccordionItem(
        title="Mean Centering",
        id="accordion-page-5",
        children=[
            html.P("Mean Centering subtracts the average level of each spectrum from the spectrum itself, so a sample is afterwards described purely by how its individual bands deviate from its own centre. Constant offsets arising from overall brightness or baseline height disappear, which prevents distance- and covariance-based methods from being dominated by absolute intensity; what remains is the relative band pattern that usually carries the class-discriminative information.", style={"text-indent": "2em"}),
            html.Hr(),
            html.Button("Calculate", id="mean-center-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Standardization",
        id="accordion-page-6",
        children=[
            html.P("Standardization gives every band a comparable footing: the column mean is subtracted and the remainder is divided by the column standard deviation, leaving each band with zero mean and unit variance. Without this rescaling, strong bands would dominate distance metrics, clustering and gradient-based optimisers simply because of their magnitude; equalising the variances restores a balanced contribution from all bands and typically improves both stability and convergence speed.", style={"text-indent": "2em"}),
            html.Hr(),
            html.Button("Calculate", id="standardize-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Min-Max Normalization",
        id="accordion-page-7",
        children=[
            html.P("Min-Max Normalization maps each feature into a fixed interval, conventionally from 0 to 1, using the smallest and largest observed values as anchors. The transformation is a linear rescale that preserves the shape of the data distribution while removing units and magnitude differences, which is convenient for methods that expect bounded inputs. One caveat is its sensitivity to outliers: a single extreme value compresses all remaining samples into a narrow sub-range.", style={"text-indent": "2em"}),
            html.Hr(),
            html.Button("Calculate", id="min-max-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Vector Normalization",
        id="accordion-page-8",
        children=[
            html.P("Vector Normalization divides every band of a spectrum by the Euclidean norm computed over all bands of that same spectrum, so each complete spectrum is scaled to unit length. Overall intensity differences caused by sample thickness, illumination strength or acquisition gain are absorbed into the norm, and only the relative proportions among bands survive, which makes spectra measured under unequal conditions directly comparable.", style={"text-indent": "2em"}),
            html.Hr(),
            html.Button("Calculate", id="vector-normalize-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Savitzky-Golay Smoothing Filter",
        id="accordion-page-9",
        children=[
            html.P("The Savitzky-Golay filter slides a window across the spectrum and fits a low-order polynomial to the points inside each window, then keeps the fitted value at the window centre. Compared with plain averaging, the polynomial fit preserves peak heights, widths and areas, so random high-frequency noise is suppressed while genuine absorption features remain essentially undistorted; the window width and polynomial order trade smoothness against retained detail.", style={"tex-indent": "2em"}),
            html.Hr(),
            dbc.Row(
                dbc.Col([
                    html.P("Window Size:"),
                    dcc.Slider(3, 21, 2, value=5, id="sg-window"),
                    html.P("Polynomial Order:"),
                    dcc.Slider(1, 7, 1, value=3, id="sg-polyorder"),
                ], width=12)
            ),
            html.Button("Calculate", id="sg-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Moving Average Filtering",
        id="accordion-page-10",
        children=[
            html.P("Moving Average Filtering replaces each band value with the mean of its neighbours inside a fixed-width window that slides along the spectral axis. Zero-mean random noise is reduced in proportion to the square root of the window length, while the cost is a slight flattening of narrow features; wider windows produce smoother curves but also clip fine structure at both ends of the retained band range.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row(
                dbc.Col([
                    html.P("Window Size:"),
                    dcc.Slider(3, 21, 2, value=5, id="moving-window"),
                ], width=12)
            ),
            html.Button("Calculate", id="moving-average-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Wavelet Thresholding",
        id="accordion-page-11",
        children=[
            html.P("Wavelet Thresholding first decomposes each spectrum into wavelet coefficients on several resolution levels, then shrinks or discards coefficients whose magnitude does not clearly exceed the estimated noise level. The threshold is selected with the SURE criterion, which adapts the trade-off between the bias of removing true detail and the variance of keeping noise. Since signal and noise separate differently at each scale, the method is effective for spectra whose noise is spread across multiple frequency bands.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Wavelet Function:"),
                    dcc.Dropdown(["haar", "db1", "db2", "db3",
                                  "sym2", "sym3", "sym4", "sym5",
                                  "coif1", "coif2", "coif3", "coif4"], value=None, id="wavelet-select"),
                    html.P("Number of Decomposition Levels:"),
                    dcc.Slider(1, max=None, step=1, id="level-select")
                ], width=12),
            ]),
            html.Button("Calculate", id="wavelet-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
])

layout_pretreatment = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Alert([
                dbc.Row([
                    html.H4("File Input", **{"data-zh": "文件输入"}),
                    dbc.InputGroup([
                        html.Button("Read Path", id="button-read3", className="btn btn-primary", **{"data-zh": "读取路径"}),
                        dbc.Input(id="input-path3", type="text", placeholder="Enter CSV File Path"),
                    ], className="mb-3", id="tip4"),
                    html.Hr(),
                    html.Button("Retrieve Current Spectral Data", id="button-get3", className="btn btn-primary", **{"data-zh": "载入当前光谱数据"}),
                ])
            ], className='alert-light', style={'height': 'calc(100vh - 80px)',
                                               'width': '15%', 'position': 'fixed'})
        ),
        dbc.Col([
            dbc.Alert([
                html.H4("Input Spectral Data", className="mb-3", **{"data-zh": "输入光谱数据"}),
                dcc.Graph(id='input_fig', style={'height': '450px'}),
                html.Hr(),
                html.H4("Output Spectral Data", className="mb-3", **{"data-zh": "输出光谱数据"}),
                dcc.Graph(id='output_fig', style={'height': '450px'}),
            ], className='alert-light'),
        ], width=7),
        dbc.Col([
            dbc.Alert([
                html.H4("Spectral Preprocessing", className="mb-3", **{"data-zh": "光谱预处理"}),
                html.Hr(),
                dbc.InputGroup([
                    html.Button("Replace Input with Output", id="button-transfer-output", className="btn btn-primary", **{"data-zh": "用输出替换输入"}),
                    html.Button("Save", id="button-save-output", className="btn btn-primary", **{"data-zh": "保存"}),
                    dbc.Input(id="input-save-path", type="text", placeholder="Enter Save Path"),
                ]),
                html.Br(),
                accordion
            ], className='alert-light'),
        ], width=3),
    ]),
    dbc.Tooltip(
        "Read Spectral File via File Path",
        target="tip4",
    ),
    dbc.Tooltip(
        "Reload the current spectral dataset from global state for continued preprocessing",
        target="button-get3",
    ),
    dbc.Tooltip(
        "Click this button to replace \"Input Spectral Data\" with \"Output Spectral Data\" for continuous processing",
        target="button-transfer-output",
    ),
], fluid=True)


def split_spectral_table(data):
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    meta_columns = [column for column in data.columns if column not in numeric_columns]
    return meta_columns, numeric_columns


def build_message_figure(message):
    """把错误/提示信息渲染成一张空图，避免回调静默失败或直接抛异常。"""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False,
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          font=dict(size=14))
    figure.update_layout(margin=dict(t=20, b=20, l=20, r=20),
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
    return figure


# 光谱数据集动辄上万个样本（Indian Pines 10249 条），全画出来浏览器会卡死，
# 且图例按行序排列、开头恰好是同一块田 → 图例「全是 Corn-mintill」。
# 这里均匀抽稀到最多 MAX_TRACES 条曲线，并按类别 legendgroup 分组、
# 每个类别只在图例里显示一次。
MAX_TRACES = 300


def build_spectra_figure(data):
    meta_columns, numeric_columns = split_spectral_table(data)
    if not numeric_columns:
        return build_message_figure("No numeric spectral columns found in the loaded data.")

    total = len(data)
    if meta_columns:
        labels = data[meta_columns[0]].astype(str).tolist()
    else:
        labels = ["sample"] * total

    if total > MAX_TRACES:
        step = int(np.ceil(total / MAX_TRACES))
        view = data.iloc[::step]
        view_labels = labels[::step]
        subsample_note = f"showing {len(view)} of {total} samples (uniform subsampling)"
    else:
        view = data
        view_labels = labels
        subsample_note = None

    seen_groups = set()
    plot_data = []
    for trace_name, (_, row) in zip(view_labels, view.iterrows()):
        is_first_of_group = trace_name not in seen_groups
        seen_groups.add(trace_name)
        plot_data.append(go.Scatter(
            y=row[numeric_columns].values,
            mode="lines",
            name=trace_name,
            legendgroup=trace_name,
            showlegend=is_first_of_group,
        ))

    figure = go.Figure(data=plot_data)
    figure.update_layout(
        xaxis_title="Band Index",
        yaxis_title="Reflectance",
        legend_title="Class" if meta_columns else "Sample",
        margin=dict(t=30, b=0, l=0, r=0),
    )
    if subsample_note:
        figure.add_annotation(
            text=subsample_note, showarrow=False,
            xref="paper", yref="paper", x=0.5, y=1.06,
            font=dict(size=12, color="gray"),
        )
    return figure


@app.callback(
    Output("level-select", "max"),
    Output("level-select", "value"),
    Input("wavelet-select", "value"),
    Input("button-read3", "n_clicks"),
    Input("button-get3", "n_clicks"),
    State("level-select", "value"),
)
def update_wavelet_max_level(wavelet, n1, n2, current_level):
    # 原来只有 wavelet-select 是 Input，导致「先加载数据、再选小波」时
    # max 永远不被计算；这里把两个加载按钮也接成 Input，并把 value 夹到合法区间。
    if not (n1 or n2) or not wavelet:
        raise PreventUpdate
    data = global_var.get_global_spectra_input()
    if data.empty:
        raise PreventUpdate
    data_len = data.select_dtypes(include=[np.number]).shape[1]
    if data_len < 2:
        raise PreventUpdate
    max_level = max(1, int(pywt.dwt_max_level(data_len=data_len, filter_len=_Wavelet(wavelet).dec_len)))
    level = min(max(_int(current_level, 1), 1), max_level)
    return max_level, level


@app.callback(
    Output("alert-spectra-save", "children"),
    Output("alert-spectra-save", "is_open"),
    Output("alert-spectra-save", "color"),
    Input("button-save-output", "n_clicks"),
    State("input-save-path", "value"),
)
def save_mean(save_button, path):
    if save_button:
        if path is not None:
            df = global_var.get_global_spectra_output()
            if not path.endswith(os.path.sep):
                path += os.path.sep
            if not df.empty:
                file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
                try:
                    df.to_csv(path + file_name, index=False)
                except Exception as exc:  # 路径不存在/无权限等：人话提示，避免 500 静默
                    return "Error: cannot save to '%s' (%s)" % (path, exc), True, 'danger'
                return "File successfully saved to" + path + file_name, True, 'success'
            return "Error: Average spectrum not yet calculated", True, 'danger'
        return "Error: No path entered", True, 'danger'
    else:
        raise PreventUpdate


@app.callback(
    Output("input_fig", "figure"),
    # Output("input_table", "data"),
    Input("button-read3", "n_clicks"),
    Input("button-get3", "n_clicks"),
    Input("button-transfer-output", "n_clicks"),
    State("input-path3", "value")
)
def update_input_data(read, get, transfer, path):
    triggered = ctx.triggered_id
    if triggered == "button-read3":
        if not path:
            return build_message_figure("Please enter a CSV file path first.")
        try:
            # 旧版 pandas 的 stub 把 read_csv 标成「DataFrame | TextFileReader」的联合，
            # 运行时这里必然拿到 DataFrame；cast 收敛后下游的 .empty/.columns 等误报一并消失。
            data = cast(pd.DataFrame, pd.read_csv(path))
        except (OSError, ValueError) as exc:  # pd.errors.ParserError 是 ValueError 的子类
            return build_message_figure(f"Failed to read {path}: {exc}")
    elif triggered == "button-get3":
        data = global_var.get_global_spectra_output()
        if data.empty:
            data = global_var.get_global_spectra_input()
        if data.empty:
            data = global_var.get_global_input_table()
    elif triggered == "button-transfer-output":
        data = global_var.get_global_spectra_output()
    else:
        raise PreventUpdate

    if data.empty:
        return build_message_figure("No spectral data available. Load a .mat file or read a CSV first.")

    global_var.set_global_spectra_input(data)
    return build_spectra_figure(data)


@app.callback(
    Output("alert-spectra", "is_open"),
    Input("snv-button", "n_clicks"),
    Input("msc-button", "n_clicks"),
    Input("diff-button", "n_clicks"),
    Input("mean-center-button", "n_clicks"),
    Input("standardize-button", "n_clicks"),
    Input("min-max-button", "n_clicks"),
    Input("vector-normalize-button", "n_clicks"),
    Input("sg-button", "n_clicks"),
    Input("moving-average-button", "n_clicks"),
    Input("wavelet-button", "n_clicks"),
    State("order", "value"),
    State("button-read3", "n_clicks"),
    State("button-get3", "n_clicks"),
    State("sg-window", "value"),
    State("sg-polyorder", "value"),
    State("moving-window", "value"),
    State("wavelet-select", "value"),
    State("level-select", "value")
)
def update_spectra(snv, msc, d, mean, standardize, min_max, vector, sg, move, wavelet, order, n1, n2,
                   sg_window, sg_polyorder, move_window, wavelet_select, wavelet_level):
    if not (n1 or n2):
        raise PreventUpdate

    triggered = ctx.triggered_id
    data = global_var.get_global_spectra_input()
    meta_columns, numeric_columns = split_spectral_table(data)
    if not numeric_columns:
        raise PreventUpdate

    # 控件可以被清空，None 直接送进算法会抛 TypeError；统一兜底成默认值。
    order = _num(order, 1.0)
    sg_window = _int(sg_window, 5)
    sg_polyorder = _int(sg_polyorder, 3)
    move_window = _int(move_window, 5)
    wavelet_select = wavelet_select or "db1"
    wavelet_level = max(1, _int(wavelet_level, 1))

    # data[list] 在旧版 stub 里是 DataFrame | Series | None 的联合，先 cast 成 DataFrame
    numeric_frame = cast(pd.DataFrame, data[numeric_columns])
    spectra_values = cast(pd.DataFrame, numeric_frame.apply(pd.to_numeric, errors='coerce')).to_numpy()

    if triggered == "snv-button":
        data_pretreated = pretreatment.snv(spectra_values)
    elif triggered == "msc-button":
        data_pretreated = pretreatment.msc(spectra_values)
    elif triggered == "diff-button":
        data_pretreated = pretreatment.diffs(spectra_values, order)
    elif triggered == "mean-center-button":
        data_pretreated = pretreatment.mean_center(spectra_values)
    elif triggered == "standardize-button":
        data_pretreated = pretreatment.standardize(spectra_values)
    elif triggered == "min-max-button":
        data_pretreated = pretreatment.min_max(spectra_values)
    elif triggered == "vector-normalize-button":
        data_pretreated = pretreatment.vector_normalize(spectra_values)
    elif triggered == "sg-button":
        data_pretreated = pretreatment.sg_smooth(spectra_values, sg_window, sg_polyorder)
    elif triggered == "moving-average-button":
        data_pretreated = pretreatment.moving_average(spectra_values, move_window)
    elif triggered == "wavelet-button":
        data_pretreated = pretreatment.wavelet_threshold(spectra_values, wavelet_select, wavelet_level)
    else:
        # 不再用 else 去兜底某个具体算法，避免未知触发源时静默执行错误的预处理
        raise PreventUpdate

    data_pretreated = np.asarray(data_pretreated)

    # 列数可能因算法而变（滑动平均/差分/小波重构）。按输出列数取波段名，
    # 避免出现 `[:-0]` 这种空切片（窗口/阶数为 0 时会得到 0 列）。
    n_out = data_pretreated.shape[1]
    if triggered == "moving-average-button":
        cut = max(0, move_window - 1)
    elif triggered == "diff-button":
        cut = max(0, int(np.floor(order)))
    else:
        cut = 0
    if cut > 0 and len(numeric_columns) - cut == n_out:
        tolist = numeric_columns[:len(numeric_columns) - cut]
    else:
        tolist = numeric_columns[:n_out]

    df_pretreated = pd.DataFrame(data_pretreated, columns=tolist)
    if meta_columns:
        # 同理：旧版 stub 里 reset_index() 的返回值带 None，cast 后才能安全 concat
        meta_frame = cast(pd.DataFrame, cast(pd.DataFrame, data[meta_columns]).reset_index(drop=True))
        df_pretreated = pd.concat([meta_frame, df_pretreated], axis=1)
    global_var.set_global_spectra_output(df_pretreated)
    return True


@app.callback(
    Output("output_fig", "figure"),
    # Output("output_table", "data"),
    Input("alert-spectra", "is_open"),
)
def update_output_data(is_open):
    if is_open:
        data = global_var.get_global_spectra_output()
        return build_spectra_figure(data)
    else:
        raise PreventUpdate
