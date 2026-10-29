import os
from datetime import datetime
from typing import cast

import dash_bootstrap_components as dbc
import pandas as pd
from dash import html, dcc, dash_table, ctx, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import global_var
from myapp import app
from unit import reduce_dimensionality


def _num(value, default=0.0):
    """dcc.Input(type='number') / Slider 回调可能给出 None 或字符串，统一转 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=0):
    return int(_num(value, default))


def _records(frame):
    """DataFrame -> list[dict]（先 reset_index，与原来行为一致）。

    旧版 pandas（1.1.x）的 stub 里 reset_index() 的返回值带 None，to_dict() 也没声明
    orient 参数，直接写会在每个调用点都报类型错。这里集中收敛一次，语义不变。
    """
    df = cast(pd.DataFrame, cast(pd.DataFrame, frame).reset_index())
    return cast(list, df.to_dict("records"))  # pyright: ignore[reportArgumentType]


# 表格只送预览行：Salinas 全量 54129 行 × 206 列 = 1100 万个单元格，
# to_dict + JSON 序列化要几分钟、几百 MB，浏览器直接卡死。
# 全量数据保存在 global_var 里供 Save 使用，这里只影响页面显示。
TABLE_PREVIEW_ROWS = 200


def _preview_records(frame):
    return _records(cast(pd.DataFrame, frame).head(TABLE_PREVIEW_ROWS))

accordion = dbc.Accordion([
    dbc.AccordionItem(
        title="PCA",
        children=[
            html.P("Principal Component Analysis condenses a set of correlated spectral bands into a few orthogonal components that are ranked by the amount of variance they explain. Keeping only the leading components retains the dominant variability of the data while discarding largely redundant directions, which simplifies later analysis, visualisation and computation. Note that large variance is a proxy for, not a guarantee of, class-discriminative information.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Dimensions:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="pca-n_components", type="number", min=0, value=3, max=10, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="pca-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="t-SNE",
        children=[
            html.P("t-Distributed Stochastic Neighbor Embedding is a nonlinear technique aimed at visualising high-dimensional data. It builds probability distributions over neighbouring samples in the original space and searches for a low-dimensional layout whose Student-t based neighbour distribution matches them as closely as possible. Local neighbourhood structure is reproduced faithfully, so clusters of similar samples become clearly visible, but inter-cluster distances and the global geometry should not be over-interpreted, and the result depends on the random seed.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Dimensions:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="tsne-n_components", type="number", min=0, value=3, max=10, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Random Seed:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="tsne-random_state", type="number", min=0, value=18, max=1000, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="tsne-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="UMAP",
        children=[
            html.P("Uniform Manifold Approximation and Projection constructs a fuzzy topological graph that describes the nearest-neighbour relations of the data, then optimises a low-dimensional embedding whose topology reproduces that graph. It generally preserves more of the global structure than t-SNE, scales better to large datasets, and can embed new samples into an existing model; the neighbour count and the random seed both influence the final layout.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Dimensions:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="umap-n_components", type="number", min=0, value=3, max=10, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Random Seed:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="umap-random_state", type="number", min=0, value=18, max=1000, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="umap-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="MDS",
        children=[
            html.P("MultiDimensional Scaling looks for an arrangement of samples in a low-dimensional space such that the pairwise distances there agree as closely as possible with the distances observed in the original high-dimensional space. Classical MDS reaches this goal through an eigendecomposition and is therefore deterministic; it is a natural choice when the distance relationships between samples, rather than the raw coordinates, are what matters for interpretation.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Dimensions:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="mds-n_components", type="number", min=0, value=3, max=10, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Random Seed:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="mds-random_state", type="number", min=0, value=18, max=1000, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="mds-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="Sparse PCA",
        children=[
            html.P("Sparse Principal Component Analysis extends classical PCA with a sparsity constraint on the loading vectors, so each component is built from a small number of bands instead of blending the entire spectrum. The resulting components are far easier to interpret because they point to specific wavelengths, which makes the method attractive for band screening; the price is slightly lower explained variance than unconstrained PCA with the same component count.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Dimensions:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="spca-n_components", type="number", min=0, value=3, max=10, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="spca-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="SPA",
        children=[
            html.P("Successive Projections Algorithm is a greedy forward-selection procedure for spectral bands. It projects candidate wavelengths onto the remaining ones and keeps chains whose projection residuals are smallest, so the chosen bands are mutually weakly correlated and largely free of collinearity. A calibration model evaluated on the candidates finally decides the subset, which tends to be compact, interpretable and effective for classification.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Max Features:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="SPA_max_vars", type="number", min=0, value=3, max=10, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Random Seed:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="SPA_random_state", type="number", min=0, value=18, max=1000, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="SPA-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="UVE",
        children=[
            html.P("Uninformative Variable Elimination judges each band by comparing a PLS model fitted on the real spectra with a reference PLS model fitted on a matrix of random noise of the same shape. Bands whose regression coefficients are indistinguishable from the random reference are considered uninformative and removed, and the PLS model is refitted on the surviving bands; this usually sharpens both model accuracy and interpretability.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Components in PLS Model:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="UVE_n_comps_in", type="number", min=0, value=8, max=10, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Cross-Validation Folds:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="UVE_cv_in", type="number", min=1, value=5, max=20, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="UVE-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
    dbc.AccordionItem(
        title="CARS",
        children=[
            html.P("Competitive Adaptive Reweighted Sampling selects spectral bands through a survival-of-the-fittest process: in each Monte-Carlo iteration the bands are reweighted by the absolute values of PLS regression coefficients, preferentially sampled, and the least competitive ones are eliminated. The iteration whose subset reaches the lowest cross-validation error is kept, yielding a compact band combination that concentrates on the wavelengths most relevant to prediction.", style={"text-indent": "2em"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Components in PLS Model:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="CARS_n_comps_in", type="number", min=0, value=8, max=10, placeholder=""),
                ], width=4),

            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Cross-Validation Folds:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="CARS_cv_in", type="number", min=1, value=5, max=20, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Number of Iterations:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="CARS_iteration_in", type="number", min=0, value=50, max=1000, placeholder=""),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.P("Random Seed:"),
                ], width=8),
                dbc.Col([
                    dbc.Input(id="CARS_random_state", type="number", min=0, value=50, max=1000, placeholder=""),
                ], width=4),
            ]),
            html.Button("Calculate", id="CARS-button", className="btn btn-primary", **{"data-zh": "计算"}),
        ]
    ),
])

layout_reduce_dimensionality = dbc.Container([
    dbc.Row([
        dbc.Col(
            dbc.Alert([
                dbc.Row([
                    html.H4("File Input", **{"data-zh": "文件输入"}),
                        dbc.InputGroup([
                            html.Button("Retrieve Preprocessed Data", id="button-get4", className="btn btn-primary", **{"data-zh": "载入预处理数据"}),
                        ], className="mb-3"),
                    dbc.InputGroup([
                        html.Button("Read File", id="button-read4", className="btn btn-primary", **{"data-zh": "读取文件"}),
                        dbc.Input(id="input-path4", type="text", placeholder="Enter CSV File Path"),
                    ], className="mb-3"),
                    html.Hr(),
                    html.H5("Loaded Data:", className="mb-3"),
                    html.P("Label Column(Y):"),
                    dcc.Dropdown(id="dimensionality_y", clearable=False, placeholder="Select Feature Columns(Y)", ),
                    html.P("Feature Columns(X):"),
                    dbc.Col([
                        html.Div([
                            dcc.RangeSlider(0, 10, 1, id="dimensionality_x", marks=None,
                                            tooltip={"placement": "bottom", "always_visible": True}),
                            dbc.Tooltip(id="tooltip_4_1", target="dimensionality_x", placement="right", is_open=False),
                        ]),
                    ]),
                    dash_table.DataTable(id='table_in',
                                         virtualization=True,
                                         style_table={'height': '50%', 'overflowY': 'auto'},
                                         style_cell={'minWidth': 95, 'maxWidth': 95, 'width': 95,
                                                     'textAlign': 'left'},
                                         style_as_list_view=True,
                                         fixed_rows={'headers': True},
                                         ),
                ])
            ], className='alert-light', style={'height': 'calc(100vh - 80px)',
                                               'width': '15%', 'position': 'fixed'})
        ),
        dbc.Col([
            dbc.Alert([
                # 计算失败时在这里显示具体报错。以前服务是 debug=False，
                # 回调里抛异常页面毫无反应，用户只能看到「点了没反应」却查不到原因。
                dbc.Alert(id="dim-error-alert", color="danger", is_open=False,
                          style={"whiteSpace": "pre-wrap"}),
                dcc.Graph(id='output_fig4', style={'height': '700px'}),
                html.Hr(),
                html.H4("Reduced Data", className="mb-3", **{"data-zh": "降维结果"}),
                dash_table.DataTable(id='table_out',
                                     virtualization=True,
                                     style_table={'height': '50%', 'overflowY': 'auto'},
                                     style_cell={'minWidth': 95, 'maxWidth': 95, 'width': 95,
                                                 'textAlign': 'left'},
                                     style_as_list_view=True,
                                     fixed_rows={'headers': True},
                                     ),
            ], className='alert-light'),
        ], width=7),
        dbc.Col([
            dbc.Alert([
                html.H4("Dimensionality Reduction and Band Selection", className="mb-3", **{"data-zh": "降维与波段选择"}),
                html.Hr(),
                accordion,
                html.Br(),
                dbc.InputGroup([
                    html.Button("Save", id="button-save-output4", className="btn btn-primary", **{"data-zh": "保存"}),
                    dbc.Input(id="input-save-path4", type="text", placeholder="Enter Save Path"),
                ]),
            ], className='alert-light'),
        ], width=3),
    ]),
], fluid=True)


@app.callback(
    Output("table_in", 'data'),
    Output("dimensionality_x", "min"),
    Output("dimensionality_x", "max"),
    Output("dimensionality_x", "value"),
    Output("dimensionality_y", "options"),
    Output("dimensionality_y", "value"),
    Output("pca-n_components", "max"),
    Output("tsne-n_components", "max"),
    Output("umap-n_components", "max"),
    Output("mds-n_components", "max"),
    Output("spca-n_components", "max"),
    Output("SPA_max_vars", "max"),
    Output("UVE_n_comps_in", "max"),
    Output("CARS_n_comps_in", "max"),
    Input("button-read4", "n_clicks"),
    Input("button-get4", "n_clicks"),
    State("input-path4", "value")
)
def update_input_data(read_clicks, get_clicks, path):
    triggered = ctx.triggered_id
    if triggered == "button-read4":
        if not path:
            raise PreventUpdate
        try:
            # 旧版 pandas 的 stub 把 read_csv 标成「DataFrame | TextFileReader」，运行时必是 DataFrame
            data = cast(pd.DataFrame, pd.read_csv(path))
        except (OSError, ValueError):  # pd.errors.ParserError 是 ValueError 的子类
            raise PreventUpdate
    elif triggered == "button-get4":
        data = global_var.get_global_spectra_output()
        if data.empty:
            raise PreventUpdate
    else:
        raise PreventUpdate

    if data.empty or data.shape[1] < 2:
        # 只有一列时既没有特征列也没有标签列可选
        raise PreventUpdate

    min_value = 0
    max_value = data.shape[1] - 1
    global_var.set_global_input_table(data)
    column_names = data.columns.tolist()
    global_var.set_global_column_name(column_names)
    data_dict = _preview_records(data)
    # 默认标签列：优先叫 label 的列（光谱流水线的输出固定是 label + band_N），
    # 否则退而求其次选第一个非数值列，再不行就第一列。
    default_y = column_names[0]
    for name in column_names:
        if name.strip().lower() in ("label", "y", "class", "target"):
            default_y = name
            break
    else:
        # 旧版 pandas 的 stub 不认识 pd.api.types，用 dtype.kind 判断数值列
        non_numeric = [c for c in column_names
                       if getattr(getattr(data[c], "dtype", None), "kind", "O") not in "biufc"]
        if non_numeric:
            default_y = non_numeric[0]
    return (data_dict,
            min_value, max_value, [min_value, max_value], column_names, default_y,
            max_value, max_value, max_value, max_value, max_value, max_value, max_value, max_value)


@app.callback(
    Output("table_out", "data"),
    Output("output_fig4", "figure"),
    Output("dim-error-alert", "children"),
    Output("dim-error-alert", "is_open"),
    Input("pca-button", "n_clicks"),
    Input("tsne-button", "n_clicks"),
    Input("umap-button", "n_clicks"),
    Input("mds-button", "n_clicks"),
    Input("spca-button", "n_clicks"),
    Input("SPA-button", "n_clicks"),
    Input("UVE-button", "n_clicks"),
    Input("CARS-button", "n_clicks"),
    State("dimensionality_x", "value"),
    State("dimensionality_y", "value"),
    State("pca-n_components", "value"),
    State("tsne-n_components", "value"),
    State("umap-n_components", "value"),
    State("tsne-random_state", "value"),
    State("umap-random_state", "value"),
    State("mds-n_components", "value"),
    State("mds-random_state", "value"),
    State("spca-n_components", "value"),
    State("SPA_max_vars", "value"),
    State("SPA_random_state", "value"),
    State("UVE_n_comps_in", "value"),
    State("UVE_cv_in", "value"),
    State("CARS_n_comps_in", "value"),
    State("CARS_cv_in", "value"),
    State("CARS_iteration_in", "value"),
    State("CARS_random_state", "value"),
)
def dimensionality(pca_button, tsne_button, umap_button, mds_button, spca_button, spa_button, uve_button, cars_button,
        dimensionality_x, dimensionality_y,
        pca_n_components, tsne_n_components, umap_n_components,
                   tsne_random_state, umap_random_state,
                   mds_n_components, mds_random_state, spca_n_components, SPA_max_vars, SPA_random_state,
                   UVE_n_comps, UVE_cv, CARS_n_comps, CARS_cv, CARS_iteration, CARS_random_state
                   ):
    # 注意：原来这里有一个 `if not read`（要求 Read File 按钮点过才继续），
    # 导致用「Retrieve Preprocessed Data」加载的数据点 Calculate 被静默拦截。
    # 数据是否就绪由下面的 global_input_table.empty 判断兜底，这个门槛是多余且有害的。
    triggered = ctx.triggered_id

    def _fail(message):
        # 出错时在页面中央显示红色提示，而不是让回调静默失败（debug=False 下
        # 抛异常用户什么都看不到，只会觉得「点了没反应」）。
        return no_update, no_update, message, True

    data = global_var.get_global_input_table()
    if data is None or data.empty:
        return _fail("No data loaded. Please load data first "
                     "(click 'Retrieve Preprocessed Data' or 'Read File' on the left).")
    if dimensionality_x is None:
        return _fail("Please select the Feature Columns(X) range on the left first.")
    column_name = global_var.get_global_column_name()
    if not column_name:
        return _fail("Column information is missing. Please reload the data on the left.")

    # 标签列兜底：Y 为空或已失效（例如换过一次数据）时自动挑 label 列
    if dimensionality_y is None or dimensionality_y not in column_name:
        dimensionality_y = None
        for name in column_name:
            if str(name).strip().lower() in ("label", "y", "class", "target"):
                dimensionality_y = name
                break
        if dimensionality_y is None:
            dimensionality_y = column_name[0]

    lo = min(_int(dimensionality_x[0], 0), _int(dimensionality_x[1], 0))
    hi = max(_int(dimensionality_x[0], 0), _int(dimensionality_x[1], 0))
    lo = max(0, min(lo, len(column_name) - 1))
    hi = max(0, min(hi, len(column_name) - 1))
    # 与 layout_model.prepare_dataset 保持一致：标签列不能混进特征列（会造成标签泄漏）
    feature_names = [column for column in column_name[lo:hi + 1] if column != dimensionality_y]
    if not feature_names:
        return _fail("Selected Feature Columns(X) range is empty. "
                     "Please widen the X slider range on the left.")
    data_x = data[feature_names]
    data_y = data[dimensionality_y]

    method_kwargs = {
        "pca-button": ('PCA', pca_n_components, {}),
        "tsne-button": ('t-SNE', tsne_n_components, {"random_state": tsne_random_state}),
        "umap-button": ('UMAP', umap_n_components, {"random_state": umap_random_state}),
        "mds-button": ('MDS', mds_n_components, {"random_state": mds_random_state}),
        "spca-button": ('Sparse PCA', spca_n_components, {}),
        "SPA-button": ('SPA', SPA_max_vars, {"random_state": SPA_random_state}),
        "UVE-button": ('UVE', None, {"n_comps": UVE_n_comps, "cv": UVE_cv}),
        "CARS-button": ('CARS', None,
                        {"n_comps": CARS_n_comps, "cv": CARS_cv,
                         "iteration": CARS_iteration, "random_state": CARS_random_state}),
    }
    if triggered not in method_kwargs:
        raise PreventUpdate
    method, n_components, extra = method_kwargs[triggered]

    try:
        if n_components is None:
            reduced_df, fig = reduce_dimensionality(data_x, data_y, method, **extra)
        else:
            reduced_df, fig = reduce_dimensionality(data_x, data_y, method,
                                                    n_components, **extra)
    except Exception as exc:  # noqa: BLE001 —— 任何计算错误都要让用户在页面上看到
        return _fail("Calculate failed: {}: {}".format(type(exc).__name__, exc))

    global_var.set_global_lower_table(reduced_df)
    return _preview_records(reduced_df), fig, "", False




@app.callback(
    Output("alert-spectra-save4", "children"),
    Output("alert-spectra-save4", "is_open"),
    Output("alert-spectra-save4", "color"),
    Input("button-save-output4", "n_clicks"),
    State("input-save-path4", "value"),
)
def save_lower(save_button, path):
    if save_button:
        if path is not None:
            df = global_var.get_global_lower_table()
            df = pd.DataFrame(df)
            if not path.endswith(os.path.sep):
                path += os.path.sep
            if len(df) != 0:
                file_name = datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
                try:
                    df.to_csv(path + file_name, index=False)
                except Exception as exc:  # 路径不存在/无权限等：人话提示，避免 500 静默
                    return "Error: cannot save to '%s' (%s)" % (path, exc), True, 'danger'
                return "File successfully saved to" + path + file_name, True, 'success'
            return "Error: Dimensionality Reduction not yet calculated", True, 'danger'
        return "Error: No path entered", True, 'danger'
    else:
        raise PreventUpdate


@app.callback(
    Output("tooltip_4_1", "children"),
    Output("tooltip_4_1", "is_open"),
    Input("dimensionality_x", "value"),
)
def update_tooltip(value):
    if value is None:
        raise PreventUpdate
    column_names = global_var.get_global_column_name()
    if not column_names:
        raise PreventUpdate
    # RangeSlider 可能返回浮点数，必须先转 int 才能切片
    lo, hi = _int(value[0], 0), _int(value[1], 0)
    lo, hi = max(0, min(lo, hi)), min(max(lo, hi), len(column_names) - 1)
    selected = column_names[lo:hi + 1]
    # 全选 205 个波段时把 205 个列名全列出来会变成一块盖住页面的巨型浮层，
    # 超过 12 列就只显示首尾 + 总数。
    if len(selected) > 12:
        text = ", ".join(str(n) for n in selected[:6]) + " … " \
            + str(selected[-1]) + f"  ({len(selected)} columns)"
    else:
        text = ", ".join(str(name) for name in selected)
    return text, True
