import logging
import os
import socket
import threading
import time
import webbrowser

import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

from layout_compare import layout_compare
from layout_data import layout_data
from layout_model import layout_model
from layout_pretreatment import layout_pretreatment
from layout_reduce_dimensionality import layout_reduce_dimensionality
from myapp import app

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

# 导航链接清单：(组件 id, 中文, 路由, 英文)。高亮回调按这个清单生成样式输出，
# 语言切换回调按 data-zh / data-en 属性交换显示文字。
_NAV_LINKS = [
    ("nav-link-data", "数据加载", "/data", "Data Loading"),
    ("nav-link-pretreatment", "光谱预处理", "/pretreatment", "Preprocessing"),
    ("nav-link-reduction", "降维与波段选择", "/reduction", "Reduction"),
    ("nav-link-model", "分类建模", "/model", "Classification"),
    ("nav-link-compare", "模型对比", "/compare", "Comparison"),
]
# 其余页保持普通白色；当前页仅亮黄色加粗（用户要求：不要底条、不弱化其他页）。
_NAV_STYLE_IDLE = {"color": "#f8f9fa", "fontWeight": "400",
                   "padding": "8px 14px", "textDecoration": "none", "display": "inline-block"}
_NAV_STYLE_ACTIVE = {"color": "#ffd43b", "fontWeight": "700",
                     "padding": "8px 14px", "textDecoration": "none", "display": "inline-block"}

navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("RHyperLab", href="/", className="mr-auto"),
            dbc.Nav(
                [
                    dbc.NavItem(dcc.Link(
                        html.Span(text, **{"data-zh": text, "data-en": en}),
                        href=href, id=link_id, className="nav-link"))
                    for link_id, text, href, en in _NAV_LINKS
                ] + [
                    dbc.NavItem(
                        html.Button(
                            "English", id="lang-toggle",
                            className="btn btn-sm btn-outline-light ml-2",
                            style={"marginRight": "10px"},
                            **{"data-zh": "English", "data-en": "中文"},
                        )
                    ),
                    dbc.Alert(
                        "Spectral Preprocessing Completed!",
                        id="alert-spectra",
                        dismissable=True,
                        is_open=False,
                        duration=2000,
                        style={"position": "fixed", "top": "25px", "right": "25px"},
                    ),
                    # 预处理页「Save」按钮的保存提示
                    dbc.Alert(
                        "Save Completed!",
                        id="alert-spectra-save",
                        dismissable=True,
                        is_open=False,
                        duration=2000,
                        style={"position": "fixed", "top": "25px", "right": "25px"},
                    ),
                    # 数据加载页「Export flattened CSV」的保存提示
                    # （原先与 alert-spectra-save 共用同一个 id，构成重复回调输出）
                    dbc.Alert(
                        "Export Completed!",
                        id="alert-data-save",
                        dismissable=True,
                        is_open=False,
                        duration=2000,
                        style={"position": "fixed", "top": "25px", "right": "25px"},
                    ),
                    dbc.Alert(
                        "Dimensionality Reduction Saved!",
                        id="alert-spectra-save4",
                        dismissable=True,
                        is_open=False,
                        duration=2000,
                        style={"position": "fixed", "top": "25px", "right": "25px"},
                    ),
                    # 说明：alert-rf / alert-svm / alert-bpnn / alert-lda / alert-knn
                    # 由 layout_model.py 在每个分类器面板内部创建（能显示具体报错信息）。
                    # 这里不再重复声明，否则同一页面会出现两个同 id 组件。
                ],
                navbar=True,                className="ml-auto",
            ),
        ],
        fluid=True,
    ),
    className="mb-3",
    color="primary",
    dark=True,
    sticky="top",
    style={"padding-left": "30px"},
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    pages = {
        "/": layout_data,
        "/data": layout_data,
        "/pretreatment": layout_pretreatment,
        "/reduction": layout_reduce_dimensionality,
        "/model": layout_model,
        "/compare": layout_compare,
    }
    return pages.get(pathname, html.H3("404"))


@app.callback(
    [Output(link_id, "style") for link_id, _, _, _ in _NAV_LINKS],
    Input("url", "pathname"),
)
def highlight_nav(pathname):
    """当前页导航项高亮：亮黄加粗，其余页普通白色。"""
    current = pathname or "/"
    styles = []
    for _, _, href, _en in _NAV_LINKS:
        is_active = current == href or (href == "/data" and current == "/")
        styles.append(dict(_NAV_STYLE_ACTIVE if is_active else _NAV_STYLE_IDLE))
    return styles


app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        navbar,
        html.Div(id="page-content"),
        # 中英文切换：语言状态持久化到 localStorage，前端 JS 换 textContent
        dcc.Store(id="lang-store", storage_type="local"),
        html.Div(id="lang-applied", style={"display": "none"}),
    ]
)

# 语言状态翻转：点按钮在 zh / en 之间切换（首次点击 null → en）
app.clientside_callback(
    """function(nClicks, current) {
        if (!nClicks) { return window.dash_clientside.no_update; }
        return (current === 'en') ? 'zh' : 'en';
    }""",
    Output("lang-store", "data"),
    Input("lang-toggle", "n_clicks"),
    State("lang-store", "data"),
)

# 应用语言：扫描所有带 data-zh 属性的元素交换文字。
# 页面路由切换后新 DOM 是服务端渲染的默认语言，需重新应用；
# Dash 回调触发与 DOM 渲染存在竞态，故用延时重试兜底。
app.clientside_callback(
    """function(lang, pathname) {
        var target = lang || 'en';
        function apply() {
            var nodes = document.querySelectorAll('[data-zh]');
            for (var i = 0; i < nodes.length; i++) {
                var el = nodes[i];
                if (!el.getAttribute('data-en')) {
                    el.setAttribute('data-en', el.textContent);
                }
                el.textContent = (target === 'zh')
                    ? el.getAttribute('data-zh') : el.getAttribute('data-en');
            }
        }
        apply();
        setTimeout(apply, 350);
        setTimeout(apply, 1200);
        return 'applied:' + target;
    }""",
    Output("lang-applied", "children"),
    Input("lang-store", "data"),
    Input("url", "pathname"),
)


def _open_browser_when_ready(port, timeout=45.0):
    """轮询端口，服务真正就绪后再打开浏览器。

    之前用户常见的困惑是「双击启动后立刻打开网址 → 报无法连接」，
    因为 dash/flask 首次导入 sklearn 等库要好几秒才 bind 端口。
    这里等端口可连接后再开，避免「打不开」。失败静默忽略，不影响服务。
    """
    if os.environ.get("RHyperLab_NO_BROWSER"):
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            if sock.connect_ex(("127.0.0.1", int(port))) == 0:
                webbrowser.open("http://127.0.0.1:%d/" % int(port))
                return
        finally:
            sock.close()
        time.sleep(0.5)


if __name__ == "__main__":
    PORT = 8050
    threading.Thread(
        target=_open_browser_when_ready, args=(PORT,), daemon=True
    ).start()
    # dash 2.x 的存根把 port 声明为 str，而 Flask/werkzeug 内部会做 int(port)，
    # 这里传 int 运行完全正常，只是类型标注对不上。
    app.run(debug=False, port=PORT)  # pyright: ignore[reportArgumentType]
