from __future__ import annotations  # Python 3.7 兼容：让 list[str] 等注解延迟求值

import pandas as pd
from typing import Any

# ── 全局状态 ──────────────────────────────────────────────────────────────────
# 说明：这些变量在不同页面/回调之间共享。此处显式标注类型，避免类型检查器
# 把它们推断成 list[Unknown]（例如 global_hs_X = []），从而在读取方
# （如 hs_X.shape）产生大量「属性不存在」的误报。
# 注意：初始值必须是「空但仍支持 len()」的对象（[] / () / DataFrame），
# 因为各页面用 `len(x) == 0` 或 `x.empty` 判断是否已加载数据。

global_img: Any = []
global_img_masked: Any = []
global_df: pd.DataFrame = pd.DataFrame()
global_wavelengths: Any = []
global_rgb: Any = []
global_file_path: str = ''
global_spectra_input: pd.DataFrame = pd.DataFrame()
global_spectra_output: pd.DataFrame = pd.DataFrame()
global_lower_table: Any = []
global_input_table: pd.DataFrame = pd.DataFrame()
global_column_name: list[str] = []

# ── Hyperspectral dataset state ───────────────────────────────────────────────
global_cube: Any = []          # ndarray (H, W, B)
global_gt: Any = []            # ndarray (H, W)
global_dataset_name: str = ''  # 'IP' | 'SA' | 'LK' | 'HH'
global_hs_X: Any = []          # ndarray (N, B) — flattened, background-removed
global_hs_y: Any = []          # ndarray (N,)  — labels
global_class_names: list[str] = []    # list of str
global_hs_shape: Any = ()      # (H, W, B) original cube shape


def set_global_spectra_input(value):
    global global_spectra_input
    global_spectra_input = value


def get_global_spectra_input():
    global global_spectra_input
    return global_spectra_input


def set_global_spectra_output(value):
    global global_spectra_output
    global_spectra_output = value


def get_global_spectra_output():
    global global_spectra_output
    return global_spectra_output


def set_global_img(value):
    global global_img
    global_img = value


def get_global_img():
    global global_img
    return global_img


def set_global_img_masked(value):
    global global_img_masked
    global_img_masked = value


def get_global_img_masked():
    global global_img_masked
    return global_img_masked


def set_global_df(value):
    global global_df
    global_df = value


def get_global_df():
    global global_df
    return global_df


def set_global_wavelengths(value):
    global global_wavelengths
    global_wavelengths = value


def get_global_wavelengths():
    global global_wavelengths
    return global_wavelengths


def set_global_rgb(value):
    global global_rgb
    global_rgb = value


def get_global_rgb():
    global global_rgb
    return global_rgb


def set_global_file_path(value):
    global global_file_path
    global_file_path = value


def get_global_file_path():
    global global_file_path
    return global_file_path


def set_global_input_table(value):
    global global_input_table
    global_input_table = value


def get_global_input_table():
    global global_input_table
    return global_input_table


def set_global_lower_table(value):
    global global_lower_table
    global_lower_table = value


def get_global_lower_table():
    global global_lower_table
    return global_lower_table


def set_global_column_name(value):
    global global_column_name
    global_column_name = value


def get_global_column_name():
    global global_column_name
    return global_column_name


# ── HS dataset getters/setters ────────────────────────────────────────────────
def set_global_cube(value):
    global global_cube
    global_cube = value


def get_global_cube():
    global global_cube
    return global_cube


def set_global_gt(value):
    global global_gt
    global_gt = value


def get_global_gt():
    global global_gt
    return global_gt


def set_global_dataset_name(value):
    global global_dataset_name
    global_dataset_name = value


def get_global_dataset_name():
    global global_dataset_name
    return global_dataset_name


def set_global_hs_X(value):
    global global_hs_X
    global_hs_X = value


def get_global_hs_X():
    global global_hs_X
    return global_hs_X


def set_global_hs_y(value):
    global global_hs_y
    global_hs_y = value


def get_global_hs_y():
    global global_hs_y
    return global_hs_y


def set_global_class_names(value):
    global global_class_names
    global_class_names = value


def get_global_class_names():
    global global_class_names
    return global_class_names


def set_global_hs_shape(value):
    global global_hs_shape
    global_hs_shape = value


def get_global_hs_shape():
    global global_hs_shape
    return global_hs_shape
