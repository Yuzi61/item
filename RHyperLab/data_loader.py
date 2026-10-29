"""
Hyperspectral image data loader for standard benchmark datasets.

Supports: Indian Pines (IP), Salinas (SA), LK (LongKou), HH (HongHu).
Reads .mat files, flattens H×W×B cube to N×B sample matrix,
removes background pixels (label == 0), and maps numeric labels
to human-readable class names.
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

import global_var


# ── Dataset metadata ──────────────────────────────────────────────────────────
# Each entry: (data_key_candidates, gt_key_candidates, class_names)
# IP & SA class lists follow the official documentation.
# LK & HH follow WHU-Hi documentation.

_DATASET_REGISTRY = {
    "IP": {
        "data_keys":   ["indian_pines", "indian_pines_corrected", "data"],
        "gt_keys":     ["indian_pines_gt", "ground_truth", "gt"],
        "n_classes": 16,
        "class_names": [
            "Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture",
            "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed", "Oats",
            "Soybean-notill", "Soybean-mintill", "Soybean-clean",
            "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
            "Stone-Steel-Towers",
        ],
    },
    "SA": {
        "data_keys":   ["salinas", "salinas_corrected", "data"],
        "gt_keys":     ["salinas_gt", "ground_truth", "gt"],
        "n_classes": 16,
        "class_names": [
            "Brocoli-green-weeds-1", "Brocoli-green-weeds-2", "Fallow",
            "Fallow-rough-plow", "Fallow-smooth", "Stubble", "Celery",
            "Grapes-untrained", "Soil-vineyard-develop", "Corn-senesced-weeds",
            "Lettuce-romaine-4wk", "Lettuce-romaine-5wk",
            "Lettuce-romaine-6wk", "Lettuce-romaine-7wk",
            "Vineyard-untrained", "Vineyard-vertical-trellis",
        ],
    },
    "LK": {
        # WHU-Hi LongKou
        "data_keys":   ["longkou", "lk", "data", "image"],
        "gt_keys":     ["longkou_gt", "lk_gt", "ground_truth", "gt", "map"],
        "n_classes": 9,
        "class_names": [
            "Corn", "Cotton", "Sesame", "Broad-leaf soybean",
            "Narrow-leaf soybean", "Rice", "Water", "Roads and houses",
            "Mixed weed",
        ],
    },
    "HH": {
        # WHU-Hi HongHu
        "data_keys":   ["honghu", "hh", "data", "image"],
        "gt_keys":     ["honghu_gt", "hh_gt", "ground_truth", "gt", "map"],
        "n_classes": 22,
        "class_names": [
            "Red roof", "Road", "Bare soil", "Cotton", "Cotton firewood",
            "Rape", "Chinese cabbage", "Pakchoi", "Cabbage", "Tuber mustard",
            "Brassica parachinensis", "Brassica chinensis",
            "Small brassica chinensis", "Lactuca sativa", "Celtuce",
            "Film covered lettuce", "Romaine lettuce", "Carrot",
            "White radish", "Garlic sprout", "Broad bean", "Tree",
        ],
    },
}


# ── 文件名 → 数据集 的匹配线索 ────────────────────────────────────────────────
# 用于「数据文件与标签文件分开存放」的情况（本仓库 data/ 目录即为此形态）：
#   Indian_pines_corrected.mat + Indian_pines_gt.mat
#   Salinas_corrected.mat      + Salinas_gt.mat
#   WHU_Hi_LongKou.mat         + WHU_Hi_LongKou_gt.mat
#   WHU_Hi_HongHu.mat          + WHU_Hi_HongHu_gt.mat
_FILENAME_HINTS = (
    ("IP", ("indian_pines", "indianpines")),
    ("SA", ("salinas",)),
    ("LK", ("longkou",)),
    ("HH", ("honghu",)),
)

_GT_NAME_HINTS = ("_gt", "gt.", "ground", "label", "truth", "mask")

# 通用变量名：几乎所有 .mat 都可能用它们，不能作为「是某个内置数据集」的证据，
# 否则自定义文件里叫 data/gt 的变量会被误判成 IP 等数据集（类名张冠李戴）。
_GENERIC_KEYS = {"data", "image", "ground_truth", "gt", "map"}


def _guess_dataset_from_filename(path):
    """按文件名推断数据集短码；认不出来返回 None。"""
    name = os.path.basename(path).lower()
    for ds_name, hints in _FILENAME_HINTS:
        if any(hint in name for hint in hints):
            return ds_name
    return None


def _guess_dataset_from_keys(raw):
    """按 .mat 里出现的变量名推断数据集短码；认不出来返回 None。

    只认「特指」变量名（如 indian_pines / salinas_gt）；data / gt / map 等
    通用名不参与判断（见 _GENERIC_KEYS），它们交给 _shape_pair 按维度兜底。
    """
    for ds_name, meta in _DATASET_REGISTRY.items():
        if any(k in raw and k not in _GENERIC_KEYS for k in meta["data_keys"]):
            return ds_name
        if any(k in raw and k not in _GENERIC_KEYS for k in meta["gt_keys"]):
            return ds_name
    return None


def _looks_like_gt(path):
    name = os.path.basename(path).lower()
    return any(hint in name for hint in _GT_NAME_HINTS)


def guess_dataset_from_filename(path):
    """Public helper: infer the dataset short code from a file name (None if unknown)."""
    return _guess_dataset_from_filename(path)


def _registry_pair(raw, ds_name):
    """按注册表候选 key 找出 (data, gt)，缺哪一半就返回 None。"""
    if not ds_name or ds_name not in _DATASET_REGISTRY:
        return None, None
    meta = _DATASET_REGISTRY[ds_name]
    data_key = next((k for k in meta["data_keys"] if k in raw), None)
    gt_key = next((k for k in meta["gt_keys"] if k in raw), None)
    return (raw[data_key] if data_key else None), (raw[gt_key] if gt_key else None)


def _shape_pair(raw):
    """没有数据集线索时的兜底：3 维数组当数据、2 维数组当标签。"""
    keys = [k for k in raw if not k.startswith("__")]
    cube = next((raw[k] for k in keys if raw[k].ndim == 3), None)
    gt = next((raw[k] for k in keys if raw[k].ndim == 2), None)
    return cube, gt


def _squeeze_last(array):
    """去掉 scipy 读入时可能多出的、长度为 1 的尾维。"""
    array = np.asarray(array)
    if array.ndim >= 3 and array.shape[-1] == 1:
        array = array.squeeze(-1)
    return array


def _find_partner_file(mat_path, ds_name, want_gt):
    """在同一目录里找配对文件：want_gt=True 找标签文件，否则找数据文件。"""
    directory = os.path.dirname(os.path.abspath(mat_path))
    this_file = os.path.abspath(mat_path)
    for file_name in sorted(os.listdir(directory)):
        if not file_name.lower().endswith(".mat"):
            continue
        candidate = os.path.join(directory, file_name)
        if os.path.abspath(candidate) == this_file:
            continue
        if _looks_like_gt(candidate) != want_gt:
            continue
        candidate_ds = _guess_dataset_from_filename(candidate)
        if ds_name is not None and candidate_ds not in (ds_name, None):
            continue
        return candidate
    return None


def load_mat(mat_path, dataset_name=None):
    """
    Load a .mat file and return (cube, gt, dataset_name).

    Supports two layouts:
      ① 数据与标签在同一个 .mat 里（经典数据集发布格式）；
      ② 数据文件与 *_gt.mat 标签文件分开存放（本仓库 data/ 目录的格式）。
    传入任意一半，会自动去同目录找另一半。

    Parameters
    ----------
    mat_path : str
        Path to the .mat file（数据文件或标签文件均可）。
    dataset_name : str, optional
        Force a specific dataset ("IP", "SA", "LK", "HH").

    Returns
    -------
    cube : ndarray, shape (H, W, B)
    gt   : ndarray, shape (H, W)
    dataset_name : str
    """
    if not os.path.isfile(mat_path):
        raise ValueError(f"File not found: {mat_path}")

    raw = loadmat(mat_path)
    keys = [k for k in raw if not k.startswith("__")]
    if not keys:
        raise ValueError(f"{mat_path} contains no arrays. Raw keys: {list(raw)}")

    ds_name = dataset_name or _guess_dataset_from_keys(raw) or _guess_dataset_from_filename(mat_path)

    # 用户在下拉框里强制指定了数据集，但这个文件明显属于另一个数据集 ——
    # 提前用人话报错，不要等到「形状不匹配」才暴露。
    file_ds = _guess_dataset_from_keys(raw) or _guess_dataset_from_filename(mat_path)
    if dataset_name and file_ds and dataset_name != file_ds:
        raise ValueError(
            f"The selected dataset {dataset_name} does not match this file: "
            f"\"{os.path.basename(mat_path)}\" looks like {file_ds}. "
            f"Check the Dataset dropdown or the file path."
        )

    # ① 同一个文件里同时含数据与标签
    cube, gt = _registry_pair(raw, ds_name)
    if cube is None or gt is None:
        auto_cube, auto_gt = _shape_pair(raw)
        cube = cube if cube is not None else auto_cube
        gt = gt if gt is not None else auto_gt

    # ② 数据 / 标签分文件存放 —— 去同目录找配对文件补齐缺的那一半
    if cube is None or gt is None:
        partner = _find_partner_file(mat_path, ds_name, want_gt=(gt is None))
        if partner is not None:
            partner_raw = loadmat(partner)
            p_cube, p_gt = _registry_pair(partner_raw, ds_name)
            if p_cube is None and p_gt is None:
                p_cube, p_gt = _shape_pair(partner_raw)
            cube = cube if cube is not None else p_cube
            gt = gt if gt is not None else p_gt

    if cube is None or gt is None:
        directory = os.path.dirname(os.path.abspath(mat_path))
        missing = "ground truth" if gt is None else "data cube"
        raise ValueError(
            f"Could not locate the {missing} for {mat_path}. "
            f"Keys in this file: {keys}. "
            f".mat files in the folder: {sorted(f for f in os.listdir(directory) if f.endswith('.mat'))}"
        )

    cube = _squeeze_last(cube)
    gt = _squeeze_last(gt)

    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D data cube, got shape {cube.shape}.")
    if gt.ndim != 2:
        raise ValueError(f"Expected a 2-D ground-truth map, got shape {gt.shape}.")
    if gt.shape != cube.shape[:2]:
        raise ValueError(
            f"Ground truth shape {gt.shape} does not match the cube's spatial size {cube.shape[:2]}."
        )

    return cube, gt, (ds_name or "CUSTOM")


def flatten(cube, gt, dataset_name=None):
    """
    Flatten a H×W×B cube into (X, y) sample matrix, removing background (label 0).

    Returns
    -------
    X : ndarray, shape (N, B)   — float32
    y : ndarray, shape (N,)     — int (1-indexed labels, 0 removed)
    """
    H, W, B = cube.shape
    X = cube.reshape(H * W, B).astype(np.float32)
    y = gt.reshape(H * W).astype(int)

    mask = y > 0
    X = X[mask]
    y = y[mask]

    return X, y


def get_class_names(dataset_name):
    """Return the list of class names for a given dataset.

    Unknown / custom datasets return an empty list: downstream code falls back
    to numeric labels ("Class_1", "Class_2", ...) instead of raising KeyError
    (a custom .mat that matches none of the built-in datasets gets ds_name
    "CUSTOM", which is not in the registry).
    """
    meta = _DATASET_REGISTRY.get(dataset_name)
    return meta["class_names"] if meta else []


def get_n_classes(dataset_name):
    """Return the number of classes (excluding background) for a given dataset.

    Returns None for unknown / custom datasets (class count is data-driven).
    """
    meta = _DATASET_REGISTRY.get(dataset_name)
    return meta["n_classes"] if meta else None


def get_available_datasets():
    """Return list of supported dataset short codes."""
    return list(_DATASET_REGISTRY.keys())


def to_dataframe(X, y, dataset_name=None):
    """
    Convert flattened (X, y) to a pandas DataFrame with band columns and a 'label' column.
    Optionally map numeric labels to class names.
    """
    n_samples, n_bands = X.shape
    columns = [f"band_{i+1}" for i in range(n_bands)]
    df = pd.DataFrame(X, columns=columns)

    if dataset_name and dataset_name in _DATASET_REGISTRY:
        names = _DATASET_REGISTRY[dataset_name]["class_names"]
        # labels are 1-indexed, names are 0-indexed list
        df["label"] = [names[label - 1] if 0 < label <= len(names) else f"Class_{label}"
                       for label in y]
    else:
        df["label"] = y

    return df


def default_data_dir():
    """本项目 data/ 目录（内置数据集所在位置）。"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def list_dataset_files(directory=None):
    """列出目录下的「数据」.mat 文件（排除 *_gt.mat 标签文件），按文件名排序。

    排除规则与数据加载页原先的 _default_mat_candidates 保持一字不差（只看
    _gt.mat 后缀），否则会与页面上「Auto-detect 载入哪个文件」的结果不一致。
    """
    if directory is None:
        directory = default_data_dir()
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith(".mat")
            and not name.lower().endswith("_gt.mat")]


def apply_dataset_state(cube, gt, ds_name):
    """把数据集写入全局状态。返回 (X, y, class_names)。

    数据加载页的 Load 与「按训练数据自动匹配数据集」共用这一套写入逻辑：
    两处各写一份的话，日后改漏一处就会出现「页面显示 A、内存实际是 B」的状态错位。
    """
    H, W, B = cube.shape
    X, y = flatten(cube, gt)
    class_names = get_class_names(ds_name)
    global_var.set_global_cube(cube)
    global_var.set_global_gt(gt)
    global_var.set_global_dataset_name(ds_name)
    global_var.set_global_hs_X(X)
    global_var.set_global_hs_y(y)
    global_var.set_global_class_names(class_names)
    global_var.set_global_hs_shape((H, W, B))
    # 下游（预处理页）用的光谱输入表
    global_var.set_global_spectra_input(to_dataframe(X, y, ds_name))
    # 新数据集作废上一轮建模/降维的中间结果，避免维度对不上
    global_var.set_global_lower_table([])
    global_var.set_global_spectra_output(pd.DataFrame())
    return X, y, class_names


def _gt_pixel_stats(gt_path, ds_name):
    """只读标签文件，返回 (有标签像元数, 全图像元数)；读不出来返回 (None, None)。

    标签文件通常只有几 KB（Salinas_gt.mat 4 KB），而数据立方体可达几百 MB，
    所以「核对像元数」这一步绝不能去加载立方体。
    """
    raw = loadmat(gt_path)
    _cube, gt = _registry_pair(raw, ds_name)
    if gt is None:
        _cube, gt = _shape_pair(raw)
    if gt is None:
        return None, None
    gt = _squeeze_last(gt)
    if gt.ndim != 2:
        return None, None
    return int((np.asarray(gt) > 0).sum()), int(np.asarray(gt).size)


def dataset_pixel_counts(ds_name, directory=None):
    """某内置数据集的 (有标签像元数, 全图像元数)；查不到返回 (None, None)。"""
    for mat_path in list_dataset_files(directory):
        if _guess_dataset_from_filename(mat_path) != ds_name:
            continue
        gt_path = _find_partner_file(mat_path, ds_name, want_gt=True)
        if gt_path is None:
            return None, None
        return _gt_pixel_stats(gt_path, ds_name)
    return None, None


def match_dataset_by_training_data(labels, n_rows, directory=None):
    """按「类别名集合 + 有标签像元数」把一批训练数据匹配回内置数据集。

    场景：用户先载入了数据集 A，随后用从数据集 B 导出的 CSV 训练，分类图会画在
    A 上（对不上时直接出不来）。这里用 CSV 里的类别名与行数反查它到底属于哪个
    内置数据集，命中即可自动切换，用户不必重启服务或手动重载。

    返回 (dataset_code, mat_path)；匹配不上返回 (None, None)。
    判定条件（两条都要满足，避免误判）：
      1) CSV 里出现的类别名全部属于该数据集的类别表；
      2) CSV 的行数 == 该数据集的有标签像元数（只含标签像元的 CSV）
         或 == 全图像元数（含背景行的整幅 CSV）。
    """
    wanted = {v for v in (labels if labels is not None else ())
              if isinstance(v, str) and v}
    if not wanted or int(n_rows) <= 0:
        return None, None
    for mat_path in list_dataset_files(directory):
        ds_name = _guess_dataset_from_filename(mat_path)
        if not ds_name:
            continue
        known = set(get_class_names(ds_name))
        if not known or not wanted.issubset(known):
            continue
        gt_path = _find_partner_file(mat_path, ds_name, want_gt=True)
        if gt_path is None:
            continue
        labeled, total = _gt_pixel_stats(gt_path, ds_name)
        if int(n_rows) in (labeled, total):
            return ds_name, mat_path
    return None, None
