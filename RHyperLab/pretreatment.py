import numpy as np
import pywt
import scipy.special as sc
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# pywt 的 Wavelet 由 C 扩展注入 pywt 命名空间且发行包没有 .pyi，
# 静态检查器看不到 pywt.Wavelet（运行时一定存在）。用 getattr 取回，行为一致。
_Wavelet = getattr(pywt, "Wavelet")


def _int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _num(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _odd_window(value, default):
    """窗口参数必须是正奇数（savgol_filter / 卷积都要求），偶数自动减 1。"""
    window = _int(value, default)
    if window < 1:
        window = default
    if window % 2 == 0:
        window -= 1
    return max(window, 1)


def snv(data):
    # 对数据进行标准正态变换(SNV)预处理，返回处理后的数据
    data = np.asarray(data, dtype=float)
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True, ddof=1)
    # 常数光谱（std=0）不能做除法，否则整条曲线变成 NaN
    std = np.where(std == 0, 1.0, std)
    processed_data = (data - mean) / std
    return processed_data


def msc(data):
    """
    对数据进行多元散射校正(MSC)预处理，返回处理后的数据。

    标准 MSC：以全体样本的「平均光谱」为参考，对每条光谱做一元线性回归
    （slope * ref + intercept ≈ spectrum），再用斜率/截距把散射造成的
    基线偏移校正掉。

    旧实现对「每条样本的均值标量序列」做 N 阶 Vandermonde 最小二乘——
    N=10249 时是一个 10249×10249（约 800MB）矩阵，lstsq 要算几分钟，
    服务直接被占死，且数学上毫无校正意义。
    """
    data = np.asarray(data, dtype=float)
    reference = data.mean(axis=0)
    processed_data = np.empty_like(data)
    for i, spectrum in enumerate(data):
        slope, intercept = np.polyfit(reference, spectrum, 1)
        if not np.isfinite(slope) or slope == 0:
            slope, intercept = 1.0, 0.0
        processed_data[i] = (spectrum - intercept) / slope
    return processed_data


def mean_center(data):
    """
    对数据进行均值中心化(Mean Center)预处理，返回处理后的数据
    """
    mean = np.mean(data, axis=1, keepdims=True)
    processed_data = data - mean
    return processed_data


def standardize(data):
    """
    对数据进行标准化预处理，返回处理后的数据
    """
    scaler = StandardScaler(with_mean=True)
    processed_data = scaler.fit_transform(data)
    return processed_data


def min_max(data):
    """
    对数据进行MinMax标准化预处理，返回处理后的数据
    """
    scaler = MinMaxScaler()
    processed_data = scaler.fit_transform(data)
    return processed_data


def vector_normalize(data):
    """
    对数据进行向量归一化预处理，返回处理后的数据
    """
    data = np.asarray(data, dtype=float)
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    # 零向量不能做除法
    norm = np.where(norm == 0, 1.0, norm)
    processed_data = data / norm
    return processed_data


def sg_smooth(data, window_size=5, polyorder=2):
    """
    对数据进行平滑处理，返回处理后的数据
    """
    data = np.asarray(data, dtype=float)
    n_bands = data.shape[1] if data.ndim > 1 else data.shape[0]
    window_size = _odd_window(window_size, 5)
    window_size = max(1, min(window_size, n_bands if n_bands % 2 == 1 else n_bands - 1))
    # savgol_filter 要求 polyorder < window_size
    polyorder = max(0, min(_int(polyorder, 2), window_size - 1))
    processed_data = savgol_filter(data, window_size, polyorder)
    return processed_data


def moving_average(data, window_size=5):
    """
    对数据进行滑动平均处理，返回处理后的数据
    """
    data = np.asarray(data, dtype=float)
    n_bands = data.shape[1] if data.ndim > 1 else data.shape[0]
    window_size = max(1, min(_int(window_size, 5), n_bands))
    # 构造滑动窗口
    kernel = np.ones(window_size) / window_size
    # 使用卷积运算实现滑动平均
    processed_data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='valid'), axis=1, arr=data)
    return processed_data


def sure_threshold(coeffs):
    """
    计算给定小波系数的 SURE 阈值
    """
    N = coeffs.shape[1]
    squared_coeffs = coeffs ** 2
    sorted_squared_coeffs = np.sort(squared_coeffs, axis=1)[:, ::-1]
    cumulative_sum = np.cumsum(sorted_squared_coeffs, axis=1)
    range_array = np.arange(1, N + 1)
    risk_values = (N - 2 * range_array + cumulative_sum) / N
    best_risk_indices = np.argmin(risk_values, axis=1)
    threshold_values = np.sqrt(sorted_squared_coeffs[np.arange(coeffs.shape[0]), best_risk_indices])
    return threshold_values


def wavelet_threshold(data, wavelet='db1', level=1):
    """
    对数据进行小波阈值处理，返回处理后的数据
    """
    data = np.asarray(data, dtype=float)
    wavelet = wavelet or 'db1'
    n_bands = data.shape[1] if data.ndim > 1 else data.shape[0]
    # 分解层数不能超过该小波在当前信号长度下的最大可分解层数，否则 pywt 会直接报错
    max_level = max(1, int(pywt.dwt_max_level(data_len=n_bands, filter_len=_Wavelet(wavelet).dec_len)))
    level = min(max(_int(level, 1), 1), max_level)
    # 进行小波变换
    coeffs = pywt.wavedec(data, wavelet, level=level, axis=1)
    # 对每一层小波系数进行阈值处理
    for i in range(1, len(coeffs)):
        # 计算 SURE 阈值
        threshold_values = sure_threshold(coeffs[i])
        # 对小波系数进行处理
        coeffs[i] = pywt.threshold(coeffs[i], threshold_values[:, np.newaxis], mode='soft')
    # 进行小波重构
    reconstructed_data = pywt.waverec(coeffs, wavelet, axis=1)
    return reconstructed_data


def diffs(data, order):
    """
    对数据进行差分预处理，支持整数和非整数(G-L定义)差分
    :param data: shape=(m, n) 的输入序列数据
    :param order: 差分阶数（可为整数或非整数）
    :return: shape=(m, n - floor(order)) 的差分结果
    """
    data = np.asarray(data, dtype=float)
    order = _num(order, 1.0)
    n = data.shape[1]
    coeffs = np.array([_gl_coeff(order, k) for k in range(n)])
    diff_data = np.zeros_like(data)

    # 计算离散GL分数阶（或整数阶）差分
    for i in range(n):
        diff_data[:, i] = np.sum(coeffs[:i + 1][::-1] * data[:, :i + 1], axis=1)

    # 无论整数或分数阶，都裁剪 floor(order) 列
    cut = int(np.floor(order))
    if cut > 0:
        diff_data = diff_data[:, cut:]
    return diff_data


def _gl_coeff(d, k):
    """
    计算 G-L 分数差分的系数
    """
    return (-1) ** k * sc.comb(d, k)
