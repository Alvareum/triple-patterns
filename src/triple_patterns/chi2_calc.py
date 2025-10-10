import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Tuple
from scipy_stats_vectorized import chi2_contingency_vectorized
from statsmodels.stats.multitest import multipletests


def get_label_intervals(
    dict_label_times: dict, n_frames: int
) -> dict[int, Tuple[int, list[int]]]:
    dict_label_arr = {}
    for label, times in dict_label_times.items():
        if len(times) < n_frames:
            tmp_arr = np.zeros(len(times))
        else:
            tmp_arr = np.zeros(n_frames)
        tmp_arr[times] = 1
        list_intervals = list(np.argwhere(np.diff(tmp_arr)).flatten())
        bool_even = len(list_intervals) % 2 == 0
        bool_non_zero = tmp_arr[0] != 0 and tmp_arr[-1] != 0
        if bool_even and bool_non_zero:
            n_ints = (len(list_intervals) + 2) // 2
            list_intervals.insert(0, 0)
            list_intervals.append(n_frames - 1)
        elif bool_even:
            n_ints = len(list_intervals) // 2
        else:
            n_ints = (len(list_intervals) + 1) // 2
            if tmp_arr[0] != 0:
                list_intervals.insert(0, 0)
            else:
                list_intervals.append(n_frames - 1)
        dict_label_arr[label] = (n_ints, list_intervals)
    return dict_label_arr


def get_freq_tables_triples(
    intensity_array: npt.ArrayLike,
    triples_array: npt.ArrayLike,
    dict_label_intensity: dict[int, npt.ArrayLike],
    labels: list[int],
    list_ints: list[int],
) -> dict[Tuple, npt.ArrayLike]:
    """Посчитать количество интервалов, когда горело то или иное ребро во время определенного лейбла
    только для uint8!
    intensity_array - двумерный массив uint8, где строки - интенсивность горения тройки, столбцы - время
    triples_array - массив с тройками
    dict_label_intensity - словарь с интенсивностями горения лейблов в uint8
    labels - список лейблов цифрами
    list_ints - число промежутков, когда горел какой-либо лейбл"""
    dict_freq_tables = {}
    n_labels = len(labels)
    n_edges = len(triples_array)
    for i in range(n_edges):
        if i % 100000 == 0:
            print(f"{i}: {triples_array[i, :]}")
        freq_arr = np.zeros((2, n_labels))
        for idx, label in enumerate(labels):
            label_intensity = dict_label_intensity[label]
            n_ints = 0
            res_intensity = np.unpackbits(label_intensity & intensity_array[i, :])
            res_ints = np.where(np.diff(res_intensity))[0]
            if len(res_ints) != 0:
                bool_even = len(res_ints) % 2 == 0
                bool_non_zero = res_intensity[0] != 0 and res_intensity[-1] != 0
                if bool_even and bool_non_zero:
                    n_ints = (len(res_ints) + 2) // 2
                elif bool_even:
                    n_ints = len(res_ints) // 2
                else:
                    n_ints = (len(res_ints) + 1) // 2
            else:
                n_ints = 0
            freq_arr[0, idx] = n_ints
            freq_arr[1, idx] = list_ints[idx] - n_ints
        if sum(freq_arr[0, :]) == 0:
            continue
        dict_freq_tables[tuple(triples_array[i])] = freq_arr
    return dict_freq_tables


def get_counts_and_time_packed(intensity: npt.ArrayLike) -> pd.DataFrame:
    popcnt = popcount_precalc()
    df_ans = pd.DataFrame(
        [
            get_counts_and_sum_packed(intensity[i, :], popcnt)
            for i in range(len(intensity))
        ],
        columns=["events_count", "events_duration"],
        dtype=np.int32,
    )
    return df_ans


def popcount_precalc(nbits: int = 8, dtype: npt.DTypeLike = np.uint32) -> npt.ArrayLike:
    popcnt = np.zeros(1 << nbits, dtype=dtype)
    for x in range(len(popcnt)):
        popcnt[x] = bin(x).count("1")
    return popcnt


@jit(nogil=True, nopython=True)
def get_counts_and_sum_packed(
    packed_intensity: npt.ArrayLike, popcnt: npt.ArrayLike
) -> Tuple[int, int]:
    """
    Numba version of get_counts_2 with bits packed values.
    """
    start_cnt = 0
    sum_cnt = 0
    prv = 0
    for t in range(len(packed_intensity)):
        start_mask = ~packed_intensity[t] & ((packed_intensity[t] >> 1) ^ (prv << 7))
        start_cnt += popcnt[start_mask & 255]
        sum_cnt += popcnt[packed_intensity[t]]
        prv = packed_intensity[t] & 1
    start_cnt += prv
    return start_cnt, sum_cnt


def get_df_edges_and_freq_tables(
    dict_freq_tables: dict[Tuple, npt.ArrayLike],
) -> pd.DataFrame:
    df_edges = pd.DataFrame(
        list(dict_freq_tables.items()),
        index=list(dict_freq_tables.keys()),
        columns=["edge", "freq_table"],
    )
    return df_edges


def get_chi_square_results_df(
    df_edges: pd.DataFrame, bool_gtest: bool = False
) -> pd.DataFrame:
    if bool_gtest:
        lambda_ = "log-likelihood"
        test_name = "gtest"
    else:
        lambda_ = "pearson"
        test_name = "chi2"
    df_chi_res = pd.DataFrame(
        columns=[test_name + "_pval", test_name + "_expected_freq"],
        index=df_edges.index,
        # dtype='float64'
    )

    freq_tables = np.stack(df_edges["freq_table"].values)
    pvalues, _, expected_freq = chi2_contingency_vectorized(
        freq_tables, lambda_=lambda_
    )
    df_chi_res[test_name + "_pval"] = pvalues
    df_chi_res[test_name + "_expected_freq"] = [x for x in expected_freq]

    df_chi_res[test_name + "_pval"] = df_chi_res[test_name + "_pval"].astype("float64")
    df_chi_res = add_adjusted_pvalue_df(df_chi_res, test_name + "_pval")
    return df_chi_res


def add_adjusted_pvalue_df(
    df_res: pd.DataFrame, pval_column: str, method: str = "hs"
) -> pd.DataFrame:
    if len(df_res) > 0:
        df_res[pval_column + "_adj"] = multipletests(
            df_res[pval_column].values, method=method
        )[1]
    else:
        df_res[pval_column + "_adj"] = []
    return df_res
