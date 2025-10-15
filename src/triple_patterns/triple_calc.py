import numpy as np
import numpy.typing as npt
from numba import jit
from typing import Callable, Tuple


@jit(nogil=True, nopython=True)
def nonzero_intersect_packed_110(
    data1: npt.ArrayLike,
    data2: npt.ArrayLike,
    data3: npt.ArrayLike,
    ids1: npt.ArrayLike,
    ids2: npt.ArrayLike,
    output: npt.ArrayLike = None,
) -> bool:
    """
    Check if there is any with condition:
    (data1 == 1) & (data2 == 1) & (data3 == 0)

    Parameters:
        data1, data2, data3 : (nt, ) bit packed arrays of uint8
        ids1 , ids2         : (nt, ) indicies of nonzero items in corresponding packed arrays

    Returns:
        ok : bool
    """
    nt = len(data1)
    ok = False
    k1 = k2 = 0
    while ids1[k1] < nt and ids2[k2] < nt:
        if ids1[k1] < ids2[k2]:
            k1 += 1
        if ids1[k1] > ids2[k2]:
            k2 += 1
        if ids1[k1] == ids2[k2] and ids1[k1] < nt:
            t = ids1[k1]
            k1 += 1
            k2 += 1
            cur = (data1[t] & data2[t]) & ~data3[t]
            ok |= cur != 0
            if not output is None:
                output[t] = cur
    return ok


def find_triples(
    packed_graph: npt.ArrayLike, is_good_triple: Callable = nonzero_intersect_packed_110
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Search for triples that satisfy to predicate `is_good_triple`

    Parameters:
        packed_graph     : (nt, n, n) array of uint8
        is_good_triple   : predicate definition specified in `find_triples_packed`
                           default is 110:
                           two edges exist at the same moment and the third edge is absent

    Returns:
        triples          : (s, 3) array of `s` triples
        triples_activity : (s, nt) array of triple activity in time
    """
    packed_graph_ids = make_packed_graph_ids(packed_graph)
    triples, triples_activity = find_triples_packed(
        packed_graph, packed_graph_ids, is_good_triple
    )
    return np.array(triples), np.array(triples_activity)


@jit(nopython=True, nogil=True)
def make_packed_graph_ids(
    packed_graph: npt.ArrayLike, dtype: npt.DTypeLike = np.uint16
) -> npt.ArrayLike:
    """
    Search for nonzero items in packed graph due to sparsity
    and return array with ids of nonzero items over time for each graph edge
    padded with `nt` values in the end.

    Parameters:
        packed_graph     : (nt, n, n) array of uint8
        dtype            : type of result array default np.uint16

    Returns:
        packed_graph_ids : (n, n, max_ids_count) array of `dtype` where
                           max_ids_count is the length of the longest
                           array of nonzero items over edges
    """
    nt, nc, nc = packed_graph.shape
    packed_ids_list = []
    max_len = 0
    for i in range(nc):
        res = []
        for j in range(nc):
            ids = [dtype(k) for k in range(nt) if packed_graph[k, i, j]]
            max_len = max(len(ids), max_len)
            res.append(ids)
        packed_ids_list.append(res)

    packed_graph_ids = np.full((nc, nc, max_len + 1), nt, dtype=dtype)
    for i in range(nc):
        for j in range(nc):
            cur = packed_ids_list[i][j]
            for k in range(len(cur)):
                packed_graph_ids[i, j, k] = cur[k]
    return packed_graph_ids


@jit(nogil=True, nopython=True)
def find_triples_packed(
    packed_graph: npt.ArrayLike,
    packed_graph_ids: npt.ArrayLike,
    is_good_triple: Callable,
) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
    """
    Search for triples confirmed by predicate function `is_good_triple`

    Parameters:
        packed_graph     : (nt, n, n) array of uint8
        packed_graph_ids : (n, n, m) array of counts of nonzero items in packed_graph
                        padded by nt values
        is_good_triple   : predicate takes 3 packed edges in time and nonzero ids

    Returns:
        triples          : (s, 3) array of triples
        triples_activity : (s, nt) array of triples activity in time
    """
    triples = []
    triples_activity = []
    edge_count = np.sum(packed_graph > 0, axis=0)
    print(packed_graph.shape)
    nt, nc, nc = packed_graph.shape
    res = np.zeros(nt, dtype=np.uint8)
    cnt = 0
    for i in range(nc):
        for j in range(nc):
            if j == i or edge_count[i, j] == 0:
                continue
            edges1 = packed_graph[:, i, j]
            for k in range(i + 1, nc):
                if k == j or edge_count[j, k] == 0:
                    continue
                cnt += 1
                edges2 = packed_graph[:, j, k]
                edges3 = packed_graph[:, i, k]
                ids1 = packed_graph_ids[i, j, : edge_count[i, j]]
                ids2 = packed_graph_ids[j, k, : edge_count[j, k]]
                if not is_good_triple(edges1, edges2, edges3, ids1, ids2):
                    continue
                triple = (i, j, k)
                res[:] = 0
                is_good_triple(edges1, edges2, edges3, ids1, ids2, res)
                triples.append(triple)
                triples_activity.append(res.copy())
                if len(triples) % 100000 == 0:
                    print(len(triples), cnt)
    print("Good triples:", len(triples))
    return triples, triples_activity


def get_triple_edges(
    triple_data: npt.ArrayLike,
    dict_label_intensity: dict,
    is_markup_packed: bool = True,
) -> npt.ArrayLike:
    if is_markup_packed:
        list_label_times = [value for value in dict_label_intensity.values()]
    elif not is_markup_packed:
        list_label_times = [
            np.packbits(value) for value in dict_label_intensity.values()
        ]
    n_triples = len(triple_data)
    n_labels = len(list_label_times)
    list_triple_info = np.zeros((n_triples, n_labels))
    for i in range(n_triples):
        list_dur = []
        for label in list_label_times:
            res_array = triple_data[i, :] & label
            t = np.sum(np.unpackbits(res_array[np.nonzero(res_array)]))
            list_dur.append(t)
        list_triple_info[i, :] = list_dur
    return list_triple_info
