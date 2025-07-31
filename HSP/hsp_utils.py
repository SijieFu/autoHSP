"""
utility functions for helping `hsp_solver.py` to solve the HSP space
"""

import functools
from typing import Any, Callable, Literal, TypeVar, ParamSpec

import numpy as np
from scipy.spatial import distance_matrix
from scipy.special import gamma

from .info import info

P = ParamSpec("P")
R = TypeVar("R")


def _hash_numpy_ndarray_input(func: Callable[P, R]) -> Callable[P, R]:
    """
    hash the numpy ndarray input to the decorated function for caching purposes.
    """

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args = tuple(
            tuple(map(tuple, arg)) if isinstance(arg, np.ndarray) else arg
            for arg in args
        )
        kwargs = {
            k: tuple(map(tuple, v)) if isinstance(v, np.ndarray) else v
            for k, v in kwargs.items()
        }
        return func(*args, **kwargs)

    return wrapper


@_hash_numpy_ndarray_input
@functools.lru_cache
def _get_possible_solvents(
    nsolvents: int,
    max_solvent: int = 1,
    mix_matrix: tuple | np.ndarray | None = None,
    mix_step_size: float = 0.1,
    HSPs: tuple | np.ndarray | None = None,
    tested_HSPs: tuple | np.ndarray | None = None,
    precision: float = 0.1,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    get all possible solvents that can be formed by mixing pure solvents.
    (This function is cached to improve performance and may have issues with type hints/docstrings
    because of `functools.lru_cache` limitations. Use `get_possible_solvents` instead for type hints.)

    :param int nsolvents:
        the number of pure solvents, 1 <= nsolvents <= 300
    :param int max_solvent:
        the maximum number of pure solvents to use to form solvents (only 1 and 2 are supported)
    :param tuple | np.ndarray | None  mix_matrix:
        the matrix of compatible solvents.
        The matrix is a boolean matrix where the (i, j) element is True if the i-th and j-th solvents are compatible.
        If None, all solvents are considered compatible
    :param float mix_step_size:
        the step size to mix pure solvents when making a new solvent, 0.05 <= mix_step_size <= 0.5
    :param tuple | np.ndarray | None HSPs:
        the HSP values of the pure solvents.
        If provided, the HSP values of the new solvents will be calculated and returned

        shape: (nsolvents, HSP_LENGTH)
    :param tuple | np.ndarray | None tested_HSPs:
        the HSP values of the tested solvents. Only used when `HSPs` is provided.
        If provided, any new solvent that has HSP values that are within `precision` of any tested solvent will be ignored

        shape: (ntested, HSP_LENGTH)
    :param float precision:
        the precision to compare the HSP values of the new solvents with the tested solvents
    :return np.ndarray | tuple[np.ndarray, np.ndarray]:
        the possible solvents that can be formed by mixing pure solvents

        - shape: (N, nsolvents); N is the number of possible solvents or solvent mixtures
        - if `HSPs` is provided, the HSP values of the possible solvents will also be returned as a second output
    """
    if HSPs is not None:
        HSPs = np.asarray(HSPs, dtype=np.float64)
        assert HSPs.shape == (nsolvents, info.HSP_LENGTH), "incorrect shape of HSPs"

    # when only 1 solvent is used to form a new solvent
    possible_solvents = np.identity(nsolvents, dtype=np.float64)

    if max_solvent < 2:
        if HSPs is None:
            return possible_solvents
        return possible_solvents, HSPs

    # when 2 solvents are used to form a new solvent
    if nsolvents > 300 or nsolvents < 1:  # to avoid memory error
        raise ValueError("`nsolvents` must be less than or equal to 300")

    if mix_matrix is None:
        mix_matrix = np.ones((nsolvents, nsolvents), dtype=bool)
    else:
        mix_matrix = np.asarray(mix_matrix, dtype=bool)
    np.fill_diagonal(mix_matrix, False)
    assert mix_matrix.shape == (nsolvents, nsolvents), "incorrect shape of mix_matrix"
    assert 0.05 <= mix_step_size <= 0.5, "incorrect mix_step_size"

    mix_matrix = np.triu(mix_matrix, k=1)
    pairs = np.argwhere(mix_matrix)

    portions1 = np.arange(mix_step_size, 1.0, mix_step_size)
    portions2 = 1.0 - portions1
    portions = np.stack((portions1, portions2), axis=-1)
    npairs, nportions = len(pairs), len(portions1)
    paired_solvents = np.zeros((npairs, nportions, nsolvents), dtype=np.float64)
    paired_solvents[
        np.arange(npairs).reshape(-1, 1, 1),
        np.arange(nportions).reshape(1, -1, 1),
        pairs.reshape(-1, 1, 2),
    ] = portions.reshape(1, -1, 2)
    paired_solvents = paired_solvents.reshape(-1, nsolvents)

    possible_solvents = np.concatenate((possible_solvents, paired_solvents), axis=0)

    if HSPs is None:
        return possible_solvents

    possible_HSPs = np.dot(possible_solvents, HSPs)

    if tested_HSPs is None:
        return possible_solvents, possible_HSPs

    tested_HSPs = np.array(tested_HSPs, dtype=np.float64).reshape(-1, info.HSP_LENGTH)
    if tested_HSPs.size == 0:
        return possible_solvents, possible_HSPs

    worthy_indices = (
        calc_comb_HSP_distances(tested_HSPs, possible_HSPs).min(axis=1) >= precision
    )
    possible_solvents = possible_solvents[worthy_indices]
    possible_HSPs = possible_HSPs[worthy_indices]

    return possible_solvents, possible_HSPs


def get_possible_solvents(
    nsolvents: int,
    max_solvent: int = 1,
    mix_matrix: tuple | np.ndarray | None = None,
    mix_step_size: float = 0.1,
    HSPs: tuple | np.ndarray | None = None,
    tested_HSPs: tuple | np.ndarray | None = None,
    precision: float = 0.1,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    get all possible solvents that can be formed by mixing pure solvents.
    (This function is a type-hinted wrapper for `_get_possible_solvents` since `functools.lru_cache` does not support type hints well)

    :param int nsolvents:
        the number of pure solvents, 1 <= nsolvents <= 300
    :param int max_solvent:
        the maximum number of pure solvents to use to form solvents (only 1 and 2 are supported)
    :param tuple | np.ndarray | None  mix_matrix:
        the matrix of compatible solvents.
        The matrix is a boolean matrix where the (i, j) element is True if the i-th and j-th solvents are compatible.
        If None, all solvents are considered compatible
    :param float mix_step_size:
        the step size to mix pure solvents when making a new solvent, 0.05 <= mix_step_size <= 0.5
    :param tuple | np.ndarray | None HSPs:
        the HSP values of the pure solvents.
        If provided, the HSP values of the new solvents will be calculated and returned

        shape: (nsolvents, HSP_LENGTH)
    :param tuple | np.ndarray | None tested_HSPs:
        the HSP values of the tested solvents. Only used when `HSPs` is provided.
        If provided, any new solvent that has HSP values that are within `precision` of any tested solvent will be ignored

        shape: (ntested, HSP_LENGTH)
    :param float precision:
        the precision to compare the HSP values of the new solvents with the tested solvents
    :return np.ndarray | tuple[np.ndarray, np.ndarray]:
        the possible solvents that can be formed by mixing pure solvents

        - shape: (N, nsolvents); N is the number of possible solvents or solvent mixtures
        - if `HSPs` is provided, the HSP values of the possible solvents will also be returned as a second output
    """
    return _get_possible_solvents(
        nsolvents,
        max_solvent,
        mix_matrix,
        mix_step_size,
        HSPs,
        tested_HSPs,
        precision,
    )


def parse_solvent_HSP(
    solvents: np.ndarray | tuple, HSPs: np.ndarray | tuple
) -> np.ndarray:
    """
    parse the HSP values of the solvents, for example, from the output of `get_possible_solvents`

    :param np.ndarray | tuple solvents:
        the conformations/mixing matrix of the solvents (e.g., the output of `get_possible_solvents`)

        - shape: (N, nsolvents); N is the number of conformations, nsolvents is the number of pure solvents
    :param np.ndarray | tuple HSPs:
        the HSP values of the solvents

        - shape: (nsolvents, HSP_LENGTH); nsolvents is the number of pure solvents
    :return np.ndarray (N, HSP_LENGTH):
        the HSP values of the solvents
    """
    solvents = np.asarray(solvents, dtype=np.float64)
    assert solvents.ndim == 2, "solvents must be a 2D array"
    # normalize the solvents
    solvents = solvents / solvents.sum(axis=1, keepdims=True)

    HSPs = np.asarray(HSPs, dtype=np.float64)
    assert HSPs.shape == (solvents.shape[1], info.HSP_LENGTH), "incorrect shape of HSPs"

    # each solvent HSP is a weighted sum of the pure solvent HSPs
    return np.dot(solvents, HSPs)


def calc_comb_HSP_distances(
    source_HSPs: np.ndarray | Any,
    target_HSPs: np.ndarray | Any,
    threshold: int = 3e7,
    agg: None | str | Literal["mean", "min", "max", "sum"] = None,
    axis: Literal[0, 1] | None = 0,
) -> np.ndarray | np.float64:
    """
    calculate the conbinational distances between the HSP values of two sets of solvents.

    NOTE: IN some cases, this implementation is slightly faster than `scipy.spatial.distance_matrix`
    (implemented by `calc_HSP_distance_matrix`)

    :param np.ndarray | Any source_HSPs:
        the HSP values of the first set of source solvents. Must be convertible to a 2D numpy array.

        - shape: (N1, HSP_LENGTH); N1 is the number of solvents
    :param np.ndarray | Any target_HSPs:
        the HSP values of the second set of target solvents. Must be convertible to a 2D numpy array.

        - shape: (N2, HSP_LENGTH); N2 is the number of solvents
    :param int threshold:
        the threshold to calculate the distance,
        when `N1 * N2 * HSP_LENGTH` is greater than the threshold, the calculation will be handled over to `calc_HSP_distance_matrix`
    :param None | str | Literal["mean", "min", "max", "sum"] agg:
        after calculating the distance matrix, whether to aggregate the distances along the specified axis.
        The distance matrix will be a 2D numpy array of shape (N2, N1), so `aggregation` must be
        a valid numpy aggregation function name or None (no aggregation will be performed).
    :param Literal[0, 1] | None axis:
        the axis along which to aggregate the distances if `aggregation` is not None.

        - 0 means to aggregate along the rows. The output will be a 1D array of shape (N1).
        - 1 means to aggregate along the columns. The output will be a 1D array of shape (N2).
        - None means the aggregation should be done over the entire matrix, resulting in a single float value.
    :return np.ndarray | np.float64:
        the distances between the HSP values of the two sets of solvents.

        - shape: (N2, N1), where the (i, j) element is the distance between
        the i-th solvent in target_HSPs and the j-th solvent in source_HSPs
        - or if `agg` is not None, a 1D array of shape (N1) or (N2) depending on the `axis` parameter,
        or a single float value if `axis` is also None.
    """
    source_HSPs = np.asarray(source_HSPs, dtype=np.float64).reshape(-1, info.HSP_LENGTH)
    target_HSPs = np.asarray(target_HSPs, dtype=np.float64).reshape(-1, info.HSP_LENGTH)
    n_source, n_target = len(source_HSPs), len(target_HSPs)

    if n_source * n_target * info.HSP_LENGTH > threshold:
        mat = calc_HSP_distance_matrix(source_HSPs, target_HSPs, threshold=threshold)
    elif n_source == 0 or n_target == 0:
        mat = np.zeros((n_target, n_source), dtype=np.float64)
    else:
        mat = np.linalg.norm(
            source_HSPs.reshape(1, -1, info.HSP_LENGTH)
            - target_HSPs.reshape(-1, 1, info.HSP_LENGTH),
            axis=-1,
        )

    if agg is None:
        return mat

    if not hasattr(mat, agg):
        raise ValueError(
            f"invalid aggregation function: {agg=}. Is it a valid numpy aggregation function?"
        )

    return getattr(mat, agg)(axis=axis)


def calc_HSP_distance_matrix(
    source_HSPs: np.ndarray | Any,
    target_HSPs: np.ndarray | Any,
    threshold: int = 3e7,
    agg: None | str | Literal["mean", "min", "max", "sum"] = None,
    axis: Literal[0, 1] | None = 0,
) -> np.ndarray | np.float64:
    """
    calculate the conbinational distances between the HSP values of two sets of solvents.

    NOTE: Implemented with `scipy.spatial.distance_matrix`. In some cases, it may be slower than `calc_comb_HSP_distances`.

    :param np.ndarray | Any source_HSPs:
        the HSP values of the first set of source solvents. Must be convertible to a 2D numpy array.

        - shape: (N1, HSP_LENGTH); N1 is the number of solvents
    :param np.ndarray | Any target_HSPs:
        the HSP values of the second set of target solvents. Must be convertible to a 2D numpy array.

        - shape: (N2, HSP_LENGTH); N2 is the number of solvents
    :param int threshold:
        the threshold to calculate the distance. When `N1 * N2 * HSP_LENGTH` is greater than the threshold,
        the calculation will be done in loops.
    :param None | str | Literal["mean", "min", "max", "sum"] agg:
        after calculating the distance matrix, whether to aggregate the distances along the specified axis.
        The distance matrix will be a 2D numpy array of shape (N2, N1), so `aggregation` must be
        a valid numpy aggregation function name or None (no aggregation will be performed).
    :param Literal[0, 1] | None axis:
        the axis along which to aggregate the distances if `aggregation` is not None.

        - 0 means to aggregate along the rows. The output will be a 1D array of shape (N1).
        - 1 means to aggregate along the columns. The output will be a 1D array of shape (N2).
        - None means the aggregation should be done over the entire matrix, resulting in a single float value.
    :return np.ndarray | np.float64:
        the distances between the HSP values of the two sets of solvents.

        - shape: (N2, N1), where the (i, j) element is the distance between
        the i-th solvent in target_HSPs and the j-th solvent in source_HSPs
        - or if `agg` is not None, a 1D array of shape (N1) or (N2) depending on the `axis` parameter,
        or a single float value if `axis` is also None.
    """
    source_HSPs = np.asarray(source_HSPs, dtype=np.float64).reshape(-1, info.HSP_LENGTH)
    target_HSPs = np.asarray(target_HSPs, dtype=np.float64).reshape(-1, info.HSP_LENGTH)
    n_source, n_target = len(source_HSPs), len(target_HSPs)
    if n_source == 0 or n_target == 0:
        mat = np.zeros((n_target, n_source), dtype=np.float64)
    else:
        mat = distance_matrix(target_HSPs, source_HSPs, threshold=threshold)

    if agg is None:
        return mat

    if not hasattr(mat, agg):
        raise ValueError(
            f"invalid aggregation function: {agg=}. Is it a valid numpy aggregation function?"
        )

    return getattr(mat, agg)(axis=axis)


def to_standard_HSP(HSPs: np.ndarray | None) -> np.ndarray:
    """
    transform a raw HSP matrix to a standard HSP matrix of (N, HSP_LENGTH)

    :param np.ndarray | None HSPs:
        the HSP values of the solvents

        - shape: (N, HSP_LENGTH); N is the number of solvents
    :return np.ndarray:
        the standard HSP values

        - shape: (N, HSP_LENGTH)
    """
    if HSPs is None or len(HSPs) == 0:
        return np.zeros((0, info.HSP_LENGTH), dtype=np.float64)

    HSPs = np.asarray(HSPs, dtype=np.float64).reshape(-1, info.HSP_LENGTH)
    return HSPs


def to_transformed_HSP(HSPs: np.ndarray | None) -> np.ndarray:
    """
    transform the HSP values to the transformed HSP space (double `dD`)

    :param np.ndarray | None HSPs:
        the HSP values of the solvents

        - shape: (N, HSP_LENGTH); N is the number of solvents
    :return np.ndarray:
        the transformed HSP values

        - shape: (N, HSP_LENGTH)
    """
    HSPs = to_standard_HSP(HSPs)
    if info.ENABLE_HSP_DISTANCE:
        HSPs[:, 0] *= 2.0
    return HSPs


def to_original_HSP(HSPs: np.ndarray | None) -> np.ndarray:
    """
    transform the HSP values to the original HSP space (half `dD`)

    :param np.ndarray | None HSPs:
        the HSP values of the solvents

        - shape: (N, HSP_LENGTH); N is the number of solvents
    :return np.ndarray:
        the original HSP values

        - shape: (N, HSP_LENGTH)
    """
    HSPs = to_standard_HSP(HSPs)
    if info.ENABLE_HSP_DISTANCE:
        HSPs[:, 0] *= 0.5
    return HSPs


def calc_hypersphere_volume(radius: float, dimension: int) -> float:
    """
    calculate the volume of a hypersphere given the radius and dimension

    :param float radius:
        the radius of the hypersphere
    :param int dimension:
        the dimension of the hypersphere
    :return float:
        the volume of the hypersphere
    """
    return np.pi ** (dimension / 2) / gamma(dimension / 2 + 1) * radius**dimension


def calc_hyper_radius(volume: float, dimension: int) -> float:
    """
    calculate the radius of a hypersphere given the volume and dimension

    :param float volume:
        the volume of the hypersphere
    :param int dimension:
        the dimension of the hypersphere
    :return float:
        the radius of the hypersphere
    """
    return (volume * gamma(dimension / 2 + 1) / np.pi ** (dimension / 2)) ** (
        1 / dimension
    )
