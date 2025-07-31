"""
solver for manuvering the HSP space
"""

import math
import logging
from typing import Literal
import random

import numpy as np
from sklearn.cluster import KMeans

from .info import info
from .hsp_math import is_good_for_exploit, get_in_hull_flags
from .hsp_math import get_convex_hull
from .hsp_utils import get_possible_solvents
from .hsp_utils import calc_comb_HSP_distances
from .hsp_utils import to_standard_HSP, to_transformed_HSP


def reached_EOE_or_not(
    compatible: np.ndarray,
    incompatible: np.ndarray,
    available: np.ndarray,
    max_solvent: int = 1,
    mix_matrix: np.ndarray = None,
    mix_step_size: float = 0.1,
    precision: float = 0.1,
    strategy: Literal["smart", "greedy"] = "smart",
    strictness: float = 0.8,
    random_seed: int | None = None,
) -> bool:
    """
    Check if the EOE (end of experiment) is reached

    :param np.ndarray compatible:
        the HSP values of compatible/miscible solvents
    :param np.ndarray incompatible:
        the HSP values of incompatible/immiscible solvents
    :param np.ndarray available:
        the HSP values of all available **pure solvents**
    :param int max_solvent:
        the maximum number of pure solvents to use to form solvents (only 1 and 2 are supported)
    :param np.ndarray mix_matrix:
        the matrix of compatible solvents.
        The matrix is a boolean matrix where the (i, j) element is True if the i-th and j-th solvents are compatible.
        If None, all solvents are considered compatible
    :param float mix_step_size:
        the step size to mix pure solvents when making a new solvent
    :param float precision:
        the precision of the HSP space.
        For example, when a solvent is found to be miscible, any other solvent within the precision will not be tested
    :param Literal["smart", "greedy"] strategy:
        the strategy to propose new solvents. Default is "smart" - enable customized EOE check,

        - "smart": use custimized strategy to check if the EOE is reached
        - "greedy": use the greedy strategy: if there are solvents not tested, propose them. In other
        words, EOE is not reached if there are solvents to test.
    :param float strictness:
        when the `strategy` is "smart", this parameter controls how strict the EOE check is.
        Should be a float value between 0 and 1. The higher the value, the more strict the check is.
    :param int | None random_seed:
        the random seed for any random operations
    :return bool:
        if the EOE is reached
    """
    compatible_HSPs = to_transformed_HSP(compatible)
    incompatible_HSPs = to_transformed_HSP(incompatible)
    available_HSPs = to_transformed_HSP(available)
    assert len(available_HSPs) > 0, "no available solvents"
    nsolvents = available_HSPs.shape[0]

    tested_HSPs = np.vstack((compatible_HSPs, incompatible_HSPs))
    if len(tested_HSPs) == 0:
        if info.DEBUG:
            logging.warning("no solvents tested yet - EOE is not reached")
        return False

    possible_solvents, possible_HSPs = get_possible_solvents(
        nsolvents=nsolvents,
        max_solvent=max_solvent,
        mix_matrix=mix_matrix,
        mix_step_size=mix_step_size,
        HSPs=available_HSPs,
        tested_HSPs=tested_HSPs,
        precision=precision,
    )

    # exclude solvents in the convex hull of the compatible solvents
    if len(compatible_HSPs) >= 2:
        _in_hull_flags = get_in_hull_flags(possible_HSPs, compatible_HSPs)
        _out_hull_flags = np.logical_not(_in_hull_flags)
        possible_solvents = possible_solvents[_out_hull_flags]
        possible_HSPs = possible_HSPs[_out_hull_flags]
    npossible = len(possible_solvents)

    if npossible == 0:
        if info.DEBUG:
            logging.warning("no solvents to test - EOE is reached")
        return True
    elif info.DEBUG:
        logging.warning(
            f"missed EOE criterion - there are {npossible} solvents to test, not empty"
        )

    if strategy == "greedy":
        if info.DEBUG:
            logging.warning("greedy strategy - EOE is not reached")
        return False

    strictness = float(np.clip(strictness, 0, 1))
    try:
        total_hull = get_convex_hull(available_HSPs)
    except:
        # exception should be triggered when `available_HSPs` is too small to form a convex hull
        _dist = calc_comb_HSP_distances(
            available_HSPs, tested_HSPs, agg="min", axis=None
        )
        if np.all(_dist <= precision * (2 - strictness)):
            if info.DEBUG:
                logging.warning(
                    "all available solvents (N<=4) are tested - EOE is reached"
                )
            return True  # not enough solvents to test so EOE is reached
        else:  # not enough solvents tested but still have solvents to test
            if info.DEBUG:
                logging.warning("not enough solvents tested - EOE is not reached")
            return False

    # now there are solvents to test, "smart"-ly check if the EOE is reached
    # this will be similar the exploitative strategy in `propose_new_solvents`
    good_for_exploit, pseudo_comp, pseudo_incomp = is_good_for_exploit(
        compatible=compatible_HSPs,
        incompatible=incompatible_HSPs,
        pending=to_transformed_HSP(None),
        available=available_HSPs,
        random_state=random_seed,
    )
    if not good_for_exploit:  # need to explore more
        if info.DEBUG:
            logging.warning(
                "not good for exploitation, need to explore more - EOE is not reached"
            )
        return False

    # exploitative strategy passed
    strictness_dimension = min(0.9, 0.5 + strictness * 2)
    try:
        comp_hull = get_convex_hull(pseudo_comp)
        comp_hull_ratio = comp_hull.volume / total_hull.volume
        _cutoff = strictness_dimension**info.HSP_LENGTH
        if comp_hull_ratio >= _cutoff:
            # criteria: compatible hull has been almost fully explored
            if info.DEBUG:
                logging.warning(
                    f"the HSP space is almost fully explored ({comp_hull_ratio}>={_cutoff}, A) - EOE is reached"
                )
            return True
        elif info.DEBUG:
            logging.warning(
                f"missed EOE criterion: compatible hull ratio ({comp_hull_ratio}<{_cutoff}, A), "
                f"the compatible hull is not large enough"
            )

        incomp_hull = get_convex_hull(pseudo_incomp)
    except:
        # failed to get the convex hulls (not enough solvents to form a hull)
        if info.DEBUG:
            logging.warning(
                "not enough solvents to form a convex hull - EOE is not reached"
            )
        return False

    comp_hull_ratio = comp_hull.volume / max(
        info.HSP_CUTOFF_SPHERE_VOLUME, incomp_hull.volume
    )
    _cutoff = strictness_dimension**info.HSP_LENGTH
    if comp_hull_ratio >= _cutoff:
        # criteria: compatible hull is reasonably large
        if info.DEBUG:
            logging.warning(
                f"the HSP space is almost fully explored ({comp_hull_ratio}>={_cutoff}, B) - EOE is reached"
            )
        return True
    elif info.DEBUG:
        logging.warning(
            f"missed EOE criterion: compatible hull ratio ({comp_hull_ratio}<{_cutoff}, B), "
            f"the HSP space is not adequately explored"
        )

    # note that all points in `pseudo_comp` shoud be in the convex hull of `pseudo_incomp`
    # possible solvents in `pseudo_incomp` but not in `pseudo_comp` (in between the two)
    in_between_flags = np.logical_and(
        get_in_hull_flags(possible_HSPs, pseudo_incomp),
        np.logical_not(get_in_hull_flags(possible_HSPs, pseudo_comp)),
    )
    in_between_indices = np.where(in_between_flags)[0]
    if len(in_between_indices) == 0:
        # exploit strategy kicked in but no solvents in between
        if info.DEBUG:
            logging.warning(
                "exploit strategy kicked in but no solvents in-between - EOE is reached"
            )
        return True
    elif info.DEBUG:
        logging.warning(
            f"missed EOE criterion: exploit strategy kicked in but there are still "
            f"{len(in_between_indices)} exploitable solvents to pick from, not empty"
        )

    in_between_to_comp_distances = calc_comb_HSP_distances(
        possible_HSPs[in_between_indices], tested_HSPs, agg="min", axis=0
    )
    _in_between_distance = np.percentile(in_between_to_comp_distances, 50)
    _cutoff_upper = np.percentile(available_HSPs, 20 + 10 * strictness, axis=0)
    _cutoff_lower = np.percentile(available_HSPs, 80 - 10 * strictness, axis=0)
    _cutoff = (_cutoff_upper - _cutoff_lower) * (1 - strictness / 4)
    _cutoff = np.linalg.norm(_cutoff / (3 + 2 * strictness))
    if _in_between_distance <= _cutoff:
        # criteria: the in-between strip is narrow
        if info.DEBUG:
            logging.warning(
                f"the in-between strip is narrow ({_in_between_distance}<={_cutoff}) - EOE is reached"
            )
        return True
    elif info.DEBUG:
        logging.warning(
            f"missed EOE criterion: the in-between strip is wide ({_in_between_distance}>{_cutoff})"
        )

    if info.DEBUG:
        logging.warning("no EOE criterion met, all criteria failed - EOE is not reached")
    return False


def propose_new_solvents(
    compatible: np.ndarray | None,
    incompatible: np.ndarray | None,
    pending: np.ndarray | None,
    available: np.ndarray,
    target_n_tasks: int,
    max_solvent: int = 1,
    mix_matrix: np.ndarray = None,
    mix_step_size: float = 0.1,
    precision: float = 0.1,
    random_seed: int | None = None,
    no_exploit: bool = False,
    explore_temp: float = 0.5,
    distance_percentile: float = 0.34,
) -> tuple[np.ndarray, str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    propose new solvents to test

    :param np.ndarray compatible:
        the HSP values of compatible/miscible solvents
    :param np.ndarray incompatible:
        the HSP values of incompatible/immiscible solvents
    :param np.ndarray pending:
        the HSP values of pending solvents (solvents that are proposed but not tested yet)
    :param np.ndarray available:
        the HSP values of all available **pure solvents**
    :param int target_n_tasks:
        the target number of tasks to propose
    :param int max_solvent:
        the maximum number of pure solvents to use to form solvents (only 1 and 2 are supported)
    :param np.ndarray mix_matrix:
        the matrix of compatible solvents.
        The matrix is a boolean matrix where the (i, j) element is True if the i-th and j-th solvents are compatible.
        If None, all solvents are considered compatible
    :param float mix_step_size:
        the step size to mix pure solvents when making a new solvent
    :param float precision:
        the precision of the HSP space.
        For example, when a solvent is found to be miscible, any other solvent within the precision will not be tested
    :param int | None random_seed:
        the random seed for any random operations
    :param bool no_exploit:
        if True, do not use the exploitative strategy
    :param float explore_temp:
        the temperature for the explore+ strategy. Should be a float value between 0 and 1.

        In the `explore+` strategy, unlabeled solvents are tentatively guessed to be compatible or incompatible.
        The `explore_temp` parameter controls how many of the *compatible* should be selected. The higher the value,
        the fewer compatible solvents are selected, i.e., the more exploration on the incompatible space is done.


        - If `explore_temp` is 0, there is a balanced exploration of both (tentatively) compatible and incompatible solvents.
        - If `explore_temp` is 1, only the (tentatively) incompatible solvents.
    :param float distance_percentile:
        for conditioned clustering/clustering with existing centroids (tested solvents), each possible solvent is calculated
        its closest distance to the existing centroids. The distances are ranked, and solvents with distances higher
        than the `distance_percentile` are selected for the clustering.

        - should be a float value between 0 and 1, instead of 0 to 100. Check `get_conditioned_n_centroids` for more details.
        - The higher the value, the most aggressive the selection is, i.e., only very distant/empirically uncertain solvents will be selected.
    :return np.ndarray:
        the composition of the proposed solvents, shape (<= target_n_tasks, n_solvents,).
        Each row is the composition of a proposed solvent with a sum of 1.0.
    :return str:
        the strategy used to propose the solvents
    :return tuple[np.ndarray, np.ndarray, np.ndarray]:
        more information on which solvents were proposed and selected

        - 1st array: all possible solvents that are available to test but not yet tested, shape (n_possible, n_solvents,).
        - 2nd array: a boolean mask of the refined possible solvents to select from, shape (n_possible,).
        - 3rd array: a boolean mask of the selected solvents to test, shape (n_possible,).
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    compatible_HSPs = to_transformed_HSP(compatible)
    incompatible_HSPs = to_transformed_HSP(incompatible)
    pending_HSPs = to_transformed_HSP(pending)
    available_HSPs = to_transformed_HSP(available)
    nsolvents = available_HSPs.shape[0]

    assert len(available_HSPs) > 0, "no available solvents given in `available`"
    assert (
        isinstance(target_n_tasks, int) and target_n_tasks > 0
    ), f"invalid `target_n_tasks`: {target_n_tasks}"

    tested_HSPs = np.vstack((compatible_HSPs, incompatible_HSPs))
    generated_HSPs = np.vstack((tested_HSPs, pending_HSPs))

    possible_solvents, possible_HSPs = get_possible_solvents(
        nsolvents=nsolvents,
        max_solvent=max_solvent,
        mix_matrix=mix_matrix,
        mix_step_size=mix_step_size,
        HSPs=available_HSPs,
        tested_HSPs=generated_HSPs,
        precision=precision,
    )
    npossible = len(possible_solvents)
    if npossible <= target_n_tasks:
        possible_subset_mask = np.ones(npossible, dtype=bool)
        selected_subset_mask = np.ones(npossible, dtype=bool)
        return (
            possible_solvents,
            "explore",
            (possible_solvents, possible_subset_mask, selected_subset_mask),
        )

    # if no solvents are proposed, `explore` the HSP space
    if len(generated_HSPs) == 0:
        solvent_indices = get_n_centroids(
            possible_HSPs,
            n_clusters=target_n_tasks,
            random_state=random_seed,
        )
        possible_subset_mask = np.ones(npossible, dtype=bool)
        selected_subset_mask = np.zeros(npossible, dtype=bool)
        selected_subset_mask[solvent_indices] = True
        return (
            possible_solvents[solvent_indices],
            "explore",
            (possible_solvents, possible_subset_mask, selected_subset_mask),
        )

    # check if there is enough results for an exploitative strategy
    good_for_exploit = False
    if not no_exploit:
        good_for_exploit, pseudo_comp, pseudo_incomp = is_good_for_exploit(
            compatible=compatible_HSPs,
            incompatible=incompatible_HSPs,
            pending=pending_HSPs,
            available=available_HSPs,
            random_state=random_seed,
        )
    if good_for_exploit:  # must have both compatible and incompatible solvents
        min_distance_to_comp = calc_comb_HSP_distances(
            possible_HSPs, compatible_HSPs, agg="min", axis=0
        )
        min_distance_to_incomp = calc_comb_HSP_distances(
            possible_HSPs, incompatible_HSPs, agg="min", axis=0
        )

        # note that all points in `pseudo_comp` shoud be in the convex hull of `pseudo_incomp`
        # possible solvents in the convex hull of `pseudo_incomp` --> all should be in the convex hull of `pseudo_comp`
        in_incomp_flags = get_in_hull_flags(possible_HSPs, pseudo_incomp)

        # possible solvents in the convex hull of `pseudo_comp`
        in_comp_flags = np.logical_and(
            get_in_hull_flags(possible_HSPs, pseudo_comp),
            min_distance_to_comp <= min_distance_to_incomp,
        )
        in_comp_indices = np.where(in_comp_flags)[0]

        # possible solvents in `pseudo_incomp` but not in `pseudo_comp` (in between the two)
        in_between_flags = np.logical_and(
            in_incomp_flags, np.logical_not(in_comp_flags)
        )
        possible_subset_indices = np.where(in_between_flags)[0]
        if len(possible_subset_indices) < target_n_tasks:
            # not enough solvents in between, add some from the compatible solvents
            possible_subset_indices = np.concatenate(
                (possible_subset_indices, in_comp_indices)
            )

        selected_indices, considered_indices = get_conditioned_n_centroids(
            HSPs=possible_HSPs[possible_subset_indices],
            fixed_centroids=generated_HSPs,
            cutoff_percentile=distance_percentile,
            n_clusters=target_n_tasks,
            random_state=random_seed,
        )

        considered_indices = possible_subset_indices[considered_indices]
        possible_subset_mask = np.zeros(npossible, dtype=bool)
        possible_subset_mask[considered_indices] = True
        selected_indices = possible_subset_indices[selected_indices]
        selected_subset_mask = np.zeros(npossible, dtype=bool)
        selected_subset_mask[selected_indices] = True
        return (
            possible_solvents[selected_indices],
            "exploit",
            (possible_solvents, possible_subset_mask, selected_subset_mask),
        )

    # not good for exploitation? stick with exploration!
    if len(compatible_HSPs) == 0 or len(incompatible_HSPs) == 0:
        # we have either compatible or incompatible solvents (and/or pending)
        # but not both compatible and incompatible solvents
        possible_subset_indices = np.arange(npossible)
    else:
        # currently, we have both compatible and incompatible solvents
        min_distance_to_comp = calc_comb_HSP_distances(
            possible_HSPs, compatible_HSPs, agg="min", axis=0
        )
        min_distance_to_incomp = calc_comb_HSP_distances(
            possible_HSPs, incompatible_HSPs, agg="min", axis=0
        )

        # solvents closer to the incompatible solvents are probably(?) incompatible too
        # so they should be put down in the priority list
        # in other words, we prioritize solvents that are closer to the compatible solvents
        _deduced_incomp = min_distance_to_incomp < min_distance_to_comp
        try:
            get_convex_hull(compatible_HSPs)
            _deduced_incomp |= ~get_in_hull_flags(possible_HSPs, compatible_HSPs)
        except:
            pass
        _deduced_incomp_indices = np.where(_deduced_incomp)[0]
        _deduced_comp_indices = np.where(~_deduced_incomp)[0]
        possible_subset_indices = np.array([], dtype=int)

        # since `_deducted_incomp_indices` could contain compatible solvents,
        # we should allow some of them to be in the subset with some probability
        # the closer they are to compatible solvents, the higher the probability
        # on the oppsite, since `_deducted_comp_indices` could contain incompatible solvents,
        # ..., but in this case, the closer they are to compatible solvents, the lower the prob.
        _comp_ratio = (1 - np.clip(explore_temp, 0, 1)) / 2
        for _closer_is_better, _indices, _ratio in zip(
            [True, False],
            [_deduced_incomp_indices, _deduced_comp_indices],
            [1 - _comp_ratio, _comp_ratio],
        ):
            _n_indices = len(_indices)
            _n_extra = math.ceil(_ratio * _n_indices)
            if _closer_is_better:
                _n_extra = max(target_n_tasks * 2, _n_extra)
            elif len(possible_subset_indices) < target_n_tasks:
                _n_extra = max(target_n_tasks * 2, _n_extra, _n_indices // 2)

            if _n_extra >= _n_indices:
                _extra_indices = _indices
            else:
                _distances: np.ndarray = min_distance_to_comp[_indices]
                _weights: np.ndarray = (
                    _distances - _distances.mean()
                ) / _distances.std()
                if _closer_is_better:  # larger distance, lower weight
                    _weights = -_weights
                _weights = _distances - _distances.min()
                _weights += _weights.max() * 0.2
                _weights = _weights / _weights.sum()
                # a random seed, if applicable, is already set above
                _extra_indices = np.random.choice(
                    _indices, size=_n_extra, replace=False, p=_weights
                )
            possible_subset_indices = np.concatenate(
                (possible_subset_indices, _extra_indices)
            )

    selected_indices, considered_indices = get_conditioned_n_centroids(
        HSPs=possible_HSPs[possible_subset_indices],
        fixed_centroids=generated_HSPs,
        cutoff_percentile=distance_percentile,
        n_clusters=target_n_tasks,
        random_state=random_seed,
    )

    considered_indices = possible_subset_indices[considered_indices]
    possible_subset_mask = np.zeros(npossible, dtype=bool)
    possible_subset_mask[considered_indices] = True
    selected_indices = possible_subset_indices[selected_indices]
    selected_subset_mask = np.zeros(npossible, dtype=bool)
    selected_subset_mask[selected_indices] = True
    return (
        possible_solvents[selected_indices],
        "explore+",
        (possible_solvents, possible_subset_mask, selected_subset_mask),
    )


def get_n_centroids(
    HSPs: np.ndarray,
    n_clusters: int | float = 0.5,
    random_state: int | None = None,
    **kwargs,
) -> np.ndarray:
    """
    get the n centroids of the HSP space from the HSP values of the solvents

    :param np.ndarray HSPs:
        the HSP values of the solvents
    :param int | float n_clusters:
        the number of centroids to get
        if int, the number of centroids to get
        if float, the proportion of the number of centroids to get
        NOTE: n_clusters will be at least 0 and at most the number of solvents
    :param int | None random_state:
        the random seed for the KMeans algorithm
    :param dict kwargs:
        the keyword arguments for the KMeans algorithm
    :return np.ndarray:
        the indices of the n centroids in `HSPs`
    """
    HSPs = to_standard_HSP(HSPs)
    if isinstance(n_clusters, float):
        n_clusters = min(1, max(0, n_clusters))
        n_clusters = math.ceil(n_clusters * HSPs.shape[0])
    assert isinstance(n_clusters, int) and n_clusters >= 0, "invalid `n_clusters`"
    if n_clusters == 0:
        return np.array([])
    if HSPs.shape[0] <= n_clusters:  # not enough solvents
        return np.arange(HSPs.shape[0])

    # by default: init="k-means++", n_init="auto", max_iter=300
    # kwargs.setdefault("random_state", 0)
    kwargs.setdefault("n_init", 25)
    kwargs.setdefault("max_iter", 100000)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)
    kmeans.fit(HSPs)

    # get the closest solvents to each of the centroids
    cluster_centers = kmeans.cluster_centers_
    indices = calc_comb_HSP_distances(HSPs, cluster_centers, agg="argmin", axis=1)
    # remove the duplicates in the indices, although the KMeans algorithm should not result in duplicates
    indices = np.unique(indices)
    return indices


def get_conditioned_n_centroids(
    HSPs: np.ndarray,
    fixed_centroids: np.ndarray | None = None,
    cutoff_percentile: float = 0.34,
    n_clusters: int | float = 0.5,
    random_state: int | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    the purpose of `get_n_centroids` is to get the n centroids of the HSP space from the HSP values of the solvents

    the addition of `get_conditioned_n_centroids` is to steer the centroids away from the fixed centroids if specified

    :param np.ndarray HSPs:
        the HSP values of the solvents. Each row is a solvent, each column is a HSP value.
    :param np.ndarray | None fixed_centroids:
        the HSP values of the fixed centroids. They are the "conditions" for getting the centroids.
        Any point close to these fixed centroids will be excluded from the final clustering.
        Check `cutoff_percentile` for more details.

        - if None, this function is equivalent to `get_n_centroids`
        - NOTE: any provided `fixed_centroids` will be removed from the `HSPs` (precision: 1e-3)
    :param float cutoff_percentile:
        for the input `HSPs`, each solvent is calculated its closest distance to the `fixed_centroids`, which are
        from solvents that were already tested.  The distances are ranked, and solvents with distances higher
        than the `distance_percentile` are selected for the clustering.
        In other words, the closest `cutoff_percentile` solvents to the fixed centroids will be removed.

        - should be a float value between 0 and 1, instead of 0 to 100.
        - The higher the value, the most aggressive the selection is, i.e., only very distant/empirically uncertain solvents will be selected.
    :param int | float n_clusters:
        the number of centroids to get

        - if int, the number of centroids to get
        - if float, the proportion of the number of centroids to get
        - NOTE: n_clusters will be at least 0 and at most the number of solvents
    :param int | None random_state:
        the random seed for the KMeans algorithm
    :param dict kwargs:
        the keyword arguments for the KMeans algorithm
    :return np.ndarray:
        the indices of the n centroids in `HSPs`
    :return np.ndarray:
        the indices of the solvents in `HSPs` that were considered to get the centroids,
        i.e., indices for a refined subset of `HSPs` that were used to get the centroids
    """
    HSPs = to_standard_HSP(HSPs)
    fixed_HSPs = to_standard_HSP(fixed_centroids)
    if fixed_centroids is None or fixed_HSPs.size == 0:
        return get_n_centroids(HSPs, n_clusters, random_state, **kwargs), np.arange(
            HSPs.shape[0]
        )

    assert len(fixed_HSPs) > 0, "no fixed centroids given in `fixed_centroids`"

    # remove the fixed centroids from the HSPs
    comb_min_distances = calc_comb_HSP_distances(HSPs, fixed_HSPs, agg="min", axis=0)
    valid_indices = comb_min_distances >= 1e-3
    valid_indices = np.where(valid_indices)[0]
    if len(valid_indices) <= 1:
        return valid_indices, valid_indices

    HSPs = HSPs[valid_indices]
    comb_min_distances = comb_min_distances[valid_indices]
    if isinstance(n_clusters, float):
        n_clusters = np.clip(n_clusters, 0, 1)
        n_clusters = math.ceil(n_clusters * HSPs.shape[0])
    assert isinstance(n_clusters, int) and n_clusters >= 0, "invalid `n_clusters`"
    if n_clusters == 0:
        return np.array([]), valid_indices
    if HSPs.shape[0] <= n_clusters:  # not enough solvents
        return valid_indices, valid_indices

    # next, filter out the solvents that are too close to the fixed centroids
    # and then perform the KMeans algorithm on the remaining solvents
    n_points_avail = len(valid_indices)
    cutoff_percentile = np.clip(cutoff_percentile, 0, 1)
    n_points_to_remove = math.ceil(cutoff_percentile * n_points_avail)
    if n_points_avail - n_points_to_remove < n_clusters * 2:
        # not enough points to cluster, so just return all points
        n_points_to_remove = 0
    if n_points_to_remove == 0:
        indices_to_cluster = np.arange(n_points_avail)
    else:
        argsorted = np.argpartition(comb_min_distances, n_points_to_remove)
        indices_to_cluster = argsorted[n_points_to_remove:]

    indices = get_n_centroids(
        HSPs[indices_to_cluster], n_clusters, random_state, **kwargs
    )
    return valid_indices[indices_to_cluster[indices]], valid_indices[indices_to_cluster]
