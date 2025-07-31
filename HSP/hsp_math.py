"""
math functions for the HSP solver in `hsp_solver.py`
"""

import warnings
import logging
from typing import Any, Generator, Literal
import functools

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
from sklearn.svm import SVC

from .info import info
from .hsp_utils import _hash_numpy_ndarray_input
from .hsp_utils import calc_comb_HSP_distances
from .hsp_utils import to_standard_HSP, to_original_HSP, to_transformed_HSP
from .hsp_utils import calc_hyper_radius


@_hash_numpy_ndarray_input
@functools.lru_cache
def _is_good_for_exploit(
    compatible: np.ndarray,
    incompatible: np.ndarray,
    pending: np.ndarray,
    available: np.ndarray,
    random_state: int | None = None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    check if the current results are good for an exploitative solvent-proposing strategy.
    In general, `good for exploit` means current results provide a good sketch of the compatible/incompatible decision boundary.
    (This function is for caching and does have good docstrings or type hints. Use `is_good_for_exploit` instead.)

    :param np.ndarray compatible:
        the HSP values of compatible/miscible solvents
    :param np.ndarray incompatible:
        the HSP values of incompatible/immiscible solvents
    :param np.ndarray pending:
        the HSP values of pending solvents (solvents that are proposed but not tested yet)
    :param np.ndarray available:
        the HSP values of available **pure solvents**
    :param int | None random_state:
        the random state to use for reproducibility
    :return bool:
        True if the current results are good for an exploitative solvent-proposing strategy
    :return np.ndarray | None:
        the HSP values of pseudo-compatible/miscible solvents (None if not good for exploit)
    :return np.ndarray | None:
        the HSP values of pseudo-incompatible/immiscible solvents (None if not good for exploit)

        - The `pseudo` part means the HSPs are adjusted from the input such that all HSPs in the pseudo-compatible group are
        in the convex hull of the pseudo-incompatible group. In other words, you can find a perfect decision boundary between
        the pseudo-compatible and pseudo-incompatible groups with a convex hull.
    """
    compatible_HSPs = to_standard_HSP(compatible)
    incompatible_HSPs = to_standard_HSP(incompatible)
    pending_HSPs = to_standard_HSP(pending)
    available_HSPs = to_standard_HSP(available)
    n_comp, n_incomp = len(compatible_HSPs), len(incompatible_HSPs)

    # check if the available pure solvents are tested or pending
    _HSPs = np.concatenate([compatible_HSPs, incompatible_HSPs, pending_HSPs], axis=0)
    _HSPs = np.unique(_HSPs, axis=0)
    tested_flag = (
        calc_comb_HSP_distances(available_HSPs, _HSPs, agg="min", axis=0) <= 1e-3
    )
    available_HSPs = np.vstack(
        [compatible_HSPs, incompatible_HSPs, pending_HSPs, available_HSPs[~tested_flag]]
    )

    # criteria 1: not enough available pure solvents to exploit
    try:
        avail_hull = get_convex_hull(available_HSPs)
    except:
        if info.DEBUG:
            logging.warning(
                "Not enough available pure solvents to exploit --> not good for exploit"
            )
        return False, None, None

    # creteria 2: the number of compatible should be at least 4
    # and the number of incompatible should be at least 1
    try:
        assert n_comp >= 4 and n_incomp >= 1
        comp_hull = get_convex_hull(compatible_HSPs)
        comp_hull_radius = 1.2 * calc_hyper_radius(comp_hull.volume, info.HSP_LENGTH)
    except:
        if info.DEBUG:
            logging.warning(
                "The number of compatible solvents should be at least 4 "
                "and the number of incompatible solvents should be at least 1 "
                "--> not good for exploit"
            )
        return False, None, None

    pseudo_comp = compatible_HSPs
    comp_hull_vertices = comp_hull.points[comp_hull.vertices]
    _HSP_vertices = np.vstack(
        [avail_hull.points[avail_hull.vertices], comp_hull_vertices]
    )
    _HSP_vertices = np.unique(_HSP_vertices, axis=0)
    _HSP_vertices = _HSP_vertices[~get_in_hull_flags(_HSP_vertices, incompatible_HSPs)]
    try:
        _hull = get_convex_hull(_HSP_vertices)
        _HSP_vertices = _hull.points[_hull.vertices]
    except:
        pass

    pseudo_incomp = np.vstack([incompatible_HSPs, _HSP_vertices])
    if info.DEBUG and len(_HSP_vertices) > 0:
        logging.warning(
            f"add the vertices of the available hull and compatible hull to the pseudo-incompatible group: "
            f"{len(_HSP_vertices)} points added"
        )

    # to avoid wrong estimation of the decision in some cases
    # add the neighboring points of `vertices_from_comp` to the pseudo-incompatible group
    vertices_from_comp = comp_hull.vertices[
        calc_comb_HSP_distances(comp_hull_vertices, _HSP_vertices, agg="min", axis=0)
        <= 1e-3
    ]
    if len(vertices_from_comp) > 0:
        _contains = np.any(np.isin(comp_hull.simplices, vertices_from_comp), axis=1)
        _neighbors = np.unique(comp_hull.simplices[_contains])
        _neighbors = np.setdiff1d(_neighbors, vertices_from_comp, assume_unique=True)
        _neighbors_HSPs = comp_hull.points[_neighbors]
        _not_in_pseudo_incomp = (
            calc_comb_HSP_distances(_neighbors_HSPs, pseudo_incomp, agg="min", axis=0)
            > 1e-3
        )
        pseudo_incomp = np.vstack(
            [pseudo_incomp, _neighbors_HSPs[_not_in_pseudo_incomp]]
        )
        if info.DEBUG and len(_neighbors_HSPs[_not_in_pseudo_incomp]) > 0:
            logging.warning(
                f"add the neighboring points of `vertices_from_comp` to the pseudo-incompatible group: "
                f"{len(_neighbors_HSPs[_not_in_pseudo_incomp])} points added"
            )

    if not all(is_in_hull(comp_hull_vertices, pseudo_incomp)):
        # the pseudo-compatible group is not fully in the pseudo-incompatible hull
        # more exploration is needed to establish a good decision boundary and exploit
        if info.DEBUG:
            logging.warning(
                "pseudo-compatible group is not fully in the pseudo-incompatible hull --> not good for exploit"
            )
        return False, None, None

    # now, the pseudo-compatible group is fully in the pseudo-incompatible hull
    # let's try recursively removing points from the pseudo-incompatible group to refine the decision boundary
    _distances = calc_comb_HSP_distances(
        pseudo_incomp, compatible_HSPs, agg="min", axis=0
    )
    _argdistances = np.argsort(_distances)[::-1]
    _is_true_incomp = _argdistances < n_incomp
    _argdistances = np.concatenate(
        [_argdistances[~_is_true_incomp], _argdistances[_is_true_incomp]]
    )
    _argdistances = _argdistances[-n_incomp:]
    unremoveable_flags = np.ones(pseudo_incomp.shape[0], dtype=bool)
    _vertices = set(get_convex_hull(pseudo_incomp).vertices)

    def _print_debug(_i: int, _removed: bool):
        if info.DEBUG:
            logging.warning(
                f"point `{_i}` {'' if _removed else 'NOT '}removed from the pseudo-incompatible group: "
                f"(dD, dP, dH) = ({', '.join(str(round(float(c), 2)) for c in to_original_HSP(pseudo_incomp)[_i])})"
            )

    for _ind in _argdistances:
        _flags = unremoveable_flags.copy()
        _flags[_ind] = False
        _flags_indices = np.where(_flags)[0]
        if _ind >= n_incomp:
            # if point is pseudo-incomp (inferred incomp)
            if get_in_hull_flags(comp_hull_vertices, pseudo_incomp[_flags]).all():
                unremoveable_flags = _flags
            _print_debug(_ind, not unremoveable_flags[_ind])
            continue

        # skip when a true incompatible point is in the compatible hull
        if get_in_hull_flags(pseudo_incomp[_ind], compatible_HSPs)[0]:
            _print_debug(_ind, False)
            continue

        try:
            _new_vertices = (
                set(
                    _flags_indices[
                        get_convex_hull(pseudo_incomp[_flags_indices]).vertices
                    ]
                )
                - _vertices
            )
            if all(_ < n_incomp for _ in _new_vertices):
                # if removing the point does not add new vertices from the pseudo-incompatible group
                unremoveable_flags = _flags
        except:
            pass
        _print_debug(_ind, not unremoveable_flags[_ind])
    pseudo_incomp = pseudo_incomp[unremoveable_flags]

    # if all the available pure solvents are tested or pending, exploit!
    if np.all(tested_flag):
        if info.DEBUG:
            logging.warning(
                "All the available pure solvents are tested or pending --> exploit!"
            )
        return True, pseudo_comp, pseudo_incomp

    import miniball

    # smallest enclosing sphere of the compatible HSPs
    _, _radius = miniball.get_bounding_ball(
        compatible_HSPs[comp_hull.vertices], rng=np.random.default_rng(random_state)
    )
    _radius = (_radius + comp_hull_radius) / 2

    comp_hull_rev_ratio = avail_hull.volume / comp_hull.volume
    if (
        comp_hull_rev_ratio >= 1.6**info.HSP_LENGTH
        and _radius <= info.HSP_CUTOFF_SPHERE_RADIUS
    ):
        # the found compatible hull is too small --> need more compatible solvents
        # an exploitative strategy may be too biased and risk being stuck in a local minimum
        if info.DEBUG:
            logging.warning(
                "The found compatible hull is too small --> need more compatible solvents --> not good for exploit"
            )
        return False, None, None

    # # NOTE: the following code is not used because it seems a bit too aggressive in locating the decision boundary

    if info.DEBUG:
        logging.warning(
            "the HSP space has been reasonably explored --> good for exploit"
        )
    return True, pseudo_comp, pseudo_incomp


def is_good_for_exploit(
    compatible: np.ndarray,
    incompatible: np.ndarray,
    pending: np.ndarray,
    available: np.ndarray,
    random_state: int | None = None,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    """
    check if the current results are good for an exploitative solvent-proposing strategy
    In general, `good for exploit` means current results provide a good sketch of the compatible/incompatible decision boundary.

    :param np.ndarray compatible:
        the HSP values of compatible/miscible solvents
    :param np.ndarray incompatible:
        the HSP values of incompatible/immiscible solvents
    :param np.ndarray pending:
        the HSP values of pending solvents (solvents that are proposed but not tested yet)
    :param np.ndarray available:
        the HSP values of available **pure solvents**
    :param int | None random_state:
        the random state to use for reproducibility
    :return bool:
        True if the current results are good for an exploitative solvent-proposing strategy
    :return np.ndarray | None:
        the HSP values of pseudo-compatible/miscible solvents (None if not good for exploit)
    :return np.ndarray | None:
        the HSP values of pseudo-incompatible/immiscible solvents (None if not good for exploit)

        - The `pseudo` part means the HSPs are adjusted from the input such that all HSPs in the pseudo-compatible group are
        in the convex hull of the pseudo-incompatible group. In other words, you can find a perfect decision boundary between
        the pseudo-compatible and pseudo-incompatible groups with a convex hull.
    """

    return _is_good_for_exploit(
        compatible, incompatible, pending, available, random_state
    )


def is_in_hull(points: np.ndarray, hull: np.ndarray) -> Generator[bool, None, None]:
    """
    check if the points are in the hull.
    The dtype will be `np.float64` so make sure the points are not too large.
    Recommended if you have a large number of points to check or if you are checking if all points or any points are in the hull with `all` or `any`.

    Reference: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl

    :param np.ndarray, (n_points, n_dimensions) points:
        the points to be checked
    :param np.ndarray, (n_hull_points, n_dimensions) hull:
        the points to form the hull
    :return Generator[bool, None, None]:
        the boolean array indicating if the points are in the hull
    """
    points = np.atleast_2d(points).astype(np.float64)
    hull = np.atleast_2d(hull).astype(np.float64)

    assert (
        points.ndim == hull.ndim and points.shape[-1] == hull.shape[-1]
    ), "dimension mismatch between points and hull"

    # Ax = b, this is what `linprog` will try to solve
    n_points, n_hullpoints = len(points), len(hull)
    A = np.concatenate([hull, np.ones((n_hullpoints, 1))], axis=1).T
    B = np.concatenate([points, np.ones((n_points, 1))], axis=1)
    c = np.zeros(n_hullpoints)
    for b in B:
        if n_hullpoints == 0:
            yield False
        else:
            res = linprog(c, A_eq=A, b_eq=b, bounds=(0, 1))
            yield res.success


def get_in_hull_flags(points: np.ndarray, hull: np.ndarray) -> np.ndarray:
    """
    get the boolean array indicating if the points are in the hull by calling `is_in_hull`
    The dtype will be `np.float64` so make sure the points are not too large.

    Reference: https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl

    :param np.ndarray, (n_points, n_dimensions) points:
        the points to be checked
    :param np.ndarray, (n_hull_points, n_dimensions) hull:
        the points to form the hull
    :return np.ndarray (n_points,) dtype=bool:
        the boolean array indicating if the points are in the hull
    """
    return np.fromiter(is_in_hull(points, hull), dtype=bool)


@_hash_numpy_ndarray_input
@functools.lru_cache
def _get_convex_hull(points: np.ndarray | tuple, **kwargs: Any) -> ConvexHull:
    """
    get the convex hull of the points, with caching.
    (This function is for caching and does have good docstrings or type hints. Use `get_convex_hull` instead.)

    :param np.ndarray | tuple points:
        the points to form the hull
    :param Any kwargs:
        additional keyword arguments to pass to `scipy.spatial.ConvexHull`
    :return ConvexHull:
        the convex hull of the points
    """
    points = np.atleast_2d(points).astype(np.float64)
    return ConvexHull(points, **kwargs)


def get_convex_hull(points: np.ndarray | tuple, **kwargs: Any) -> ConvexHull:
    """
    get the convex hull of the points

    :param np.ndarray | tuple points:
        the points to form the hull
    :param Any kwargs:
        additional keyword arguments to pass to `scipy.spatial.ConvexHull`
    :return ConvexHull:
        the convex hull of the points
    """
    return _get_convex_hull(points, **kwargs)


def _calc_HSP_estimation_loss(
    x: np.ndarray,
    compatible_HSPs: np.ndarray,
    incompatible_HSPs: np.ndarray,
    initial_guess: np.ndarray | None = None,
    tol: float = 0.0,
    alpha: float = 0.1,
) -> float:
    """
    calculate the loss function for an estimated HSP sphere center and radius to
    separate the compatible and incompatible HSPs.

    WARNING: This is NOT for private use. Check `interpret_HSP` for the public interface.
    For example, this loss function assumes at least 4 compatible solvents and at least 1 incompatible solvent.

    :param np.ndarray, (HSP_LENGTH + 1,) x:
        the estimated HSP sphere center and radius
    :param np.ndarray, (n_comp, HSP_LENGTH) compatible_HSPs:
        the transformed HSP values of compatible solvents, (2*dD, dP, dH).

        - Note: `2*dD` only applies when `info.ENABLE_HSP_DISTANCE` is True.
    :param np.ndarray, (n_incomp, HSP_LENGTH) incompatible_HSPs:
        the transformed HSP values of incompatible solvents, (2*dD, dP, dH)

        - Note: `2*dD` only applies when `info.ENABLE_HSP_DISTANCE` is True.
    :param np.ndarray, (HSP_LENGTH + 1,) | None initial_guess:
        the initial guess from the compatible HSPs and the radius from the miniball.

        This initial guess is used to penalize the distance from the center to the compatible HSPs,
        if the center is too far from the compatible HSPs.

        If `None`, the initial guess will not be used and no penalty will be applied. The final estimation
        may drift away from the compatible HSPs.

        Related to the `alpha` parameter.
    :param float, optional tol:
        misclassification tolerance, similar to the epsilon in SVM
    :param float, optional alpha:
        the weight for penalizing distance from the center to the compatible HSPs.

        Related to the `initial_guess` parameter.
    :return float:
        the loss function value
    """
    loss = 0.0
    center, radius = x[:-1], x[-1]

    if radius < info.HSP_CUTOFF_SPHERE_RADIUS or tol < 0:
        return float("inf")

    comp_distances = calc_comb_HSP_distances(compatible_HSPs, center).flatten()
    incomp_distances = calc_comb_HSP_distances(incompatible_HSPs, center).flatten()

    # penalize the misclassification in compatible group
    incorrect_comp = comp_distances >= radius + tol
    loss += np.sum((comp_distances[incorrect_comp] - radius) ** 2)
    # penalize the misclassification in incompatible group
    incorrect_incomp = incomp_distances <= radius - tol
    loss += np.sum((radius - incomp_distances[incorrect_incomp]) ** 2)

    if initial_guess is None or alpha <= 0:
        return loss

    initial_center = initial_guess[:-1]
    cutoff_radius = initial_guess[-1] / 3
    distance = np.linalg.norm(center - initial_center)
    if distance > cutoff_radius:
        # penalize the distance from the center to the initial guess
        loss += (
            alpha
            * ((distance - cutoff_radius) ** 2)
            * max(np.sum(incorrect_comp), np.sum(incorrect_incomp), 1)
        )
    return loss


def interpret_HSP(
    compatible_HSPs: np.ndarray,
    incompatible_HSPs: np.ndarray | None = None,
    random_state: int | None = None,
    method: Literal["scipy-minimize", "svm"] = "scipy-minimize",
) -> tuple[tuple[float, ...], float]:
    """
    analyze the current results and give estimates of the HSPs (dD, dP, dH) of the target resin and the cutoff sphere radius

    :param np.ndarray compatible_HSPs:
        the HSP values of compatible/miscible solvents
    :param np.ndarray | None incompatible_HSPs:
        the HSP values of incompatible/immiscible solvents
    :param int | None random_state:
        the random state to use for reproducibility
    :param Literal["scipy-minimize", "svm"] method:
        the method to use for estimating the cutoff sphere radius
    :return tuple[float, ...]:
        the estimated HSPs of the target resin. The shape is (info.HSP_LENGTH,).
    :return float:
        the estimated cutoff sphere radius
    """
    compatible_HSPs = to_transformed_HSP(compatible_HSPs)
    incompatible_HSPs = to_transformed_HSP(incompatible_HSPs)
    combined_HSPs = np.concatenate([compatible_HSPs, incompatible_HSPs], axis=0)

    if len(compatible_HSPs) == 0:
        raise ValueError("No compatible solvents provided")

    # incomp_in_comp_flags = get_in_hull_flags(incompatible_HSPs, compatible_HSPs)
    # incompatible_HSPs = incompatible_HSPs[~incomp_in_comp_flags]

    if len(compatible_HSPs) < 4:
        warnings.warn(
            f"Only {len(compatible_HSPs)} compatible solvents are provided, "
            f"which is less than 4. Be cautious when interpreting the results."
        )

    import miniball

    # the radius estimation from miniball is not reliable when there are only a few points
    center, _ = miniball.get_bounding_ball(
        compatible_HSPs, rng=np.random.default_rng(random_state)
    )
    r_comp = calc_comb_HSP_distances(center, compatible_HSPs, agg="max", axis=None)
    r_incomp = calc_comb_HSP_distances(center, incompatible_HSPs, agg="min", axis=None)
    radius = max(r_comp, (r_comp + r_incomp) / 2, info.HSP_CUTOFF_SPHERE_RADIUS)
    if len(incompatible_HSPs) == 0:
        # get the minumum enclosing sphere of the compatible HSPs
        warnings.warn(
            "No incompatible solvents provided. "
            "The HSP sphere will be estimated based on the `bounding ball` of the compatible solvents. "
            f"And the radius will be stretched further by 10%."
        )
        center = tuple(round(float(c), 2) for c in to_original_HSP(center).flatten())
        radius = round(float(radius * 1.1), 2)
        return center, radius

    # use the center as the estimate as well
    comp_distances = calc_comb_HSP_distances(compatible_HSPs, center).flatten()
    incomp_distances = calc_comb_HSP_distances(incompatible_HSPs, center).flatten()

    comp_max = comp_distances.max()
    incomp_min = incomp_distances.min()
    if comp_max <= incomp_min:
        # the compatible group can be fully separated from the incompatible group
        # the center is the best estimate
        center = tuple(round(float(c), 2) for c in center)
        radius = round(float((comp_max + incomp_min) / 2), 2)

    if method == "scipy-minimize":
        # use scipy.optimize.minimize to find the best center and radius
        # NOTE: this method is not as stable as the SVM method yet
        x0 = np.concatenate([center, [radius]])
        _mins = np.min(combined_HSPs, axis=0)
        _maxs = np.max(combined_HSPs, axis=0)
        _ranges = _maxs - _mins
        _mins, _maxs = _mins - _ranges / 4, _maxs + _ranges / 4
        bounds = [(x, y) for x, y in zip(_mins, _maxs)]
        bounds += [
            (
                info.HSP_CUTOFF_SPHERE_RADIUS,
                max(np.linalg.norm(_ranges / 1.5), 1.5 * info.HSP_CUTOFF_SPHERE_RADIUS),
            )
        ]

        initial_guess = np.concatenate([center, [radius]])
        res = minimize(
            _calc_HSP_estimation_loss,
            x0,
            args=(compatible_HSPs, incompatible_HSPs, initial_guess),
            bounds=bounds,
            options={"disp": False, "maxiter": 100000},
        )
        center, radius = res.x[:-1], res.x[-1]
        center = tuple(round(float(c), 2) for c in to_original_HSP(center).flatten())
        radius = round(float(radius), 2)
        return center, radius

    # use support vector machine to estimate the cutoff sphere radius
    # by finding the best distance to separate `comp_distances` and `incomp_distances`
    # the distance is the cutoff sphere radius
    X = np.concatenate([comp_distances, incomp_distances])
    y = np.concatenate([np.ones_like(comp_distances), np.zeros_like(incomp_distances)])

    svc = SVC(kernel="linear", C=1.0, random_state=random_state)
    svc.fit(X.reshape(-1, 1), y)

    # get the cutoff sphere radius
    _rs = np.linspace(np.min(X), np.max(X), 10000)
    _ys = svc.decision_function(_rs.reshape(-1, 1))
    # pick the radius where the decision function is closest to 0
    center = tuple(round(float(c), 2) for c in to_original_HSP(center).flatten())
    radius = round(float(_rs[np.argmin(np.abs(_ys))]), 2)
    return center, radius
