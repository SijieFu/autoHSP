# %% [markdown]
# ## Batch-Model Active Learning in autoHSP - Examples
#

# %%
import os
import sys
from typing import Any
import logging
import shutil

import numpy as np
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = "Arial"

import HSP
from HSP import hsp_math, hsp_solver, hsp_utils
from HSP.info import info
import HSP.utils

AGG_DIST = os.environ.get("TEST_AGG_DIST", "0").lower()[:1] in ("1", "t", "y")

df_solvents = HSP.utils.get_solvents_df()

n = 22
pure_solvents = df_solvents.loc[:, ["dP", "dH"]].values[:n]
info.HSP_LENGTH = pure_solvents.shape[1]
info.ENABLE_HSP_DISTANCE = False
info.DEBUG = True

solvents_mix_matrix = np.ones((n, n), dtype=bool)
np.fill_diagonal(solvents_mix_matrix, False)

_min, _max = -3.5, 25
mix_step_size = 0.1
precision = 0.1
target_n = 6
explore_temp = 0.5
distance_percentile = 0.66 if AGG_DIST else 0.34
random_state = 42
n_points = 1000
fig_dir = f"figures{'_aggressive' if AGG_DIST else ''}"
os.makedirs(fig_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename=os.path.join(fig_dir, "_log.log"),
    filemode="w",
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

all_solvents_ingredients, all_solvents = hsp_utils.get_possible_solvents(
    nsolvents=n,
    max_solvent=2,
    mix_step_size=mix_step_size,
    mix_matrix=solvents_mix_matrix,
    HSPs=pure_solvents,
    precision=precision,
)

# define a test material
target_coord = np.array([7.5, 9.5])
target_radius = 7.2


def get_miscible_indices(
    solvents: np.ndarray,
    center: np.ndarray = target_coord,
    radius: float = target_radius,
) -> np.ndarray:
    """
    Get the indices of the miscible solvents within a certain radius from the target coordinate.

    :param np.ndarray solvents:
        array of solvents, shape (n_solvents, HSP_LENGTH)
    :param np.ndarray center:
        center coordinate, shape (HSP_LENGTH,)
    :param float radius:
        radius to consider for miscibility
    :return np.ndarray:
        array of indices of the miscible solvents
    """
    solvents = np.array(solvents).reshape(-1, info.HSP_LENGTH)
    distances = hsp_utils.calc_comb_HSP_distances(
        source_HSPs=solvents, target_HSPs=center.reshape(1, -1)
    ).flatten()
    return np.flatnonzero(distances <= radius)


def get_HSPs_from_ingredients(
    ingredients: np.ndarray, pure_solvents: np.ndarray = pure_solvents
) -> np.ndarray:
    """
    Get the HSPs of the solvents from the ingredients (portion of pure solvents).

    :param np.ndarray ingredients:
        array of ingredients (portion of pure solvents). Shape (n_solvents, n_pure_solvents)
    :param np.ndarray pure_solvents:
        array of pure solvents, shape (n_pure_solvents, HSP_LENGTH)
    :return np.ndarray:
        array of HSPs of the solvents, shape (n_solvents, HSP_LENGTH)
    """
    return np.dot(ingredients, pure_solvents)


def get_considered_area(
    possible_solvents: np.ndarray | None = None,
    tested_solvents: np.ndarray | None = None,
    considered_mask: np.ndarray | None = None,
    pure_solvents: np.ndarray = pure_solvents,
    n_points: int = 100,
) -> np.ndarray:
    """
    Get the considered area of the HSP space for solvent selection.

    :param np.ndarray | None possible_solvents:
        array of possible solvents, shape (n_possible_solvents, HSP_LENGTH)
    :param np.ndarray | None tested_solvents:
        array of tested solvents, shape (n_tested_solvents, HSP_LENGTH)
    :param np.ndarray | None considered_mask:
        boolean mask indicating which solvents are considered, shape (n_possible_solvents,)
    :param np.ndarray pure_solvents:
        array of pure solvents, shape (n_pure_solvents, HSP_LENGTH)
    :return np.ndarray:
        meshgrid of the considered area, shape (n_points*n_points, n_points*n_points)
    """
    xx, yy = np.meshgrid(
        np.linspace(_min, _max, n_points), np.linspace(_min, _max, n_points)
    )
    points = np.vstack([xx.ravel(), yy.ravel()]).T

    convex_hull = ConvexHull(pure_solvents)
    convex_hull_path = Delaunay(pure_solvents[convex_hull.vertices])
    mask = convex_hull_path.find_simplex(points) >= 0

    if possible_solvents is None or len(possible_solvents) == 0:
        return mask.reshape(xx.shape)
    if (
        considered_mask is None
        or len(considered_mask) == 0
        or np.all(considered_mask)
        or not np.any(considered_mask)
    ):
        return mask.reshape(xx.shape)

    indices = np.flatnonzero(mask)
    points = points[indices]

    considered_solvents = possible_solvents[considered_mask]
    min_dist_considered = hsp_utils.calc_comb_HSP_distances(
        source_HSPs=points, target_HSPs=considered_solvents
    ).min(axis=0)
    excluded_solvents = possible_solvents[~considered_mask]
    if tested_solvents is not None and len(tested_solvents) > 0:
        excluded_solvents = np.vstack([excluded_solvents, tested_solvents])
    min_dist_excluded = hsp_utils.calc_comb_HSP_distances(
        source_HSPs=points, target_HSPs=excluded_solvents
    ).min(axis=0)

    indices = indices[min_dist_considered <= min_dist_excluded]
    mask = np.zeros(xx.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(xx.shape)


def get_plot_shape(
    solvents: np.ndarray, pure_solvents: np.ndarray = pure_solvents
) -> dict[str, np.ndarray]:
    """
    for pure solvents, use a circle shape; for mixture solvents, use a dimond shape

    :param np.ndarray solvents:
        array of solvents to plot
    :param np.ndarray pure_solvents:
        array of pure solvents
    :return dict[str, np.ndarray]:
        dictionary for the corresponding plot shape

        - keys: "o" for circle, "D" for diamond
        - values: indices of the `solvents` array that correspond to the shape
    """
    distances = (
        hsp_utils.calc_comb_HSP_distances(
            source_HSPs=solvents, target_HSPs=pure_solvents
        ).min(axis=0)
        < 0.01
    )
    return {"o": np.flatnonzero(distances), "D": np.flatnonzero(~distances)}


def plot_solvents(
    pure_solvents: np.ndarray = pure_solvents,
    ax: plt.Axes | None = None,
    possible_solvents: np.ndarray | None = None,
    considered_mask: np.ndarray | None = None,
    n_points: int = 1000,
    miscible_solvents: np.ndarray | None = None,
    immiscible_solvents: np.ndarray | None = None,
    pending_solvents: np.ndarray | None = None,
    plot_annotations: bool = True,
    show_sphere: bool = False,
    show_legend: bool = False,
    a_color: str = "none",
    m_color: str = "green",
    im_color: str = "red",
    p_color: str = "steelblue",
    **kwargs: Any,
) -> tuple[tuple[plt.Axes, plt.Figure], bool]:
    """
    Plot the HSP solvents on a 2D plane.

    - The pure solvents are plotted as hollow circles with a black border.
    - The miscible solvents are plotted as green circles or diamonds.
    - The immiscible solvents are plotted as red circles or diamonds.
    - The pending solvents are plotted as gray circles or diamonds.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    if miscible_solvents is None:
        miscible_solvents = np.array([]).reshape(0, info.HSP_LENGTH)
    miscible_solvents = miscible_solvents.reshape(-1, info.HSP_LENGTH)

    if immiscible_solvents is None:
        immiscible_solvents = np.array([]).reshape(0, info.HSP_LENGTH)
    immiscible_solvents = immiscible_solvents.reshape(-1, info.HSP_LENGTH)

    if pending_solvents is None:
        pending_solvents = np.array([]).reshape(0, info.HSP_LENGTH)
    pending_solvents = pending_solvents.reshape(-1, info.HSP_LENGTH)

    validated_solvents = np.vstack([miscible_solvents, immiscible_solvents])
    tested_solvents = np.vstack([validated_solvents, pending_solvents])

    if possible_solvents is not None:
        Z = get_considered_area(
            possible_solvents=possible_solvents,
            tested_solvents=tested_solvents,
            considered_mask=considered_mask,
            pure_solvents=pure_solvents,
            n_points=n_points,
        )
        ax.imshow(
            Z,
            extent=(_min, _max, _min, _max),
            origin="lower",
            cmap="Greys",
            alpha=0.2,
            zorder=-2,
        )

    ax.set_title("HSP Solvents", fontsize=14, fontweight="bold")
    # Place axis labels inside the plot frame, near the axes
    # ax.yaxis.label.set_rotation(0)
    ax.set_xlabel("X1", fontsize=12, fontdict={"weight": "bold"})
    ax.set_ylabel("X2", fontsize=12, fontdict={"weight": "bold"})
    # Adjust label positions manually for more control
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.set_label_coords(0.07, 0.035)
    ax.yaxis.set_label_position("left")
    ax.yaxis.set_label_coords(0.0415, 0.08)
    # no x/y ticks
    ax.set_xticks([])
    ax.set_yticks([])
    # set frame weight
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)

    ax.set_xlim(_min, _max)
    ax.set_ylim(_min, _max)
    ax.set_aspect("equal")
    ax.grid(zorder=-1)

    ax.scatter([], [], c="none", edgecolors="black", marker="o", label="Pure Solvent")
    ax.scatter(
        [], [], c="none", edgecolors="black", marker="D", label="Solvent Mixture"
    )
    ax.scatter([], [], c="gray", marker="s", label="Refined Query Space")
    ax.scatter(
        [], [], marker=r"$\#1$", c="black", s=100, label="Latest Query", linewidths=0.3
    )

    s = kwargs.setdefault("s", 120)
    edgecolors = kwargs.setdefault("edgecolors", "black")
    linewidths = kwargs.setdefault("linewidths", 1.5)
    alpha = kwargs.setdefault("alpha", 0.6)

    ax.scatter([], [], c=m_color, marker="o", label="Miscible Solvent", alpha=alpha)
    ax.scatter([], [], c=im_color, marker="o", label="Immiscible Solvent", alpha=alpha)
    ax.scatter([], [], c=p_color, marker="o", label="Pending Solvent", alpha=alpha)

    # Add invisible legend entry for "Estimated Sphere"
    ax.plot([], [], color="black", linestyle="-", label='Target "Sphere"')
    ax.plot([], [], color="black", linestyle="--", label='Estimated "Sphere"')

    # original pure solvents -> hollow circles with black border
    ax.scatter(*pure_solvents.T, c=a_color, marker="o", zorder=1000, **kwargs)

    # miscible solvents -> green circles/diamonds
    if miscible_solvents.size > 0:
        miscible_shapes = get_plot_shape(miscible_solvents, pure_solvents)
        for marker, indices in miscible_shapes.items():
            if len(indices) == 0:
                continue
            ax.scatter(
                *miscible_solvents[indices, :2].T, c=m_color, marker=marker, **kwargs
            )

    # immiscible solvents -> red circles/diamonds
    if immiscible_solvents.size > 0:
        immiscible_shapes = get_plot_shape(immiscible_solvents, pure_solvents)
        for marker, indices in immiscible_shapes.items():
            if len(indices) == 0:
                continue
            ax.scatter(
                *immiscible_solvents[indices, :2].T, c=im_color, marker=marker, **kwargs
            )

    # pending solvents -> gray circles/diamonds
    if pending_solvents.size > 0:
        pending_shapes = get_plot_shape(pending_solvents, pure_solvents)
        _kwargs = kwargs.copy()
        _kwargs["linewidths"] *= 1.25
        for marker, indices in pending_shapes.items():
            if len(indices) == 0:
                continue
            _psolvents = pending_solvents[indices]
            if len(validated_solvents) == 0:
                _psolvents_mask = np.ones(_psolvents.shape[0], dtype=bool)
            else:
                _psolvents_mask = (
                    hsp_math.calc_comb_HSP_distances(
                        _psolvents, validated_solvents
                    ).min(axis=0)
                    > 1e-3
                )
            for _mask, _color in [
                (_psolvents_mask, p_color),
                (~_psolvents_mask, "none"),
            ]:
                if not np.any(_mask):
                    continue
                ax.scatter(*_psolvents[_mask, :2].T, c=_color, marker=marker, **_kwargs)
            if not plot_annotations:
                continue
            # Add index annotations
            for idx in indices:
                ax.annotate(
                    str(idx + 1),
                    pending_solvents[idx, :2],
                    textcoords="offset points",
                    xytext=(0, -s / 100),
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                    color="black",
                    zorder=1002,
                )

    # check if end-of-experiment (EOE) criteria are met
    EOE = hsp_solver.reached_EOE_or_not(
        compatible=miscible_solvents,
        incompatible=immiscible_solvents,
        available=pure_solvents,
        max_solvent=2,
        mix_matrix=solvents_mix_matrix,
        mix_step_size=mix_step_size,
        precision=precision,
        strategy="smart",
        random_seed=random_state,
    )

    if show_sphere or EOE:
        center = tuple(round(float(x), 2) for x in target_coord)
        radius = round(float(target_radius), 2)
        _kwargs = kwargs.copy()
        ax.scatter(*center[:2], marker=r"$\odot$", c="black", **_kwargs)
        # add annotation for the center and radius
        ax.annotate(
            f"Ground Truth\nCenter: {center}\nRadius: {radius:.2f}",
            xy=(0.98, 0.02),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            color="black",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="none"),
        )
        circle = plt.Circle(
            center[:2], radius, color="black", fill=False, linestyle="-", zorder=-2
        )
        ax.add_artist(circle)

    if EOE:
        center, radius = hsp_math.interpret_HSP(
            compatible_HSPs=miscible_solvents,
            incompatible_HSPs=immiscible_solvents,
            random_state=random_state,
            method="scipy-minimize",
        )
        center = tuple(round(float(x), 2) for x in center)
        radius = round(float(radius), 2)
        _kwargs = kwargs.copy()
        ax.scatter(*center[:2], marker=r"x", c="black", **_kwargs)
        # add annotation for the center and radius
        ax.annotate(
            f"EOE Criteria Met\nCenter: {center}\nRadius: {radius:.2f}",
            xy=(0.02 if show_legend else 0.98, 0.98),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
            color="black",
            ha="left" if show_legend else "right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="none"),
        )
        circle = plt.Circle(
            center[:2], radius, color="black", fill=False, linestyle="--", zorder=-1
        )
        ax.add_artist(circle)

    if show_legend:
        legend = ax.legend(
            loc="upper right",
            fontsize=10,
            frameon=True,
            edgecolor="black",
            facecolor="none",
        )
        # Set font weight for legend text
        for text in legend.get_texts():
            text.set_fontweight("bold")
    return (ax, fig), EOE


(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents, show_sphere=True, show_legend=True
)
ax.set_title("")
save_path = os.path.join(fig_dir, "0_initial_hsp_space_0_pure_solvents")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
_all_solvents_dist = hsp_utils.calc_comb_HSP_distances(
    all_solvents, pure_solvents, agg="min", axis=0
)
_all_solvents = all_solvents[_all_solvents_dist > 1e-3]

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    pending_solvents=_all_solvents,
    plot_annotations=False,
    a_color="yellow",
    p_color="none",
    linewidths=0.5,
)
ax.set_title("")
save_path = os.path.join(fig_dir, "0_initial_hsp_space_1_all_solvents")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
compatible_solvents = np.array([[]]).reshape(-1, info.HSP_LENGTH)
incompatible_solvents = np.array([[]]).reshape(-1, info.HSP_LENGTH)
pending_solvents = np.array([[]]).reshape(-1, info.HSP_LENGTH)

ith_round = 1
logging.info(f"***** Selection Round {ith_round} *****")
selected_solvent_ingredients, strategy, _detail = hsp_solver.propose_new_solvents(
    compatible=compatible_solvents,
    incompatible=incompatible_solvents,
    pending=pending_solvents,
    available=pure_solvents,
    target_n_tasks=target_n,
    max_solvent=2,
    mix_matrix=solvents_mix_matrix,
    mix_step_size=mix_step_size,
    precision=precision,
    random_seed=random_state,
    explore_temp=explore_temp,
    distance_percentile=distance_percentile,
)
logging.info(f"Strategy: {strategy}")
possible_solvent_ingredients, possible_subset_mask, selected_subset_mask = _detail
possible_solvents = np.dot(possible_solvent_ingredients, pure_solvents)

selected_solvents = np.dot(selected_solvent_ingredients, pure_solvents)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=np.array([]).reshape(0, info.HSP_LENGTH),
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_1_space")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_2_selection")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
miscible_indices = get_miscible_indices(selected_solvents)
logging.info(f"Miscible Solvents Indices: {miscible_indices}")

miscible_mask = np.zeros(target_n, dtype=bool)
miscible_mask[miscible_indices] = True

compatible_solvents = np.vstack([compatible_solvents, selected_solvents[miscible_mask]])
incompatible_solvents = np.vstack(
    [incompatible_solvents, selected_solvents[~miscible_mask]]
)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_3_result")
if EOE:
    with open(f"{save_path}_EOE", "w") as f:
        f.write("EOE Criteria Met\n")
else:
    shutil.rmtree(f"{save_path}_EOE", ignore_errors=True)

fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
ith_round += 1
logging.info(f"***** Selection Round {ith_round} - high temp test *****")
selected_solvent_ingredients, strategy, _detail = hsp_solver.propose_new_solvents(
    compatible=compatible_solvents,
    incompatible=incompatible_solvents,
    pending=pending_solvents,
    available=pure_solvents,
    target_n_tasks=target_n,
    max_solvent=2,
    mix_matrix=solvents_mix_matrix,
    mix_step_size=mix_step_size,
    precision=precision,
    random_seed=random_state,
    explore_temp=1.0,
    distance_percentile=distance_percentile,
)
# logging.info(f"Strategy: {strategy}")
possible_solvent_ingredients, possible_subset_mask, selected_subset_mask = _detail
possible_solvents = np.dot(possible_solvent_ingredients, pure_solvents)

selected_solvents = np.dot(selected_solvent_ingredients, pure_solvents)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=pending_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_0_space_high_temp")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
logging.info(f"***** Selection Round {ith_round} *****")
selected_solvent_ingredients, strategy, _detail = hsp_solver.propose_new_solvents(
    compatible=compatible_solvents,
    incompatible=incompatible_solvents,
    pending=pending_solvents,
    available=pure_solvents,
    target_n_tasks=target_n,
    max_solvent=2,
    mix_matrix=solvents_mix_matrix,
    mix_step_size=mix_step_size,
    precision=precision,
    random_seed=random_state,
    explore_temp=explore_temp,
    distance_percentile=distance_percentile,
)
logging.info(f"Strategy: {strategy}")
possible_solvent_ingredients, possible_subset_mask, selected_subset_mask = _detail
possible_solvents = np.dot(possible_solvent_ingredients, pure_solvents)

selected_solvents = np.dot(selected_solvent_ingredients, pure_solvents)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=pending_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_1_space")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_2_selection")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
miscible_indices = get_miscible_indices(selected_solvents)
logging.info(f"Miscible Solvents Indices: {miscible_indices}")

miscible_mask = np.zeros(target_n, dtype=bool)
miscible_mask[miscible_indices] = True

compatible_solvents = np.vstack([compatible_solvents, selected_solvents[miscible_mask]])
incompatible_solvents = np.vstack(
    [incompatible_solvents, selected_solvents[~miscible_mask]]
)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_3_result")
if EOE:
    with open(f"{save_path}_EOE", "w") as f:
        f.write("EOE Criteria Met\n")
else:
    shutil.rmtree(f"{save_path}_EOE", ignore_errors=True)
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
ith_round += 1
logging.info(f"***** Selection Round {ith_round} *****")
selected_solvent_ingredients, strategy, _detail = hsp_solver.propose_new_solvents(
    compatible=compatible_solvents,
    incompatible=incompatible_solvents,
    pending=pending_solvents,
    available=pure_solvents,
    target_n_tasks=target_n,
    max_solvent=2,
    mix_matrix=solvents_mix_matrix,
    mix_step_size=mix_step_size,
    precision=precision,
    random_seed=random_state,
    explore_temp=explore_temp,
    distance_percentile=distance_percentile,
)
logging.info(f"Strategy: {strategy}")
possible_solvent_ingredients, possible_subset_mask, selected_subset_mask = _detail
possible_solvents = np.dot(possible_solvent_ingredients, pure_solvents)

selected_solvents = np.dot(selected_solvent_ingredients, pure_solvents)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=pending_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_1_space")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_2_selection")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
miscible_indices = get_miscible_indices(selected_solvents)
logging.info(f"Miscible Solvents Indices: {miscible_indices}")

miscible_mask = np.zeros(target_n, dtype=bool)
miscible_mask[miscible_indices] = True

compatible_solvents = np.vstack([compatible_solvents, selected_solvents[miscible_mask]])
incompatible_solvents = np.vstack(
    [incompatible_solvents, selected_solvents[~miscible_mask]]
)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_3_result")
if EOE:
    with open(f"{save_path}_EOE", "w") as f:
        f.write("EOE Criteria Met\n")
else:
    shutil.rmtree(f"{save_path}_EOE", ignore_errors=True)
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
ith_round += 1
logging.info(f"***** Selection Round {ith_round} *****")
selected_solvent_ingredients, strategy, _detail = hsp_solver.propose_new_solvents(
    compatible=compatible_solvents,
    incompatible=incompatible_solvents,
    pending=pending_solvents,
    available=pure_solvents,
    target_n_tasks=target_n,
    max_solvent=2,
    mix_matrix=solvents_mix_matrix,
    mix_step_size=mix_step_size,
    precision=precision,
    random_seed=random_state,
    explore_temp=explore_temp,
    distance_percentile=distance_percentile,
)
logging.info(f"Strategy: {strategy}")
possible_solvent_ingredients, possible_subset_mask, selected_subset_mask = _detail
possible_solvents = np.dot(possible_solvent_ingredients, pure_solvents)

selected_solvents = np.dot(selected_solvent_ingredients, pure_solvents)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=pending_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_1_space")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_2_selection")
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
miscible_indices = get_miscible_indices(selected_solvents)
logging.info(f"Miscible Solvents Indices: {miscible_indices}")

miscible_mask = np.zeros(target_n, dtype=bool)
miscible_mask[miscible_indices] = True

compatible_solvents = np.vstack([compatible_solvents, selected_solvents[miscible_mask]])
incompatible_solvents = np.vstack(
    [incompatible_solvents, selected_solvents[~miscible_mask]]
)

(ax, fig), EOE = plot_solvents(
    pure_solvents=pure_solvents,
    possible_solvents=possible_solvents,
    considered_mask=possible_subset_mask,
    n_points=n_points,
    miscible_solvents=compatible_solvents,
    immiscible_solvents=incompatible_solvents,
    pending_solvents=selected_solvents,
    plot_annotations=True,
)
ax.set_title("")
save_path = os.path.join(fig_dir, f"{ith_round}_{strategy}_query_3_result")
if EOE:
    with open(f"{save_path}_EOE", "w") as f:
        f.write("EOE Criteria Met\n")
else:
    shutil.rmtree(f"{save_path}_EOE", ignore_errors=True)
fig.savefig(f"{save_path}.svg", bbox_inches="tight", pad_inches=0.0)
fig.savefig(f"{save_path}.png", bbox_inches="tight", pad_inches=0.0, dpi=300)

# %%
logging.info("===== End of HSP Solvent Selection Example =====")
