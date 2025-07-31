"""
Configuration file for HSP solver
"""

import os
from zoneinfo import ZoneInfo


class _Info:
    """
    instance for dealing with configuration settings
    """

    WORKING_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    # configuration dir for the test materials
    EXP_CONFIG_DIR = os.path.join(DATA_DIR, "exp_config")
    os.makedirs(EXP_CONFIG_DIR, exist_ok=True)
    EXP_CONFIG_EXT = "toml"
    EXP_CONFIG_REGEX = r"^R\d+$"
    EXP_CONFIG_MUTABLE_KEYS = [
        "N_max",
        "N_min",
        "n_per_thread",
        "n_per_solvent",
        "strategy",
        "mix_step_size",
        "precision",
        "priority",
        "sample_sit_time",
        "listen_to",
    ]

    # experiment record data dir
    EXP_DATA_DIR = os.path.join(DATA_DIR, "exp_record")
    os.makedirs(EXP_DATA_DIR, exist_ok=True)
    EXP_DATA_SUMMARY_PATH = os.path.join(EXP_DATA_DIR, "summary.json")
    EXP_RESIN_COLNAME = "Code"
    EXP_SOLVENT_COLNAME = "Code"
    EXP_RANDOM_SEED = None  # no random seed
    EXP_TEST_RESIN_NAME = "R0"

    # lab experiment setting - in addition to the .toml config file
    # check the .toml config file for more details
    # NOTE: the values in the .toml config file will override these defaults
    EXP_RESIN_AMOUNT = 5  # milliliters or grams
    EXP_SOLVENT_AMOUNT = 5  # milliliters or grams in total
    EXP_MIX_STEP_SIZE = 0.1  # 10% A + 90% B, 20% A + 80% B, etc
    EXP_PRECISION = 0.1
    EXP_RESIN_PRIORITY = 1
    # see also `HSP.hsp_solver.reached_EOE_or_not`
    # see also `HSP.hsp_solver.propose_new_solvents`
    # see also `HSP.hsp_solver.get_conditioned_n_centroids`
    EXP_SELECTION_STRATEGY = "smart"
    EXP_SELECTION_STRICTNESS = 0.8
    EXP_EXPLORE_TEMP = 0.5
    EXP_DISTANCE_PERCENTILE = 0.34

    # special tasks for the experiment scheduling
    EXP_TASK_EOE = {"task": "EOE", "taskId": "taskid:EOE"}
    EXP_TASK_PAUSE = {"task": "pause", "taskId": "taskid:pause"}

    # lab experiment timezone information
    EXP_TIMEZONE = ZoneInfo("America/New_York")
    EXP_RESIN_NAME_REGEX = r"^R\d+$"
    EXP_RESIN_CAPTURE_REGEX = r"([R]\d+)"
    EXP_SOLVENT_NAME_REGEX = r"^S\d+$"
    EXP_SOLVENT_CAPTURE_REGEX = r"([S]\d+)"

    # whether to print debug information for the HSP solver
    DEBUG: bool = False

    # set the number of values in one set of HSPs
    # by default, it should be 3 (dD, dP, dH)
    # but technically this could other numbers for a more general case
    # recommended values are 2 or 3
    HSP_LENGTH: int = 3

    # when calculating HSP distance, dD will be converted to 2 * dD first
    # turn off this feature by setting this to False if you need to keep
    # the first value as it is
    # this is useful if you also change HSP_LENGTH to other values
    ENABLE_HSP_DISTANCE: bool = True

    # when exploring the HSP space, what should be the cutoff radius
    # for deciding if reasonable space has been explored
    # this is useful for early stopping/termination of the exploration/etc
    HSP_CUTOFF_SPHERE_RADIUS: float = 6.0

    @property
    def HSP_CUTOFF_SPHERE_VOLUME(self) -> float:
        from .hsp_utils import calc_hypersphere_volume

        return calc_hypersphere_volume(self.HSP_CUTOFF_SPHERE_RADIUS, self.HSP_LENGTH)


info = _Info()
