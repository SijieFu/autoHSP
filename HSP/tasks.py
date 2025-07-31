"""
solver for task-related functions
"""

import os
from datetime import datetime
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .info import info
from .utils import get_resin_config_dict, get_resin_config_info
from .utils import get_summary_log_json, update_summary_log_json
from .utils import get_resin_log_df, update_resin_log_df
from .utils import compose_experiment_label
from .utils import get_which_solvents_to_use, get_solvent_HSP
from .utils import parse_current_solvent_HSP, get_current_results_for_resin
from .hsp_solver import reached_EOE_or_not, propose_new_solvents


def get_taskid(task: str, labels: str | list[str]) -> str:
    """
    Get the taskid from the labels

    :param list[str] labels:
        the labels of the task
    :return str:
        the taskid
    """
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    labels = sorted(labels)
    label_string = f"{task}:" + ":".join(labels) + f":{task}"
    hash_object = hashlib.md5(label_string.encode())

    summary_json_keys = get_summary_log_json().keys()

    # in case of MD5 hash collision, add some random string at the end until no collision
    while True:
        taskmd5 = hash_object.hexdigest()
        if (taskid := f"{task}:{taskmd5}") not in summary_json_keys:
            break
        hash_object.update(os.urandom(6))

    return taskid


def _push_image_task_to_resin(
    resin: str,
    thread: str,
    labels: str | list[str],
    time: datetime = None,
    in_test_mode: bool = False,
) -> dict[str, Any]:
    """
    push the image task to log files

    :param str resin:
        the resin name
    :param str thread:
        the thread name that claims the task
    :param str | list[str] labels:
        the labels of the task in the `Label` column
    :param datetime time:
        the time of the claim
    :param bool in_test_mode:
        if True, do not update the summary log
    :return dict[str, Any]:
        the task dictionary
    """
    labels = labels if isinstance(labels, (list, tuple)) else [labels]
    labels = list(set(labels))  # remove duplicates

    time = time or datetime.now(info.EXP_TIMEZONE)
    if isinstance(time, str):
        time = datetime.fromisoformat(time)
    time = time.astimezone(info.EXP_TIMEZONE).isoformat()

    resin_log_df = get_resin_log_df(resin)

    # make sure all labels are in the `Label` column
    assert all(
        label in resin_log_df["Label"].values for label in labels
    ), f"some label(s) in `{labels}` not in the `Label` column of the resin log"

    task = "image"
    taskid = get_taskid(task, labels)

    # get row indices and set value for `ImageTaskId`, `ImageByThread`, `ImageStart`
    row_indices = resin_log_df[resin_log_df["Label"].isin(labels)].index
    _cols = ["ImageTaskId", "ImageByThread", "ImageStart"]
    resin_log_df[_cols] = resin_log_df[_cols].astype(str)
    resin_log_df.loc[row_indices, "ImageTaskId"] = taskid
    resin_log_df.loc[row_indices, "ImageByThread"] = thread
    resin_log_df.loc[row_indices, "ImageStart"] = time
    update_resin_log_df(resin, resin_log_df, keep_previous=True)

    task_dict = {
        "task": task,
        "taskId": taskid,
        "samples": labels,
        "thread": thread,
        "time": time,
    }

    # if IN_TEST_MODE, do not update the summary log
    if in_test_mode:
        return task_dict

    summary_json = get_summary_log_json()
    summary_json[taskid] = task_dict

    summary_json.setdefault(f"exception:{thread}", [])
    summary_json.setdefault(f"history:{thread}", [])
    summary_json[f"history:{thread}"].append(taskid)
    if prev_thread_taskid := summary_json.get(thread, False):
        # previous taskid claimed by the thread should be removed if taskid succeeded
        summary_json[f"exception:{thread}"].append(prev_thread_taskid)

    # update the summary log -> `thread` is working on `taskid`
    summary_json[thread] = taskid
    update_summary_log_json(summary_json, keep_previous=True)

    return task_dict


def check_image_task_for_resin(
    resin: str, claiming_thread: str = None, in_test_mode: bool = False
) -> tuple[bool, dict[str, Any] | None]:
    """
    Check if the resin has image task available

    :param str resin:
        the resin name
    :param str claiming_thread:
        the thread name to claim the task, if provided, this task will be directly pushed if available
    :param bool in_test_mode:
        if True, do not update the summary log
    :return bool:
        True if the resin has image task available, False otherwise
    :return dict[str, Any] | None:
        the task dictionary if available, None otherwise
    """
    resin_log_df = get_resin_log_df(resin)
    n_per_thread = get_resin_config_dict(resin).get("n_per_thread", 1)
    sample_sit_time = get_resin_config_dict(resin).get(
        "sample_sit_time", 1440
    )  # in minutes, default 24 hours

    # remove rows with NA value at `Label` column or `PrepEnd` column
    df = resin_log_df.dropna(subset=["Label", "PrepEnd"], how="any", axis=0)

    # get rows with a NA value at `ImageStart` column
    df = df[df["ImageStart"].isna()]

    if len(df) == 0:
        return False, None

    # transform the string in `PrepEnd` column to datetime object
    # for rows with a time stamp at `PrepEnd` column, check if `sample_sit_time` has passed
    now = datetime.now(info.EXP_TIMEZONE)
    df = df[
        df["PrepEnd"].apply(datetime.fromisoformat)
        + pd.Timedelta(minutes=sample_sit_time)
        <= now
    ]

    if len(df) == 0:
        return False, None
    elif not (isinstance(claiming_thread, str) and bool(claiming_thread)):
        return True, None

    # get the `n_per_thread` rows with the earliest `PrepEnd` time
    # or all rows if there are less than `n_per_thread` rows
    indices = df["PrepEnd"].sort_values().head(n_per_thread).index
    labels = df.loc[indices, "Label"].tolist()
    task_dict = _push_image_task_to_resin(
        resin, claiming_thread, labels, time=now, in_test_mode=in_test_mode
    )
    return True, task_dict


def _push_prep_task_to_resin(
    resin: str,
    thread: str,
    labels: str | list[str],
    time: datetime = None,
    in_test_mode: bool = False,
) -> dict[str, Any]:
    """
    push the prep task to log files

    :param str resin:
        the resin name
    :param str thread:
        the thread name that claims the task
    :param str | list[str] labels:
        the labels of the task in the `Label` column
    :param datetime time:
        the time of the claim
    :param bool in_test_mode:
        if True, do not update the summary log
    :return dict[str, Any]:
        the task dictionary
    """
    labels = labels if isinstance(labels, (list, tuple)) else [labels]
    labels = list(set(labels))  # remove duplicates

    time = time or datetime.now(info.EXP_TIMEZONE)
    if isinstance(time, str):
        time = datetime.fromisoformat(time)
    time = time.astimezone(info.EXP_TIMEZONE).isoformat()

    resin_log_df = get_resin_log_df(resin)

    # make sure all labels are in the `Label` column
    assert all(
        label in resin_log_df["Label"].values for label in labels
    ), f"some label(s) in `{labels}` not in the `Label` column of the resin log"

    task = "prep"
    taskid = get_taskid(task, labels)

    # get row indices and set value for `PrepTaskId`, `PrepByThread`, `PrepStart`
    row_indices = resin_log_df[resin_log_df["Label"].isin(labels)].index
    _cols = ["PrepTaskId", "PrepByThread", "PrepStart"]
    resin_log_df[_cols] = resin_log_df[_cols].astype(str)
    resin_log_df.loc[row_indices, "PrepTaskId"] = taskid
    resin_log_df.loc[row_indices, "PrepByThread"] = thread
    resin_log_df.loc[row_indices, "PrepStart"] = time
    update_resin_log_df(resin, resin_log_df, keep_previous=True)

    samples = []
    for label in labels:
        row_ind = resin_log_df[resin_log_df["Label"] == label].index[0]
        resin = resin_log_df.loc[row_ind, "Resin"]
        ramount = round(float(resin_log_df.loc[row_ind, "ResinAmount"]), 2)
        solvent = resin_log_df.loc[row_ind, "Solvent"].split(":")
        samount = [
            round(float(x), 2)
            for x in str(resin_log_df.loc[row_ind, "SolventAmount"]).split(":")
        ]
        samples.append(
            {
                "label": label,
                "resin": resin,
                "ramount": ramount,
                "solvent": solvent,
                "samount": samount,
            }
        )

    task_dict = {
        "task": task,
        "taskId": taskid,
        "thread": thread,
        "time": time,
        "samples": samples,
    }

    # if IN_TEST_MODE, do not update the summary log
    if in_test_mode:
        return task_dict

    summary_json = get_summary_log_json()
    summary_json[taskid] = task_dict
    summary_json.setdefault(f"exception:{thread}", [])
    summary_json.setdefault(f"history:{thread}", [])
    summary_json[f"history:{thread}"].append(taskid)
    if prev_thread_taskid := summary_json.get(thread, False):
        summary_json[f"exception:{thread}"].append(prev_thread_taskid)

    summary_json[thread] = taskid
    update_summary_log_json(summary_json, keep_previous=True)

    return task_dict


def check_prep_task_for_resin(
    resin: str, claiming_thread: str = None, in_test_mode: bool = False
) -> tuple[bool, dict[str, Any] | None]:
    """
    Check if the resin has prep task available

    :param str resin:
        the resin name
    :param str claiming_thread:
        the thread name to claim the task, if provided, this task will be directly pushed if available
    :param bool in_test_mode:
        if True, do not update the summary log
    :return bool:
        True if the resin has prep task available, False otherwise
    :return dict[str, Any] | None:
        the task dictionary if available, None otherwise
    """
    resin_log_df = get_resin_log_df(resin)
    n_per_thread = get_resin_config_dict(resin).get("n_per_thread", 1)

    # remove rows with NA value at `Label` column
    df = resin_log_df.dropna(
        subset=["Label", "Resin", "ResinAmount", "Solvent", "SolventAmount"],
        how="any",
        axis=0,
    )

    # get rows with a NA value at `PrepStart` column
    df = df[df["PrepStart"].isna()]

    if len(df) == 0:
        return False, None
    elif not (isinstance(claiming_thread, str) and bool(claiming_thread)):
        return True, None

    # get the first `n_per_thread` rows
    labels = df.head(n_per_thread)["Label"].tolist()
    time = datetime.now(info.EXP_TIMEZONE)
    task_dict = _push_prep_task_to_resin(
        resin, claiming_thread, labels, time=time, in_test_mode=in_test_mode
    )
    return True, task_dict


def _push_initiator_task_to_resin(
    resin: str,
    resin_amount: int | float | str,
    solvent: str | list[str],
    solvent_amount: int | float | str | list[int | float | str],
    initiator: str = "serendipity",
):
    """
    push the one single experiment task to the resin log

    :param str resin:
        the resin name
    :param int | float resin_amount:
        the amount of resin to use
    :param str | list[str] solvent:
        the solvent(s) to use
    :param int | float | list[int | float] solvent_amount:
        the amount(s) of solvent(s) to use
    :param str | list[str] initiator:
        the initiator(s) to use
    """
    # get the resin and solvent string
    resin_str = resin
    ramount = float(resin_amount)
    ramount_str = f"{ramount:.2f}"

    solvent = solvent if isinstance(solvent, (list, tuple)) else solvent.split(":")
    solvent_str = ":".join(solvent)
    if isinstance(solvent_amount, (int, float)):
        samount = [solvent_amount]
    elif isinstance(solvent_amount, str):
        samount = [x for x in solvent_amount.split(":")]
    else:
        assert isinstance(solvent_amount, (list, tuple))
        samount = solvent_amount
    samount = [float(x) for x in samount]
    samount_str = ":".join([f"{x:.2f}" for x in samount])

    # get the label
    label = compose_experiment_label(resin_str, ramount, solvent, samount)
    dD, dP, dH = get_solvent_HSP(solvent, samount)

    # add a new row to the resin log
    resin_log_df = get_resin_log_df(resin)

    new_row = {
        "Label": label,
        "Resin": resin_str,
        "ResinAmount": ramount_str,
        "Solvent": solvent_str,
        "SolventAmount": samount_str,
        "dD": dD,
        "dP": dP,
        "dH": dH,
        "Initiator": initiator,
    }
    if resin_log_df.empty:
        columns = resin_log_df.columns
        if len(columns) == 0:
            columns = new_row.keys()
        resin_log_df = pd.DataFrame([new_row], columns=columns)
    else:
        resin_log_df = pd.concat(
            [resin_log_df, pd.DataFrame([new_row])], ignore_index=True
        )

    _cols = ["dD", "dP", "dH"]
    resin_log_df[_cols] = resin_log_df[_cols].astype(float).round(2)
    update_resin_log_df(resin, resin_log_df, keep_previous=True)


def check_required_solvents_for_resin(
    resin: str, required_solvents: list[str] = None, push_found_tasks: bool = True
) -> int:
    """
    Initialize the required solvents for the resin

    :param str resin:
        the resin name
    :param list[str] required_solvents:
        the list of required solvents to use.
        If None, the required solvents will be computed from the resin configuration
    :param bool push_found_tasks:
        if True, push new tasks if find any
    :return int:
        the number of required solvents that are generated
    """
    required_solvents = required_solvents or get_which_solvents_to_use(resin)[1]
    resin_log_df = get_resin_log_df(resin)
    resin_amount, solvent_amount = get_resin_config_info(resin, "amount")

    # find required solvents that are not in the resin log `Solvent` column
    required_solvents = set(required_solvents) - set(resin_log_df["Solvent"])
    if push_found_tasks:
        for solvent in required_solvents:
            _push_initiator_task_to_resin(
                resin,
                resin_amount,
                solvent,
                solvent_amount,
                initiator="required",
            )

    return len(required_solvents)


def check_resin_tmp_pause_status(resin: str) -> bool:
    """
    if a resin has current experiments running, pause it

    :param str resin:
        the resin name
    :return bool:
        True if the resin should be temporarily paused, False otherwise
    """
    resin_log_df = get_resin_log_df(resin)

    # the only case for a temp pause is when some existing rows are running `prep`/`image` tasks
    # and there is no more available `prep`/`image` task to start
    if not resin_log_df["PrepStart"].hasnans and resin_log_df["PrepEnd"].hasnans:
        return True
    if not resin_log_df["ImageStart"].hasnans and resin_log_df["ImageEnd"].hasnans:
        return True

    return False


def check_resin_EOE_status(resin: str, push_found_tasks: bool = True) -> int:
    """
    Check if the resin has met the end of experiment criteria.

    - NOTE: this function does more than just checking the EOE status
    - NOTE: this function can also register new tasks if available (not EOE yet)

    :param str resin:
        the resin name
    :param bool push_found_tasks:
        if True, push new tasks if available (in such case, the EOE status must NOT be met)
    :return int:
        the EOE status code of the resin:

        - `1` for EOE criteria met
        - `0` for EOE criteria not met
        - `-1` for a temporary pause
    """
    available_solvents, required_solvents = get_which_solvents_to_use(resin)

    # if there are required solvents that are not in the resin log, push them
    ntasks_required = check_required_solvents_for_resin(
        resin, required_solvents, push_found_tasks=push_found_tasks
    )

    if check_resin_tmp_pause_status(resin):
        # a temporary pause is required
        return -1

    resin_config = get_resin_config_dict(resin)
    resin_amount, solvent_amount = get_resin_config_info(resin_config, "amount")

    # a priority of -1 (less than 0) means the resin reached EOE
    if resin_config.get("priority", 0) < 0 and ntasks_required <= 0:
        return 1

    N_max = resin_config.get("N_max", 1)
    N_min = resin_config.get("N_min", 1)
    if N_min > N_max:  # just in case the configuration is wrong
        N_min, N_max = N_max, N_min
    n_per_thread = resin_config.get("n_per_thread", 1)
    n_per_solvent = resin_config.get("n_per_solvent", 1)
    n_per_solvent = min(max(1, n_per_solvent), 2)  # only 1 or 2 are supported
    strategy = get_resin_config_info(resin_config, "strategy")
    resin_log_df = get_resin_log_df(resin)
    resin_log_df.dropna(subset=["Label"], how="any", axis=0, inplace=True)

    # wrong configuration
    if n_per_thread <= 0 or (N_max <= 0 and ntasks_required <= 0):
        return 1

    n_valid_results = (
        resin_log_df["Result"].isin({"Y", "N", "y", "n", "1", "0", 1, 0}).sum()
    )
    # all the rows have been tested with a valid `Result` value
    if not resin_log_df["Result"].isin({np.nan, None, ""}).any():
        # reached the maximum number of solvents to test and all results are valid
        if n_valid_results >= N_max:
            return 1

        # if there is a result that needs human intervention but not provided yet
        if any(resin_log_df["ResultRevised"].isin({-1, "-1"})):
            return -1  # a temporary pause is required for human intervention

        # if the number of tested solvents reached N_min but not N_max
        # and if strategy is "greedy", then EOE is not reached
        if n_valid_results >= N_min and strategy == "smart":
            avail_HSP, mix_matrix = parse_current_solvent_HSP(
                available_solvents, compute_mix_matrix=n_per_solvent > 1
            )
            compat_HSP, incompat_HSP, pending_HSP = get_current_results_for_resin(resin)

            # there should not be any pending HSPs, but just in case
            if len(pending_HSP) > 0:
                raise ValueError(f"Missing HSPs found in {resin}.csv")

            _EOE = reached_EOE_or_not(
                compatible=compat_HSP,
                incompatible=incompat_HSP,
                available=avail_HSP,
                max_solvent=n_per_solvent,
                mix_matrix=mix_matrix,
                strategy=strategy,
                strictness=get_resin_config_info(resin_config, "strictness"),
                random_seed=get_resin_config_info(resin_config, "random_seed"),
            )
            if _EOE or not push_found_tasks:
                return int(_EOE)

    # Up to this point, every EOE criteria has been tested
    # if there is not need to push new tasks, return 0 (not EOE)
    if not push_found_tasks:
        return 0

    # some rows do not have a valid `Result` value
    # what if all samples are prepared but some are not imaged yet?
    if not resin_log_df["PrepEnd"].hasnans:
        if resin_log_df["ImageStart"].hasnans:
            # these samples are ready for `image` task (not EOE yet)
            return 0

    # if all the rows have been tested with a valid `Result` value (or no row at all)
    # then the HSP solver is required to determine whether or not to continue
    ntasks_not_started = sum(resin_log_df["PrepStart"].isna())

    # this is the target number of new tasks to generate
    target_n_tasks = n_per_thread - ntasks_not_started

    if target_n_tasks <= 0:  # no new tasks to generate for now
        return 0

    avail_HSP, mix_matrix = parse_current_solvent_HSP(
        available_solvents,
        compute_mix_matrix=n_per_solvent > 1,
    )

    for max_solvent in [1, 2]:
        if max_solvent > n_per_solvent:
            # skip the case of max_solvent > n_per_solvent
            continue

        compa_HSP, incompa_HSP, pendind_HSP = get_current_results_for_resin(resin)
        if (
            max_solvent == 1
            and n_per_solvent > 1
            and n_valid_results >= N_min
            and len(incompa_HSP) > 0
        ):
            # skip the case of max_solvent=1 if n_per_solvent>1 and N_min is reached
            continue

        new_solvents, sub_strategy, *_ = propose_new_solvents(
            compatible=compa_HSP,
            incompatible=incompa_HSP,
            pending=pendind_HSP,
            available=avail_HSP,
            target_n_tasks=target_n_tasks,
            max_solvent=max_solvent,
            mix_matrix=mix_matrix,
            mix_step_size=get_resin_config_info(resin_config, "mix_step_size"),
            precision=get_resin_config_info(resin_config, "precision"),
            random_seed=get_resin_config_info(resin_config, "random_seed"),
            explore_temp=get_resin_config_info(resin_config, "explore_temp"),
            distance_percentile=get_resin_config_info(
                resin_config, "distance_percentile"
            ),
        )
        new_solvents = np.array(new_solvents).reshape(-1, len(available_solvents))
        # make sure each row sums to 1.0
        new_solvents = new_solvents / new_solvents.sum(axis=1, keepdims=True)

        for new_solvent in new_solvents:
            pure_solvents_which = np.where(new_solvent > 0.001)[0]
            pure_solvents_ratios = new_solvent[pure_solvents_which]
            pure_solvents_names = [available_solvents[i] for i in pure_solvents_which]
            pure_solvents_amounts = (solvent_amount * pure_solvents_ratios).tolist()
            _push_initiator_task_to_resin(
                resin,
                resin_amount,
                pure_solvents_names,
                pure_solvents_amounts,
                initiator=sub_strategy,
            )
        target_n_tasks -= len(new_solvents)

        if target_n_tasks <= 0:
            return 0

    if not resin_log_df["Result"].hasnans and target_n_tasks == n_per_thread:
        # all the rows have been tested with a valid `Result` value
        # and no new tasks are generated
        # just a fail-safe in case `reached_EOE_or_not` fails
        return 1

    return 0
