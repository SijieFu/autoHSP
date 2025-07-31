"""
experiment solver module for the HSP to get the next set of experiments, and determine HSP values
"""

import os
import re
import shutil
from typing import Literal
from datetime import datetime

from .info import info
from .tasks import check_image_task_for_resin, check_prep_task_for_resin
from .tasks import check_resin_EOE_status
from .utils import get_resin_config_dict, update_resin_config_toml
from .utils import get_resin_log_df, update_resin_log_df
from .utils import get_summary_log_json, update_summary_log_json
from .utils import extract_resins_from_samples


def get_resins_with_config(
    dir: str = None, ext: str = None, regex: str = None
) -> list[str]:
    """
    Get the resins with configuration files, excluding the test resin

    :param str dir:
        the directory to look for the resins
    :param str ext:
        the extension of the configuration files
    :param str regex:
        the regex pattern for the resin names
    :return list[str]:
        the list of resins
    """
    dir = dir or info.EXP_CONFIG_DIR
    ext = ext or info.EXP_CONFIG_EXT
    regex = regex or info.EXP_CONFIG_REGEX

    resins = []
    for file in os.listdir(dir):
        filename, fileext = os.path.splitext(file)
        fileext = fileext.replace(".", "")
        if filename == info.EXP_TEST_RESIN_NAME:
            continue
        if re.match(regex, filename) and fileext.lower() == (ext).lower():
            resins.append(filename)
    return sorted(resins, key=lambda x: int(re.findall(r"\d+", x)[-1]))


def get_resins_for_thread(
    thread: str,
    dir: str = None,
    ext: str = None,
    regex: str = None,
    in_test_mode: bool = False,
) -> list[str]:
    """
    Get resins that accept the thread (compatible to run on the thread)

    :param str thread:
        the thread name
    :param str dir:
        the directory to look for the resins
    :param str ext:
        the extension of the configuration files
    :param str regex:
        the regex pattern for the resin names
    :param bool in_test_mode:
        whether the function is called in test mode. If True, only the test resin is considered.
    :return list[str]:
        the list of resins
    """
    if in_test_mode:
        # IN_TEST_MODE, "info.EXP_TEST_RESIN_NAME" is the only resin if it exists
        if not os.path.exists(
            os.path.join(info.EXP_CONFIG_DIR, f"{info.EXP_TEST_RESIN_NAME}.toml")
        ):
            shutil.copy(
                os.path.join(info.EXP_CONFIG_DIR, "default.toml"),
                os.path.join(info.EXP_CONFIG_DIR, f"{info.EXP_TEST_RESIN_NAME}.toml"),
            )
        return [info.EXP_TEST_RESIN_NAME]

    dir = dir or info.EXP_CONFIG_DIR
    ext = ext or info.EXP_CONFIG_EXT
    regex = regex or info.EXP_CONFIG_REGEX

    resins = get_resins_with_config(dir, ext, regex)
    resins_for_thread = []
    for resin in resins:
        thread_regex = get_resin_config_dict(resin).get("listen_to", [])
        thread_regex = [thread_regex] if isinstance(thread_regex, str) else thread_regex
        if any(re.match(regex, thread) for regex in thread_regex):
            resins_for_thread.append(resin)
    return resins_for_thread


def get_tasks_for_thread(
    thread: str,
    ability: str | list[str] | None = None,
    dir: str | None = None,
    ext: str | None = None,
    regex: str | None = None,
    in_test_mode: bool = False,
) -> dict | None:
    """
    Get tasks that accept the thread (compatible to run on the thread)

    :param str thread:
        the thread name
    :param str | list[str] ability:
        the ability of the tasks
    :param str dir:
        the directory to look for the resins
    :param str ext:
        the extension of the configuration files. Default is "toml" (from `info.EXP_CONFIG_EXT`)
    :param str regex:
        the regex pattern for the resin names. Default is r"^R\d+$" (from `info.EXP_CONFIG_REGEX`)
    :param bool in_test_mode:
        whether the function is called in test mode. If True, only the test resin is considered.
    :return dict | None:
        the task dictionary. If no task is found, return None (this should not happen as there is always a pause task)
    """
    ability = ability or ["prep", "image"]
    dir = dir or info.EXP_CONFIG_DIR
    ext = ext or info.EXP_CONFIG_EXT
    regex = regex or info.EXP_CONFIG_REGEX

    resins = get_resins_for_thread(thread, dir, ext, regex, in_test_mode=in_test_mode)
    resins_config = {resin: get_resin_config_dict(resin) for resin in resins}
    resins_priority = [resins_config[resin].get("priority", -1) for resin in resins]

    # check if any resin meets the EOE criteria: having been tested over with `N_max` solvent
    for i, resin in enumerate(resins):
        if resins_priority[i] < 1:
            continue

        _EOE_status = check_resin_EOE_status(resin, push_found_tasks=True)
        if _EOE_status == 1:
            # reached the EOE criteria
            resins_priority[i] = -1
            update_resin_config_toml(resin, "priority", "-1")
        elif _EOE_status == 0:
            # not reached the EOE criteria
            continue
        elif _EOE_status == -1:
            # temporary pause -> should not update config toml
            resins_priority[i] = 0

    # check if all resins have priority < 0 for the EOE criteria
    # NOTE: if there is no resin, i.e. len(resins) == 0, then this condition is also True
    if all(priority < 0 for priority in resins_priority):
        return info.EXP_TASK_EOE  # end of experiment

    # check if all resins have priority < 1 for the pause criteria
    if all(priority < 1 for priority in resins_priority):
        return info.EXP_TASK_PAUSE  # pause the experiment

    # filter resins with priority >= 1
    active_resin_indices = [
        i for i, priority in enumerate(resins_priority) if priority >= 1
    ]
    resins = [resins[i] for i in active_resin_indices]
    resins_priority = [resins_priority[i] for i in active_resin_indices]

    # sort resins according to priority
    resins = [resin for _, resin in sorted(zip(resins_priority, resins))]
    ability = ability if isinstance(ability, (list, tuple)) else str(ability).split(",")
    assert len(ability) > 0, "ability should not be empty"

    found_task, task = False, None

    # chech for image tasks first
    if "image" in ability and not found_task:
        for resin in resins:
            found_task, task = check_image_task_for_resin(
                resin, thread, in_test_mode=in_test_mode
            )
            if found_task:
                break

    # no resin has image task? then check for prep tasks
    if "prep" in ability and not found_task:
        for resin in resins:
            found_task, task = check_prep_task_for_resin(
                resin, thread, in_test_mode=in_test_mode
            )
            if found_task:
                break

    if found_task:
        summary_log_dict = get_summary_log_json()
        summary_log_dict.setdefault("THREADS", [])
        if thread not in summary_log_dict["THREADS"]:
            summary_log_dict["THREADS"].append(thread)
            update_summary_log_json(summary_log_dict)
        return task

    return info.EXP_TASK_PAUSE  # pause the experiment if no task is found


def update_image_results(
    samples: str | list[str] | tuple[str],
    results: bool | list[bool] | tuple[bool],
    results_are_revised: int | list[int] | tuple[int] = 0,
):
    """
    deals with `results` messages coming from the lab script.

    If you are providing multiple samples at the same time, make sure they belong to the same resin.

    :param str | list[str] | tuple[str] samples: the sample names
    :param bool | list[bool] | tuple[bool] results: the results of the samples
    :param int | list[int] | tuple[int] results_are_revised:
        whether the results are revised or need revision

        -  0: the corresponding result in `results` does not need revision and is not revised
        -  1: the corresponding result in `results` is revised
        - -1: the corresponding result in `results` needs revision (pending revision)
    """
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    samples = [str(sample) for sample in samples]
    if not isinstance(results, (list, tuple)):
        assert isinstance(
            results, bool
        ), "results must be a boolean, or a list of booleans"
        results = [results]
    results = ["Y" if bool(result) else "N" for result in results]

    if not isinstance(results_are_revised, (list, tuple)):
        results_are_revised = [results_are_revised]
    results_are_revised = [int(result) for result in results_are_revised]
    assert all(
        x in [0, 1, -1] for x in results_are_revised
    ), "results_are_revised must be in [0, 1, -1]"
    assert (
        len(samples) == len(results) == len(results_are_revised)
    ), "samples, results, and needs_intervention must have the same length"

    # _dict = {0: "N", 1: "Y", -1: "P"}
    # results_are_revised = [_dict[x] for x in results_are_revised]

    resin = extract_resins_from_samples(samples, check_single_resin=True)

    # update the results in the resin log
    resin_log_df = get_resin_log_df(resin)
    for sample, res, res_rev in zip(samples, results, results_are_revised):
        # find sample in the `Label` column
        _whichrows = resin_log_df["Label"].eq(sample)
        if not _whichrows.any():
            raise ValueError(
                f"sample `{sample}` not found in the resin log for resin `{resin}`"
            )
        resin_log_df.loc[_whichrows, ["Result", "ResultRevised"]] = [res, res_rev]
    update_resin_log_df(resin, resin_log_df)


def task_succeeded(
    task: Literal["prep", "image"],
    taskid: str,
    protocolid: str,
    samples: str | list[str] | tuple[str],
    sampleids: str | list[str] | tuple[str],
    time: str | datetime = None,
):
    """
    deals with `success` messages coming from the lab script

    NOTE: all samples must belong to the same protocol and the same resin

    :param str task:
        the task type
    :param str taskid:
        the task id
    :param str protocolid:
        the protocol id
    :param str | list[str] | tuple[str] samples:
        the sample names
    :param str | list[str] | tuple[str] sampleids:
        the sample ids
    """
    assert task in ["prep", "image"], "task must be either 'prep' or 'image'"
    task = task.capitalize()

    # find the current time
    time = time or datetime.now(info.EXP_TIMEZONE)
    if isinstance(time, str):
        time = datetime.fromisoformat(time)
    time = time.astimezone(info.EXP_TIMEZONE).isoformat()

    if not isinstance(samples, (list, tuple)):
        samples = [samples]
    samples = [str(sample) for sample in samples]
    if not isinstance(sampleids, (list, tuple)):
        sampleids = [sampleids]
    sampleids = [str(sampleid) for sampleid in sampleids]

    resin = extract_resins_from_samples(samples, check_single_resin=True)
    resin_log_df = get_resin_log_df(resin)

    columns = ["LabId", f"{task}ProtocolId", f"{task}End"]
    resin_log_df[columns] = resin_log_df[columns].fillna("").astype(str)
    for s, sid in zip(samples, sampleids):
        # find sample in the `Label` column
        _whichrows = resin_log_df["Label"].eq(s)
        if not _whichrows.any():
            raise ValueError(f"sample {s} not found in the resin log")
        resin_log_df.loc[_whichrows, columns] = [sid, protocolid, time]
    update_resin_log_df(resin, resin_log_df)

    if isinstance(taskid, str) and taskid:
        # update the task id in the resin log
        summary_log_dict = get_summary_log_json()
        if taskid in summary_log_dict.keys():
            thread = summary_log_dict[taskid].get("thread", "")
            if summary_log_dict.get(thread, False) == taskid:
                summary_log_dict.pop(thread)
                update_summary_log_json(summary_log_dict, keep_previous=True)
