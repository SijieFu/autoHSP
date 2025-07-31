"""
utility functions for the Flask app to handle test requests with a fake resin.

This module is used to test the autoHSP workflow without actually executing the experiments
(with a fake resin and fake results).
"""

import os
import random
from datetime import datetime

import pandas as pd
from werkzeug.datastructures import MultiDict

from info import info

from HSP.api import get_tasks_for_thread
from HSP.utils import init_resin_log


def _get_test_unknowns() -> list[str]:
    """
    get the list of samples with unknown test results for the test resin `info.EXP_TEST_RESIN_NAME`
    """
    resin_log_path = os.path.join(info.EXP_DATA_DIR, f"{info.EXP_TEST_RESIN_NAME}.csv")
    try:
        resin_log = pd.read_csv(resin_log_path)
    except FileNotFoundError:
        return []

    return resin_log.loc[resin_log["Result"].isna(), "Label"].tolist()


def _update_test_results(results: str) -> str | pd.DataFrame:
    """
    HTTP request to update the test results for resin `info.EXP_TEST_RESIN_NAME`

    :param str results:
        a string of test results separated by commas; each result is either "Y" or "N"
    :return str | pd.DataFrame:
        a message or a DataFrame of the updated test results
        <br>str: a message if the resin log file does not exist
        <br>pd.DataFrame: a DataFrame of the updated test results
    """
    results = [res.strip().lower() if res.strip() else "" for res in results.split(",")]

    _true_vals = {"yes", "y", "true", "t", "1"}
    _false_vals = {"no", "n", "false", "f", "0"}
    results_lst = []
    for res in results:
        if res in _true_vals:
            results_lst.append("Y")
        elif res in _false_vals:
            results_lst.append("N")
        else:
            results_lst.append("")
    results = results_lst
    nresults = len(results)

    resin_log_path = os.path.join(info.EXP_DATA_DIR, f"{info.EXP_TEST_RESIN_NAME}.csv")
    try:
        resin_log = pd.read_csv(resin_log_path)
    except FileNotFoundError:
        return (
            f"The resin log file `{info.EXP_TEST_RESIN_NAME}.csv` does not exist. "
            f"You need to access `/test?next=y` first."
        )

    # get the first `nresults` rows with nan values in the `Result` column
    nan_rows = resin_log[resin_log["Result"].isna()].index.tolist()
    results = results[: len(nan_rows)] + [""] * max(0, len(nan_rows) - nresults)
    if not all(results[: len(nan_rows)]):
        random_choices = random.choices(["Y", "N"], weights=[0.6, 0.4], k=len(nan_rows))
        results = [res or rand for res, rand in zip(results, random_choices)]

    # update the `Result` column with the test results
    _cols = ["PrepStart", "PrepEnd", "ImageStart", "ImageEnd"]
    _now = datetime.now().astimezone(info.EXP_TIMEZONE).isoformat()
    resin_log = resin_log.fillna("")
    for i, idx in enumerate(nan_rows):
        resin_log.loc[idx, "Result"] = results[i]
        resin_log.loc[idx, _cols] = _now

    # save the updated resin log
    resin_log.to_csv(resin_log_path, index=False)

    resin_log = resin_log.loc[nan_rows]
    resin_log.set_index("Label", inplace=True)
    return resin_log[["Initiator", "Result", "dD", "dP", "dH"]]


def _restart_test() -> None:
    """
    delete files associated with the test resin `info.EXP_TEST_RESIN_NAME`
    """
    resin_config_path = os.path.join(
        info.EXP_CONFIG_DIR, f"{info.EXP_TEST_RESIN_NAME}.toml"
    )
    resin_log_path = os.path.join(info.EXP_DATA_DIR, f"{info.EXP_TEST_RESIN_NAME}.csv")
    resin_log_history_path = os.path.join(
        info.EXP_DATA_DIR, f"{info.EXP_TEST_RESIN_NAME}_history.csv"
    )
    resin_log_prior_path = os.path.join(
        info.EXP_DATA_DIR, f"{info.EXP_TEST_RESIN_NAME}_prior.csv"
    )
    for _path in [
        resin_config_path,
        resin_log_path,
        resin_log_history_path,
        resin_log_prior_path,
    ]:
        if os.path.isfile(_path):
            os.remove(_path)
    init_resin_log(info.EXP_TEST_RESIN_NAME)


def _test(request_args: MultiDict, **kwargs: str) -> tuple[dict, int]:
    """
    test the server with a test resin `info.EXP_TEST_RESIN_NAME`
    :param MultiDict request_args:
                the request arguments
    :param str kwargs:
                the keyword arguments (will replace the request arguments if conflict)
    :return tuple[dict, int]:
        a dictionary of the response and the HTTP status code
    """

    return_dict = {}
    possible_yes_set = {"yes", "y", "true", "t", "1"}

    restart_flag = kwargs.get("restart", request_args.get("restart", "n"))
    if restart_flag.lower() in possible_yes_set:
        _restart_test()
        return_dict["restart"] = (
            f"The test resin {info.EXP_TEST_RESIN_NAME} has been restarted."
        )

    if "results" in request_args or "results" in kwargs:
        results = kwargs.get("results", request_args.get("results", ""))
        if results != "":
            status = _update_test_results(results)
            if isinstance(status, str):
                return_dict["results"] = status
                return return_dict, 405
            return_dict["results"] = status.to_dict(orient="index")

    next_flag = kwargs.get("next", request_args.get("next", "n"))
    if not next_flag.split(",")[0] in possible_yes_set:
        return return_dict, 200

    # get the next test task
    task = get_tasks_for_thread(
        thread="test",
        ability=request_args.getlist("ability", kwargs.get("ability", None)),
        in_test_mode=True,
    )
    if "," in next_flag:
        faked_results = next_flag.split(",", 1)[1]
        faked_status = _update_test_results(faked_results)
        if isinstance(faked_status, pd.DataFrame) and not faked_status.empty:
            task["fakedResults"] = faked_status.to_dict(orient="index")
        elif isinstance(faked_status, str):
            task["fakedResults"] = faked_status
    return_dict["next"] = task
    return return_dict, 200
