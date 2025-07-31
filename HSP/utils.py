"""
Utility functions for the HSP solver
"""

import os
import re
import shutil
import tomllib
import json
from collections import abc
import warnings
from typing import Any, Literal

import numpy as np
import pandas as pd

from .info import info
from .hsp_math import interpret_HSP


def init_resin_config(
    resin: str, force: bool = False, verbose: bool = True, in_test_mode: bool = False
) -> None:
    """
    Initialize the configuration file for the resin.

    - NOTE: you must also manually update the configuration file after it is created.
    - NOTE: you need to delete the `_copy` text in the configuration file name.

    :param str resin:
        the resin name
    :param bool force:
        force to create the configuration file
    :param bool verbose:
        print the message if the configuration file is created
    :param bool in_test_mode:
        whether the function is in test mode. In other words, no `_copy` text will be added to the configuration file name.
    """
    assert isinstance(resin, str) and len(resin) > 0, "Invalid resin name"
    default_toml_path = os.path.join(
        info.EXP_CONFIG_DIR, f"default.{info.EXP_CONFIG_EXT}"
    )
    if not os.path.exists(default_toml_path):
        raise FileNotFoundError("Default configuration file not found")

    config_filename = f"{resin}.{info.EXP_CONFIG_EXT}"
    if os.path.exists(os.path.join(info.EXP_CONFIG_DIR, config_filename)):
        if force:
            shutil.move(
                os.path.join(info.EXP_CONFIG_DIR, config_filename),
                os.path.join(
                    info.EXP_CONFIG_DIR,
                    f"{resin}_previous.{info.EXP_CONFIG_EXT}",
                ),
            )
        else:
            if verbose:
                print(
                    f"Configuration file for resin `{resin}` already exists as `{config_filename}`. "
                    f"You can set argument `force=True` to overwrite the configuration file."
                )
            return

    config_filename = f"{resin}{'_copy' * (not in_test_mode)}.{info.EXP_CONFIG_EXT}"
    shutil.copy(
        default_toml_path,
        os.path.join(info.EXP_CONFIG_DIR, config_filename),
    )

    if verbose:
        print(
            f"Resin configuration file `{config_filename}` created. "
            f"Please update the configuration file and rename it to `{resin}.{info.EXP_CONFIG_EXT}`."
        )


def _deep_update(original: dict | abc.Mapping, update: dict | abc.Mapping) -> dict:
    """
    Recursively update a dictionary

    :param dict original:
        the original dictionary
    :param dict update:
        the dictionary to update with
    :return dict:
        the updated dictionary
    """
    for key, value in update.items():
        if isinstance(value, abc.Mapping):
            original[key] = _deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


def get_resin_config_dict(resin: str, in_test_mode: bool = False) -> dict:
    """
    Get the configuration dictionary for the resin

    :param str resin:
        the resin name
    :param bool in_test_mode:
        whether the function is in test mode.
    :return dict:
        the configuration dictionary
    """
    default_toml_path = os.path.join(
        info.EXP_CONFIG_DIR, f"default.{info.EXP_CONFIG_EXT}"
    )
    if not os.path.exists(default_toml_path):
        raise FileNotFoundError("Default configuration file not found")
    with open(default_toml_path, "rb") as f:
        default_toml = tomllib.load(f)

    resin_toml_path = os.path.join(
        info.EXP_CONFIG_DIR, f"{resin}.{info.EXP_CONFIG_EXT}"
    )
    if not os.path.exists(resin_toml_path):
        init_resin_config(resin, force=False, verbose=False, in_test_mode=in_test_mode)
    if not os.path.exists(resin_toml_path):
        raise FileNotFoundError(
            f"Configuration file for resin `{resin}` not found. A copy of the default configuration file has been created. "
            f"Please update the configuration file and rename it to `{resin}.{info.EXP_CONFIG_EXT}`."
        )
    with open(resin_toml_path, "rb") as f:
        resin_toml = tomllib.load(f)

    return _deep_update(default_toml, resin_toml)


def update_resin_config_toml(resin: str, key: str, value: str) -> None:
    """
    Update the configuration file for the resin
    :param str resin:
        the resin name
    :param str key:
        the key to update
    :param str value:
        the value to update with
    """
    assert key in info.EXP_CONFIG_MUTABLE_KEYS, f"Key `{key}` is not mutable"
    resin_toml_path = os.path.join(
        info.EXP_CONFIG_DIR, f"{resin}.{info.EXP_CONFIG_EXT}"
    )
    if not os.path.exists(resin_toml_path):
        raise FileNotFoundError(f"Configuration file for resin `{resin}` not found")

    with open(resin_toml_path, "r") as f:
        lines = f.readlines()

    no_match_found = True
    for i, line in enumerate(lines):
        if re.match(f"^{key}\\s*=\\s*.*", line):
            lines[i] = f"{key} = {value}\n"
            no_match_found = False
            break

    with open(resin_toml_path, "w") as f:
        if no_match_found:
            f.write(
                f"# added by `update_resin_config_toml` function\n{key} = {value}\n\n"
            )
        f.writelines(lines)


def get_resin_config_info(
    resin: str | dict, which: str, default: Any = None, in_test_mode: bool = False
) -> Any:
    """
    Get the configuration information for the resin

    :param str resin:
        the resin name
    :param str which:
        the configuration information to get
    :param Any default:
        the default value if the configuration information is not found
    :param bool in_test_mode:
        whether the function is in test mode (the resin is a test resin).
    :return Any:
        the configuration information
    """
    if isinstance(resin, str):
        resin_config_dict = get_resin_config_dict(resin, in_test_mode=in_test_mode)
    else:
        assert isinstance(resin, dict), "Invalid resin configuration"
        resin_config_dict = resin

    match which:
        case "ramount":
            resin_info = resin_config_dict.get("resin", {})
            return resin_info.get("amount", default or info.EXP_RESIN_AMOUNT)
        case "samount":
            solvent_info = resin_config_dict.get("solvent", {})
            return solvent_info.get("amount", default or info.EXP_SOLVENT_AMOUNT)
        case "amounts" | "amount":
            ramount = get_resin_config_info(resin_config_dict, default or "ramount")
            samount = get_resin_config_info(resin_config_dict, default or "samount")
            return ramount, samount
        case "random_state" | "random_seed" | "seed":
            return resin_config_dict.get("random_seed", default or info.EXP_RANDOM_SEED)
        case "mix_step_size":
            _size = resin_config_dict.get(
                "mix_step_size", default or info.EXP_MIX_STEP_SIZE
            )
            return min(0.5, max(0.02, _size))
        case "precision":
            _precision = resin_config_dict.get(
                "precision", default or info.EXP_PRECISION
            )
            return max(1e-3, _precision)
        case "priority":
            _default = default or info.EXP_RESIN_PRIORITY
            return resin_config_dict.get("priority", _default)
        case "strategy":
            _default = default or info.EXP_SELECTION_STRATEGY
            return resin_config_dict.get("strategy", _default)
        case "strictness":
            _default = default or info.EXP_SELECTION_STRICTNESS
            return resin_config_dict.get("strictness", _default)
        case "explore_temp":
            _default = default or info.EXP_EXPLORE_TEMP
            return resin_config_dict.get("explore_temp", _default)
        case "distance_percentile":
            _default = default or info.EXP_DISTANCE_PERCENTILE
            return resin_config_dict.get("distance_percentile", _default)
        case _:
            if which not in resin_config_dict.keys():
                raise ValueError(f"Invalid configuration information `{which}`")
    return resin_config_dict.get(which, default)


def get_resins_df() -> pd.DataFrame:
    """
    Get the DataFrame for the resins

    :return pd.DataFrame:
        the DataFrame for the resins
    """
    df_path = os.path.join(info.DATA_DIR, "resins.csv")
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, sep=",", header=0, index_col=False).dropna(
            axis=0, how="all"
        )
    else:
        raise FileNotFoundError("Resin DataFrame not found in `data` directory")

    assert (
        info.EXP_RESIN_COLNAME in df.columns
    ), f"Column `{info.EXP_RESIN_COLNAME}` not found in the `resins.csv` file. It is used as identifiers."

    return df


def get_solvents_df() -> pd.DataFrame:
    """
    Get the DataFrame for the solvents

    :return pd.DataFrame:
        the DataFrame for the solvents
    """
    df_path = os.path.join(info.DATA_DIR, "solvents.csv")
    if os.path.exists(df_path):
        df = pd.read_csv(df_path, sep=",", header=0, index_col=False).dropna(
            axis=0, how="all"
        )
    else:
        raise FileNotFoundError("Solvent DataFrame not found in `data` directory")

    assert (
        info.EXP_SOLVENT_COLNAME in df.columns
    ), f"Column `{info.EXP_SOLVENT_COLNAME}` not found in the `solvents.csv` file. It is used as identifiers."
    assert all(col in df.columns for col in ["dD", "dP", "dH"]), (
        f"Column(s) `dD`, `dP`, `dH` not found in the `solvents.csv` file."
        + f"These columns are used for the Hansen solubility parameters."
    )
    assert (
        df[["dD", "dP", "dH"]].notna().all().all()
    ), f"Invalid HSP values in the `solvents.csv` file. Remove or replace the invalid/missing HSP values."

    return df


def update_solvents_df(df: pd.DataFrame, keep_previous: bool = True):
    """
    Update the DataFrame for the solvents

    :param pd.DataFrame df:
        the DataFrame to update with
    :param bool keep_previous:
        keep the previous DataFrame if exists
    """
    df_path = os.path.join(info.DATA_DIR, "solvents.csv")
    if os.path.exists(df_path) and keep_previous:
        shutil.move(
            df_path,
            os.path.join(
                info.DATA_DIR,
                "solvents_previous.csv",
            ),
        )

    df.to_csv(df_path, sep=",", index=False, header=True)


def _parse_solvent_match_regex(
    df: pd.DataFrame, match_regex: list[dict], default_col: str = "Code"
) -> pd.Index:
    """
    Parse the solvent match regex to get the index of the solvents

    :param pd.DataFrame df:
        the DataFrame for the solvents
    :param list[dict] match_regex:
        the list of regex to match the solvents
    :return pd.Index:
        the index of the solvents that match the regex
    """
    assert all(
        isinstance(d, dict) and "regex" in d.keys() for d in match_regex
    ), "Invalid match regex format"

    match_index = pd.Index([])
    for d in match_regex:  # `d` for dictionary
        col = d.get("colname", default_col)
        match_index = match_index.union(
            df[df[col].astype(str).str.match(d["regex"])].index
        )

    return match_index


def get_which_solvents_to_use(resin: str) -> tuple[list[str], list[str]]:
    """
    Get the compatible solvents and required solvents for the resin

    :param str resin:
        the resin name
    :return tuple:
        the compatible solvents and required solvents for the resin
    """
    resin_config_dict = get_resin_config_dict(resin)
    about_solvents = resin_config_dict.get("solvent", {})
    solvents_default_col = about_solvents.get("colname", info.EXP_SOLVENT_COLNAME)
    solvents_include = about_solvents.get("include", [])
    solvents_exclude = about_solvents.get("exclude", [])
    solvents_required = about_solvents.get("required", [])

    solvents_df = get_solvents_df()
    if "Priority" in solvents_df.columns:
        solvents_df = solvents_df.sort_values("Priority", ascending=False)

    solvents_include_index = _parse_solvent_match_regex(
        solvents_df, solvents_include, solvents_default_col
    )
    solvents_exclude_index = _parse_solvent_match_regex(
        solvents_df, solvents_exclude, solvents_default_col
    )
    solvents_required_index = _parse_solvent_match_regex(
        solvents_df, solvents_required, solvents_default_col
    )

    compatible_solvents_index = solvents_include_index.difference(
        solvents_exclude_index
    )
    required_solvents_index = compatible_solvents_index.intersection(
        solvents_required_index
    )

    compatible_solvents = solvents_df.loc[
        compatible_solvents_index, solvents_default_col
    ]
    required_solvents = solvents_df.loc[required_solvents_index, solvents_default_col]
    return compatible_solvents.to_list(), required_solvents.to_list()


def init_summary_log(force: bool = False):
    """
    Initialize the log file for all lab experiments

    :param bool force:
        force to (re)create the log file
    """
    if os.path.exists(info.EXP_DATA_SUMMARY_PATH):
        if force:
            shutil.move(
                info.EXP_DATA_SUMMARY_PATH,
                os.path.join(
                    info.EXP_DATA_DIR,
                    "summary_previous.json",
                ),
            )
        else:
            return

    summary = {
        "taskid:EOE": info.EXP_TASK_EOE,
        "taskid:pause": info.EXP_TASK_PAUSE,
    }
    with open(info.EXP_DATA_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)


def get_summary_log_json() -> dict:
    """
    Get the summary JSON log for the lab experiments

    :return dict:
        the JSON for the summary log
    """
    if not os.path.exists(info.EXP_DATA_SUMMARY_PATH):
        init_summary_log(force=False)

    with open(info.EXP_DATA_SUMMARY_PATH, "r") as f:
        return json.load(f)


def update_summary_log_json(summary: dict, keep_previous: bool = False):
    """
    Update the summary JSON log for the lab experiments

    :param dict summary:
        the summary log to update with
    :param bool keep_previous:
        keep the previous log file if exists
    """
    if os.path.exists(info.EXP_DATA_SUMMARY_PATH) and keep_previous:
        shutil.move(
            info.EXP_DATA_SUMMARY_PATH,
            os.path.join(
                info.EXP_DATA_DIR,
                "summary_previous.json",
            ),
        )

    with open(info.EXP_DATA_SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=4)


def init_resin_log(resin: str, force: bool = False):
    """
    Initialize the log file for the resin that will be tested in the lab

    :param str resin:
        the resin name
    :param bool force:
        force to (re)create the log file
    """
    assert isinstance(resin, str) and len(resin) > 0, "Invalid resin name"

    log_path = os.path.join(info.EXP_DATA_DIR, f"{resin}.csv")
    if os.path.exists(log_path):
        if force:  # move the existing log file to the previous log file
            shutil.move(
                log_path,
                os.path.join(
                    info.EXP_DATA_DIR,
                    f"{resin}_previous.csv",
                ),
            )
        else:
            return

    columns = [
        "Label",
        "Resin",
        "ResinAmount",
        "Solvent",
        "SolventAmount",
        "dD",
        "dP",
        "dH",
        "Initiator",
        "Result",
        "ResultRevised",
        "LabId",
        "PrepTaskId",
        "PrepByThread",
        "PrepStart",
        "PrepEnd",
        "PrepProtocolId",
        "ImageTaskId",
        "ImageByThread",
        "ImageStart",
        "ImageEnd",
        "ImageProtocolId",
    ]
    df = pd.DataFrame(columns=columns)
    df.to_csv(log_path, sep=",", index=False, header=True)


def get_resin_log_df(resin: str) -> pd.DataFrame:
    """
    Get the experiment records for a resin as a pandas DataFrame

    :param str resin:
        the resin name
    :return pd.DataFrame:
        the DataFrame for the resin log
    """
    if not isinstance(resin, str) and resin:
        raise ValueError("Invalid resin name. It should be a non-empty string.")

    log_path = os.path.join(info.EXP_DATA_DIR, f"{resin}.csv")
    if not os.path.exists(log_path):
        # create the log file if not exists
        init_resin_log(resin, force=False)

    df = pd.read_csv(log_path, sep=",", header=0, index_col=False)
    if "Result" in df.columns:
        df["Result"] = df["Result"].fillna("").astype(str)
    if "ResultRevised" in df.columns:
        # Convert the column to numeric, forcing errors to NaN
        df["ResultRevised"] = pd.to_numeric(df["ResultRevised"], errors="coerce")

        # Convert the column to integers, preserving NaN values
        df["ResultRevised"] = df["ResultRevised"].astype("Int64")

    # check file integrity (disabled for now)
    # if a image is taken with `ImageEnd` but not `Result`, then raise an error
    # _results = df[["ImageEnd", "Result"]].isna().values.reshape(-1, 2)
    # if np.any(_results[:, 0] != _results[:, 1]):
    #     raise ValueError(
    #         "Invalid log file. If an image is taken, then the `Result` must be provided. Vice versa."
    #     )
    return df


def update_resin_log_df(resin: str, df: pd.DataFrame, keep_previous: bool = False):
    """
    Update the experiment records for a resin with a pandas DataFrame

    :param str resin:
        the resin name
    :param pd.DataFrame df:
        the DataFrame to update with
    :param bool keep_previous:
        keep the previous log file if exists
    """
    if not isinstance(resin, str) and resin:
        raise ValueError("Invalid resin name. It should be a non-empty string.")

    log_path = os.path.join(info.EXP_DATA_DIR, f"{resin}.csv")
    if os.path.exists(log_path) and keep_previous:
        log_history_path = os.path.join(
            info.EXP_DATA_DIR,
            f"{resin}_history.csv",
        )
        with open(log_path, "r") as f:
            previous_lines = f.readlines()
        with open(log_history_path, "a") as f:
            f.writelines(previous_lines)

    if "ResultRevised" in df.columns:
        # Convert the column to numeric, forcing errors to NaN
        df["ResultRevised"] = pd.to_numeric(df["ResultRevised"], errors="coerce")

        # Convert the column to integers, preserving NaN values
        df["ResultRevised"] = df["ResultRevised"].astype("Int64")
    df.to_csv(log_path, sep=",", index=False, header=True)


def init_resin_for_exp(*resins: str | int, force: bool = False):
    """
    Initialize the resin(s) for experiment

    :param str *resins:
        the resin name(s) to initialize (e.g., "R1", "R2", "R3")

        You can also pass integers, which will be converted to strings with a prefix "R" (e.g., 1 -> "R1")
    :param bool force:
        force to (re)create the log file, configuration file, and summary file
    """
    init_summary_log(force=force)

    for resin in resins:
        if isinstance(resin, int):
            resin = f"R{resin}"
        assert isinstance(resin, str) and len(resin) > 0, "Invalid resin name"
        init_resin_config(resin, force=force, verbose=False)
        init_resin_log(resin, force=force)


def compose_experiment_label(
    resin: str,
    ramount: int | float | str,
    solvent: str | list[str],
    samount: int | float | str | list[int | float | str],
) -> str:
    """
    compose the label for the experiment given the resin and solvent information

    :param str resin:
        the resin name
    :param int | float ramount:
        the resin amount
    :param str | list[str] | str solvent:
        the solvent name(s)
    :param int | float | str | list[int | float | str] samount:
        the solvent amount(s)
    :return str:
        the label for the experiment, which will has the format `{component_part}-{amount_part}`

        - component_part: the resin and solvent names concatenated together (the solvent names are sorted)
        - amount_part: the resin amount (with a leading `r`) and solvent amounts (with a leading `s`) concatenated together.
        - Kept to the second decimal
    """
    ramount = float(ramount)
    if not isinstance(solvent, (list, tuple)):
        solvent = solvent.split(":")
    if not isinstance(samount, (list, tuple)):
        if isinstance(samount, str):
            samount = samount.split(":")
        else:
            samount = [samount]
    samount = list(map(float, samount))

    assert re.match(info.EXP_RESIN_NAME_REGEX, resin) is not None, "Invalid resin name"
    assert all(
        re.match(info.EXP_SOLVENT_NAME_REGEX, s) is not None for s in solvent
    ), "Invalid solvent name"

    assert len(solvent) == len(
        samount
    ), "Length of `solvent` and `samount` must be the same"

    # order solvent and samount according to the solvent name
    solvent, samount = zip(
        *sorted(zip(solvent, samount), key=lambda x: int(re.findall(r"\d+", x[0])[-1]))
    )

    # keep ramount and samout to the second decimal
    component_part = f"{resin}:" + ":".join(solvent)
    amount_part = f"r{ramount:.2f}:" + ":".join(map(lambda x: f"s{x:.2f}", samount))
    return f"{component_part}-{amount_part}"


def extract_resins_from_samples(
    samples: str | list[str] | tuple[str],
    check_single_resin: bool = False,
) -> str | list[str]:
    """
    extract the resin names from the sample names

    :param str | list[str] | tuple[str] samples:
        the sample name(s)
    :param bool check_single_resin:
        check if the samples contain only one resin name
    :return str | list[str]:
        the resin name(s)

        - return `str` if `check_single_resin` is `True` or if the input `samples` is a single string
    """
    return_single_str = check_single_resin
    if not isinstance(samples, (list, tuple)):
        samples = [samples]
        return_single_str = True
    samples = [str(sample) for sample in samples]

    resins = []
    for sample in samples:
        resin = re.findall(info.EXP_RESIN_CAPTURE_REGEX, sample)
        assert (
            len(resin) == 1
        ), "sample name must contain one and only one resin name of the form 'R#'"
        resins.append(resin[0])

    if check_single_resin:
        assert len(set(resins)) == 1, "Samples must contain only one resin name"
    return resins[0] if return_single_str else resins


def get_solvent_HSP(
    solvent: str | list[str],
    samount: int | float | str | list[int | float | str] = None,
) -> tuple[float, float, float]:
    """
    get the Hansen solubility parameters for a solvent or a mixture of solvents

    :param str | list[str] solvent:
        the pure solvent name(s) in the solvent or solvent mixture. These should be code names
        of the form "S#". If a mixture of solvents is provided, it should be a list of strings.
    :param int | float | str | list[int | float | str] samount:
        the corresponding solvent amount(s) for the individual `solvent`s
    :return tuple[float, float, float]:
        the Hansen solubility parameters (HSPs) for the solvent or solvent mixture.
        The first value is dD, the second value is dP, and the third value is dH
    """
    if not isinstance(solvent, (list, tuple)):
        solvent = solvent.split(":")
    solvents_df = get_solvents_df()
    solvents = solvents_df.set_index(info.EXP_SOLVENT_COLNAME)
    solvents_hsp = solvents.loc[solvent, ["dD", "dP", "dH"]].values

    if samount is None:
        return solvents_hsp.mean(axis=0)

    if not isinstance(samount, (list, tuple)):
        if isinstance(samount, str):
            samount = samount.split(":")
        else:
            samount = [samount]
    samount = list(map(float, samount))
    assert (stotal := sum(samount)) > 1e-3, "Invalid solvent amount"
    sratio = np.array(samount) / stotal

    assert len(solvent) == len(
        samount
    ), "Length of `solvent` and `samount` must be the same"

    return np.dot(solvents_hsp.T, sratio.reshape(-1, 1)).flatten()


def parse_current_solvent_HSP(
    solvents: list[str] | None = None,
    compute_mix_matrix: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse the current solvents and their Hansen solubility parameters

    :param list[str] | None solvents:
        the list of solvents to parse. If `None`, all solvents in the database will be parsed.
    :param bool compute_mix_matrix:
        compute the HSP matrix for the mixtures
    :return np.ndarray:
        the HSPs os the `solvents` (if provided) or all solvents in the database
    :return np.ndarray:
        the mix matrix of the solvents (if two solvents can be mixed to form a new solvent)

        - shape: (nsolvents, nsolvents); dtype: bool
    """
    solvents_df = get_solvents_df()
    solvents_df.set_index(info.EXP_SOLVENT_COLNAME, inplace=True)
    if solvents is not None:
        solvents_df = solvents_df.loc[solvents, :]

    solvents = solvents_df.index
    nsolvents = len(solvents)

    solvents_HSP = solvents_df.loc[:, ["dD", "dP", "dH"]].values
    solvents_mix_matrix = np.ones((nsolvents, nsolvents), dtype=bool)
    np.fill_diagonal(solvents_mix_matrix, False)

    if "IncompatibleWith" not in solvents_df.columns or not compute_mix_matrix:
        return solvents_HSP, solvents_mix_matrix

    # fill na entries with empty string
    solvents_df["IncompatibleWith"] = solvents_df["IncompatibleWith"].fillna("")
    for i, solvent in enumerate(solvents_df.index):
        incomp_with = solvents_df.loc[solvent, "IncompatibleWith"].split(",")
        # get the index of the incompatible solvents in solvents (if any)
        incomp_indices = np.where(np.isin(solvents, incomp_with))[0]
        if len(incomp_indices) > 0:
            solvents_mix_matrix[i, incomp_indices] = False
            solvents_mix_matrix[incomp_indices, i] = False
    return solvents_HSP, solvents_mix_matrix


def get_current_results_for_resin(
    resin: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    gather the current results for the resin from both `{resin}_prior.csv` and `{resin}.csv`

    :param str resin:
        the resin name
    :return np.ndarray:
        the HSPs of solvents that are compatible/miscible with the resin
    :return np.ndarray:
        the HSPs of solvents that are incompatible/immiscible with the resin
    :return np.ndarray:
        the HSPs of pending solvents (not yet tested but were selected for testing)
    """
    columns = ["dD", "dP", "dH", "Result"]

    log_dir = info.EXP_DATA_DIR
    if os.path.exists(os.path.join(log_dir, f"{resin}_prior.csv")):
        prior_df = pd.read_csv(
            os.path.join(log_dir, f"{resin}_prior.csv"), sep=",", index_col=False
        )
        assert all(
            col in prior_df.columns for col in columns
        ), "Invalid prior log file provided"
        prior_df = prior_df[columns].dropna(axis=0, how="any")
    else:
        prior_df = pd.DataFrame(columns=columns)

    resin_log_df = get_resin_log_df(resin)
    assert all(
        col in resin_log_df.columns for col in columns
    ), "Invalid resin log file provided"
    resin_log_df = resin_log_df[columns].dropna(axis=0, how="any")

    if resin_log_df.empty:
        resin_log_df = prior_df
    elif not prior_df.empty:
        resin_log_df = pd.concat([prior_df, resin_log_df], axis=0)

    compatible_HSPs = resin_log_df[resin_log_df["Result"].isin({"Y", "y", "1", 1})]
    incompatible_HSPs = resin_log_df[resin_log_df["Result"].isin({"N", "n", "0", 0})]
    pending_HSPs = resin_log_df[
        ~resin_log_df.index.isin(compatible_HSPs.index.union(incompatible_HSPs.index))
    ]

    return (
        compatible_HSPs.loc[:, columns[:3]].values.reshape(-1, 3),
        incompatible_HSPs.loc[:, columns[:3]].values.reshape(-1, 3),
        pending_HSPs.loc[:, columns[:3]].values.reshape(-1, 3),
    )


def update_incompatible_solvents(cutoff: float = 15.0):
    """
    update the `IncompatibleWith` column in the solvents DataFrame based on the Hansen solubility parameters (HSPs)
    and the cutoff value for the Hansen distance

    :param float cutoff:
        the default cutoff value for the Hansen distance, if the `dT` column has a missing value.
    """
    solvents_df = get_solvents_df()
    # add the `IncompatibleWith` column if not exists
    if "IncompatibleWith" not in solvents_df.columns:
        solvents_df["IncompatibleWith"] = ""
    else:
        solvents_df["IncompatibleWith"] = solvents_df["IncompatibleWith"].fillna("")
    if "dT" not in solvents_df.columns:
        _cols = solvents_df.columns
        _ind = _cols.get_loc("dH")
        solvents_df["dT"] = 15.0
        solvents_df = solvents_df[
            _cols[: _ind + 1].to_list() + ["dT"] + _cols[_ind + 1 :].to_list()
        ]
    else:
        solvents_df["dT"] = solvents_df["dT"].fillna(cutoff)

    # append the `dT` after `dH` column
    if os.path.exists(os.path.join(info.DATA_DIR, "solvent_library.csv")):
        solvent_library_df = pd.read_csv(
            os.path.join(info.DATA_DIR, "solvent_library.csv"),
            sep=",",
            header=0,
            index_col=False,
        )
        # using data from `InChIKey` column to match two DataFrames and append `dT` values from `solvent_library.csv` to `solvents.csv`
        solvent_library_df.set_index("InChIKey", inplace=True, drop=False)
        solvents_df.set_index("InChIKey", inplace=True, drop=False)
        for inchikey in solvents_df.index:
            if inchikey in solvent_library_df[solvent_library_df["dT"].isna()].index:
                solvents_df.loc[inchikey, "dT"] = solvent_library_df.loc[inchikey, "dT"]
        solvents_df.reset_index(drop=True, inplace=True)
        # round `dT` to the fisrt decimal
        solvents_df["dT"] = solvents_df["dT"].round(1)

    from .hsp_utils import calc_comb_HSP_distances

    solvents = solvents_df[info.EXP_SOLVENT_COLNAME].values
    solvent_HSPs = solvents_df[["dD", "dP", "dH"]].values
    solvent_HSPs[:, 0] *= 2.0
    distances = calc_comb_HSP_distances(solvent_HSPs, solvent_HSPs)
    incompatibles = []
    for sol, dist, (thresh, incomp) in zip(
        solvents, distances, solvents_df[["dT", "IncompatibleWith"]].values
    ):
        new_incomp = solvents[dist >= thresh].tolist()
        incomp = [s for s in str(incomp).split(",") if s]
        incomp += [s for s in new_incomp if s not in incomp and s != sol]
        incompatibles.append(",".join(incomp))
    solvents_df["IncompatibleWith"] = incompatibles
    update_solvents_df(solvents_df, keep_previous=True)


def get_HSP_estimates(
    resin: str,
    method: Literal["scipy-minimize", "svm"] = "scipy-minimize",
    in_test_mode: bool = False,
) -> tuple[tuple[float, float, float], float]:
    """
    get the estimated HSPs for the resin and the cutoff value for the Hansen distance

    :param str resin:
        the resin name
    :param Literal["scipy-minimize", "svm"] method:
        the method to use for estimating the cutoff sphere radius
    :param bool in_test_mode:
        whether the function is in test mode with a test resin.
    :return tuple[float, float, float]:
        the estimated Hansen solubility parameters (HSPs) of the target resin
    :return float:
        the estimated cutoff sphere radius for the Hansen distance (miscible sphere radius)
    """
    # check if reached the end of the experiment
    resin_config_dict = get_resin_config_dict(resin, in_test_mode=in_test_mode)
    priority = get_resin_config_info(
        resin_config_dict, "priority", in_test_mode=in_test_mode
    )
    random_seed = get_resin_config_info(
        resin_config_dict, "seed", in_test_mode=in_test_mode
    )

    if priority >= 0:
        warnings.warn(
            f"Resin `{resin}` has NOT yet reached the end of the experiment. "
            f"The current HSP estimates are based on the current results and are not final. "
            f"Be cautious when using these estimates."
        )
    compat_HSP, incompat_HSP, _ = get_current_results_for_resin(resin)
    return interpret_HSP(
        compatible_HSPs=compat_HSP,
        incompatible_HSPs=incompat_HSP,
        random_state=random_seed,
        method=method,
    )
