import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from .info import info
from .utils import check_user_privilege, make_sphere
from .utils import _button_clicked


def view_experimental_record():
    """
    function to view the experiment records for the test materials
    """
    st.write(f"## View Past/Current Experiment Threads")
    if not os.path.isfile(os.path.join(info.IMG_DIR, info.README)):
        readme = {}
    else:
        with open(os.path.join(info.IMG_DIR, info.README), "r") as f:
            readme = json.load(f)

    record_tab, alias_tab = st.tabs([f"Resin Record", "Chemical Aliases"])

    alias_dict = {}
    resins_df = pd.read_csv(os.path.join(info.DATA_DIR, "resins.csv"))
    resins_df = resins_df[["Code", "Resin", "Type"]].dropna(
        axis=0, how="any", subset=["Code", "Resin"]
    )
    resins_df.set_index("Code", inplace=True)
    for code, name in resins_df.iterrows():
        alias_dict[code] = name["Resin"]

    solvents_df_raw = pd.read_csv(os.path.join(info.DATA_DIR, "solvents.csv"))
    solvents_df = solvents_df_raw[["Code", "Solvent"]].dropna(axis=0, how="any")
    solvents_df.set_index("Code", inplace=True)
    for code, name in solvents_df.iterrows():
        alias_dict[code] = name["Solvent"]

    # work on the synonym reference tab
    with alias_tab:
        if not any(check_user_privilege(p, prompt=False) for p in ["admin", "user"]):
            st.error(f"You are not authorized to view the true chemical names.")
        else:
            table_or_dict = st.radio(
                "View the chemical aliases as tables or a dictionary?",
                ("Table", "Dictionary"),
                index=0,
                key="view_thread_alias_table_or_dict",
                horizontal=True,
            )
            if table_or_dict == "Table":
                st.info(f"You may need to scroll/swipe to view the full tables.")
                col1, col2 = st.columns(2)
                col1.dataframe(
                    resins_df,
                    use_container_width=True,
                    key="view_thread_alias_resins_df",
                )
                col2.dataframe(
                    solvents_df,
                    use_container_width=True,
                    key="view_thread_alias_solvents_df",
                )
            else:
                st.write(alias_dict)

    exp_record_dir = info.EXP_DATA_DIR
    resins = set()

    for resin in os.listdir(exp_record_dir):
        filename, fileext = os.path.splitext(resin)
        if not fileext.lower().endswith(".csv"):
            continue
        if not re.match(info.EXP_RESIN_NAME_REGEX, filename):
            continue
        resins.add(filename)

    if len(resins) == 0:
        record_tab.warning(
            f"Oops, no resin has been run in the lab yet. Please check back later."
        )
        st.stop()

    resin_options = sorted(resins, key=lambda x: int(x[1:]) if x[1:].isdigit() else x)

    query_resin = st.query_params.get("resin", None)
    if query_resin is not None and query_resin in resin_options:
        st.session_state.select_resin = query_resin
    else:
        st.query_params.pop("resin", None)

    def _reset_resin_choices():
        _ = st.session_state.get("select_resin", None)
        if _:
            st.query_params["resin"] = _
        else:
            st.query_params.pop("resin", None)

    resin = record_tab.selectbox(
        "Select a resin to view its experimental record",
        options=resin_options,
        index=None,
        key="select_resin",
        on_change=_reset_resin_choices,
        placeholder="Select a resin",
    )

    if not resin:
        st.query_params.pop("resin", None)
        record_tab.warning(
            f"Please select a resin to view its experimental record. "
            f"For references on the chemical aliases/codenames, please refer to the 'Chemical Aliases' tab."
        )
        st.stop()

    IS_TEST_RESIN = resin == info.EXP_TEST_RESIN_NAME
    if IS_TEST_RESIN:
        record_tab.warning(
            f"NOTE: Resin `{resin}` is a test resin and has NOT been run in the lab."
        )

    try:
        from HSP.utils import get_resin_log_df

        resin_log_df = get_resin_log_df(resin).dropna(subset=["dD", "dP", "dH"])
    except:
        record_tab.error(f"Failed to load the experimental record for resin `{resin}`.")
        st.stop()

    resin_log_df["Batch#"] = "Unknown"
    resin_log_df.loc[resin_log_df["Result"].isin({np.nan, ""}), "Result"] = "P"
    with open(os.path.join(exp_record_dir, f"summary.json"), "r") as file:
        summary_log_dict = json.load(file)

    with record_tab.expander(
        f"How to interpret the experimental record for resin {resin} (click to expand)"
    ):
        st.write(f"**:red[Result]**: if the solvent is miscible with resin {resin}")
        st.write(f"- **:green[Y]** for miscible (a good mixing)")
        st.write(f"- **:red[N]** for immiscible (not mixing well)")
        st.write(f"- **:violet[P]** for pending (still being tested)")
        st.write(
            f"**:red[ResultRevised]**: if the previous **:red[Result]** was revised"
        )
        st.write(f"- **:green[0]** for no revision required and performed")
        st.write(f"- **:red[1]** for revision required and revised")
        st.write(f"- **:red[-1]** for revision required but NOT revised")
        st.write(
            f"**:red[Initiator]**: the solvent selection method used to prosope the experiment"
        )

    # plot_df will be used to plot the experimental record
    plot_columns = [
        "Label",
        "Solvent",
        "SolventAmount",
        "dD",
        "dP",
        "dH",
        "Result",
        "ResultRevised",
        "Initiator",
        "Batch#",
    ]
    plot_df = pd.DataFrame(columns=plot_columns)
    try:
        resin_log_prior = pd.read_csv(
            os.path.join(exp_record_dir, f"{resin}_prior.csv")
        )
        assert len(resin_log_prior) > 0, "No prior record found."
        record_tab.info(
            f"Resin {resin} started with the following :red[prior information] before the lab run:"
        )
        with record_tab.expander(
            f"Resin {resin}: Prior Information (N={len(resin_log_prior)}, click to expand)"
        ):
            st.dataframe(resin_log_prior, hide_index=True)

        if "Label" in resin_log_prior.columns:
            pass
        elif "Sample" in resin_log_prior.columns:
            resin_log_prior.rename(columns={"Sample": "Label"}, inplace=True)
        elif "Code" in resin_log_prior.columns:
            resin_log_prior.rename(columns={"Code": "Label"}, inplace=True)
        elif "Solvent" in resin_log_prior.columns:
            resin_log_prior["Label"] = resin_log_prior["Solvent"]
        else:
            resin_log_prior["Label"] = [
                f"Unknown prior {i}" for i in range(1, len(resin_log_prior) + 1)
            ]

        if "Solvent" not in resin_log_prior.columns:
            resin_log_prior["Solvent"] = "N.A."
        if "SolventAmount" not in resin_log_prior.columns:
            resin_log_prior["SolventAmount"] = "N.A."
        resin_log_prior["Initiator"] = "prior"
        resin_log_prior["Batch#"] = f"Batch 0 (N={resin_log_prior.shape[0]})"

        plot_df = resin_log_prior.loc[:, plot_columns]
    except FileNotFoundError:
        record_tab.info(
            f"No prior record found for resin {resin}. "
            f"The lab run started from scratch."
        )

    resin_log_batches = resin_log_df["PrepTaskId"].unique()
    for i, batch in enumerate(resin_log_batches, start=1):
        batch_mask = resin_log_df["PrepTaskId"].eq(batch)
        if batch_mask.sum() == 0:
            batch_mask = resin_log_df["PrepTaskId"].isna()
        resin_log_df["Batch#"] = f"Batch {i} (N={batch_mask.sum()})"
        log_df: pd.DataFrame = resin_log_df[batch_mask].copy()
        nsamples = log_df.shape[0]

        images_ready = False
        if not log_df["ImageEnd"].hasnans:
            status = "samples have been prepared and imaged with results uploaded to the backend server"
            images_ready = not IS_TEST_RESIN
        elif not log_df["PrepEnd"].hasnans:
            if not log_df["ImageStart"].hasnans:
                status = "samples have been prepared and are currently being imaged in the lab"
            else:
                status = "samples have been prepared and are waiting to be imaged in the lab"
        elif not log_df["PrepStart"].hasnans:
            status = "samples are currently being prepared in the lab"
        else:
            status = "samples are waiting to be prepared in the lab"

        record_tab.info(
            f":red[Experiment batch {i}] of resin {resin} has status: {nsamples} {status}."
        )
        with record_tab.expander(
            f"Resin {resin}: Experiment Batch {i} (N={nsamples}, click to expand)"
        ):
            if not images_ready:
                st.caption(":green[scroll/swipe] to view the full table")
                st.dataframe(log_df, hide_index=True)
            else:
                st.caption(
                    f":green[scroll/swipe] to view the full table, :green[use the checkbox on "
                    f"the left to select a sample and view its image]"
                )
                df_event = st.dataframe(
                    log_df,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                )
                selected_rows = df_event.selection["rows"]
                if selected_rows:
                    row_idx = selected_rows[0]
                    _label = f"{log_df.iloc[row_idx]['Label']}.jpg"
                    if _label in readme:
                        st.query_params["image"] = _label
                        st.query_params["resin"] = resin
                        st.query_params["task"] = "View uploaded experiment images"
                        st.rerun()

        if plot_df.empty:
            plot_df = log_df.loc[:, plot_columns]
        else:
            plot_df = pd.concat(
                [plot_df, log_df.loc[:, plot_columns]], ignore_index=True
            )

        try:
            thread = summary_log_dict[batch]["thread"]
            thread_exceptions = summary_log_dict.get(f"exception:{thread}", [])
            if batch in thread_exceptions:
                record_tab.error(
                    f"Batch {i} encountered an exception during the lab run. "
                    f"Were you able to resolve the issue? The task id is `{batch}`."
                )
        except:
            if resin != info.EXP_TEST_RESIN_NAME and batch not in ["", np.nan]:
                record_tab.error(
                    f"Failed to identify if batch {i} was successfully executed in the lab or not. "
                    f"Were you able to confirm its status? The task id is `{batch}`."
                )

    if plot_df.empty:
        record_tab.error(
            f"No experimental record found for resin {resin} yet. Please check back later."
        )
        st.stop()

    FINISHED = False
    try:
        from HSP.utils import get_resin_config_info, get_HSP_estimates

        resin_config_status = get_resin_config_info(resin, "priority")
        assert isinstance(resin_config_status, (int, float, np.number))
        if resin_config_status >= 1:
            msg = f":bee: The lab worker bee is :red[still working] on **resin {resin}**. Stay tuned for more updates! :bee:"
        elif resin_config_status >= 0:
            msg = (
                f":bee: The lab worker bee is :red[taking a break] (pause) from working on **resin {resin}**. "
                f"You will need to manually wake it up by changing the resin priority to 1 or higher in `{resin}.toml`. :bee:"
            )
        else:
            FINISHED = True
            msg = f":bee: The lab worker bee has :red[finished] working on **resin {resin}**. Congratulations! :bee:"
            if info.SHOW_BALLOONS and resin not in st.session_state.setdefault(
                "select_resin_balloons_shown", set()
            ):
                st.session_state["select_resin_balloons_shown"].add(resin)
                st.balloons()
    except:
        msg = ":bee: Emmm... The lab worker bee is currently in a state of :red[confusion]. Please check back later. :bee:"
    finally:
        record_tab.info(msg)

    # choose the batch to view the 3D plot
    batch_options = plot_df.loc[:, "Batch#"].unique()
    _regex = re.compile(r"^Batch (\d+)")
    _is_batch = np.array([bool(_regex.match(b)) for b in batch_options])
    _default = batch_options[~_is_batch].tolist()
    batch_options = batch_options[_is_batch].tolist()
    batch_options_dict = {int(_regex.match(b).group(1)): b for b in batch_options}
    batch_options_int = sorted(batch_options_dict.keys())

    plot_expander = record_tab.expander(
        f"Experimental Record for Resin {resin} (click to expand)", expanded=True
    )
    hsp_estimation_method = plot_expander.radio(
        "Which method to use for HSP sphere estimation?",
        options=["svm", "scipy-minimize"],
        index=0,
        key="select_hsp_estimation_method",
        horizontal=True,
    )
    if not hsp_estimation_method in ["svm", "scipy-minimize"]:
        plot_expander.error(
            f"Invalid HSP estimation method selected: `{hsp_estimation_method}`. "
            f"Please select either `svm` or `scipy-minimize`."
        )
        hsp_estimation_method = "svm"
    if not FINISHED:
        plot_expander.button(
            f":warning: Force HSP estimation for {resin} :warning:",
            on_click=_button_clicked("btn_estimate_HSP"),
            key="btn_estimate_HSP",
            help=f"Click to enable/disable the HSP sphere estimation for resin {resin}",
        )

    ESTIMATE_HSP = FINISHED or st.session_state.get("btn_estimate_HSP_clicked", False)
    if ESTIMATE_HSP:
        try:
            HSP_center, HSP_radius = get_HSP_estimates(
                resin, method=hsp_estimation_method, in_test_mode=IS_TEST_RESIN
            )
        except Exception as e:
            plot_expander.error(
                f"Failed to estimate the HSP sphere for resin {resin}. "
                f"Please check the error message: {e}"
            )
            ESTIMATE_HSP = False
            st.session_state.pop("btn_estimate_HSP_clicked", None)
        else:
            hsp_estimator_msg = (
                f"({hsp_estimation_method}) HSP estimations for resin **{resin}**: **:red[dD]**: {HSP_center[0]:.2f}, **:red[dP]**: "
                f"{HSP_center[1]:.2f}, **:red[dH]**: {HSP_center[2]:.2f}, **:red[Ro]**: {HSP_radius:.2f}"
            )

    if ESTIMATE_HSP and not FINISHED:
        plot_expander.warning(
            f"The remote lab is still working on resin :red[{resin}]. "
            f"The sphere estimation is for :red[reference] only and is :red[NOT FINAL]."
        )
        hsp_estimator_msg = (
            f":warning: :red[EXPERIMENT STILL ON-GOING] :warning: {hsp_estimator_msg}"
        )

    highest_batch = plot_expander.radio(
        "View up to which batch? (0 is for prior information)",
        options=["^"] + batch_options_int + ["$"],
        index=None,
        key="select_batch_experimental_record",
        horizontal=True,
    )
    if highest_batch is None:
        selected_batches_int = batch_options_int
    elif highest_batch == "^":
        selected_batches_int = []
    elif highest_batch == "$":
        selected_batches_int = batch_options_int
    else:
        selected_batches_int = [i for i in batch_options_int if i <= highest_batch]

    selected_batches = _default + [batch_options_dict[i] for i in selected_batches_int]
    plot_df = plot_df.loc[plot_df["Batch#"].isin(selected_batches), :]

    solvents_df_raw = solvents_df_raw.rename(columns={"Code": "Label"})
    try:
        from HSP.utils import get_which_solvents_to_use

        solvents_df_raw = solvents_df_raw[
            solvents_df_raw["Label"].isin(get_which_solvents_to_use(resin)[0])
        ]
    except:
        pass
    solvents_df_raw = solvents_df_raw.loc[:, ["Label", "dD", "dP", "dH"]]
    solvents_df_raw["Solvent"] = solvents_df_raw["Label"]
    solvents_df_raw["SolventAmount"] = "N.A."
    solvents_df_raw["Result"] = "N.A."
    solvents_df_raw["ResultRevised"] = "N.A."
    solvents_df_raw["Initiator"] = "N.A."
    solvents_df_raw["Batch#"] = "N.A."
    plot_df = pd.concat([plot_df, solvents_df_raw[plot_columns]], ignore_index=True)
    plot_df[["dD", "dP", "dH"]] = (
        plot_df[["dD", "dP", "dH"]].apply(pd.to_numeric, errors="coerce").round(2)
    )
    plot_df = plot_df.drop_duplicates(subset=["dD", "dP", "dH"], keep="first")
    plot_df = plot_df.dropna(subset=["dD", "dP", "dH"], how="any")
    plot_df["2dD"] = 2 * plot_df["dD"]
    plot_df = plot_df.fillna("N.A.")

    def _result_converter(_res: str | int) -> str:
        if _res.lower() in {"y", "yes", "t", "true", "1", 1}:
            return "miscible"
        elif _res.lower() in {"n", "no", "f", "false", "0", 0}:
            return "immiscible"
        elif _res.lower() in {"pending", "p", "-1", -1}:
            return "pending"
        return "not tested"

    color_map = {
        "miscible": "green",
        "immiscible": "red",
        "pending": "violet",
        "not tested": "blue",
    }

    def _shape_converter(_label: str) -> str:
        _nsol = len(re.findall(info.EXP_SOLVENT_CAPTURE_REGEX, _label))
        if _nsol > 1:
            return "diamond"
        return "circle"

    plot_df["Result"] = plot_df["Result"].map(_result_converter)
    plot_df["Color"] = plot_df["Result"].map(color_map)
    plot_df["Shape"] = plot_df["Label"].map(_shape_converter)

    # put dot to shape of the last selected batch
    if isinstance(highest_batch, int):
        _batch = batch_options_dict[highest_batch]
        plot_df.loc[plot_df["Batch#"] == _batch, "Shape"] += "-open"

    fig = go.Figure()
    scatter = go.Scatter3d(
        x=plot_df["2dD"],
        y=plot_df["dP"],
        z=plot_df["dH"],
        mode="markers",
        marker=dict(
            size=6,
            color=plot_df["Color"],
            # colorscale="Viridis",
            opacity=0.8,
            symbol=plot_df["Shape"],
        ),
        text=plot_df["Result"],
        customdata=plot_df[
            [
                "dD",
                "Label",
                "Solvent",
                "SolventAmount",
                "Initiator",
                "ResultRevised",
                "Batch#",
            ]
        ],
        hovertemplate=(
            "<b>Label</b>: %{customdata[1]}<br>"
            "<b>Solvent</b>: %{customdata[2]}<br>"
            "<b>SAmount</b>: %{customdata[3]}<br>"
            "<b>Initiator</b>: %{customdata[4]}<br>"
            "<b>2*dD</b>: %{x:.2f}<br>"
            "<b>dD</b>: %{customdata[0]:.2f}<br>"
            "<b>dP</b>: %{y:.2f}<br>"
            "<b>dH</b>: %{z:.2f}<br>"
            "<b>Result</b>: %{text}<br>"
            "<b>ResultRevised</b>: %{customdata[6]}<br>"
            "<b>Batch#</b>: %{customdata[6]}"
            "<extra></extra>"
        ),
    )
    fig.add_trace(scatter)

    if ESTIMATE_HSP:
        show_sphere = plot_expander.checkbox(
            f"Show the HSP sphere for resin {resin} :red[(Uncheck to inspect the points)]",
            value=True,
        )
        if show_sphere:
            HSP_center = list(HSP_center)
            HSP_center[0] = 2 * HSP_center[0]
            (x_surface, y_surface, z_surface) = make_sphere(
                *HSP_center, HSP_radius, resolution=30
            )
            hovertemplate = (
                f"HSP estimation: {resin}<br>"
                f"method: {hsp_estimation_method}<br>"
                f"2*dD: {HSP_center[0]:.2f}<br>"
                f"dD: {HSP_center[0]/2.:.2f}<br>"
                f"dP: {HSP_center[1]:.2f}<br>"
                f"dH: {HSP_center[2]:.2f}<br>"
                f"dT(radius): {HSP_radius:.2f}<extra></extra>"
            )

            fig.add_trace(
                go.Surface(
                    x=x_surface,
                    y=y_surface,
                    z=z_surface,
                    colorscale="Sunsetdark",
                    showscale=False,
                    opacity=0.1,
                    hovertemplate=hovertemplate,
                )
            )

    fig.update_layout(
        scene=dict(
            xaxis_title="2 * dD: dispersion",
            yaxis_title="dP: polarity",
            zaxis_title="dH: hydrogen bonding",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=600,
    )

    plot_expander.plotly_chart(fig, use_container_width=True)
    plot_expander.write(
        f"**:large_green_circle: :green[(green markers): miscible]**, "
        f"**:red_circle: :red[(red markers): immiscible]**, "
        f"**:large_purple_circle: :violet[(violet markers): pending]**, "
        f"**:large_blue_circle: :blue[(blue markers): not tested]**"
    )
    plot_expander.write(
        f":large_blue_circle: (circle markers): single pure solvent, "
        f":large_blue_diamond: (diamond markers): mixture solvents"
    )
    if ESTIMATE_HSP and show_sphere:
        plot_expander.info(hsp_estimator_msg)
