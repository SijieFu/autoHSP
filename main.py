from collections import defaultdict

import streamlit as st

from info import info
from login import integrated_login, add_preauthorized_email

info.check_login_config()

from st_app_funcs import (
    view_experimental_record,
    view_uploaded_images,
    correct_image_analysis,
    playground,
)
from st_app_funcs import view_solvent_library
from st_app_funcs.info import info as secondary_info


def INVALID_FUNCTION():
    """
    Function for invalid function
    """
    st.warning(f"Please select a valid function from the dropdown list.")
    st.stop()


if __name__ == "__main__":
    import os

    path = os.path.join(info.WORKING_DIR, ".streamlit", "app.toml")
    info.update_from_toml(path, no_warning=True)
    secondary_info.update_from_toml(path, no_warning=True)

    st.set_page_config(
        page_title="autoHSP",
        page_icon=":test_tube:",
        layout="centered",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get help": info.CONTACT_LINK,
            "About": (
                f"## Welcome to the autoHSP web app!\n"
                f"This app is developed for visualizing and analyzing vials images for the autoHSP project.\n\n"
                f"GitHub repository: {info.PROJECT_HYPERLINK}"
            ),
        },
    )

    integrated_login(config_path=info.CONFIG_FILEPATH)

    functions = defaultdict(lambda: INVALID_FUNCTION)
    functions.update(
        {
            "View past/current experimental records": view_experimental_record,
            "View uploaded experiment images": view_uploaded_images,
            "Correct image analysis results": correct_image_analysis,
            "View a comprehensive solvent library": view_solvent_library,
            "Playground": playground,
            "Pre-authorize user to register": add_preauthorized_email,
        }
    )

    options = list(functions.keys())
    task = st.query_params.get("task", None)
    if task is not None and task in options:
        st.session_state.select_functions = task
    else:
        st.query_params.pop("task", None)

    def _reset_task_choice():
        _ = st.session_state.get("select_functions", None)
        if _:
            st.query_params["task"] = _
        else:
            st.query_params.pop("task", None)

    choice = st.selectbox(
        "What can I do for you today? (Select from the dropdown list below)",
        options,
        index=None,
        key="select_functions",
        on_change=_reset_task_choice,
        help="Select an option from the dropdown list to perform your desired task.",
    )

    if choice is None:
        st.error("Please select your desired task from the dropdown list.")
        st.stop()

    functions[choice]()
