import os
import json
import re
import io
import time

import streamlit as st
import cv2

from .utils import _button_clicked
from .utils import check_user_privilege
from .utils import get_image_analysis_result, update_image_analysis_result
from .utils import visualize_analysis, update_human_intervention_result
from .info import info


def _go_to_uploaded_images():
    st.query_params.task = "View uploaded experiment images"
    st.session_state["select_functions"] = "View uploaded experiment images"


def correct_image_analysis():
    """
    Function to correct the image analysis results
    """
    st.write(f"## Correct Image Analysis Results")
    if not os.path.isfile(os.path.join(info.IMG_DIR, info.README)):
        readme = {}
    else:
        with open(os.path.join(info.IMG_DIR, info.README), "r") as f:
            readme = json.load(f)

    images_md5 = []
    for image in os.listdir(info.IMG_DIR):
        if not os.path.isfile(os.path.join(info.IMG_DIR, image)):
            continue
        if (
            not os.path.splitext(image)[-1].lstrip(".").lower()
            in info.ALLOWED_IMG_EXTENSIONS
        ):
            continue
        if image != readme.get(str(readme.get(image, "")), None):
            continue
        images_md5.append(image)
    if len(images_md5) == 0:
        st.warning("No images found. Please upload images and analyze them first.")
        st.stop()
    images_original = sorted([readme[image] for image in images_md5])

    # categorize the images according to resin names
    images_original_dict = {"test": [], "others": []}
    for image in images_original:
        if image.lower().startswith("test"):
            resin = "test"
        else:
            resin_name = re.findall(info.EXP_RESIN_CAPTURE_REGEX, image)
            resin = resin_name[0] if resin_name else "others"
        images_original_dict.setdefault(resin, [])
        images_original_dict[resin].append(image)

    possible_resins = list(images_original_dict.keys())
    possible_resins_1, possible_resins_2 = [], []
    for resin in possible_resins:
        if re.match(info.EXP_RESIN_NAME_REGEX, resin):
            possible_resins_1.append(resin)
        else:
            possible_resins_2.append(resin)
    possible_resins_1 = sorted(
        possible_resins_1, key=lambda x: int(re.findall(r"(\d+)", x)[0])
    )
    possible_resins_2 = sorted(possible_resins_2)
    possible_resins = possible_resins_1 + possible_resins_2

    resins = st.query_params.get_all("resin")
    resins = [r for r in resins if r in possible_resins]
    if len(resins) == 0:
        st.query_params.pop("resin", None)
    else:
        st.session_state.select_resin = resins

    def _reset_resin_choices():
        _ = st.session_state.get("select_resin", None)
        if _:
            st.query_params["resin"] = _
        else:
            st.query_params.pop("resin", None)

    resin_choices = st.multiselect(
        "Select resin(s) to view images",
        options=possible_resins,
        default=None,
        key="select_resin",
        placeholder="Select resin(s) to view images",
        on_change=_reset_resin_choices,
    )

    if not resin_choices:
        st.query_params.pop("resin", None)
        st.warning(f"Please select at least one resin to view images.")
        st.stop()

    chosen_images_original = []
    for resin in resin_choices:
        chosen_images_original.extend(images_original_dict[resin])

    if len(chosen_images_original) == 0 and len(images_original) == 0:
        st.warning(f"No images uploaded yet. Please upload images first.")
        return
    elif len(chosen_images_original) == 0:
        placeholder_text = "No images for the selected resin(s)"
    else:
        placeholder_text = "Select an image"

    format_func = lambda x: x
    if len(resin_choices) == 1:
        resin = resin_choices[0]
        if re.match(info.EXP_RESIN_NAME_REGEX, resin):
            resin = re.findall(info.EXP_RESIN_CAPTURE_REGEX, resin)[0]
        try:
            from HSP.utils import get_resin_log_df

            resin_log_df = get_resin_log_df(resin).loc[:, ["Label", "PrepTaskId"]]
            batch_dict = resin_log_df.loc[:, "PrepTaskId"].unique()
            batch_dict = {batch: f"Batch {i}" for i, batch in enumerate(batch_dict, 1)}
            resin_log_df["Batch#"] = resin_log_df.loc[:, "PrepTaskId"].map(batch_dict)
            resin_log_df["Batch#"] = resin_log_df.apply(
                lambda row: f"[{row['Batch#']}] {row['Label']}", axis=1
            )
            resin_labels = resin_log_df.loc[:, "Label"]
            _image_choices = {
                os.path.splitext(choice)[0]: choice for choice in chosen_images_original
            }

            _resin_idx_mask = resin_labels.isin(_image_choices)
            resin_labels = resin_labels[_resin_idx_mask]
            chosen_images_original = [_image_choices[x] for x in resin_labels]
            _dict = {
                chosen_images_original[i]: resin_log_df.loc[idx, "Batch#"]
                for i, idx in enumerate(resin_labels.index)
            }
            format_func = lambda x: _dict.get(x, x)

        except Exception as e:
            st.error(f"Error: {e}")
            pass

    _image = st.query_params.get("image", None)
    if _image is not None and _image in chosen_images_original:
        st.session_state.select_image = _image
    else:
        st.query_params.pop("image", None)

    def _reset_image_choice():
        _button_clicked("btn_correct_image_analysis_human", False)()
        _ = st.session_state.get("select_image", None)
        if _:
            st.query_params["image"] = _
        else:
            st.query_params.pop("image", None)

    image_choice = st.selectbox(
        "Select an image to correct",
        chosen_images_original,
        index=None,
        format_func=format_func,
        key="select_image",
        on_change=_reset_image_choice,
        placeholder=placeholder_text,
        help="Select the image to correct the analysis results.",
    )

    if image_choice is None:
        st.query_params.pop("image", None)
        st.warning(f"Please select an image to view.")
        st.stop()

    image_choice_md5 = readme[image_choice]
    image_filepath = os.path.join(info.IMG_DIR, image_choice_md5)
    st.session_state["_correct_image_analysis"] = image_choice_md5
    image_md5, image_ext = os.path.splitext(image_choice_md5)
    image_ext = str(image_ext).lstrip(".").lower() or "jpg"

    result_json_path = os.path.join(info.IMG_RESULT_DIR, f"{image_md5}.json")
    if not os.path.isfile(result_json_path):
        with st.spinner("Analyzing vials... First time may take a while..."):
            image_slot = st.empty()
            image_slot.image(image_filepath, caption=image_choice)
            result, response_code, response_error = get_image_analysis_result(
                file=(image_choice, open(image_filepath, "rb"), f"image/{image_ext}"),
                result_json_path=result_json_path,
                server=info.INTERFACE_DETECTION_SERVER,
                returnimage=False,
                force_rerun=False,
                saveflag=True,
                form_data=None,
                timeout=60,
            )
        # the post request to the interface detection server failed
        if response_code != 200:
            st.error(response_error or f"Connection error: {response_code}")
            return

        image_slot.empty()
        if response_error:
            st.error(response_error)

        # update image result log
        sample_name = os.path.splitext(image_choice)[0]
        update_success, update_msg = update_image_analysis_result(
            sample_name=sample_name, result=result
        )
        if not update_success:
            st.error(update_msg)

    # make two tabs
    image_tab, correction_tab = st.tabs(["Image", "Vial Correction Actions"])

    annotated_image = visualize_analysis(
        img=image_filepath, result=result_json_path, zoomin=True
    )

    _success, annotated_image = cv2.imencode(f".jpg", annotated_image)
    if not _success:
        st.error(
            f"Failed to encode the annotated image, probably due to an unsupported file format. "
            "Please try again with a different image and make sure the image extension is valid."
        )
        return

    annotated_image = io.BytesIO(annotated_image.tobytes())
    annotated_image.seek(0)

    image_slot = image_tab.empty()
    image_slot.image(annotated_image, caption=f"Image of `{image_choice}`")
    image_tab.warning(
        f"The :red[**Vial Correction Actions**] are on the next tab (NOT here). "
        f"Please review the image **carefully** and decide if the current results need correction."
    )

    def _pause_resin_workflow(_resin_choices: list[str]):
        warning_slot = image_tab.empty()
        if not check_user_privilege("admin", prompt=False):
            msg = "Permission denied: Only admins can pause a workflow."
            st.toast(msg)
            with st.spinner():
                warning_slot.error(msg)
                time.sleep(3)
                warning_slot.empty()
            return

        from HSP.utils import update_resin_config_toml
        from HSP.tasks import check_resin_EOE_status

        for _resin in _resin_choices:
            if check_resin_EOE_status(_resin) > 0:
                continue
            update_resin_config_toml(_resin, "priority", "0")

        msg = f"Paused the lab experiment workflow on the selected resin(s): {', '.join(_resin_choices)}."
        st.toast(msg)
        with st.spinner():
            warning_slot.success(msg)
            time.sleep(3)
            warning_slot.empty()

    image_tab.button(
        "Go back to uploaded images",
        help="Go back to the uploaded images page",
        on_click=_go_to_uploaded_images,
    )
    pause_button = image_tab.empty()
    pause_button.button(
        "Emergency PAUSE",
        help=f"Pause the lab experiment workflow on the selected resin(s): {', '.join(resin_choices)}.",
        on_click=_pause_resin_workflow,
        args=(resin_choices,),
    )

    with open(result_json_path, "r") as f:
        image_result = json.load(f)
        image_result.setdefault("human_intervention", {})
        if not isinstance(image_result["human_intervention"], dict):
            image_result["human_intervention"] = {}

    # pick a vial to correct
    vials = list(image_result["vials"])
    if len(vials) == 0:
        correction_tab.warning(
            f"No vials found in the image analysis results. Please re-analyze the image."
        )
        return
    vial_choice = correction_tab.selectbox(
        "Which vial would you like to correct?",
        vials,
        index=0,
        key="correct_image_analysis_vial",
        on_change=_button_clicked("btn_correct_image_analysis_human", False),
        help="Select the vial to correct the analysis results.",
        placeholder=vials[0],
    )

    if vial_choice is None:
        st.stop()

    # give a warning about the current status of the vial
    is_miscible = image_result[vial_choice]["is_miscible"]
    needs_intervention = image_result[vial_choice]["needs_intervention"]
    human_intervention = image_result["human_intervention"].get(vial_choice, -1)
    correction_tab.info(
        f"Currently, vial :test_tube: :red[**{vial_choice}**] is analyzed by :robot_face: as holding "
        f"{'a :green[**miscible**]' if is_miscible else 'an :red[**immiscible**]'} resin/solvent mixture."
        f" Human intervention is {'' if needs_intervention else 'NOT '}required."
    )

    human_intervention_info_slot = correction_tab.empty()

    def _render_human_intervention_info(
        slot=human_intervention_info_slot,
        vial_choice: str = vial_choice,
        human_intervention: int = human_intervention,
    ):
        if human_intervention != -1:
            slot.warning(
                f"A human :alien: decided that the mixture in :red[**{vial_choice}**] is "
                f"{':green[**miscible**]' if human_intervention else ':red[**immiscible**]'}."
                f" Are you SURE that you would like to change this AGAIN?"
            )
        else:
            slot.info(
                f"No human :alien: has yet decided the status of the vial. Care to make a correction?"
            )

    _render_human_intervention_info(
        human_intervention_info_slot, vial_choice, human_intervention
    )

    correction_tab.button(
        "Yes, I'd like a correction!",
        key="btn_correct_image_analysis_human",
        on_click=_button_clicked("btn_correct_image_analysis_human"),
    )

    human_intervention_slot = correction_tab.empty()
    _correction_options = ["miscible", "immiscible", "rescind correction"]
    if st.session_state.get("btn_correct_image_analysis_human_clicked", False):
        if not check_user_privilege("admin", prompt=False):
            correction_tab.error(
                "Permission denied: Only admins can correct image analysis results."
            )
            st.stop()

        # ask for the correct status of the vial
        correct_status = human_intervention_slot.radio(
            "What is the correct status of the vial?",
            _correction_options,
            index=None,
            key="correct_image_analysis_status",
            help="Select the correct status of the vial.",
        )

        if not correct_status or correct_status not in _correction_options:
            st.stop()

        # update the image result
        match correct_status:
            case "miscible":
                human_intervention = 1
            case "immiscible":
                human_intervention = 0
            case "rescind correction" | _:
                human_intervention = -1
        _render_human_intervention_info(
            human_intervention_info_slot, vial_choice, human_intervention
        )

        image_result["human_intervention"][vial_choice] = human_intervention
        with open(result_json_path, "w") as f:
            json.dump(image_result, f, indent=4)

        annotated_image = visualize_analysis(
            img=image_filepath, result=image_result, zoomin=True
        )

        _success, annotated_image = cv2.imencode(f".jpg", annotated_image)
        if not _success:
            st.error(
                f"Failed to encode the annotated image, probably due to an unsupported file format. "
                "Please try again with a different image and make sure the image extension is valid."
            )
        else:
            annotated_image = io.BytesIO(annotated_image.tobytes())
            annotated_image.seek(0)
            image_slot.image(annotated_image, caption=f"Image of `{image_choice}`")

        # update image result log
        sample_name = os.path.splitext(image_choice)[0]
        update_success, update_msg = update_human_intervention_result(
            sample_name=sample_name,
            human_intervention=human_intervention,
            is_miscible=is_miscible,
        )
        if not update_success:
            correction_tab.error(update_msg)

        st.session_state.pop("btn_correct_image_analysis_human_clicked", None)
