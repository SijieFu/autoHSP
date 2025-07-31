import os
import re
import json
import io

import streamlit as st
import cv2

from .info import info
from .utils import _button_clicked
from .utils import get_image_analysis_result, update_image_analysis_result
from .utils import visualize_analysis
from .utils import check_user_privilege


def _go_to_correct_image_analysis():
    st.query_params.task = "Correct image analysis results"
    st.session_state["select_functions"] = "Correct image analysis results"


def view_uploaded_images():
    """
    Function to view the uploaded images
    """
    st.write(f"## View Uploaded Images")
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
        if resin == "others":
            continue
        chosen_images_original.extend(images_original_dict[resin])
    if "others" in resin_choices:
        chosen_images_original.extend(images_original_dict["others"])

    if len(chosen_images_original) == 0 and len(images_original) == 0:
        st.warning(f"No images uploaded yet. Please upload images first.")
        st.stop()
    elif len(chosen_images_original) == 0:
        placeholder_text = "No images for the selected resin(s)"
    else:
        placeholder_text = "Select an image"

    format_func = lambda x: x
    if len(resin_choices) == 1 and re.match(
        info.EXP_RESIN_NAME_REGEX, resin_choices[0]
    ):
        resin = re.findall(info.EXP_RESIN_CAPTURE_REGEX, resin_choices[0])[0]
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
        _ = st.session_state.get("select_image", None)
        if _:
            st.query_params["image"] = _
        else:
            st.query_params.pop("image", None)

    image_choice = st.selectbox(
        "Select an image",
        chosen_images_original,
        index=None,
        format_func=format_func,
        on_change=_reset_image_choice,
        key="select_image",
        placeholder=placeholder_text,
    )

    if image_choice is None:
        st.query_params.pop("image", None)
        st.warning(f"Please select an image to view.")
        st.stop()

    filename = readme[image_choice]
    filemd5, fileext = os.path.splitext(filename)
    fileext = str(fileext).lstrip(".").lower() or "jpg"
    filepath = os.path.join(info.IMG_DIR, filename)
    filetype = f"image/{fileext}"

    force_rerun = st.session_state.get("btn_view_image_fix_clicked", False)
    force_rerun = force_rerun and any(
        check_user_privilege(p, prompt=False) for p in ["admin", "user"]
    )
    result_json_path = os.path.join(info.IMG_RESULT_DIR, f"{filemd5}.json")
    st.session_state["btn_view_image_fix_clicked"] = False

    if force_rerun or not os.path.exists(result_json_path):
        with st.spinner("Analyzing vials... First time may take a while..."):
            image_slot = st.empty()
            image_slot.image(filepath, caption=image_choice)
            result, response_code, response_error = get_image_analysis_result(
                file=(image_choice, open(filepath, "rb"), filetype),
                result_json_path=result_json_path,
                server=info.INTERFACE_DETECTION_SERVER,
                returnimage=False,
                force_rerun=force_rerun,
                saveflag=True,
                form_data=None,
                timeout=60,
            )

            # the post request to the interface detection server failed
            if response_code != 200:
                st.error(response_error or f"Connection error: {response_code}")
                return

            if response_error:
                st.error(response_error)
            image_slot.empty()

        # update image result log
        sample_name = os.path.splitext(image_choice)[0]
        update_success, update_msg = update_image_analysis_result(
            sample_name=sample_name, result=result
        )
        if not update_success:
            st.error(update_msg)

    annotated_image = visualize_analysis(
        img=filepath, result=result_json_path, zoomin=True
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

    analyzed_image_tab, original_image_tab = st.tabs(
        ["Annotated Image", "Original Image"]
    )
    original_image_tab.image(filepath, caption=image_choice)
    analyzed_image_tab.image(
        annotated_image, caption=f"Annotated image analysis of {image_choice}"
    )

    # guest users cannot correct the image analysis results
    if not any(check_user_privilege(p, prompt=False) for p in ["admin", "user"]):
        return

    st.button(
        f"Wrong image? Re-analyze it!",
        help=f"If you see a different image of `{image_choice}` (md5 collision), click this button to re-analyze the image.",
        on_click=_button_clicked("btn_view_image_fix", True),
    )

    st.button(
        "Wrong analysis? Correct it!",
        help=f"Correct the image analysis result of `{image_choice}` if you see any wrong miscibility results.",
        on_click=_go_to_correct_image_analysis,
    )
