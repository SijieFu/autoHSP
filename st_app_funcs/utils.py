import os
import json
from functools import wraps, partial
from typing import Any, Literal, Callable
import requests
import re

import streamlit as st
import numpy as np
import cv2

from .info import info


def _button_clicked(key: str, how: str | bool = "switch") -> Callable:
    """
    Will return a function such that when the button is clicked, the function will be called
    <br>Will create a session state variable `{key}_clicked` if it does not exist
    <br>Every time the button is clicked, the session state variable will be switched to the opposite of its current value

    :param str key:
        the key for the button
    :param str | bool how:
        how to handle the button click, default is 'switch', other options boolean True or False
    :return Callable:
        the function to monitor the button click
    """
    key = f"{key}_clicked" if not key.endswith("_clicked") else key

    def wrapper(key: str = key, how: str | bool = how):
        st.session_state.setdefault(key, False)
        st.session_state[key] = not st.session_state[key] if how == "switch" else how

    return wrapper


def check_user_privilege(privilege: str, prompt: bool = True) -> bool:
    """
    Check if the user has the privilege to access the page

        :param str privilege:
        the privilege to check
    :param bool prompt:
        whether to prompt the user if the user does not have the privilege
    :return bool:
        whether the user has the privilege to access the page
    """
    if not st.session_state.get("authentication_status", False):
        st.error(":x: You are not logged in. Please log in to access this page.")
        return False
    elif not st.session_state.get(
        f"USER_{st.session_state['username']}_IS_{privilege.upper()}", False
    ):
        if prompt:
            st.error(
                f"You do not have the :x:**{privilege}**:x: privilege to access this function/page. "
                f"If you believe this is an error, please contact **{info.CONTACT_NAME_HYPERLINK}**."
            )
        return False
    return True


def requires_privilege(
    func: callable, privilege: str | list[str], how: Literal["all", "any"] = "all"
) -> Callable:
    """
    Decorator to check if the user has the privilege to access the page

    :param function func:
        the function to decorate
    :param str | list[str] privilege:
        the privilege to check
    :param Literal["all", "any"] how:
        how to check the privileges
    :return Callable:
        the decorated function
    """
    if isinstance(privilege, str):
        privilege = [privilege]
    else:
        privilege = list(privilege)

    assert (
        len(privilege) > 0
    ), "The `privilege` parameter must be a string or a list of strings."
    assert all(
        isinstance(p, str) for p in privilege
    ), "The `privilege` parameter must be a string or a list of strings."
    assert isinstance(how, str), "The `how` parameter must be a string."
    how = {"all": all, "any": any}.get(how.lower(), all)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not how(check_user_privilege(p) for p in privilege):
            st.stop()
        return func(*args, **kwargs)

    return wrapper


requires_admin = partial(requires_privilege, privilege="admin")


def get_image_analysis_result(
    file: tuple[str, bytes, str],
    result_json_path: str | None = None,
    server: str | None = None,
    returnimage: Literal["yes", "no"] | bool = "no",
    force_rerun: Literal["yes", "no"] | bool = "no",
    saveflag: Literal["yes", "no"] | bool = "no",
    params: dict[str, Any] | None = None,
    form_data: dict[str, Any] | None = None,
    timeout: int | float = 60,
) -> tuple[dict[str, Any], int, str | None]:
    """
    Get the image analysis result from the interface detection server.

    :param tuple[str, bytes, str] file:
        the file to analyze, in the format of (filename, file_bytes, mimetype)
    :param str | None result_json_path:
        the path to save the result json file. Also used to check if the result file already exists.
        The human intervention results from previous analysis will be kept in the new result file.
    :param str | None server:
        the server to send the request to. If None, the default server will be used.
        Check `info.INTERFACE_DETECTION_SERVER` for the default server URL.
    :param Literal["yes", "no"] | bool returnimage:
        whether to return the annotated image in the result.
        If set to "yes", the result JSON will contain a key "annotated_image" with the base64 encoded image.
    :param Literal["yes", "no"] | bool force_rerun:
        whether to force rerun the analysis even if the result file already exists.
        If set to "yes", the analysis will be performed regardless of the existence of the result file.
    :param Literal["yes", "no"] | bool saveflag:
        whether to cache the result file on the server.
        If set to "yes", the server will cache the result file for future requests.
    :param dict[str, Any] | None params:
        additional query parameters to send with the request. This can be used to pass additional parameters to the server.
        Check the `params` argument in the `requests.post` function for more details.
    :param dict[str, Any] | None form_data:
        additional form data to send with the request. This can be used to pass additional parameters to the server.
        If None, no additional form data will be sent.
        Check the `interface_detection` module for the available parameters.
    :param int | float timeout:
        the timeout for the request in seconds. Default is 60 seconds.
        If the request takes longer than this, it will be aborted.

    :return dict[str, Any]:
        the analysis result as a dictionary (JSON object).
    :return int:
        the HTTP status code of the response from the server.
    :return str | None:
        any error message. If everything is fine, this will be None.
    """
    server = server or info.INTERFACE_DETECTION_SERVER
    form_data = form_data or {}
    params = params or {}

    # for non-autoHSP sample images, inform the server to use the default parameters
    sample_name = os.path.splitext(file[0])[0]
    params.setdefault(
        "usedefault", not bool(re.match(info.EXP_SAMPLE_NAME_REGEX, sample_name))
    )
    params["returnjson"] = "yes"
    params["returnimage"] = _parse_bool(returnimage)
    params["force"] = _parse_bool(force_rerun)
    params["saveflag"] = _parse_bool(saveflag)

    response = requests.post(
        server, files={"file": file}, data=form_data, params=params, timeout=timeout
    )
    response_code = response.status_code
    error_msg = None

    if response_code != 200:
        error_msg = (
            f"Something went wrong when communicating with the interface detection server. "
            f"Error message: {response.text}. "
            f"Error code: {response_code}."
        )
        return {}, response_code, error_msg

    try:
        response = response.json()
    except json.JSONDecodeError as e:
        error_msg = (
            f"Failed to decode the response from the interface detection server as JSON. "
            f"Error message: {str(e)}. "
            f"Response content: {response.text}"
        )
        return {}, response_code, error_msg

    result = response.get(file[0], {})
    if not result_json_path:
        return result, response_code, error_msg

    result.setdefault("human_intervention", {})
    if os.path.isfile(result_json_path):
        try:
            with open(result_json_path, "r") as f:
                existing_result = json.load(f)
            result["human_intervention"].update(
                existing_result.get("human_intervention", {})
            )
            if result.get("nvials", 0) != existing_result.get("nvials", 0):
                error_msg = (
                    f"Number of vials changed from {existing_result['nvials']} to {result['nvials']}. "
                    f"Please check the image and correct the analysis results if necessary."
                )
        except:
            pass

    # save the result to the file
    with open(result_json_path, "w") as f:
        json.dump(result, f, indent=4)
    return result, response_code, error_msg


def update_image_analysis_result(
    sample_name: str,
    result: str | dict[str, Any],
    vial_idx: int = 0,
) -> tuple[bool, str]:
    """
    Update the image analysis result for a specific sample and vial index, typically after running
    the `get_image_analysis_result` function.

    :param str sample_name:
        the name of the sample. Refer to `info.EXP_SAMPLE_NAME_REGEX` for the format.
        Additionally, if the sample name starts with "test", it will not be updated.
    :param str | dict[str, Any] result:
        the analysis result, either a path to a JSON file or a dictionary.
        Check the `get_image_analysis_result` function for the expected format.
    :param int vial_idx:
        the index of the vial to update (default is 0)
    :return bool:
        whether the update was successful.
        If sample name is not valid, True is returned to inducate no update is needed.
    :return str:
        a message indicating the result of the update.
    """
    if sample_name.startswith("test"):
        return True, "Sample name starts with 'test', no update is needed."
    if not re.match(info.EXP_SAMPLE_NAME_REGEX, sample_name):
        return (
            True,
            f"Not a valid experiment sample, no update is needed for sample {sample_name}.",
        )

    try:
        from HSP.api import update_image_results

        if isinstance(result, str):
            with open(result, "r") as f:
                result = json.load(f)

        vial = result["vials"][vial_idx]
        human_intervention = result["human_intervention"].get(vial, -1)
        is_miscible = result[vial]["is_miscible"]
        update_image_results(
            samples=sample_name,
            results=(
                bool(human_intervention) if human_intervention >= 0 else is_miscible
            ),
            results_are_revised=int(
                human_intervention >= 0 and int(is_miscible) != human_intervention
            ),
        )
    except Exception as e:
        return False, (
            f"Failed to update the image analysis result to the experiment records. "
            f"Error: {e}. Is :memo:**{sample_name}**:memo: a valid sample name?",
        )
    else:
        return True, f"Updated the image analysis result for sample {sample_name}."


def update_human_intervention_result(
    sample_name: str, human_intervention: int, is_miscible: bool
) -> tuple[bool, str]:
    """
    similar to `update_image_analysis_result`, but for updating the human intervention result

    :param str sample_name:
        the name of the sample. Refer to `info.EXP_SAMPLE_NAME_REGEX` for the format.
        Additionally, if the sample name starts with "test", it will not be updated.
    :param int human_intervention:
        the human intervention result

        - 1: miscible by human intervention
        - 0: not miscible by human intervention
        - -1: human intervention is not provided/needed or is rescinded
    :param bool is_miscible:
        the automatic analysis result of the vial, whether it is miscible or not.
        This should NOT be the human intervention result.

    :return bool:
        whether the update was successful.
        If sample name is not valid, True is returned to indicate no update is needed.
    :return str:
        a message indicating the result of the update.
    """
    result = {
        "vials": ["A1"],
        "human_intervention": {"A1": human_intervention},
        "A1": {"is_miscible": is_miscible},
    }
    return update_image_analysis_result(
        sample_name=sample_name, result=result, vial_idx=0
    )


def _parse_bool(value: str | bool) -> bool:
    """
    Parse a string or boolean value to a boolean value. This is designed to handle URL query parameters.

    :param str | bool value: the value to parse. Can be a literal string or a boolean.

        > Acceptable True strings (not case sensitive): "true", "yes", "1", "y", "t"
    :return bool: the parsed boolean value
    """
    if isinstance(value, str):
        return True if value.lower() in ["true", "yes", "1", "y", "t"] else False
    return bool(value)


def _get_adaptive_font_scale(
    text: str, font: int, thickness: int, max_width: int, max_height: int
) -> float:
    """
    Get the adaptive font scale for a text

    :param str text: the text to display
    :param int font: the font face
    :param int thickness: the thickness of the text
    :param int max_width: the maximum width of the text
    :param int max_height: the maximum height of the text
    :return float: the adaptive font scale
    """
    scale = 0.05
    while True:
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        if w > max_width or h > max_height:
            break
        scale += 0.05
    return scale


def visualize_analysis(
    img: str | np.ndarray,
    result: str | dict,
    zoomin: bool = False,
    save_path: str | None = None,
    force_rerun: str | bool = "false",
    vial_detection_only: bool = False,
    annotated_img_only: bool = False,
    debug: bool = False,
    title: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """
    Visualize the analysis result by annotating the image with the detected vials and their properties.

    :param str | np.ndarray img:
        the image to visualize, either the path to the image or the numpy array of the image
    :param str | dict result:
        the analysis result from the function `analyze_image`, either the path to the result json file or the dictionary
    :param bool zoomin:
        whether to zoom in the image (zoom in the detected vial instead of the whole image)
    :param str | None save_path: the path to save the result
    :param str | bool force_rerun:
        in case of found result file, whether to force rerun the visualization even if the result file exists
    :param bool vial_detection_only:
        whether to only visualize the vial detection result (without the interface detection result)
    :param bool annotated_img_only:
        whether to only visualize the annotated image (without the original image)
    :param bool debug:
        whether to show the debug information. If True, the image will be displayed in a window on the screen,
        blocking the program until the user closes the window.
    :param str | None title:
        the title of the visualization image
    :param Any kwargs:
        additional keyword arguments for visualization. For compatibility with other functions and not used in this function.
    :return np.ndarray: the annotated image
    """
    force_rerun = _parse_bool(force_rerun)

    if save_path and not force_rerun and os.path.isfile(save_path):
        if zoomin == ("zoomin" in save_path):
            return cv2.imread(save_path)

    if isinstance(img, str):
        img = cv2.imread(img)
    if isinstance(result, str):
        with open(result, "r") as f:
            result = json.load(f)

    # check the img width and height, resize to at least 720px height if necessary
    if img.shape[0] < 20 or img.shape[1] < 20:
        return img  # return the original image if it is too small

    factor: float = 1.0
    new_height = int(np.clip(img.shape[0], 480, 2160))
    factor = new_height / img.shape[0]
    if not np.isclose(factor, 1.0):
        img = cv2.resize(img, (int(img.shape[1] * factor), int(img.shape[0] * factor)))

    img_original = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    height, width = img.shape[:2]
    if result["nvials"] == 0:  # if no vial is detected
        boxwidth = np.clip(1920, width // 3, width)
        boxheight = np.clip(height // 10, min(30, height // 3), height // 2)
        boxheight = min(boxheight, boxwidth // 3)
        cv2.rectangle(img, (0, 0), (boxwidth, boxheight), (0, 0, 255), -1)
        text, text_thickness = ("No vial detected", np.clip(int(boxheight / 15), 1, 4))
        cv2.putText(
            img,
            text,
            (int(boxwidth * 0.05), int(boxheight * 0.95)),
            font,
            _get_adaptive_font_scale(
                text, font, text_thickness, int(boxwidth * 0.9), int(boxheight * 0.9)
            ),
            (255, 255, 255),
            text_thickness,
        )
        if save_path:
            cv2.imwrite(save_path, img)

        if debug:
            cv2.imshow(title or "result", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img

    height_original, width_original = result["image_dimension_height_width"]
    # check if the aspect ratio is maintained
    if not np.isclose(width_original / height_original, width / height, rtol=0.05):
        raise ValueError(
            f"Aspect ratio mismatch: [original]{width_original / height_original} vs. [current]{width / height}"
        )
    factor: float = height / height_original

    max_boxheight = 0
    for vial in result["vials"]:
        x1, y1, x2, y2 = (np.array(result[vial]["xyxy"]) * factor).astype(int)
        w, h = x2 - x1 + 1, y2 - y1 + 1
        conf = result[vial]["confidence"]
        cap_top, cap_bottom = (np.array(result[vial]["cap_abs"]) * factor).astype(int)
        interfaces = (np.array(result[vial]["interfaces_abs"]) * factor).astype(int)
        polylines = [
            (np.array(polyline) * factor).astype(int)
            for polyline in result[vial]["label_mask_abs"]
        ]

        thickness = max(1, (w + h) // 150)
        if not vial_detection_only:
            # draw the label mask in white
            for poly in polylines:
                cv2.polylines(
                    img,
                    [np.array(poly, dtype=int).reshape(-1, 1, 2)],
                    True,
                    (255, 255, 255),
                    thickness,
                )
            # draw the cap top and bottom in blue
            cv2.line(img, (x1, cap_top), (x2, cap_top), (255, 0, 0), thickness)
            cv2.line(img, (x1, cap_bottom), (x2, cap_bottom), (255, 0, 0), thickness)
            # draw the interfaces in blue
            for interface in interfaces:
                cv2.line(
                    img, (x1, interface), (x2, interface), (255, 255, 0), thickness
                )

        # draw the bounding box in red or green
        _is_miscible = bool(result[vial]["is_miscible"])
        _human_intervention = result["human_intervention"].get(vial, -1)
        _needs_intervention = result[vial]["needs_intervention"]
        if vial_detection_only:
            color = (255, 0, 0)  # blue if only vial detection
        elif _human_intervention == 1 or (_human_intervention < 0 and _is_miscible):
            color = (0, 160, 0)  # green if miscible
        else:
            color = (0, 0, 255)  # red if not miscible
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # draw vial name and confidence within a red filled rectangle with transparent text
        text = f"{vial} {conf:.2f}"
        if vial_detection_only:
            pass
        elif _needs_intervention or _human_intervention >= 0:
            if _human_intervention < 0:
                text = f"?{text}"  # human assessment is needed but not provided
                color = tuple(int(_x * 0.5) for _x in color)  # darken the color
            elif bool(_human_intervention) != _is_miscible:
                text = f"!{text}"  # human assessment is inconsistent with automatic assessment
            else:
                text = f"${text}"  # human assessment is consistent with automatic assessment

        if y1 < 30:  # too few pixels to display the label above the bounding box
            pt1 = (int(x1 * 0.85 + x2 * 0.15), 0)
            pt2 = (int(x1 * 0.15 + x2 * 0.85), min(h // 3, max(30, h // 10)))
        else:  # display the label above the bounding box
            pt1 = (x1, max(0, y1 - max(30, h // 10)))
            pt2 = (x2, y1)

        boxwidth, boxheight = pt2[0] - pt1[0] + 1, pt2[1] - pt1[1] + 1
        if boxheight > boxwidth // 3:
            boxheight = boxwidth // 3
            pt1 = (pt1[0], pt2[1] - boxheight + 1)
        text_thickness = np.clip(int(boxheight / 15), 1, 4)
        cv2.rectangle(img, pt1, pt2, color, -1)
        cv2.putText(
            img,
            text,
            (int(pt1[0] + boxwidth * 0.05), int(pt2[1] - boxheight * 0.05)),
            font,
            _get_adaptive_font_scale(
                text,
                font,
                text_thickness,
                max_width=int(boxwidth * 0.9),
                max_height=int(boxheight * 0.9),
            ),
            (255, 255, 255),
            text_thickness,
        )

        max_boxheight = max(max_boxheight, boxheight * 1.2)

    if zoomin:
        vial_height, vial_width = (
            np.array(result["vial_dimension_height_width"]) * factor
        ).astype(int)
        xx1, yy1, xx2, yy2 = (np.array(result["xyxy"]) * factor).astype(int)
        xx1, yy1, xx2, yy2 = (
            max(0, xx1 - vial_width // 4),
            max(0, yy1 - int(max_boxheight)),
            min(img.shape[1], xx2 + vial_width // 4),
            min(img.shape[0], yy2 + vial_height // 16),
        )
        img = img[yy1 : yy2 + 1, xx1 : xx2 + 1]
        img_original = img_original[yy1 : yy2 + 1, xx1 : xx2 + 1]

    # concatenate the original image and the result image together along thr longer axis
    if annotated_img_only:
        pass
    elif img.shape[0] >= img.shape[1] * 0.8:
        # vertically concatenate & add padding betteen the two images
        img = np.concatenate(
            (
                img_original,
                np.zeros((img.shape[0], 10, *[3] * (img.ndim - 2)), dtype=img.dtype),
                img,
            ),
            axis=1,
        )
    else:
        # horizontally concatenate & add padding betteen the two images
        img = np.concatenate(
            (
                img_original,
                np.zeros((10, img.shape[1], *[3] * (img.ndim - 2)), dtype=img.dtype),
                img,
            ),
            axis=0,
        )

    if save_path:
        cv2.imwrite(save_path, img)

    if debug:
        cv2.imshow(title or "result", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


def make_sphere(
    x: float | np.number,
    y: float | np.number,
    z: float | np.number,
    radius: float | np.number,
    resolution: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the coordinates for plotting a sphere centered at (x,y,z)

    :param float | np.number x:
        the x coordinate of the center of the sphere
    :param float | np.number y:
        the y coordinate of the center of the sphere
    :param float | np.number z:
        the z coordinate of the center of the sphere
    :param float | np.number radius:
        the radius of the sphere
    :param int resolution:
        the resolution of the sphere
    :return tuple[np.ndarray, np.ndarray, np.ndarray]:
        the coordinates of the sphere
    """
    u, v = np.mgrid[0 : 2 * np.pi : resolution * 2j, 0 : np.pi : resolution * 1j]
    X = radius * np.cos(u) * np.sin(v) + x
    Y = radius * np.sin(u) * np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)
