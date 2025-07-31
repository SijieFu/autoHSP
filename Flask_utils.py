"""
Utilities for Flask
"""

import os
import hashlib
import json
import logging
from typing import Any

import numpy as np
import cv2
from PIL import Image
from werkzeug.datastructures import MultiDict

from HSP.api import update_image_results


def get_md5_hash(file: str | bytes, chunk_size: int = 4096) -> str:
    """
    Get the MD5 hash of a file (designed for images)

    :param str | bytes file: the filepath or the file content
    :param int chunk_size: chunk size in kb
    :return str: the MD5 hash of the file
    """
    md5 = hashlib.md5()
    if isinstance(file, str):
        with open(file, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
    elif isinstance(file, bytes):
        md5.update(file)
    return md5.hexdigest()


def get_image_md5_hash(image: str | bytes) -> str:
    """
    Get the MD5 hash of an image, regardless of the file format or EXIF data.

    :param str | bytes image: the image path or the image content.
        Note: `image` should be a valid argument for `PIL.Image.open()`.
    :return str: the MD5 hash of the image
    """
    img = Image.open(image)
    return hashlib.md5(img.tobytes()).hexdigest()


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


def get_interface_detection_parameter(
    weights: str = "yolov8n.pt",
    name: str = "vial",
    conf: float = 0.7,
    iou: float = 0.5,
    device: str = "cpu",
    max_det: int = 100,
    width_expansion: float = 0.0,
    alignment_cutoff: float = 0.5,
    config: str | list[int] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    This function returns the parameters (configuration) for running the interface detection algorithm
    for the computer vision module of autoHSP.

    Check `interface_detection.vial_detection.detect_vials` for more details on how these parameters are used.

    :param str weights:
        the weight filename. By default, weights should be under the `runs` directory.
    :param str name:
        name of the class to detect. This is the name of the class when training or tuning the object detection model.
    :param float conf:
        confidence threshold for the detection.
    :param float iou:
        IoU threshold for the detection. Should be between 0 and 1. The IoU threshold is used to filter out overlapping detections.
        The higher the IoU threshold, the more strict the filtering is.
    :param str device:
        device to use for inference. By default, it is set to "cpu". If you have a GPU, you can set it to "cuda" to use the GPU for inference.
    :param int max_det:
        maximum number of detections to return. This is used to limit the number of detections returned by the model.
    :param float width_expansion:
        width expansion for the vial. This is used to expand the vial width by a certain percentage to both left and right.
        Should be between 0 and 0.25. If 0, no expansion is applied.
    :param float alignment_cutoff:
        cutoff for alignment ratio, should be between 0 and 1 (recommended >= 0.5).
        For the detected vials, how to group them into rows. If the vertical alignment ratio of the vials is greater than this value, they are considered to be in the same row.
        See `align_detected_vials` for more details
    :param str | list[int] | None config:
        how many vials in each row; if not provided, the function will try to infer. See `align_detected_vials` for more details
    :param bool return_parameters:
        if True, return the parameters used for detection. This is useful to compile the parameters for comparison with previous runs,
        without having to go through the actual detection process.
    :param Any **kwargs:
        additional keyword arguments for `vial_contour.get_interfaces`
    :return dict[str, Any]:
        If `return_parameters` is True, return a dictionary containing the parameters used for detection.

        If `return_parameters` is False, return a dictionary containing the detected vials and their interfaces.

        Specifically, for the key `human_intervention`, the value will be a dictionary with:
            - key as the string index for a vial
            - value as if human intervention was performed on the vial
                -1: not assessed by human,
                0: not miscible by human intervention,
                1: miscible by human intervention.
    """
    # Check input arguments
    if not isinstance(weights, str) or not weights:
        raise ValueError("`weights` must be a non-empty string")
    if not 0 <= conf <= 1:
        raise ValueError("confidence threshold `conf` must be between 0 and 1")
    if not 0 <= iou <= 1:
        raise ValueError("`iou` threshold must be between 0 and 1")
    if device not in ["cpu", "cuda"]:
        raise ValueError("`device` must be 'cpu' or 'cuda'")
    if not isinstance(max_det, int) or max_det <= 0:
        raise ValueError("`max_det` must be a positive integer")
    if not isinstance(width_expansion, float) or not 0 <= width_expansion <= 0.25:
        raise ValueError("`width_expansion` must be a float between 0 and 0.25.")
    if not 0 <= alignment_cutoff <= 1:
        raise ValueError("alignment_cutoff must be between 0 and 1, recommended >= 0.5")
    if isinstance(config, str) and config != "":
        config = list(map(int, config.split(",")))

    max_det = min(max_det, 300)
    if isinstance(config, list):
        max_det = sum(config)

    return {
        "vial_detection": {
            "weights": weights,
            "name": name,
            "conf": conf,
            "iou": iou,
            "device": device,
            "max_det": max_det,
            "alignment_cutoff": alignment_cutoff,
            "width_expansion": width_expansion,
            "config": config,
        },
        "interface_detection": kwargs,
    }


def update_image_analysis_result(request_args: MultiDict, image_result: dict) -> None:
    """
    update the image analysis result to the experiment records

    :param MultiDict request_args:
        the request arguments
    :param dict image_result:
        the image analysis result
    """
    if not "sample" in request_args:
        raise ValueError("No sample is provided")
    if (
        "tid" not in request_args
        or not request_args.get("tid")
        or request_args.get("tid").lower().startswith("test")
    ):
        return

    samples = request_args.getlist("sample")
    nsamples = len(samples)
    nvials = image_result.get("nvials", 0)
    if nsamples != nvials:
        logging.warning(
            f"Number of samples does not match the number of vials detected. "
            f"Image (md5:{image_result.get('md5_hash', 'n.a.')}) detected with {image_result.get('nvials', 0)} vials, "
            f"but by truth, there are {nsamples} samples: {', '.join(samples)}"
        )
    human_intervention = image_result.get("human_intervention", {})
    results, results_are_revised = [True] * nsamples, [-1] * nsamples
    for i in range(nsamples):
        if i >= nvials:
            break
        vial = image_result["vials"][i]
        result = bool(image_result[vial]["is_miscible"])
        result_needs_intervention = -int(bool(image_result[vial]["needs_intervention"]))
        if human_intervention.get(vial, -1) != -1:
            result = bool(human_intervention[vial])
            result_needs_intervention = -result_needs_intervention
        results[i] = result
        results_are_revised[i] = result_needs_intervention
    update_image_results(samples, results, results_are_revised)
