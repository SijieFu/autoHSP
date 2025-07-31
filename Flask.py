"""
entry point for the flask app to handle HTTP requests from the designated (remote) lab for autoHSP

API endpoints:
- `/`: a simple hello world message to check if the server is up
- `/next`: get the next experiment task for a resin
- `/notify`: update experiment status
- `/upload`: upload an image from an experiment
- `/test`: testing `/next` (experiment scheduling) with a fake resin
"""

import os
import json, base64
from collections import defaultdict
import requests as Req
from urllib.parse import unquote
from io import BytesIO
import logging
from typing import Any

import cv2
from flask import Flask, request
from flask import jsonify, send_file, render_template
from werkzeug.middleware.proxy_fix import ProxyFix

from Flask_utils import get_image_md5_hash, _parse_bool
from Flask_utils import update_image_analysis_result
from Flask_utils import visualize_analysis, get_interface_detection_parameter
from Flask_test import _test, _get_test_unknowns
from utils import send_notification_email
from info import info

info.check_smtp_server(timeout=2)

from HSP.api import get_tasks_for_thread
from HSP.api import task_succeeded


app = Flask(__name__, template_folder=info.HTML_DIR, static_folder=info.IMG_DIR)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)


@app.errorhandler(404)
def page_not_found(e):
    # Log the error
    app.logger.warning("Route not found: %s", request.url)
    return (
        jsonify({"message": f"404 NOT FOUND: `{request.url}` is not a valid route."}),
        404,
    )


@app.route("/", methods=["GET"])
def index():
    """
    HTTP request to check if the server is up and running
    """
    return jsonify({"message": "Hi there! The autoHSP Flask server is up and running!"})


@app.route("/next", methods=["GET"])
def next():
    """
    HTTP request to get the next experiment task for a resin
    """
    possible_yes_set = {"yes", "y", "Y", "true", "t", "T", "1"}

    thread = request.args.get("thread", "test")
    if thread.lower().startswith("test"):
        return_dict, code = _test(request_args=request.args, next="y")
        return jsonify(return_dict), code

    concurrent_path = os.path.join(info.DATA_DIR, "latest_tasks.json")
    try:
        with open(concurrent_path, "r") as file:
            concurrent_tasks = json.load(file)
    except:
        concurrent_tasks = {}

    use_last_task = (
        request.args.get("uselast", "no").lower() in possible_yes_set
        and thread in concurrent_tasks.keys()
    )
    if use_last_task:
        task = concurrent_tasks[thread]
    else:
        task = get_tasks_for_thread(
            thread=thread, ability=request.args.getlist("ability", None)
        )
        concurrent_tasks[thread] = task
        _task_type = task.get("task", "test")
        if _task_type in ["prep", "image"]:
            send_notification_email(
                status=_task_type,
                update=(
                    f"Thread `{thread}` just picked up a new `{_task_type}` task "
                    f"for the following samples: {task.get('samples', 'N.A.')}."
                ),
            )
        elif _task_type == "EOE":
            request_str = unquote(request.query_string.decode())
            send_notification_email(
                status="FINISHED",
                update=(
                    f"An End-Of-Experiment (EOE) token has been returned to thread `{thread}` "
                    f"for the following request: `{request_str}`."
                ),
            )

    with open(concurrent_path, "w") as file:
        json.dump(concurrent_tasks, file, indent=4)

    logging.info(
        f"Returning task \"{task.get('taskId', 'N.A.')}\" to thread \"{thread}\" "
        f'to the following request: "{request.path}?{request.query_string.decode()}"'
    )
    return jsonify(task)


@app.route("/notify", methods=["GET", "POST"])
def notify():
    """
    HTTP request to update experiment status
    """
    status = request.args.get("status", "exception").lower()
    if status not in {"success", "exception", "error"}:
        return (
            jsonify(
                {
                    "message": f'Invalid status: received {status}, but not in ["success", "exception", "error"]'
                }
            ),
            405,
        )

    match status:
        case "success":
            message = request.args.get(
                "message", "An experiment has been successfully completed in the lab!"
            )
            if request.args.get("task", "test") != "test":
                task_succeeded(
                    task=request.args.get("task", "", type=str),
                    taskid=request.args.get("taskId", "", type=str),
                    protocolid=request.args.get("protocolId", "", type=str),
                    samples=request.args.getlist("samples", type=str),
                    sampleids=request.args.getlist("sampleIds", type=str),
                    time=request.args.get("time", None, type=str),
                )
        case "exception":
            message = request.args.get(
                "message",
                "An [exception] occurred during the experiment and your attention is required!",
            )
        case "error":
            message = request.args.get(
                "message",
                "An [error] occurred during the experiment and your attention is required!",
            )
        case _:
            message = request.args.get(
                "message", "An unknown status has been received!"
            )
            pass

    message = f"{message}<br>{unquote(request.query_string.decode())}"
    send_notification_email(status, message)
    return (
        jsonify(
            {"message": f"Your `{status}` notification has been successfully processed"}
        ),
        200,
    )


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """
    HTTP request to upload an image from an experiment
    """

    arguments = defaultdict(lambda: False)
    arguments.update(request.form.to_dict())
    # URL parameters overwrite form parameters
    arguments.update(request.args.to_dict())

    # parameters designed for autoHSP images
    kwargs_for_analysis = {
        # Flask_utils.analyze_image --> interface_detection.vial_detection.detect_vials
        "weights": "yolov8n.pt",
        "name": "vial",
        "conf": 0.7,
        "iou": 0.5,
        "device": "cpu",
        "max_det": 100,
        "width_expansion": 0.025,
        "alignment_cutoff": 0.5,
        "config": "",
        # Flask_utils.analyze_image --> interface_detection.vial_detection.detect_vials --> interface_detection.vial_contour.get_interfaces
        "over_exposure_threshold": 200,
        "bg_correction": "True",
        "bg_tolerance": "40",
        "bg_resolution": 0.02,
        "bg_sobel_ratio": 0.55,
        "cap_ratio": "0.1,0.25",
        "cap_target": "-1",
        "cap_tolerance": "30",
        "vessel_sobel_threshold": 31,
        "sobel_threshold": 24,
        "sobel_xy_ratio": 1.0,
        "dilation": "True",
        "phase_resolution": 0.08,
        "boundary_resolution": "0.4,0.1",
        "label_correction": "True",
        "label_low": "150",
        "label_high": "220",
        "label_check_gray": "False",
        "interface_signal": 0.55,
    }
    # default parameters for the general image analysis
    if use_default := _parse_bool(arguments.pop("usedefault", False)):
        kwargs_for_analysis.update(
            {
                "width_expansion": 0.0,
                "bg_tolerance": "30",
                "cap_ratio": "0.1,0.3",
                "cap_tolerance": "40",
                "vessel_sobel_threshold": 63,
                "sobel_xy_ratio": 0.75,
                "phase_resolution": 0.06,
                "boundary_resolution": "0.1,0.15",
                "interface_signal": 0.6,
            }
        )
    for key, value in kwargs_for_analysis.items():
        if key in arguments:
            try:
                kwargs_for_analysis[key] = type(value)(arguments[key])
            except:
                pass

    if request.method == "GET":
        return render_template(
            "img_upload_get.html",
            parameters=kwargs_for_analysis,
            use_default=use_default,
        )

    if "file" not in request.files:
        return f"No file part in request", 405

    file = request.files.get("file", False)
    filename = request.form.get("filename", "").strip() or file.filename
    filetype = request.form.get("mimetype", "").strip() or file.mimetype

    if not file or not filename:
        return "No selected file", 405

    if not file or filetype.lower() not in info.ALLOWED_EXTENSIONS:
        return "Invalid file type", 405

    # how to handle the image in the server
    saveflag = _parse_bool(
        arguments.pop("saveflag", False) or arguments.pop("nosaveflag", False) or "yes"
    )  # default to yes
    force_rerun = _parse_bool(arguments.pop("force", False) or "no")  # default to no
    # visualization parameters
    return_image = _parse_bool(arguments.pop("returnimage", False))
    zoomin_initial = _parse_bool(arguments.pop("zoomin", False))
    zoomin = return_image and zoomin_initial
    return_json = _parse_bool(
        arguments.pop("returnjson", False)
        or arguments.pop("donot_returnjson", False)
        or "yes"
    )  # default to yes
    kwargs_for_analysis["bg_correction"] = _parse_bool(
        arguments["bg_correction"] or kwargs_for_analysis["bg_correction"]
    ) and not _parse_bool(arguments["no_bg_correction"] or "no")
    kwargs_for_analysis["dilation"] = _parse_bool(
        arguments["dilation"] or kwargs_for_analysis["dilation"]
    ) and not _parse_bool(arguments["no_dilation"] or "no")
    kwargs_for_analysis["label_correction"] = _parse_bool(
        arguments["label_correction"] or kwargs_for_analysis["label_correction"]
    ) and not _parse_bool(arguments["no_label_correction"] or "no")
    kwargs_for_analysis["label_check_gray"] = _parse_bool(
        arguments["label_check_gray"] or kwargs_for_analysis["label_check_gray"]
    )

    file_md5 = get_image_md5_hash(file)
    file_ext = (
        os.path.splitext(filename)[-1]
        or filetype.split("/")[-1].strip().split()[0].split(";")[0]
    )
    file_ext = file_ext.lstrip(".").lower() or "jpg"
    save_dir = info.IMG_DIR if saveflag else info.TMP_DIR
    os.makedirs(save_dir, exist_ok=True)
    file_save_name = f"{file_md5}.{file_ext}"
    file_save_path = os.path.join(save_dir, file_save_name)

    file.seek(0)
    if force_rerun or not os.path.isfile(file_save_path):
        file.save(file_save_path)
    file.close()

    # Analyze the image, result_path = info.IMG_RESULT_DIR / {file_md5}.json
    result_dir = os.path.join(save_dir, info.IMG_RESULT_DIRNAME)
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{file_md5}.json")

    readme_path = os.path.join(save_dir, info.README)
    try:
        with open(readme_path, "r") as f:
            readme_data = json.load(f)
    except FileNotFoundError:
        readme_data = {}

    # clean up the previous files if the file has been uploaded before
    _prev_file = readme_data.get(filename, "")
    _prev_md5 = os.path.splitext(_prev_file)[0]
    if _prev_file and _prev_md5 != file_md5:
        readme_data.pop(filename, None)
        readme_data.pop(_prev_file, None)
        if os.path.isfile(_tmp_file := os.path.join(save_dir, _prev_file)):
            os.remove(_tmp_file)
        if os.path.isfile(_tmp_file := os.path.join(result_dir, f"{_prev_md5}.json")):
            os.remove(_tmp_file)
        if os.path.isfile(
            _tmp_file := os.path.join(result_dir, f"annotated_zoomin_{_prev_file}")
        ):
            os.remove(_tmp_file)
        if os.path.isfile(
            _tmp_file := os.path.join(result_dir, f"annotated_{_prev_file}")
        ):
            os.remove(_tmp_file)

    readme_data[file_save_name] = filename
    readme_data[filename] = file_save_name
    with open(readme_path, "w") as f:
        json.dump(readme_data, f, indent=4)

    previous_result = defaultdict(dict)
    argument_mistmatch = True
    try:
        with open(result_path, "r") as f:
            previous_result.update(json.load(f))
        if (
            get_interface_detection_parameter(**kwargs_for_analysis)
            == previous_result["parameters"]
        ):
            argument_mistmatch = False
    except:
        pass

    # redirect the request to the HSP server
    if force_rerun or argument_mistmatch:
        response = Req.post(
            # force the HSP server to return the result as json
            f"{info.INTERFACE_DETECTION_SERVER}?returnjson=yes&returnimage=False",
            files={"file": (filename, open(file_save_path, "rb"), filetype)},
            data=kwargs_for_analysis,
            timeout=60,
        )

        # the post request to the interface detection server failed
        if response.status_code != 200:
            return (
                f"Something went wrong when communicating with the interface detection server: {response.text}",
                500,
            )

        # the post request to the interface detection server succeeded
        response: dict[str, Any] = response.json()
        response.setdefault(filename, {})
        response[filename].setdefault("human_intervention", {})
        # preserve the previous human intervention results
        response[filename]["human_intervention"].update(
            previous_result["human_intervention"]
        )

        image_result = response[filename]
        with open(result_path, "w") as f:
            json.dump(image_result, f, indent=4)

        _human_intervention_needed = False
        for vial in image_result.get("vials", []):
            if image_result.get(vial, {}).get("needs_intervention", False):
                if image_result["human_intervention"].get(vial, -1) < 0:
                    _human_intervention_needed = True
                    break
        if _human_intervention_needed:
            send_notification_email(
                status="exception`image",
                update=f"Your attention is required ASAP to correct the image analysis results for image `{request.args.get('sample', 'N.A.')}`.",
            )

        # deal with result update if the request is from the designated lab
        if "sample" in request.args:
            update_image_analysis_result(
                request_args=request.args, image_result=image_result
            )

    # the result json must be cached or was just created throught POST request
    with open(result_path, "r") as f:
        result = json.load(f)

    # deal with result update if the request is from the designated lab
    if "sample" in request.args:
        update_image_analysis_result(request_args=request.args, image_result=result)

    result = {filename: result}
    if not return_image:  # return json analysis
        return jsonify(result)

    return_image = visualize_analysis(
        file_save_path,
        result[filename],
        zoomin=zoomin,
        force_rerun=force_rerun,
        debug=False,
    )

    default_img_ext = "jpg"
    _success, return_image = cv2.imencode(f".{default_img_ext}", return_image)
    if not _success:
        result["error"] = "Failed to encode image"
        return jsonify(result)

    return_image_io = BytesIO(return_image.tobytes())
    return_image_io.seek(0)

    return_image_name = f"annotated{'_zoomin' if zoomin else ''}_{filename}"
    if not return_json:
        return send_file(
            return_image_io,
            mimetype=f"image/{default_img_ext}",
            as_attachment=False,
            download_name=return_image_name,
        )

    # return both json and image
    result["annotated_image"] = {
        "file": base64.b64encode(return_image_io.read()).decode("utf-8"),
        "filename": return_image_name,
        "filetype": f"image/{default_img_ext}",
        "mimetype": f"image/{default_img_ext}",
    }
    return jsonify(result)


@app.route("/test", methods=["GET", "POST"])
@app.route("/test/", methods=["GET", "POST"])
def test():
    """
    test the server with a test resin `info.EXP_TEST_RESIN_NAME`
    """
    possible_yes_set = {"yes", "y", "true", "t", "1"}
    if request.method == "GET":
        _next, _next_results = f"{request.args.get('next', 'n')},".split(",", 1)
        return render_template(
            "test_run.html",
            restart=request.args.get("restart", "n").lower() in possible_yes_set,
            results=request.args.get("results", ""),
            unknown_samples=_get_test_unknowns(),
            next=_next.lower() in possible_yes_set,
            next_results=_next_results,
        )

    kwargs = request.form.to_dict()
    if "next" in kwargs and kwargs.get("next_results", ""):
        kwargs["next"] = (
            f"{kwargs.get('next', 'n').strip(',')},{kwargs.get('next_results', '').strip(',')}"
        )
    return_dict, code = _test(request_args=request.args, **kwargs)
    return jsonify(return_dict), code


if __name__ == "__main__":
    logging.basicConfig(filename="flask.log", level=logging.INFO)
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=True, threaded=False)
