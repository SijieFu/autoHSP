import os
import io
import requests

import numpy as np
import cv2
import streamlit as st

from .info import info
from .utils import _button_clicked
from .utils import get_image_analysis_result
from .utils import visualize_analysis


def playground():
    """
    Playground for testing
    """
    st.write(f"## Playground")
    st.write(
        f"In this page, you can test the image analysis feature yourself with images of your choice. "
        f"Here is an example link: <https://cdn11.bigcommerce.com/s-neo29sbo9q/images/stencil/1280x1280/products/7388/19317/4447B17CLR-WB__32737.1537453464.jpg>"
    )
    uploaded_image_is_valid, get_image_from_url_is_valid = False, False
    uploaded_image = st.file_uploader(
        "Choose an image for the fun ...",
        type=info.ALLOWED_IMG_EXTENSIONS,
        key="file_uploader_playground",
        help="Upload an image to analyze the vials in the image for fun!",
        on_change=_button_clicked("playground_use_image_url", False),
    )
    get_image_from_url = st.text_input(
        f"Or enter the image URL (not webpage URL; [you can pick one from here]"
        f"(https://www.google.com/search?q=vial+liquid&tbm=isch))",
        help="Enter the URL of the image to analyze the vials in the image for fun!",
        placeholder="https://example.com/image.jpg",
        autocomplete="on",
        on_change=_button_clicked("playground_use_image_url", True),
    )

    if get_image_from_url and (
        not uploaded_image
        or st.session_state.get("playground_use_image_url_clicked", False)
    ):
        with st.spinner(
            "Requesting the image from the URL... This might fail if the URL is invalid or the image is not accessible"
        ):
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:10.0) Gecko/20100101 Firefox/33.0",
                }
                response = requests.get(get_image_from_url, timeout=30, headers=headers)
                # check if the request was successful
                response.raise_for_status()
                # check if the response is an image
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise TypeError(
                        f"Invalid image file type: `{fileext}` from your URL. "
                        f"Please make sure the URL points to an image file."
                    )
                filename = get_image_from_url.split("?")[0].split("/")[-1]
                fileext = filename.split(".")[-1].lower()
                fileext = os.path.splitext(filename)[-1].lstrip(".").lower()
                if not fileext in info.ALLOWED_IMG_EXTENSIONS:
                    fileext = (
                        content_type.split("/")[-1].strip().split()[0].split(";")[0]
                    )
                    if not fileext in info.ALLOWED_IMG_EXTENSIONS:
                        raise TypeError(
                            f"Invalid image file type: `{fileext}` from your URL"
                        )
                image_binary = response.content
                filename = (
                    f"{filename}.{fileext}"
                    if not filename.endswith(f".{fileext}")
                    else filename
                )
            except requests.exceptions.ReadTimeout:
                st.warning(
                    f"Request Time Out: URL took too long to respond possibly due to their anti-scraping measures. "
                    f"Please try another URL or upload an image."
                )
            except Exception as e:
                st.warning(
                    f"Failed to fetch the image from the URL. Please try another URL or upload an image. (error: {e})"
                )
            else:
                st.session_state["playground_use_image_url_clicked"] = True
                uploaded_image_is_valid, get_image_from_url_is_valid = False, True

    if uploaded_image and (
        not get_image_from_url_is_valid
        or not st.session_state.get("playground_use_image_url_clicked", False)
    ):
        image_binary = uploaded_image.read()
        filename = uploaded_image.name
        fileext = os.path.splitext(filename)[-1].lstrip(".").lower() or "jpg"
        uploaded_image_is_valid, get_image_from_url_is_valid = True, False

    if not uploaded_image_is_valid and not get_image_from_url_is_valid:
        st.warning("Please upload an image or enter a valid image URL")
        return

    if get_image_from_url and uploaded_image:
        st.info(
            f"The image from {'your uploaded image' if uploaded_image_is_valid else 'the URL'} will be used for the analysis."
        )

    # put out some information to the user before the analysis is done
    analyzed_image_tab, original_image_tab = st.tabs(
        ["Annotated Image", "Original Image"]
    )
    original_image_tab.image(image_binary, caption=filename)

    with analyzed_image_tab:
        with st.spinner(
            "Analyzing vials... First time may take a while... Here's a cake while you wait :cake:"
        ):
            force_rerun = st.session_state.get("btn_playground_fix_clicked", False)
            st.session_state["btn_playground_fix_clicked"] = False

            analyzed_image_slot = st.empty()
            analyzed_image_slot.image(
                image_binary,
                caption=f"{filename} is still being cooked... :fire: Hang tight and enjoy the cake :cake:",
            )

            # force the interface detection server to return the result as json
            result, response_code, response_error = get_image_analysis_result(
                file=(filename, image_binary, f"image/{fileext}"),
                server=info.INTERFACE_DETECTION_SERVER,
                returnimage=False,
                force_rerun=force_rerun,
                saveflag=False,
                params={"usedefault": "true"},
                form_data=None,
                timeout=60,
            )

    # the post request to the interface detection server failed
    if response_code != 200:
        st.error(response_error or f"Connection error: {response_code}")
        return

    if response_error:
        st.error(response_error)

    annotated_image = visualize_analysis(
        img=cv2.imdecode(np.frombuffer(image_binary, np.uint8), cv2.IMREAD_COLOR),
        result=result,
        zoomin=True,
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
    analyzed_image_slot.image(
        annotated_image, caption=f"Annotated image analysis of {filename}"
    )

    analyzed_image_tab.button(
        f"Fix an error?",
        help=f"Correct the image analysis of {filename}",
        on_click=_button_clicked("btn_playground_fix", True),
    )
