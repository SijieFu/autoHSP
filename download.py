"""
for downloading optional files/resources in addition to the main repository.
"""

import os
import requests
from typing import Iterable


def download_file(
    urls: Iterable[str],
    filename: str,
    dest_dir: str = ".",
    unzip: bool = False,
    cleanup: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Download files from the given URLs to the specified destination directory.

    NOTE: The URLs should point to the SAME resource, and the first one that succeeds will be used.

    :param Iterable[str] urls:
        An iterable of URLs to download the file from. They all need to point to the same resource.
        The first URL that succeeds will be used.
    :param str filename:
        The name of the file to save the downloaded content as.
    :param str dest_dir:
        The directory where the downloaded file will be saved. Defaults to the current working directory.
    :param bool unzip:
        If True and `filename` ends with `.zip`, the downloaded file will be unzipped and then deleted.
        Defaults to False.
    :param bool cleanup:
        If True, the downloaded file will be deleted after extraction if `unzip` is True.
        Defaults to False.
    :param bool overwrite:
        If True, existing files with the same name will be overwritten. Defaults to False.
    """
    os.makedirs(dest_dir, exist_ok=True)
    filepath = os.path.join(dest_dir, filename)
    if os.path.exists(filepath) and not overwrite:
        raise FileExistsError(
            f"File `{filepath}` already exists. "
            f"Please remove it, choose a different filename, or set `overwrite=True`."
        )

    for url in urls:
        try:
            if "google.com" in url:
                import gdown

                gdown.download(url, output=filepath, quiet=False, fuzzy=True)
            else:
                response = requests.get(url, timeout=10)
                # Raise an error for bad responses
                response.raise_for_status()

                with open(filepath, "wb") as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            file.write(chunk)
            print(f"Downloaded: `{filepath}` from `{url}`")

            if unzip and filepath.endswith(".zip"):
                import zipfile

                with zipfile.ZipFile(filepath, "r") as zip_ref:
                    zip_ref.extractall(dest_dir)
                print(f"Unzipped: `{filepath}` to `{dest_dir}`")
                if cleanup:
                    os.remove(filepath)
                    print(f"Deleted: `{filepath}` after extraction")
            return
        except ImportError as e:
            print(
                f"Failed to import the required module: {e}. "
                f"Please ensure all dependencies are installed."
            )
        except requests.RequestException as e:
            print(f"Failed to download from `{url}`: {e}")

    raise RuntimeError("Failed to download the file from all provided URLs.")


def download_images_zip(
    urls: Iterable[str] | None = None,
    filename: str = "images.zip",
    dest_dir: str = ".",
    unzip: bool = True,
    cleanup: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Download a zip file containing images from the given URLs.

    :param Iterable[str] | None urls:
        An iterable of URLs to download the zip file from. The first URL that succeeds will be used.
        If None, the default URLs will be used.
    :param str filename:
        The name of the zip file to save the downloaded content as. Defaults to "images.zip".
    :param str dest_dir:
        The directory where the downloaded zip file will be saved. Defaults to the current working directory.
    :param bool unzip:
        If True, the downloaded zip file will be unzipped. Defaults to True.
    :param bool cleanup:
        If True, the downloaded zip file will be deleted after extraction if `unzip` is True.
        Defaults to False.
    :param bool overwrite:
        If True, existing files with the same name will be overwritten. Defaults to False.
    """
    google_template = "https://drive.google.com/uc?id={}"
    github_template = (
        "https://github.com/SijieFu/autoHSP/releases/download/v1.0.0-image/{}"
    )
    if urls is None:
        urls = [
            google_template.format("1B6lmzqUwZa2AnPA3JCMA4DTyVrzNDh93"),
            github_template.format("image.zip"),
        ]

    download_file(
        urls=urls,
        filename=filename,
        dest_dir=dest_dir,
        unzip=unzip,
        cleanup=cleanup,
        overwrite=overwrite,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download additional files/resources for the project."
    )
    parser.add_argument(
        "-i", "--images", action="store_true", help="Download the images zip file."
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite existing files if they already exist.",
    )
    parser.add_argument(
        "-a", "--all", action="store_true", help="Download all resources."
    )
    parser.add_argument(
        "-c",
        "--cleanup",
        action="store_true",
        help="Delete downloaded zip files after extraction.",
    )
    args = parser.parse_args()

    actions = {
        "images": download_images_zip,
    }
    for action, func in actions.items():
        if not args.all and not getattr(args, action, False):
            continue
        print(f"\n===== Downloading `{action}` =====")
        try:
            func(cleanup=args.cleanup, overwrite=args.overwrite)
            print(f">>> {action.capitalize()} downloaded successfully.")
        except FileExistsError as e:
            print(
                f">>> Resource already exists. "
                f"Use `-o` or `--overwrite` if you want to overwrite it: {e}"
            )
        except Exception as e:
            print(f">>> Failed to download {action}: {e}")
