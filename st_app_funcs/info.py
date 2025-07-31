import os
import tempfile
import warnings
from zoneinfo import ZoneInfo


class _info:
    """
    instance for dealing with configuration settings
    """

    WORKING_DIR = os.path.dirname(os.path.dirname(__file__))
    WORKING_DIRNAME = os.path.basename(WORKING_DIR)
    if not WORKING_DIRNAME:
        WORKING_DIRNAME = "vial_detection"

    IMG_DIR = os.path.join(WORKING_DIR, "images")
    README = f"readme.json"  # IMG_DIR / README
    IMG_RESULT_DIRNAME = "analysis"
    IMG_RESULT_DIR = os.path.join(IMG_DIR, IMG_RESULT_DIRNAME)
    os.makedirs(IMG_RESULT_DIR, exist_ok=True)
    HTML_DIR = os.path.join(WORKING_DIR, "templates")
    TMP_DIR = os.path.join(tempfile.gettempdir(), WORKING_DIRNAME)
    os.makedirs(TMP_DIR, exist_ok=True)
    DATA_DIR = os.path.join(WORKING_DIR, "data")
    os.makedirs(DATA_DIR, exist_ok=True)

    # allowed image extensions for file uploads to streamlit
    ALLOWED_IMG_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"}

    # who to contact - you might need to change this in production
    CONTACT_NAME = "sijie"
    CONTACT_NAME_FULL = "Sijie Fu"
    CONTACT_LINK = "https://SijieFu.github.io/autoHSP"
    CONTACT_NAME_HYPERLINK = f"[{CONTACT_NAME}]({CONTACT_LINK})"
    CONTACT_NAME_FULL_HYPERLINK = f"[{CONTACT_NAME_FULL}]({CONTACT_LINK})"

    # configuration dir for the test materials
    EXP_CONFIG_DIR = os.path.join(DATA_DIR, "exp_config")
    os.makedirs(EXP_CONFIG_DIR, exist_ok=True)

    # experiment record data dir
    EXP_DATA_DIR = os.path.join(DATA_DIR, "exp_record")
    os.makedirs(EXP_DATA_DIR, exist_ok=True)

    # lab experiment timezone information
    EXP_TIMEZONE = ZoneInfo("America/New_York")
    EXP_TEST_RESIN_NAME = "R0"
    EXP_RESIN_NAME_REGEX = r"^R\d+$"
    EXP_RESIN_CAPTURE_REGEX = r"([R]\d+)"
    EXP_SOLVENT_NAME_REGEX = r"^S\d+$"
    EXP_SOLVENT_CAPTURE_REGEX = r"([S]\d+)"
    EXP_SAMPLE_NAME_REGEX = r"^(test.*_)?R\d+(:?S\d+){1,2}-r[\d\.]+(:?s[\d\.]+){1,2}$"

    # interface detection server (should not end with a slash)
    INTERFACE_DETECTION_SERVER = "http://localhost:5001"

    # miscillaneous
    SHOW_BALLOONS = False

    @classmethod
    def update_from_toml(cls, config_file: str, no_warning: bool = False) -> None:
        """
        Update the configuration from a TOML file.

        Only top-level keys in the TOML file will be used to update the class attributes.
        Additionally, only keys that match the existing class attributes will be updated.

        :param str config_file:
            Path to the TOML configuration file.
        :param bool no_warning:
            Whether to suppress warnings for keys that do not match existing class attributes.
        """
        import toml

        try:
            with open(config_file, "r") as f:
                config = toml.load(f)
            for key, value in config.items():
                if not isinstance(key, str) or not hasattr(cls, key):
                    if not no_warning:
                        warnings.warn(
                            f"Key `{key}` in {config_file} does not match any existing class attribute. "
                            "Skipping this key."
                        )
                    continue
                setattr(cls, key, value)
        except Exception as e:
            if not no_warning:
                warnings.warn(f"Failed to load configuration from {config_file}: {e}")


info = _info()
