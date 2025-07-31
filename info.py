import os
import tempfile
import smtplib
import warnings
from zoneinfo import ZoneInfo


class _Info:
    """
    instance for dealing with configuration settings
    """

    # `Flask.py` configuration
    WORKING_DIR = os.path.dirname(__file__)
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

    # allowed MIME types for file uploads
    ALLOWED_EXTENSIONS = {
        "image/png",
        "image/jpg",
        "image/jpeg",
        "image/gif",
        "image/svg",
        "image/bmp",
        "image/tiff",
        "image/webp",
        "application/json",
        "application/xml",
        "application/zip",
        "application/pdf",
        "text/plain",
    }

    # Streamlit configuration
    # APP_NAME must correspond to `server.baseUrlPath` in `.streamlit/config.toml`
    APP_NAME = "HSP"
    TIMEZONE = ZoneInfo("America/New_York")

    # login configuration file .yaml
    CONFIG_FILEPATH = os.path.join(WORKING_DIR, ".streamlit", "login.yaml")
    ENABLE_CAPTCHA = True
    ENABLE_TWO_FACTOR_AUTH = True

    # username regex for registration
    USERNAME_REGEX = r"^[a-zA-Z][a-zA-Z0-9_]{2,19}$"
    # email regex for registration
    EMAIL_REGEX = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
    # elevated privileges for the streamlit app
    SPECIAL_PRIVILEGES: list[str] = ["admin", "inviter"]

    # who to contact - you might need to change this in production
    CONTACT_NAME = "sijie"
    CONTACT_NAME_FULL = "Sijie Fu"
    CONTACT_LINK = "https://github.com/SijieFu/autoHSP"
    CONTACT_NAME_HYPERLINK = f"[{CONTACT_NAME}]({CONTACT_LINK})"
    CONTACT_NAME_FULL_HYPERLINK = f"[{CONTACT_NAME_FULL}]({CONTACT_LINK})"
    # email sender, default from HSP@{DOMAIN_NAME}
    SEND_EMAIL_FROM = "HSP"
    # set `DOMAIN_NAME` to your actual domain name in production
    DOMAIN_NAME = "localhost"

    # general project information
    PROJECT_NAME = "autoHSP: Autonomous Determination of Hansen Solubility Parameters in an Automated Lab"
    PROJECT_URL = "https://github.com/SijieFu/autoHSP"
    PROJECT_HYPERLINK = f"[SijieFu/autoHSP]({PROJECT_URL})"

    # configuration dir for the test materials
    EXP_CONFIG_DIR = os.path.join(DATA_DIR, "exp_config")
    os.makedirs(EXP_CONFIG_DIR, exist_ok=True)

    # experiment record data dir
    EXP_DATA_DIR = os.path.join(DATA_DIR, "exp_record")
    os.makedirs(EXP_DATA_DIR, exist_ok=True)

    # timezone information for experiments/lab
    EXP_TIMEZONE = ZoneInfo("America/New_York")
    EXP_TEST_RESIN_NAME = "R0"

    # interface detection server (should not end with a slash)
    INTERFACE_DETECTION_SERVER = "http://localhost:5001"
    # SMTP server configuration; set to None to disable email sending
    SMTP_HOST = "localhost"
    SMTP_PORT = 0

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

    @classmethod
    def check_login_config(cls, no_warning: bool = False) -> bool:
        """
        Check if the login configuration file exists and is valid.

        :param bool no_warning:
            Whether to suppress the warning prompts.
        :return bool:
            True if the configuration file exists and is valid, False otherwise.
        """
        if os.path.exists(cls.CONFIG_FILEPATH):
            return True

        CONFIG_FILEPATH = os.path.join(
            os.path.dirname(__file__), ".streamlit", "default_login.yaml"
        )
        if os.path.exists(CONFIG_FILEPATH):
            cls.CONFIG_FILEPATH = CONFIG_FILEPATH
            if not no_warning:
                warnings.warn(
                    f"`{CONFIG_FILEPATH}` is for testing only. NOT PRODUCTION READY!!!"
                )
            return True

        if not no_warning:
            warnings.warn(
                f"Login configuration file `{cls.CONFIG_FILEPATH}` does not exist. "
                "Your app will fail to start without a valid configuration file."
            )
        return False

    @classmethod
    def check_smtp_server(
        cls, timeout: int | float = 3, no_warning: bool = False
    ) -> bool:
        """
        Check if the SMTP server is available.

        :param int | float timeout:
            Timeout for the SMTP connection in seconds.
        :param bool no_warning:
            Whether to suppress the warning prompts.
        :return bool:
            True if the SMTP server is available, False otherwise.
        """
        if not cls.SMTP_HOST or cls.SMTP_PORT <= 0:
            if not no_warning:
                warnings.warn(
                    "SMTP server is not configured. Email sending will be disabled."
                )
            return False

        try:
            with smtplib.SMTP(cls.SMTP_HOST, cls.SMTP_PORT, timeout=timeout) as server:
                server.noop()  # This will raise an exception if the server is not available
            return True
        except (smtplib.SMTPException, ConnectionRefusedError, TimeoutError) as e:
            if not no_warning:
                warnings.warn(
                    f"SMTP server {cls.SMTP_HOST}:{cls.SMTP_PORT} is not available. "
                    f"Email sending will be disabled. (Error: {e})"
                )
            cls.SMTP_HOST = None
            cls.SMTP_PORT = 0
            return False


info = _Info()
