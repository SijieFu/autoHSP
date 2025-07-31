import os
import yaml
from yaml.loader import SafeLoader
import re
from datetime import datetime
import time
from types import ModuleType
from typing import Any, Iterable, Literal
import urllib.parse

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import streamlit_authenticator as stauth
import extra_streamlit_components as stx

from info import info
from utils import _button_clicked, _update_yaml, get_updated_mapping
from utils import send_registration_email
from utils import is_logged_in, is_logged_in_user
from utils import check_user_privilege, requires_inviter

CONFIG_FILEPATH: None | str = None
CONFIG_UPDATE_PATH: None | str = None
CONFIG: None | dict[str, Any] = None
AUTHENTICATOR: None | stauth.Authenticate = None
COOKIE_MANAGER: None | stx.CookieManager = None


def generate_authenticator(
    config_path: str | None = None, regenerate: bool = False
) -> None:
    """
    Generate an authenticator. If the authenticator is already generated from the same config,
    it will not be regenerated unless `regenerate` is set to True.

    For live deployment (update the config file manually without restarting the app),
    you can create a new update file with at the same directory as the config file,
    and add `update_` prefix to the filename. You can put your changes (edits) in the update file.
    It can be a partial update, so you don't need to put the whole config file in the update file.
    When you finished editing the update file, **add `_update: true` to the first line of the update file**.

    :param str | None config:
        path to the configuration file
    :param bool regenerate:
        force to regenerate the authenticator
    """
    global CONFIG_FILEPATH, CONFIG_UPDATE_PATH, CONFIG, AUTHENTICATOR, COOKIE_MANAGER

    # for compatibility with streamlit_authenticator
    # bugfix for 'st.session_state has no key "logout".'
    st.session_state.setdefault("logout", None)

    config_path = config_path or info.CONFIG_FILEPATH
    if not config_path and not CONFIG_FILEPATH:
        raise ValueError("Authenticator not initialized. Please provide a config path.")

    if (
        CONFIG_FILEPATH == config_path
        and CONFIG is not None
        and AUTHENTICATOR is not None
        and not os.path.isfile(CONFIG_UPDATE_PATH)
        and not regenerate
    ):
        # no need to regenerate the authenticator
        return

    if CONFIG_FILEPATH != config_path:
        CONFIG_FILEPATH = config_path
        filedir, filename = os.path.split(CONFIG_FILEPATH)
        CONFIG_UPDATE_PATH = os.path.join(filedir, f"update_{filename}")

    if not os.path.isfile(CONFIG_FILEPATH):
        raise FileNotFoundError(
            f"Configuration file `{CONFIG_FILEPATH}` not found. Please check the path."
        )
    with open(CONFIG_FILEPATH, "r") as file:
        CONFIG = yaml.load(file, Loader=SafeLoader)

    config_updated = False
    try:
        # try update CONFIG with the update file
        with open(CONFIG_UPDATE_PATH, "r") as file:
            CONFIG_UPDATE = yaml.load(file, Loader=SafeLoader)
        if CONFIG_UPDATE and CONFIG_UPDATE.pop("_update", False):
            # perform a recursive update
            CONFIG = get_updated_mapping(
                mapping=CONFIG, update=CONFIG_UPDATE, is_root=True
            )
            # remove the update file
            os.remove(CONFIG_UPDATE_PATH)
            config_updated = True
    except:
        pass

    # for any guest user that is authorized to login or register
    # remove their guest user accounts
    for email in CONFIG.get("pre-authorized", {}).get("emails", []):
        if CONFIG["credentials"]["usernames"].pop(email, None) is not None:
            config_updated = True
    if config_updated:
        # update the configuration file
        _update_yaml(CONFIG, config_path=CONFIG_FILEPATH)

    if AUTHENTICATOR is not None:
        # AUTHENTICATOR is updated whenever CONFIG is updated in place
        return

    AUTHENTICATOR = stauth.Authenticate(
        credentials=CONFIG["credentials"],
        cookie_name=CONFIG["cookie"]["name"],
        cookie_key=CONFIG["cookie"]["key"],
        cookie_expiry_days=CONFIG["cookie"]["expiry_days"],
        api_key=CONFIG.get("api_key", None),
    )
    COOKIE_MANAGER = AUTHENTICATOR.cookie_controller.cookie_model.cookie_manager


class MODULE_SUCCESS(Exception):
    pass


def integrated_login(
    config_path: str | None = None,
) -> tuple[str | None, bool, str | None]:
    """
    Integrated login; master login function for the application.
    If not verified, the whole app will be blocked. In other words,
    if the function returns, it means the user is logged in and verified.

    :param str | None config_path:
        path to the configuration file
    :return str | None:
        name of the user
    :return bool:
        authentication status
    :return str | None:
        username of the user
    """
    config_path = config_path or info.CONFIG_FILEPATH
    try:
        generate_authenticator(config_path=config_path)
    except Exception as e:
        st.error(
            f"Error: {e}. Login configuration failed to load."
            f" If you believe this is a mistake, please contact **{info.CONTACT_NAME_HYPERLINK}**."
        )
        st.stop()

    login_request = st.query_params.get("login", "") or st.session_state.get(
        "login_page", ""
    )
    username = st.query_params.pop("username", "")

    try:
        if login_request == "fromlogout":
            logout(config_path=config_path)
            st.query_params.pop("login", None)
            st.session_state.pop("login_page", None)

        if login_request == "login":
            login(config_path=config_path)
            raise MODULE_SUCCESS  # jump to `finally` block

        if login_request == "register":
            logout(config_path=config_path)
            register(config_path=config_path)
            raise MODULE_SUCCESS  # jump to `finally` block

        if login_request.startswith("guest"):
            guest_login(login_request=login_request, config_path=config_path)
            raise MODULE_SUCCESS  # jump to `finally` block

        if login_request == "forgotusername":
            logout(config_path=config_path)
            forgot_username(config_path=config_path)
            raise MODULE_SUCCESS  # jump to `finally` block

        if login_request == "forgotpassword":
            logout(config_path=config_path)
            forgot_password(config_path=config_path)
            raise MODULE_SUCCESS  # jump to `finally` block

        # if none of the above conditions apply, then go to the login page
        login(config_path=config_path)
    except MODULE_SUCCESS:
        # this exception actually means things went fine, so do nothing
        pass
    except Exception as e:
        st.error(
            f"Error: {e}. If you believe this is a mistake, please contact **{info.CONTACT_NAME_HYPERLINK}**."
        )
        st.stop()

    name, authentication_status, username = (
        st.session_state.get("name", None),
        st.session_state.get("authentication_status", False),
        st.session_state.get("username", None),
    )
    if authentication_status and username and name:
        if not login_request.startswith("guest"):
            st.query_params.pop("login", None)
            st.session_state.pop("login_page", None)
        for privilege_type in CONFIG["permissions"].keys():
            st.session_state[f"USER_{username}_IS_{privilege_type.upper()}"] = (
                username in CONFIG["permissions"].get(privilege_type, [])
            )
        return name, authentication_status, username
    else:
        st.stop()


def greet_user(name: str, username: str, config_path: str | None = None):
    """
    When a user is logged in as a valid user (not a guest user),
    greet the user and provide options to reset password or update user information.

    :param str name:
        name of the user
    :param str username:
        username of the user
    :param str | None config_path:
        path to the configuration file
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    if AUTHENTICATOR is not None:
        AUTHENTICATOR.logout("Logout", "sidebar", key="btn_logout_sidebar")

    st.write(
        f"## Hi **`{name.split()[0]}`**, how are you doing today?"
    )  # st.session_state["name"]
    st.markdown(
        f"You are logged in as **`{username}`**. "
        f"You can log out, reset your password, or update your information from the sidebar `>`. "
        f"Or, click <a href='/{info.APP_NAME}/?login=fromlogout' target='_self'>here</a> to log out.",
        unsafe_allow_html=True,
    )
    reset_password(username=username, config_path=config_path)
    update_user_info(username=username, config_path=config_path)


def login(config_path: str | None = None):
    """
    Login to the application
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    username_prompt = st.session_state.get(
        "query_params_username", False
    ) or st.query_params.get("username", False)

    try:
        AUTHENTICATOR.login(
            location="main",
            max_concurrent_users=CONFIG.get("max_concurrent_users", None),
            fields={
                "Form name": f"{info.APP_NAME} App - Login",
                "Username": (
                    f"Username ({username_prompt})" if username_prompt else "Username"
                ),
            },
        )
    except Exception as e:
        st.error(f"(login failed) {e}")
    login_warning = st.empty()

    name = st.session_state.get("name", None)
    authentication_status = st.session_state.get("authentication_status", None)
    username = st.session_state.get("username", None)

    if authentication_status and username in CONFIG.get("require-password-reset", []):
        login_warning.warning(
            f"You recently requested a temporary password. Please reset your password immediately."
        )
        reset_password(username=username, config_path=config_path, must_fire=True)
        st.stop()

    if authentication_status == None:
        login_warning.warning(
            "Please enter your username and password, or find an option below :arrow_down:"
        )
    elif authentication_status == False:
        _update_yaml(CONFIG, config_path=config_path)
        login_warning.error(
            "Username/password is incorrect. Need help? Find an option below :arrow_down:"
        )
    elif authentication_status and re.match(info.USERNAME_REGEX, username):
        # valid username --> not oauth guest user
        _update_yaml(CONFIG, config_path=config_path)
        greet_user(name=name, username=username, config_path=config_path)
    elif authentication_status:
        # email username means a guest user from oauth
        st.query_params["login"] = "guest-"
        guest_login(login_request="guest-", config_path=config_path)
        _update_yaml(CONFIG, config_path=config_path)
        return name, authentication_status, username

    if not authentication_status:
        add_guest_login = bool(CONFIG.get("guest-credentials", None))
        if add_guest_login:
            cols = st.columns([1, 1, 1, 1])
        else:
            cols = st.columns([1, 1, 1])

        cols[0].button(
            "Register",
            key="btn_registration",
            help="Register as a new user",
            on_click=_button_clicked("login_page", "register", original_key=True),
            use_container_width=True,
        )
        if add_guest_login:
            cols[1].button(
                "Login as guest",
                key="btn_guest_login",
                help="Login as a guest user",
                on_click=_button_clicked("login_page", "guest", original_key=True),
                use_container_width=True,
            )

        cols[-2].button(
            "Forgot username",
            key="btn_forgot_username",
            help="Forgot your username?",
            on_click=_button_clicked("login_page", "forgotusername", original_key=True),
            use_container_width=True,
        )
        cols[-1].button(
            "Forgot password",
            key="btn_forgot_password",
            help="Forgot your password?",
            on_click=_button_clicked("login_page", "forgotpassword", original_key=True),
            use_container_width=True,
        )
        st.stop()

    login_warning.empty()
    return name, authentication_status, username


def logout(config_path: str | None = None):
    """
    Logout from the application

    :param str | None config_path:
        path to the configuration file
    """
    generate_authenticator(config_path=config_path)

    try:
        AUTHENTICATOR.authentication_controller.logout()
        AUTHENTICATOR.cookie_controller.delete_cookie()
        _update_yaml(CONFIG, config_path=config_path)
    except Exception as e:
        st.error(
            f"Error: {e}. Logout failed. If you believe this is a mistake, please contact **{info.CONTACT_NAME_HYPERLINK}**."
        )
        st.stop()


def redirect(
    count_down: int = 0,
    where: str | None = None,
    logout_first: bool = False,
    username: str | None = None,
    config_path: str | None = None,
    **kwargs,
):
    """
    Function to redirect to the login (or `where`) page with countdown

    :param int count_down:
        the countdown time in seconds
    :param str | None where:
        the page to redirect to. By default (None), it will redirect to the login page.
        In short, this sets `st.query_params["login"]` to the value of `where`.
    :param bool logout_first:
        whether to log out the user first
    :param str | None username:
        the username to log out (if not None)
    :param str | None config_path:
        the configuration file path
    :param Any **kwargs:
        the keyword arguments to pass to the login page (st.query_params)
    """
    generate_authenticator(config_path=config_path)

    kwargs["login"] = where
    kwargs["username"] = username
    if not where or where == "login":
        st.session_state.pop("login_page", None)
        kwargs.pop("login", None)
        st.query_params.pop("login", None)

    keys_to_drop = []
    for key, value in kwargs.items():
        if value is None:
            keys_to_drop.append(key)
    for key in keys_to_drop:
        kwargs.pop(key, None)
        st.query_params.pop(key, None)
    st.query_params.update(kwargs)

    where = where or "login"
    with st.spinner(f"Redirecting to the {where} page ..."):
        st_empty = st.empty()
        query_param_string = (
            f"?{urllib.parse.urlencode(st.query_params)}" if st.query_params else ""
        )
        while count_down >= 1:
            st_empty.markdown(
                f"You will be redirected to the {where} page in `{count_down}` second{'s' if count_down > 1 else ''}... "
                f"To skip, <a href='/{info.APP_NAME}{query_param_string}' target='_self'>click here</a>.",
                unsafe_allow_html=True,
            )
            time.sleep(1)
            count_down -= 1
        st_empty.empty()
        if logout_first:
            logout(config_path=config_path)
        st.query_params.update(kwargs)


def redirect_button_to_login(
    slot: ModuleType | DeltaGenerator = st,
    use_container_width: bool = False,
    **kwargs: Any,
):
    """
    Redirect button to login page

    :param ModuleType | DeltaGenerator slot:
        place to put the button
    :param bool use_container_width:
        whether to use the full width of the container
    :param Any **kwargs:
        the keyword arguments to pass to the `redirect` function. See `redirect` for details.
    """
    return slot.button(
        "Go back to login",
        help="Go back to the login page",
        on_click=redirect,
        kwargs=kwargs,
        use_container_width=use_container_width,
    )


def register(config_path: str | None = None):
    """
    Register a new user
    """
    config_path = config_path or info.CONFIG_FILEPATH

    ENABLE_TWO_FACTOR = bool(
        CONFIG.get("api_key", None) and info.ENABLE_TWO_FACTOR_AUTH
    )
    try:
        new_email, new_username, new_name = AUTHENTICATOR.register_user(
            location="main",
            pre_authorized=CONFIG["pre-authorized"]["emails"],
            fields={
                "Form name": f"{info.APP_NAME} App - New User Registration",
                "Username": "Username (start with a letter, between 3 and 20 characters, a-Z, 0-9, _)",
                "Password": "Password (:warning: USE RANDOM PASSWORD :warning:)",
            },
            two_factor_auth=ENABLE_TWO_FACTOR,
        )
        msg_slot = st.empty()
        redirect_button_to_login()
    except Exception as e:
        st.error(f"(registration failed) {e}")
        redirect_button_to_login()
        st.stop()

    if not all([new_email, new_username, new_name]):
        msg_slot.warning(
            "Please enter your username, email address, and full name to register."
        )
        st.stop()

    if new_username and new_username.lower().startswith("guest"):
        msg_slot.error("Username `guest` is reserved. Please try another one.")
        st.stop()

    if not new_name or len(new_name.split()) < 2:
        msg_slot.error("Please enter your full name (first and last name).")
        st.stop()
    if not re.match(info.USERNAME_REGEX, new_username):
        msg_slot.error(
            f"`{info.USERNAME_REGEX}`: Username must start with a letter and only contain letters, numbers, and underscores (between 3 and 20 characters)."
        )
        st.stop()
    if not re.match(info.EMAIL_REGEX, new_email):
        msg_slot.error(
            f"`{info.EMAIL_REGEX}`: Email address is not valid. Please try again."
        )
        st.stop()

    # update preset permissions
    permissions = CONFIG["permissions"]
    for privilege_type in permissions.keys():
        try:
            loc = permissions[privilege_type].index(new_email)
            permissions[privilege_type][loc] = new_username
        except ValueError:
            # email not found in the list
            pass
    # remove the email from pre-authorized emails
    try:
        CONFIG["pre-authorized"]["emails"].remove(new_email)
    except:
        pass
    _update_yaml(CONFIG, config_path=config_path)

    msg_slot.success(
        f"Hi `{new_name.split()[0]}`, you are now registered as `{new_username}`."
    )
    redirect(count_down=3, username=new_username, config_path=config_path)
    return new_email, new_username, new_name


def forgot_password(config_path: str | None = None) -> None:
    """
    Forgot password
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    ENABLE_TWO_FACTOR = bool(
        CONFIG.get("api_key", None) and info.ENABLE_TWO_FACTOR_AUTH
    )
    try:
        username, email, password = AUTHENTICATOR.forgot_password(
            location="main",
            fields={
                "Form name": f"{info.APP_NAME} App - Forgot Password",
            },
            captcha=info.ENABLE_CAPTCHA,
            two_factor_auth=ENABLE_TWO_FACTOR,
        )
        if username == False:
            raise ValueError("Username not found.")
        msg_slot = st.empty()
        password_slot = st.empty()
        redirect_button_to_login()
    except Exception as e:
        st.error(f"(forgot password failed) {e}")
        redirect_button_to_login()
        st.stop()

    if username:
        if not isinstance(CONFIG.get("require-password-reset", False), list):
            CONFIG["require-password-reset"] = []
        if ENABLE_TWO_FACTOR:
            CONFIG["require-password-reset"].append(username)
            msg_slot.success(
                f"New password sent to your registered email at ***@{email.rsplit('@', 1)[-1]}. "
                f"You will be required to reset the password."
            )
        else:
            msg_slot.warning(
                "**TWO-FACTOR AUTHENTICATION IS NOT ENABLED.** Your new password is:"
            )
            password_slot.code(password, language="plaintext")
        _update_yaml(CONFIG, config_path=config_path)


def forgot_username(config_path: str = None) -> None:
    """
    Forgot username
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    ENABLE_TWO_FACTOR = bool(
        CONFIG.get("api_key", None) and info.ENABLE_TWO_FACTOR_AUTH
    )
    try:
        username, email = AUTHENTICATOR.forgot_username(
            location="main",
            fields={
                "Form name": f"{info.APP_NAME} App - Forgot Username",
            },
            captcha=info.ENABLE_CAPTCHA,
            two_factor_auth=ENABLE_TWO_FACTOR,
        )
        if username == False:
            raise ValueError("Email not found.")
        msg_slot = st.empty()
        redirect_button_to_login()
    except Exception as e:
        st.error(f"(forgot username failed) {e}")
        redirect_button_to_login()
        st.stop()

    if username:
        if ENABLE_TWO_FACTOR:
            msg_slot.success(
                f"New username sent to your registered email at ***@{email.rsplit('@', 1)[-1]}."
            )
        else:
            msg_slot.warning(
                f"**TWO-FACTOR AUTHENTICATION IS NOT ENABLED.** Your username is: `{username}`"
            )
        _update_yaml(CONFIG, config_path=config_path)


def reset_password(
    username: str, config_path: str | None = None, must_fire: bool = False
):
    """
    Add a button to sidebar for resetting password. Must be logged in to reset password.

    :param str username:
        username of the logged in user
    :param str | None config_path:
        path to the configuration file
    :param bool must_fire:
        ignore if button clicked, enforce user to reset their password
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    if must_fire:
        # add an option to go to login
        redirect_button_to_login(logout_first=True)
    else:
        st.sidebar.button(
            "Reset password",
            key="btn_reset_password",
            on_click=_button_clicked("btn_reset_password"),
        )
        if not st.session_state.get("btn_reset_password_clicked", False):
            return

    if not is_logged_in_user(username=username):
        st.error(
            "You need to log in first and then reset your password from the sidebar."
        )
        redirect(count_down=5, logout_first=True, config_path=config_path)
        redirect_button_to_login(logout_first=True)
        st.stop()

    try:
        state = AUTHENTICATOR.reset_password(
            username=username,
            location="main",
            fields={
                "Form name": f"{info.APP_NAME} App - Reset Password (`{username}`)",
                "New password": "New password (:warning: USE RANDOM PASSWORD :warning:)",
            },
        )
        if not state:
            raise ValueError("Please fill in your old and/or new password.")

        try:
            CONFIG["require-password-reset"].remove(username)
        except:
            pass
        st.success("Password modified successfully! Please log in again.")
        _update_yaml(CONFIG, config_path=config_path)
        st.session_state.pop("btn_reset_password_clicked", None)
        st.toast("Password was reset. Logging out ...", icon="🔒")
        redirect(
            count_down=5, logout_first=True, username=username, config_path=config_path
        )
        st.stop()
    except Exception as e:
        st.error(f"(reset password failed) {e}")
        if not must_fire:
            st.button(
                "Cancel password reset",
                on_click=_button_clicked("btn_reset_password", False),
            )
        st.stop()


def update_user_info(
    username: str, config_path: str | None = None, must_fire: bool = False
):
    """
    Update user information. Must be logged in to update user info.

    :param str username:
        username of the logged in user
    :param str | None config_path:
        path to the configuration file
    :param bool must_fire:
        ignore if button clicked, enforce user to update their info
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    if must_fire:
        # add an option to go to login
        redirect_button_to_login(logout_first=True)
    else:
        st.sidebar.button(
            "Update my information",
            key="btn_update_userinfo",
            on_click=_button_clicked("btn_update_userinfo"),
        )
        if not st.session_state.get("btn_update_userinfo_clicked", False):
            return

    if not is_logged_in_user(username=username):
        st.error(
            "You need to log in first and update your information from the sidebar."
        )
        redirect(count_down=5, logout_first=True, config_path=config_path)
        redirect_button_to_login(logout_first=True)
        st.stop()

    try:
        state = AUTHENTICATOR.update_user_details(
            username=username,
            location="main",
            fields={
                "Form name": f"{info.APP_NAME} App - Update User Info (`{username}`)"
            },
        )
        if not state:
            raise ValueError("Please fill in your update information.")

        st.success("User information modified successfully!")
        _update_yaml(CONFIG, config_path=config_path)
        st.button(
            "Return to main page",
            on_click=_button_clicked("btn_update_userinfo", False),
        )
        st.stop()
    except Exception as e:
        st.error(f"(update user info failed) {e}")
        if not must_fire:
            st.button(
                "Cancel info update",
                on_click=_button_clicked("btn_update_userinfo", False),
            )
        st.stop()


@requires_inviter
def add_preauthorized_email(config_path: str | None = None):
    """
    Add pre-authorized email for future registration

    :param str | None config_path:
        path to the configuration file
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    all_priviledge_types = list(CONFIG.get("permissions", {}).keys())
    if not check_user_privilege("admin", prompt=False):
        allowed_priviledge_types = [
            x for x in all_priviledge_types if x not in info.SPECIAL_PRIVILEGES
        ]
    else:
        allowed_priviledge_types = all_priviledge_types.copy()
    st.session_state.multiselect_preauthorized_priviledge = ["user"]

    with st.form(key="form_preauthorized_email"):
        email = st.text_input(
            "Enter email to pre-authorize (press Enter to apply)",
            key="txtin_preauthorized_email",
        ).strip()
        priviledge = st.multiselect(
            "Select priviledge type(s)",
            options=allowed_priviledge_types,
            key="multiselect_preauthorized_priviledge",
        )
        submit_button = st.form_submit_button("Submit")

    if submit_button and email:
        if not re.match(info.EMAIL_REGEX, email):
            st.warning(f"Email address `{email}` is not valid. Please try again.")
        elif email in (x["email"] for x in CONFIG["credentials"]["usernames"].values()):
            st.error(
                f"Email address `{email}` is already registered. Please try another one."
            )
        else:
            prompt_slot = st.empty()
            if email in CONFIG["pre-authorized"].setdefault("emails", []):
                prompt_slot.warning(f"Email `{email}` is already pre-authorized.")
            else:
                CONFIG["pre-authorized"]["emails"].append(email)
                for priv in all_priviledge_types:
                    if priv not in priviledge:
                        try:
                            CONFIG["permissions"][priv].remove(email)
                        except ValueError:
                            pass
                    else:
                        if email not in CONFIG["permissions"].setdefault(priv, []):
                            CONFIG["permissions"][priv].append(email)

                try:
                    send_registration_email(to=email)
                except Exception as e:
                    st.error(
                        f"Failed to send invite to register: {e}. You should notify the user to register yourself."
                    )

                _update_yaml(CONFIG, config_path=config_path)
                prompt_slot.success(
                    f"Email `{email}` has been pre-authorized with priviledge(s): `{', '.join(priviledge)}`."
                )
    else:
        st.warning("Please enter an email address to pre-authorize.")


def prompt_for_guestid(
    config: dict, check_cookie: bool = True, save_cookie: bool = False
) -> tuple[str, str]:
    """
    Prompt for guest ID and return the guest ID.

    :param dict config:
        configuration dictionary
    :param bool check_cookie:
        whether to check the cookie for the guest ID
    :param bool save_cookie:
        whether to save the guest ID in the cookie
    :return str:
        valid guest ID
    :return str:
        guest ID expiration time in isoformat
    """
    guest_info_key = "guest-credentials"

    cookie_name = f"{info.APP_NAME}-guestid"
    hint = ""
    if check_cookie:
        hint = COOKIE_MANAGER.get(cookie_name) or ""

    prompt_for_guestid = st.empty()
    guestid_input = prompt_for_guestid.text_input(
        f"Enter your guest pass (find `?login=guest-<guestpass>` from the shared URL"
        f"{f', or try `{hint}`' if hint else ''}):",
        value=hint,
    )

    if not guestid_input:
        st.stop()

    if guestid_input not in config[guest_info_key]:
        st.error(f"Invalid guest pass `{guestid_input}`. Please try again.")
        st.stop()

    _expire_str = config[guest_info_key][guestid_input].get(
        "expire_on", "200000101T00:00:00"
    )
    _expire_dt = datetime.fromisoformat(_expire_str)
    if datetime.now(info.TIMEZONE) > _expire_dt:
        st.error(
            f"Guest pass **`{guestid_input}`** expired on **`{_expire_str}`**. "
            f"Contact **{info.CONTACT_NAME_HYPERLINK}** if you need access."
        )
        st.stop()

    prompt_for_guestid.empty()
    login_request = f"guest-{guestid_input}"
    st.query_params["login"] = login_request
    if save_cookie:
        COOKIE_MANAGER.set(cookie_name, guestid_input, expires_at=_expire_dt)
    return guestid_input, _expire_str


def guest_login(login_request: str = "", config_path: str = None):
    """
    Guest login

    :param str login_request:
        login request from the query params. Should be in the format of `guest-<guestid>`.
    :param str | None config_path:
        path to the configuration file
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    user_login_button = st.empty()
    user_login_button.button(
        "login as user",
        key="btn_go_back_to_login_guest",
        help="Go back to the login page",
        on_click=redirect,
        kwargs={"logout_first": True},
    )

    # check if the guestid is already in the cookie
    cookie_name = f"{info.APP_NAME}-guestid"
    cookie_guestid = COOKIE_MANAGER.get(cookie_name)
    if not isinstance(cookie_guestid, str):
        cookie_guestid = ""

    login_request = re.sub(r"^guest[:_\s]", "guest-", login_request)
    guestid = re.match(r"^guest-\s*([^\s]*)\s*$", login_request)
    if not guestid:
        guestid = cookie_guestid
    else:
        guestid = guestid.group(1)

    guestid_from_oauth = guestid.startswith("@")
    if guestid_from_oauth:
        # oauth redirected login
        oauth_guest_login(
            config_path=config_path, oauth_providers=[guestid.lstrip("@")]
        )
        guestid = cookie_guestid
    st.query_params["login"] = f"guest-{guestid}"

    guestid_saved = False
    if not guestid:
        prompt_why_here = st.empty()
        if is_logged_in():
            prompt_why_here.warning(
                "You are logged in as a guest, but we still need to verify your guest pass."
            )
        guestid, *_ = prompt_for_guestid(CONFIG, save_cookie=not guestid_saved)
        guestid_saved = True
        if guestid_from_oauth:
            st.session_state["authentication_status"] = True
        prompt_why_here.empty()

    expire_prompt_format = "#### Guest pass `{guestid}` will expire on `{time_str}`."
    warning_prompt = (
        f"> If you need to share this guest pass, please share it to the intended audience only. "
        f" For an unintended leak of this pass, please contact **{info.CONTACT_NAME_HYPERLINK}** "
        f"to revoke the pass immediately."
    )

    guest_info_key = "guest-credentials"
    guestid_expires_on = CONFIG[guest_info_key].get(guestid, {}).get("expire_on", None)
    try:
        guestid_expires_on = datetime.fromisoformat(guestid_expires_on).astimezone(
            info.TIMEZONE
        )
    except:
        guestid_expires_on = datetime.fromtimestamp(0).astimezone(info.TIMEZONE)
    guestid_expires_on_str = guestid_expires_on.strftime("%Y-%m-%d %H:%M:%S %Z")

    error_msg_slot = st.empty()
    if guestid not in CONFIG[guest_info_key]:
        msg = f"You are trying to login as a guest, but the guest pass `{guestid}` is not found."
    elif guestid_expires_on < datetime.now(info.TIMEZONE):
        msg = (
            f"Guest ID **:red[{guestid}]** expired on {guestid_expires_on_str}. "
            f"Please contact **{info.CONTACT_NAME_HYPERLINK}** if you need access."
        )
    else:
        msg = False

    if msg:
        error_msg_slot.error(msg)
        guestid, guestid_expires_on_str = prompt_for_guestid(
            CONFIG, save_cookie=not guestid_saved
        )
        guestid_saved = True
        error_msg_slot.empty()

    if not guestid_saved:
        # save the guestid in the cookie
        COOKIE_MANAGER.set(cookie_name, guestid, expires_at=guestid_expires_on)

    # invoke oauth_guest_login
    st.write(
        expire_prompt_format.format(guestid=guestid, time_str=guestid_expires_on_str)
    )
    st.write(warning_prompt)
    st.session_state["guest_pass_expires"] = guestid_expires_on.timestamp()

    if not is_logged_in():
        oauth_guest_login(
            guest_pass=guestid,
            config_path=config_path,
            oauth_providers=CONFIG[guest_info_key][guestid].get("oauth2", None),
        )

    if is_logged_in_user():
        # should not happen, but just in case if the user is logged in as an app user
        # and try to login as a guest user
        logout(config_path=config_path)
        user_login_button.empty()
        guest_login(login_request=f"guest-{guestid}", config_path=config_path)

    st.write(
        f"## Hi **`{st.session_state['name'].split()[0]}`**, how are you doing today?"
    )


def oauth_guest_login(
    guest_pass: str | None = None,
    config_path: str | None = None,
    oauth_providers: (
        Iterable[Literal["google", "microsoft"]] | Literal[None, False, "all"]
    ) = None,
):
    """
    OAuth guest login

    :param str | None guest_pass:
        guest pass to login
    :param str | None config_path:
        path to the configuration file
    :param Iterable[Literal["google", "microsoft"]] | False | None oauth_providers:
        list of oauth providers to use for guest login.
        Check `streamlit_authenticator.Authenticate.experimental_guest_login` for more details.
        if False, do not use oauth providers - direct success
        if None, will use the query params to determine the oauth providers
        if "all", use all oauth providers ["google", "microsoft"]
    """
    config_path = config_path or info.CONFIG_FILEPATH
    generate_authenticator(config_path=config_path)

    if oauth_providers == False:
        # do not use oauth providers
        st.session_state["authentication_status"] = True
        st.session_state["username"] = "guest"
        st.session_state["name"] = "Guest Guess"
        return
    elif oauth_providers == "all":
        oauth_providers = ["google", "microsoft"]
    elif not isinstance(oauth_providers, Iterable) or len(oauth_providers) == 0:
        # wrong configuration
        st.error(
            "Error: Guest login is not configured properly. No OAuth providers found. "
            f"Please contact **{info.CONTACT_NAME_HYPERLINK}**."
        )
        st.stop()

    oauth_warning = st.empty()
    if len(oauth_providers) > 1:
        warning_msg = f"You are required to login with one of {', '.join(map(lambda x: f'**{x.title()}**', oauth_providers))}."
    else:
        warning_msg = (
            f"You are required to login with **{oauth_providers[0].title()}**."
        )
    if guest_pass:
        warning_msg += f" After logging in, re-enter the guest pass `{guest_pass}` to access the application."
    oauth_warning.warning(warning_msg)

    try:
        for provider in oauth_providers:
            if not provider in CONFIG["oauth2"]:
                raise ValueError(
                    f"OAuth provider `{provider}` not found in the configuration file. "
                    f"Please contact **{info.CONTACT_NAME_HYPERLINK}**."
                )
            AUTHENTICATOR.experimental_guest_login(
                button_name=f"Login with {provider.title()}",
                location="main",
                provider=provider,
                oauth2=CONFIG["oauth2"],
            )
    except Exception as e:
        st.error(f"Error: {e}. OAuth guest login failed. ")
        st.stop()
    finally:
        if st.button("Enter new guest pass", key="btn_new_guest_pass"):
            # user wants to enter a new guest pass
            st.query_params["login"] = "guest-"
            COOKIE_MANAGER.delete(f"{info.APP_NAME}-guestid")
            st.rerun()

    if not is_logged_in():
        st.stop()

    if is_logged_in_user():
        # app user cookies are used instead of oauth2
        # log the app user out and re-login with guest
        logout(config_path=config_path)
        oauth_guest_login(
            guest_pass=guest_pass,
            config_path=config_path,
            oauth_providers=oauth_providers,
        )

    # oauth guest users should not have an email field
    # otherwise, the email will be blocked from registration in the future
    user_email = st.session_state.get("email", None)
    if user_email and user_email in CONFIG["credentials"]["usernames"]:
        CONFIG["credentials"]["usernames"][user_email].pop("email", None)

    oauth_warning.empty()
    _update_yaml(CONFIG, config_path=config_path)
