"""
Utility functions for the app
"""

import os
import time
import random
import re
import string
import yaml
from yaml.loader import SafeLoader
from functools import wraps, partial
from types import ModuleType
from typing import Any, Callable, Literal, Mapping

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from info import info


def get_md5_hash(file: str | bytes, chunk_size: int = 4096) -> str:
    """
    Get the MD5 hash of a file (designed for images)

    :param str | bytes file:
        the filepath or the file content
    :param int chunk_size:
        chunk size in kb
    :return str:
        the MD5 hash of the file
    """
    import hashlib

    md5 = hashlib.md5()
    if isinstance(file, str):
        with open(file, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
    elif isinstance(file, bytes):
        md5.update(file)
    return md5.hexdigest()


def send_notification_email(
    status: str = "exception",
    update: str = None,
    subject: str = None,
    to: str = None,
    from_: str = None,
    smtp_host: str | None = None,
    smtp_port: int = 0,
    ntries: int = 1,
    timeout: int = 3,
) -> None:
    """
    Send a notification email to the user

    :param str status:
        the status of the update, one of `["exception", "error", "success"]`
    :param str update:
        the update message
    :param str subject:
        the subject of the email
    :param str to:
        the email address to send the username to
    :param str from_:
        the email address of the sender (on the server side), by default it is `{from_}(@{info.DOMAIN_NAME})`
    :param str | None smtp_host:
        the SMTP host to use. If None, will use the default SMTP server defined in `info.SMTP_HOST`.
        If both `info.SMTP_HOST` and `smtp_host` are None, email sending will be skipped.
    :param int smtp_port:
        the SMTP port to use, default is 0 (use default port defined in `info.SMTP_PORT`).
    :param int ntries:
        the number of tries to send the email, default is 1 try
    :param int timeout:
        the timeout in seconds for each try, default is 3 seconds
    """
    smtp_host = smtp_host or info.SMTP_HOST
    smtp_port = smtp_port or info.SMTP_PORT
    if not smtp_host:
        return

    if not to or "@" not in to:
        raise ValueError("Invalid email address provided for 'to' parameter.")
    from_ = from_ or info.SEND_EMAIL_FROM
    sender_email = f"{from_}@{info.DOMAIN_NAME}" if "@" not in from_ else from_
    receiver_email = to
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    subject = (
        subject or f'[{status}] The remote lab just sent an update "{status}" to you'
    )
    message["Subject"] = f"{subject}" if not subject.startswith("Re: ") else subject
    html = f"""\
    <html>
      <body>
        <p>Hi {info.CONTACT_NAME},</p>
        <p>The remote lab just sent you an update:</p>
        <p>[{status}]<br>{update}<br>[end:{status}]</p>
        <p>Thank you,<br>
        <a href="{info.CONTACT_LINK}">{info.CONTACT_NAME}</a></p>
      </body>
    </html>
    """
    message.attach(MIMEText(html, "html"))

    # use smtp local server
    for _ in range(ntries):
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                server.sendmail(sender_email, receiver_email, message.as_string())
            return
        except:
            pass


def _button_clicked(
    key: str, value: str | bool = "switch", original_key: bool = False
) -> Callable:
    """
    Will return a function such that when the button is clicked, the function will be called.
    Will create a session state variable `{key}_clicked` or `{key}` if it does not exist.
    Every time the button is clicked, the session state variable will be switched to the opposite of its current value.

    :param str key:
        the key for the button. By default, will try to append `_clicked` to the key if it does not end with `_clicked`.
        Check `original_key` option to turn this off.
    :param str | bool value:
        value or how to handle the button click, default is 'switch', other options boolean True or False
    :param bool original_key:
        whether to use the original key or not. If True, will not append `_clicked` to the key.
    :return Callable:
        the function to monitor the button click
    """
    if not original_key:
        key = f"{key}_clicked" if not key.endswith("_clicked") else key

    def wrapper():
        st.session_state.setdefault(key, False)
        st.session_state[key] = (
            not st.session_state[key] if value == "switch" else value
        )

    return wrapper


def get_updated_mapping(
    mapping: Mapping | Any, update: Mapping | Any, is_root: bool = True
) -> Mapping | Any:
    """
    Update the mapping with the new values recursively in place

    :param Mapping | Any mapping:
        the original mapping to update.
    :param Mapping | Any update:
        the mapping to update with.
    :param bool is_root:
        whether this is the root of the recursion. At root, `mapping` and `update` must both be mappings.
        Otherwise, the `mapping` will be returned directly. At non-root, if `mapping` is not a mapping,
        the `update` will be returned directly.
    :return Mapping | Any:
        the updated mapping
    """
    if not isinstance(mapping, Mapping) or not isinstance(update, Mapping):
        if is_root:
            return mapping
        return update

    for key, value in update.items():
        if key not in mapping or not isinstance(value, Mapping):
            mapping[key] = value
        else:
            mapping[key] = get_updated_mapping(mapping[key], value, is_root=False)
    return mapping


def _update_yaml(config: dict, config_path: str = None, max_length: int = 10_000):
    """
    Update the YAML file

    :param dict config: the configuration dictionary
    :param str config_path: the configuration file path
    :param int max_length: the maximum length of the history file
    """
    config_path = config_path or info.CONFIG_FILEPATH
    if "." not in config_path:
        config_path += ".yaml"
    ext = config_path.split(".")[-1]

    with open(config_path, "r") as file:
        previous_config = yaml.load(file, Loader=SafeLoader)

    if config == previous_config:
        return
    else:
        with open(config_path, "w") as file:
            yaml.dump(config, file)

    with open(config_path.replace(f".{ext}", f"_history.{ext}"), "a") as file1:
        file1.write(
            f"----- {time.strftime('%Y-%m-%d %H:%M:%S')} - Updated the YAML file -----\n"
        )
        with open(config_path, "r") as file2:
            lines = file2.readlines()
            max_length = max(max_length, 10 * len(lines))
            file1.writelines(lines)
        file1.write("\n")

    with open(config_path.replace(f".{ext}", f"_history.{ext}"), "r+") as file:
        lines = file.readlines()
        if len(lines) > max_length:
            file.seek(0)
            file.writelines(lines[-max_length:])
            file.truncate()


def _expire_time_format(expire: int = 60 * 5) -> str:
    """
    Function to format the expiration time to `x minute(s) y second(s)`

    :param int expire: the expiration time in seconds
    :return str: the formatted expiration time
    """
    expire_minute_prompt = (
        f"{expire//60} minute{'s' if expire//60 > 1 else ''}"
        if expire // 60 > 0
        else ""
    )
    expire_second_prompt = (
        f" {expire%60} second{'s' if expire%60 > 1 else ''}" if expire % 60 > 0 else ""
    )
    return f"{expire_minute_prompt}{expire_second_prompt}"


def _generate_code(
    length: int = 6,
    options: str = string.ascii_letters + string.digits,
    extra: str = "",
) -> str:
    """
    Function to generate a random code

    :param int length: the length of the code
    :param str options: the characters to choose from, default is ascii letters and digits
    :param str extra: extra characters to add to the options
    :return str: the random code
    """
    return "".join(random.SystemRandom().choice(options + extra) for _ in range(length))


def _send_username_email(
    username: str,
    name: str = None,
    to: str = None,
    from_: str = None,
    subject: str = None,
    password: str = None,
    smtp_host: str | None = None,
    smtp_port: int = 0,
    ntries: int = 1,
    timeout: int = 3,
) -> None:
    """
    Send the username email to the user

    :param str username:
        the username of the receiver
    :param str name:
        the name of the receiver (actual name, not username, optional, default is None)
    :param str to:
        the email address to send the username to
    :param str from_:
        the email address of the sender (on the server side), by default it is `{info.SEND_EMAIL_FROM}(@{info.DOMAIN_NAME})`
    :param str subject:
        the subject of the email, if None, will use a default subject
    :param str password:
        the password of the receiver, if None, will not include the password in the email
    :param str | None smtp_host:
        the SMTP host to use. If None, will use the default SMTP server defined in `info.SMTP_HOST`.
        If both `info.SMTP_HOST` and `smtp_host` are None, email sending will be skipped.
    :param int smtp_port:
        the SMTP port to use, default is 0 (use default port defined in `info.SMTP_PORT`).
    :param int ntries:
        the number of tries to send the email, default is 1 try
    :param int timeout:
        the timeout in seconds for each try, default is 3 seconds
    """
    smtp_host = smtp_host or info.SMTP_HOST
    smtp_port = smtp_port or info.SMTP_PORT
    if not smtp_host:
        return

    name = name or username
    if not to or "@" not in to:
        raise ValueError("Invalid email address provided for 'to' parameter.")
    from_ = from_ or info.SEND_EMAIL_FROM
    sender_email = f"{from_}@{info.DOMAIN_NAME}" if "@" not in from_ else from_
    receiver_email = to
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = (
        subject
        or f"[Do not reply] Your requested {info.APP_NAME} app username{' and password' if password else ''} is here!"
    )
    password_prompt = (
        f"Your temporary password is: <strong>{password}</strong>. <strong>Please reset your password as soon as possible.</strong> "
        if password
        else ""
    )
    html = f"""\
    <html>
      <body>
        <p>Hi{" "+name if name else ""},</p>
        <p>Welcome to the {info.APP_NAME} app.<br>
        Your username associated with this email address is: <strong>{username}</strong>. {password_prompt}If you did not request this information, please ignore this email.</p>
        <p>Thank you,<br>
        <a href="{info.CONTACT_LINK}">{info.CONTACT_NAME}</a></p>
      </body>
    </html>
    """
    message.attach(MIMEText(html, "html"))

    # use smtp local server
    for _ in range(ntries):
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                server.sendmail(sender_email, receiver_email, message.as_string())
            return
        except Exception as e:
            pass

    st.error(
        f"System error while trying to send you the username. "
        f"Please try again later or contact **{info.CONTACT_NAME_HYPERLINK}**. "
        f"Error: {e}."
    )


def _send_verification_email(
    code: int | str | None = None,
    length: int = 6,
    to: str = None,
    from_: str = None,
    name: str = "",
    username: str = "",
    expire: int = 60 * 5,
    smtp_host: str | None = None,
    smtp_port: int = 0,
    ntries: int = 1,
    timeout: int = 3,
) -> tuple[str, float, int]:
    """
    Send the verification email to the user

    :param int | str | None code:
        the verification code, if None, will generate a new code with the given length
    :param int length:
        the length of the code (ignored if code is not None)
    :param str to:
        the email address to send verification code to
    :param str from_:
        the email address of the sender (on the server side), by default it is `{info.SEND_EMAIL_FROM}(@{info.DOMAIN_NAME})`
    :param str name:
        the name of the receiver
    :param str username:
        the username of the receiver
    :param int expire:
        the expiration time of the code in seconds, default is 5 minutes/300 seconds
    :param str | None smtp_host:
        the SMTP host to use. If None, will use the default SMTP server defined in `info.SMTP_HOST`.
        If both `info.SMTP_HOST` and `smtp_host` are None, email sending will be skipped.
    :param int smtp_port:
        the SMTP port to use, default is 0 (use default port defined in `info.SMTP_PORT`).
    :param int ntries:
        the number of tries to send the email, default is 1 try
    :param int timeout:
        the timeout in seconds for each try, default is 3 seconds

    :return str:
        the verification code
    :return float:
        the time when the code was sent
    :return int:
        the expiration time of the code in seconds
    """
    code = str(code or _generate_code(length=length))
    smtp_host = smtp_host or info.SMTP_HOST
    smtp_port = smtp_port or info.SMTP_PORT
    if not smtp_host:
        return code, time.time(), expire

    if not to or "@" not in to:
        raise ValueError("Invalid email address provided for 'to' parameter.")
    from_ = from_ or info.SEND_EMAIL_FROM
    sender_email = f"{from_}@{info.DOMAIN_NAME}" if "@" not in from_ else from_
    receiver_email = to
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = (
        f"[Do not reply] Your {info.APP_NAME} app verification code is here!"
    )
    greetings = name or username  # name is preferred to username for the greeting
    html = f"""\
    <html>
      <body>
        <p>Hi{" "+greetings if greetings else ""},</p>
        <p>Welcome to the {info.APP_NAME} app.<br>
        Your verification code is: <strong>{code}</strong>. The code will expire in {_expire_time_format(expire)}. If you did not request this code, please ignore this email.</p>
        <p>Thank you,<br>
        <a href="{info.CONTACT_LINK}">{info.CONTACT_NAME}</a></p>
      </body>
    </html>
    """
    message.attach(MIMEText(html, "html"))

    # use smtp local server
    for _ in range(ntries):
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                server.sendmail(sender_email, receiver_email, message.as_string())
            return code, time.time(), expire
        except Exception as e:
            pass

    st.error(
        f"System error while trying to send you the username. "
        f"Please try again later or contact **{info.CONTACT_NAME_HYPERLINK}**. "
        f"Error: {e}."
    )
    return code, time.time(), expire


def _send_verification_code(
    to_email: str,
    to_who: str,
    to_username: str,
    from_: str = None,
    length: int = 6,
    expire: int = 60 * 5,
    force_resend: bool = False,
    warning1: ModuleType | DeltaGenerator = st,
    warning2: ModuleType | DeltaGenerator = st,
    nowarnings: bool = False,
    smtp_host: str | None = None,
    smtp_port: int = 0,
    ntries: int = 1,
    timeout: int = 3,
) -> str:
    """
    Send the verification code email to the user with status prompt and return the verification code

    :param str to_email:
        the email address to send the verification code to
    :param str to_who:
        the name of the receiver
    :param str to_username:
        the username of the receiver
    :param str from_:
        the email address of the sender (on the server side), by default it is `{info.SEND_EMAIL_FROM}(@{info.DOMAIN_NAME})`
    :param int length:
        the length of the code
    :param int expire:
        the expiration time of the code in seconds, default is 5 minutes/300 seconds (only applicable for the first time the code is sent)
    :param bool force_resend:
        whether to force resend the verification code no matter if it has expired
    :param ModuleType | DeltaGenerator warning1:
        the first warning prompt (pass st.empty() returns for future removal)
    :param ModuleType | DeltaGenerator warning2:
        the second warning prompt (pass st.empty() returns for future removal)
    :param bool nowarnings:
        whether to suppress the warning prompts
    :param str | None smtp_host:
        the SMTP host to use. If None, will use the default SMTP server defined in `info.SMTP_HOST`.
        If both `info.SMTP_HOST` and `smtp_host` are None, email sending will be skipped.
    :param int smtp_port:
        the SMTP port to use, default is 0 (use default port defined in `info.SMTP_PORT`).
    :param int ntries:
        the number of tries to send the email, default is 1 try
    :param int timeout:
        the timeout in seconds for each try, default is 3 seconds

    :return str:
        the verification code
    """
    from_ = from_ or info.SEND_EMAIL_FROM
    sender_email = f"{from_}@{info.DOMAIN_NAME}" if "@" not in from_ else from_
    identity = to_username or to_email
    st_state_identity = f"username_{identity}_verification_code"
    if to_username and st.session_state.get(
        f"username_{to_username}_email_verified", False
    ):
        return True
    if (
        (st_state_identity in st.session_state)
        and (
            time.time() - st.session_state[st_state_identity][1]
            <= st.session_state[st_state_identity][2]
        )
        and not force_resend
    ):
        # if the verification code has been sent and has not expired, and the user has not clicked the resend button
        count_down = int(
            st.session_state[st_state_identity][2]
            - (time.time() - st.session_state[st_state_identity][1])
        )
        if not nowarnings:
            warning1.warning(
                f"Verification code already sent to {to_email} from {sender_email}. "
                f"Check your spam folder if you do not see the email or contact **{info.CONTACT_NAME_HYPERLINK}**."
            )
            warning2.warning(
                f"Please do not leave or refresh the page. "
                f"Your code will expire in **`{_expire_time_format(count_down)}`**."
            )
    else:
        st.session_state[st_state_identity] = _send_verification_email(
            length=length,
            to=to_email,
            from_=sender_email,
            name=to_who,
            username=to_username,
            expire=expire,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            ntries=ntries,
            timeout=timeout,
        )
        if not nowarnings:
            warning1.warning(
                f"Verification code sent to {to_email} from {sender_email}. "
                f"Check your spam folder if you do not see the email or contact **{info.CONTACT_NAME_HYPERLINK}**."
            )
            warning2.warning(
                f"Please do not leave or refresh the page. "
                f"Your code will expire in **`{_expire_time_format(expire)}`**."
            )

    # this will be the system-generated verification code
    return st.session_state[st_state_identity][0]


def send_registration_email(
    to: str = None,
    copy: str = None,
    from_: str = None,
    smtp_host: str | None = None,
    smtp_port: int = 0,
    ntries: int = 1,
    timeout: int = 3,
) -> None:
    """
    send an email notification to the user about registration

    :param str to:
        the email address to send the email to
    :param str copy:
        the email address to send a copy of the email to
    :param str from_:
        the email address to send the email from
    :param str | None smtp_host:
        the SMTP host to use. If None, will use the default SMTP server defined in `info.SMTP_HOST`.
        If both `info.SMTP_HOST` and `smtp_host` are None, email sending will be skipped.
    :param int smtp_port:
        the SMTP port to use, default is 0 (use default port defined in `info.SMTP_PORT`).
    :param int ntries:
        the number of tries to send the email, default is 1 try
    :param int timeout:
        the timeout in seconds for each try, default is 3 seconds
    """
    smtp_host = smtp_host or info.SMTP_HOST
    smtp_port = smtp_port or info.SMTP_PORT
    if not smtp_host:
        return

    if not to or "@" not in to:
        raise ValueError("Invalid email address provided for 'to' parameter.")
    from_ = from_ or info.SEND_EMAIL_FROM

    sender_email = f"{from_}@{info.DOMAIN_NAME}" if "@" not in from_ else from_
    receiver_email = to
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    if copy and "@" in copy:
        message["Cc"] = copy
        to = [to, copy]

    link = f"https://{info.DOMAIN_NAME}/{info.APP_NAME}/?login=register"

    subject = f"[Do not reply] You are invited to register for the {info.APP_NAME} app!"
    message["Subject"] = subject
    html = f"""\
    <html>
      <body>
        <p>Greetings,</p>
        <p>{st.session_state.get("name", info.CONTACT_NAME)} is inviting you to register for the {info.APP_NAME} app
        developed by {info.CONTACT_NAME_FULL} for the project "<strong>{info.PROJECT_NAME}</strong>". Please use the following
        link with your email address ({to}) to register: <a href="{link}">{link}</a>.
        If you have any questions, please feel free to contact <a href="{info.CONTACT_LINK}">{info.CONTACT_NAME}</a>.</p>
        <p>Please ignore this email if you did not request to register for the app.</p>
        <p>Thank you,<br>
        <a href="{info.CONTACT_LINK}">{info.CONTACT_NAME}</a></p>
      </body>
    </html>
    """
    message.attach(MIMEText(html, "html"))

    # use smtp local server
    for _ in range(ntries):
        try:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                server.sendmail(sender_email, to, message.as_string())
            return
        except Exception as e:
            pass

    st.error(
        f"System error while trying to send registration invitation to {to}. "
        f"Please try again later or contact **{info.CONTACT_NAME_HYPERLINK}**. "
        f"Error: {e}."
    )


def is_logged_in(username: str | None = None) -> bool:
    """
    Check if the user is logged in. Guest user included.
    If username is provided, additionally check if the user is logged in with the given username.

    :param str | None username:
        the username to check, default is None (do not check if username is matched)
    :return bool:
        whether the user is logged in and has the same provided username
    """
    if not st.session_state.get("authentication_status", False):
        return False

    stored_username = st.session_state.get("username", None)
    if username is not None:
        return stored_username == username
    elif not stored_username:
        return False

    if not st.session_state.get("name", None):
        return False
    return True


def is_logged_in_user(username: str | None = None) -> bool:
    """
    Check if the user is logged in. Guest user excluded.

    :param str | None username:
        the username to check, default is None (do not check if username is matched)
    :return bool:
        whether the user is logged in
    """
    stored_username = st.session_state.get("username", None)
    if not isinstance(stored_username, str):
        return False

    return bool(
        is_logged_in(username)
        and stored_username.lower() != "guest"
        and re.match(info.USERNAME_REGEX, stored_username)
    )


def check_user_privilege(privilege: str, prompt: bool = True) -> bool:
    """
    Check if the user has the privilege to access the page

    :param str privilege: the privilege to check
    :param bool prompt: whether to prompt the user if the user does not have the privilege
    :return bool: whether the user has the privilege to access the page
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
    func: Callable, privilege: str | list[str], how: Literal["all", "any"] = "all"
) -> Callable:
    """
    Decorator to check if the user has the privilege to access the page

    :param Callable func: the function to decorate
    :param str privilege: the privilege to check
    :param Literal["all", "any"] how: how to check the privileges
    :return Callable: the decorated function
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
    how_func = {"all": all, "any": any}.get(how.lower(), all)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not how_func(check_user_privilege(p, prompt=False) for p in privilege):
            st.error(
                f"You do not have {how} of the following privileges: {', '.join(f':red[{x}]' for x in privilege)}. "
                f"Your access to this function/page is denied. If you believe this is an error, "
                f"please contact **{info.CONTACT_NAME_HYPERLINK}**."
            )
            st.stop()
        return func(*args, **kwargs)

    return wrapper


requires_admin = partial(requires_privilege, privilege="admin")
requires_inviter = partial(
    requires_privilege, privilege=["admin", "inviter"], how="any"
)


def catch_uncaught_exception(func: Callable):
    """
    Decorator to catch uncaught exceptions

    :param Callable func: the function to decorate
    :return Callable: the decorated function
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(
                f"{e}: An uncaught exception occurred. Please contact **{info.CONTACT_NAME_HYPERLINK}** for help if needed."
            )
            with open(os.path.join(info.WORKING_DIR, "error.log"), "a") as file:
                file.write(
                    f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {e} in {func.__name__}\n"
                )
            st.stop()

    return wrapper
