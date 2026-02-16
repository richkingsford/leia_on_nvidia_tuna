"""Shared helpers for livestream setup."""
import socket
import threading

from helper_stream_server import StreamServer, format_stream_url


def build_stream_provider(state, frame_key="frame", lock_key="lock"):
    def _provider():
        lock = state.get(lock_key)
        if lock is None:
            return state.get(frame_key)
        with lock:
            return state.get(frame_key)
    return _provider


def build_text_provider(state, text_key="text_lines", lock_key="lock"):
    def _provider():
        lock = state.get(lock_key)
        if lock is None:
            return state.get(text_key)
        with lock:
            return state.get(text_key)
    return _provider


def build_bool_getter(state, key, lock_key="lock", default=True):
    def _getter():
        lock = state.get(lock_key)
        if lock is None:
            return bool(state.get(key, default))
        with lock:
            return bool(state.get(key, default))
    return _getter


def build_bool_setter(state, key, lock_key="lock"):
    def _setter(value):
        lock = state.get(lock_key)
        if lock is None:
            state[key] = bool(value)
            return
        with lock:
            state[key] = bool(value)
    return _setter


def build_choice_getter(state, key, choices, lock_key="lock", default=None):
    allowed = {str(choice).strip().lower() for choice in (choices or []) if str(choice).strip()}
    default_value = str(default).strip().lower() if default is not None else None
    if default_value not in allowed:
        default_value = next(iter(allowed), "")

    def _getter():
        lock = state.get(lock_key)
        if lock is None:
            raw = state.get(key, default_value)
        else:
            with lock:
                raw = state.get(key, default_value)
        value = str(raw).strip().lower() if raw is not None else default_value
        if value not in allowed:
            return default_value
        return value

    return _getter


def build_choice_setter(state, key, choices, lock_key="lock"):
    allowed = {str(choice).strip().lower() for choice in (choices or []) if str(choice).strip()}

    def _setter(value):
        mode = str(value).strip().lower() if value is not None else ""
        if mode not in allowed:
            return
        lock = state.get(lock_key)
        if lock is None:
            state[key] = mode
            return
        with lock:
            state[key] = mode

    return _setter


def _normalize_choice_options(options):
    normalized = []
    if not options:
        return normalized
    for item in options:
        value = None
        label = None
        if isinstance(item, dict):
            value = item.get("value")
            label = item.get("label")
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            value = item[0]
            label = item[1]
        if value is None:
            continue
        value_norm = str(value).strip().lower()
        if not value_norm:
            continue
        label_norm = str(label).strip() if label is not None else value_norm
        if not label_norm:
            label_norm = value_norm
        normalized.append((value_norm, label_norm))
    seen_modes = set()
    deduped = []
    for value, label in normalized:
        if value in seen_modes:
            continue
        deduped.append((value, label))
        seen_modes.add(value)
    return deduped


def start_stream_server(
    state,
    title,
    header,
    footer,
    host,
    port,
    fps,
    jpeg_quality,
    img_width=800,
    sharpen=True,
    port_tries=10,
    ready_timeout_s=3.0,
    vision_mode_options=None,
    vision_mode_key="vision_mode",
    markerless_profile_options=None,
    markerless_profile_key="markerless_profile",
):
    def _port_available(host, port):
        host_raw = "" if host is None else str(host).strip()
        if not host_raw:
            host_raw = "127.0.0.1"
        try:
            port_val = int(port)
        except (TypeError, ValueError):
            return False
        try:
            addrinfo = socket.getaddrinfo(host_raw, port_val, type=socket.SOCK_STREAM)
        except OSError:
            addrinfo = []
        if not addrinfo:
            return False
        for family, socktype, proto, _canon, sockaddr in addrinfo:
            sock = socket.socket(family, socktype, proto)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
            except OSError:
                return False
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        return True

    if "lock" not in state:
        state["lock"] = threading.Lock()
    if "show_center_line" not in state:
        state["show_center_line"] = True

    normalized_vision_mode_options = _normalize_choice_options(vision_mode_options)
    normalized_markerless_profile_options = _normalize_choice_options(markerless_profile_options)

    vision_mode_getter = None
    vision_mode_setter = None
    if normalized_vision_mode_options:
        allowed_modes = [value for value, _label in normalized_vision_mode_options]
        default_mode = allowed_modes[0]
        current_mode = str(state.get(vision_mode_key, default_mode)).strip().lower()
        if current_mode not in allowed_modes:
            state[vision_mode_key] = default_mode
        vision_mode_getter = build_choice_getter(
            state,
            vision_mode_key,
            allowed_modes,
            default=default_mode,
        )
        vision_mode_setter = build_choice_setter(
            state,
            vision_mode_key,
            allowed_modes,
        )

    markerless_profile_getter = None
    markerless_profile_setter = None
    if normalized_markerless_profile_options:
        allowed_profiles = [value for value, _label in normalized_markerless_profile_options]
        default_profile = allowed_profiles[0]
        current_profile = str(state.get(markerless_profile_key, default_profile)).strip().lower()
        if current_profile not in allowed_profiles:
            state[markerless_profile_key] = default_profile
        markerless_profile_getter = build_choice_getter(
            state,
            markerless_profile_key,
            allowed_profiles,
            default=default_profile,
        )
        markerless_profile_setter = build_choice_setter(
            state,
            markerless_profile_key,
            allowed_profiles,
        )

    try:
        start_port = int(port)
    except (TypeError, ValueError):
        start_port = 5000

    try:
        port_tries = int(port_tries)
    except (TypeError, ValueError):
        port_tries = 1
    port_tries = max(1, min(50, port_tries))

    last_error = None
    for offset in range(port_tries):
        port_candidate = int(start_port + offset)
        if not _port_available(host, port_candidate):
            last_error = RuntimeError(f"Port {port_candidate} not available for host {host!r}.")
            continue
        server = StreamServer(
            build_stream_provider(state),
            text_provider=build_text_provider(state),
            host=host,
            port=port_candidate,
            fps=fps,
            jpeg_quality=jpeg_quality,
            title=title,
            header=header,
            footer=footer,
            img_width=img_width,
            sharpen=sharpen,
            show_center_line_getter=build_bool_getter(state, "show_center_line"),
            show_center_line_setter=build_bool_setter(state, "show_center_line"),
            vision_mode_getter=vision_mode_getter,
            vision_mode_setter=vision_mode_setter,
            vision_mode_options=normalized_vision_mode_options,
            markerless_profile_getter=markerless_profile_getter,
            markerless_profile_setter=markerless_profile_setter,
            markerless_profile_options=normalized_markerless_profile_options,
        )
        server.start()
        try:
            server.wait_until_ready(timeout_s=ready_timeout_s)
        except RuntimeError as exc:
            # Likely "address already in use" or bind failure.
            last_error = exc
            continue
        except TimeoutError as exc:
            last_error = exc
            break
        url = format_stream_url(host, port_candidate)
        return server, url

    if last_error is not None:
        raise last_error
    raise RuntimeError(
        f"No available stream port found for host {host!r} starting at port {start_port} (tried {port_tries})."
    )
