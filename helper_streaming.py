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
