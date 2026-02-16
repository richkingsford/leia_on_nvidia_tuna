import threading
import time
import socket
import logging
import html
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask import cli as flask_cli

DEFAULT_STREAM_HOST = "127.0.0.1"
DEFAULT_STREAM_FPS = 10
DEFAULT_JPEG_QUALITY = 90


def format_stream_url(host, port):
    try:
        port_val = int(port)
    except (TypeError, ValueError):
        port_val = 5000
    host_raw = "" if host is None else str(host).strip()
    if not host_raw or host_raw == "localhost":
        host_raw = "127.0.0.1"
    if host_raw == "0.0.0.0":
        host_raw = "127.0.0.1"
    elif host_raw in ("::", "[::]"):
        host_raw = "::1"
    if ":" in host_raw and not host_raw.startswith("["):
        host_raw = f"[{host_raw}]"
    return f"http://{host_raw}:{port_val}"


class StreamServer:
    def __init__(
        self,
        frame_provider: Callable[[], Optional[np.ndarray]],
        text_provider: Optional[Callable[[], Optional[list]]] = None,
        host: str = DEFAULT_STREAM_HOST,
        port: int = 5000,
        fps: int = DEFAULT_STREAM_FPS,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        title: str = "Robot Leia",
        header: Optional[str] = None,
        footer: Optional[str] = None,
        img_width: Optional[int] = None,
        sharpen: bool = False,
        show_center_line_getter: Optional[Callable[[], bool]] = None,
        show_center_line_setter: Optional[Callable[[bool], None]] = None,
        vision_mode_getter: Optional[Callable[[], str]] = None,
        vision_mode_setter: Optional[Callable[[str], None]] = None,
        vision_mode_options: Optional[list] = None,
        markerless_profile_getter: Optional[Callable[[], str]] = None,
        markerless_profile_setter: Optional[Callable[[str], None]] = None,
        markerless_profile_options: Optional[list] = None,
    ):
        self.frame_provider = frame_provider
        self.text_provider = text_provider
        self.host = host
        self.port = port
        self.fps = max(1, int(fps))
        try:
            jpeg_quality = int(jpeg_quality)
        except (TypeError, ValueError):
            jpeg_quality = DEFAULT_JPEG_QUALITY
        self.jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self.title = title
        self.header = title if header is None else header
        self.footer = footer
        self.img_width = img_width
        self.sharpen = sharpen
        self.show_center_line_getter = show_center_line_getter
        self.show_center_line_setter = show_center_line_setter
        self.vision_mode_getter = vision_mode_getter
        self.vision_mode_setter = vision_mode_setter
        self.vision_mode_options = self._normalize_options(vision_mode_options)
        if (self.vision_mode_getter is not None or self.vision_mode_setter is not None) and not self.vision_mode_options:
            self.vision_mode_options = [("aruco", "AruCo Markers"), ("yolo", "Markerless")]
        self._vision_mode_allowed = {value for value, _label in self.vision_mode_options}
        self.markerless_profile_getter = markerless_profile_getter
        self.markerless_profile_setter = markerless_profile_setter
        self.markerless_profile_options = self._normalize_options(markerless_profile_options)
        self._markerless_profile_allowed = {value for value, _label in self.markerless_profile_options}
        self._stop = threading.Event()
        self._thread = None
        self._startup_error = None

        self.app = Flask(__name__)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self.app.logger.disabled = True

        @self.app.route("/")
        def index():
            return self._index_html()

        @self.app.route("/video_feed")
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/text")
        def text_feed():
            lines = []
            if self.text_provider is not None:
                try:
                    payload = self.text_provider()
                except Exception:
                    payload = None
                if isinstance(payload, list):
                    lines = payload
            return jsonify({"lines": lines})

        @self.app.route("/stream_prefs", methods=["GET", "POST"])
        def stream_prefs():
            payload = request.get_json(silent=True) or {}
            if request.method == "POST":
                if self.show_center_line_setter is not None and "show_center_line" in payload:
                    flag = self._coerce_bool(payload.get("show_center_line"), default=True)
                    try:
                        self.show_center_line_setter(flag)
                    except Exception:
                        pass
                if self.vision_mode_setter is not None and "vision_mode" in payload:
                    mode = self._coerce_vision_mode(payload.get("vision_mode"))
                    if mode is not None:
                        try:
                            self.vision_mode_setter(mode)
                        except Exception:
                            pass
                if self.markerless_profile_setter is not None and "markerless_profile" in payload:
                    profile = self._coerce_markerless_profile(payload.get("markerless_profile"))
                    if profile is not None:
                        try:
                            self.markerless_profile_setter(profile)
                        except Exception:
                            pass

            show_center_line = True
            if self.show_center_line_getter is not None:
                try:
                    show_center_line = bool(self.show_center_line_getter())
                except Exception:
                    show_center_line = True

            vision_mode = self.vision_mode_options[0][0] if self.vision_mode_options else None
            if self.vision_mode_getter is not None:
                try:
                    mode = self._coerce_vision_mode(self.vision_mode_getter())
                    if mode is not None:
                        vision_mode = mode
                except Exception:
                    pass

            vision_mode_options_payload = [
                {"value": value, "label": label}
                for value, label in self.vision_mode_options
            ]
            markerless_profile = self.markerless_profile_options[0][0] if self.markerless_profile_options else None
            if self.markerless_profile_getter is not None:
                try:
                    profile = self._coerce_markerless_profile(self.markerless_profile_getter())
                    if profile is not None:
                        markerless_profile = profile
                except Exception:
                    pass
            markerless_profile_options_payload = [
                {"value": value, "label": label}
                for value, label in self.markerless_profile_options
            ]
            return jsonify(
                {
                    "show_center_line": bool(show_center_line),
                    "editable": self.show_center_line_setter is not None,
                    "show_center_line_editable": self.show_center_line_setter is not None,
                    "vision_mode": vision_mode,
                    "vision_mode_editable": self.vision_mode_setter is not None,
                    "vision_mode_options": vision_mode_options_payload,
                    "markerless_profile": markerless_profile,
                    "markerless_profile_editable": self.markerless_profile_setter is not None,
                    "markerless_profile_options": markerless_profile_options_payload,
                }
            )

    def start(self):
        def _run():
            try:
                # Silence Flask's startup banner (e.g. "* Serving Flask app ...").
                flask_cli.show_server_banner = lambda *args, **kwargs: None
                self.app.run(
                    host=self.host,
                    port=self.port,
                    debug=False,
                    use_reloader=False,
                    threaded=True,
                )
            except BaseException as exc:
                self._startup_error = exc

        self._thread = threading.Thread(
            target=_run,
            daemon=True,
        )
        self._thread.start()
        return self._thread

    def stop(self):
        self._stop.set()

    def wait_until_ready(self, timeout_s=1.0):
        connect_url = format_stream_url(self.host, self.port)
        # Strip scheme and brackets for socket connection.
        connect_host = connect_url.split("://", 1)[-1].rsplit(":", 1)[0]
        if connect_host.startswith("[") and connect_host.endswith("]"):
            connect_host = connect_host[1:-1]
        deadline = time.time() + max(0.0, float(timeout_s))
        last_error = None
        while time.time() < deadline:
            if self._startup_error is not None:
                raise RuntimeError("Stream server failed to start.") from self._startup_error
            try:
                with socket.create_connection((connect_host, int(self.port)), timeout=0.2):
                    return True
            except OSError as exc:
                last_error = exc
                time.sleep(0.05)
        if self._startup_error is not None:
            raise RuntimeError("Stream server failed to start.") from self._startup_error
        raise TimeoutError(f"Stream server not reachable at {connect_url}.") from last_error

    def _index_html(self):
        width_attr = f' width="{self.img_width}"' if self.img_width else ""
        header_html = f"<h1>{self.header}</h1>" if self.header else ""
        footer_html = f"<p>{self.footer}</p>" if self.footer else ""
        controls_html = ""
        controls_script = ""
        has_center_line_control = self.show_center_line_getter is not None or self.show_center_line_setter is not None
        has_vision_mode_control = bool(self.vision_mode_options) and (
            self.vision_mode_getter is not None or self.vision_mode_setter is not None
        )
        has_markerless_profile_control = bool(self.markerless_profile_options) and (
            self.markerless_profile_getter is not None or self.markerless_profile_setter is not None
        )
        if has_center_line_control or has_vision_mode_control or has_markerless_profile_control:
            controls_parts = ["<div class='controls'>"]
            if has_center_line_control:
                controls_parts.append(
                    "<label class='control-item'><input type='checkbox' id='showCenterLine' checked> Show center line</label>"
                )
            if has_vision_mode_control:
                radio_parts = []
                for value, label in self.vision_mode_options:
                    value_escaped = html.escape(str(value), quote=True)
                    label_escaped = html.escape(str(label))
                    radio_parts.append(
                        "<label class='control-item'>"
                        f"<input type='radio' name='visionMode' value='{value_escaped}'> {label_escaped}"
                        "</label>"
                    )
                controls_parts.append("<div class='vision-mode'>" + "".join(radio_parts) + "</div>")
            if has_markerless_profile_control:
                option_parts = []
                for value, label in self.markerless_profile_options:
                    value_escaped = html.escape(str(value), quote=True)
                    label_escaped = html.escape(str(label))
                    option_parts.append(f"<option value='{value_escaped}'>{label_escaped}</option>")
                controls_parts.append(
                    "<label class='control-item'>"
                    "Markerless Config: "
                    "<select id='markerlessProfile' class='control-select'>"
                    + "".join(option_parts)
                    + "</select>"
                    "</label>"
                )
            controls_parts.append("</div>")
            controls_html = "".join(controls_parts)

            controls_script = (
                "<script>"
                "const centerLineToggle = document.getElementById('showCenterLine');"
                "const visionModeInputs = Array.from(document.querySelectorAll(\"input[name='visionMode']\"));"
                "const markerlessProfileSelect = document.getElementById('markerlessProfile');"
                "const setVisionMode = (mode, editable) => {"
                "if (!visionModeInputs.length) return;"
                "let matched = false;"
                "visionModeInputs.forEach((input) => {"
                "const isMatch = mode !== null && mode !== undefined && input.value === String(mode);"
                "input.checked = isMatch;"
                "if (isMatch) matched = true;"
                "input.disabled = !editable;"
                "});"
                "if (!matched && visionModeInputs.length) {"
                "visionModeInputs[0].checked = true;"
                "}"
                "};"
                "const setMarkerlessProfile = (profile, editable) => {"
                "if (!markerlessProfileSelect) return;"
                "if (profile !== null && profile !== undefined) {"
                "markerlessProfileSelect.value = String(profile);"
                "}"
                "markerlessProfileSelect.disabled = !editable;"
                "};"
                "const postPrefs = async (payload) => {"
                "try {"
                "await fetch('/stream_prefs', {"
                "method:'POST',"
                "headers:{'Content-Type':'application/json'},"
                "body:JSON.stringify(payload)"
                "});"
                "} catch (e) { /* ignore */ }"
                "};"
                "const syncPrefs = async () => {"
                "try {"
                "const res = await fetch('/stream_prefs', {cache:'no-store'});"
                "if (!res.ok) return;"
                "const data = await res.json();"
                "if (centerLineToggle) {"
                "const centerEditable = (data.show_center_line_editable !== undefined)"
                "? !!data.show_center_line_editable : !!data.editable;"
                "centerLineToggle.checked = !!data.show_center_line;"
                "centerLineToggle.disabled = !centerEditable;"
                "}"
                "if (visionModeInputs.length) {"
                "setVisionMode(data.vision_mode, !!data.vision_mode_editable);"
                "}"
                "if (markerlessProfileSelect) {"
                "setMarkerlessProfile(data.markerless_profile, !!data.markerless_profile_editable);"
                "}"
                "} catch (e) { /* ignore */ }"
                "};"
                "if (centerLineToggle) {"
                "centerLineToggle.addEventListener('change', async () => {"
                "await postPrefs({show_center_line:centerLineToggle.checked});"
                "syncPrefs();"
                "});"
                "}"
                "visionModeInputs.forEach((input) => {"
                "input.addEventListener('change', async () => {"
                "if (!input.checked) return;"
                "await postPrefs({vision_mode:input.value});"
                "syncPrefs();"
                "});"
                "});"
                "if (markerlessProfileSelect) {"
                "markerlessProfileSelect.addEventListener('change', async () => {"
                "await postPrefs({markerless_profile:markerlessProfileSelect.value});"
                "syncPrefs();"
                "});"
                "}"
                "syncPrefs();"
                "</script>"
            )
        if self.text_provider is None:
            return (
                "<html><head><title>"
                f"{self.title}"
                "</title>"
                "<style>"
                "body{background:#1a1a1a;color:#eee;font-family:sans-serif;"
                "text-align:center;margin-top:40px;}"
                ".stream{display:inline-block;border:4px solid #333;border-radius:8px;"
                "overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.35);}" 
                "img{image-rendering:auto;}"
                "h1{color:#f0ad4e;}"
                ".controls{margin:0 auto 12px auto;display:inline-flex;gap:12px;"
                "align-items:center;justify-content:center;flex-wrap:wrap;}"
                ".control-item{white-space:nowrap;}"
                ".vision-mode{display:inline-flex;gap:12px;align-items:center;}"
                ".control-select{margin-left:6px;}"
                "</style>"
                "</head><body>"
                f"{header_html}"
                f"{controls_html}"
                f"<div class='stream'><img src='/video_feed'{width_attr}></div>"
                f"{footer_html}"
                f"{controls_script}"
                "</body></html>"
            )
        return (
            "<html><head><title>"
            f"{self.title}"
            "</title>"
            "<style>"
            "body{background:#1a1a1a;color:#eee;font-family:sans-serif;"
            "text-align:center;margin-top:40px;}"
            ".layout{display:flex;justify-content:center;gap:20px;align-items:flex-start;"
            "margin-top:20px;}"
            ".sidebar{min-width:260px;max-width:360px;background:#111;border:1px solid #333;"
            "border-radius:8px;padding:12px;text-align:left;box-shadow:0 6px 20px rgba(0,0,0,0.35);}"
            ".telemetry{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;"
            "font-size:13px;line-height:1.4;white-space:pre-wrap;overflow-wrap:anywhere;"
            "word-break:break-word;user-select:text;}"
            ".stream{display:inline-block;border:4px solid #333;border-radius:8px;"
            "overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.35);}" 
            "img{image-rendering:auto;}"
            "h1{color:#f0ad4e;}"
            ".controls{margin:0 auto 12px auto;display:inline-flex;gap:12px;"
            "align-items:center;justify-content:center;flex-wrap:wrap;}"
            ".control-item{white-space:nowrap;}"
            ".vision-mode{display:inline-flex;gap:12px;align-items:center;}"
            ".control-select{margin-left:6px;}"
            "</style>"
            "</head><body>"
            f"{header_html}"
            f"{controls_html}"
            "<div class='layout'>"
            "<div class='sidebar'><div id='telemetry' class='telemetry'></div></div>"
            f"<div class='stream'><img src='/video_feed'{width_attr}></div>"
            "</div>"
            f"{footer_html}"
            "<script>"
            "const telemetryEl = document.getElementById('telemetry');"
            "const esc = (s) => String(s)"
            ".replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')"
            ".replace(/\"/g,'&quot;').replace(/'/g,'&#39;');"
            "const renderLine = (line) => {"
            "if (line && Array.isArray(line.segments)) {"
            "return '<div>' + line.segments.map(seg => {"
            "const color = seg.color || '#ffffff';"
            "return '<span style=\"color:' + color + '\">' + esc(seg.text || '') + '</span>';"
            "}).join('') + '</div>';"
            "}"
            "const color = (line && line.color) ? line.color : '#ffffff';"
            "const text = (line && line.text) ? String(line.text) : '';"
            "if (!text) { return '<div>&nbsp;</div>'; }"
            "return '<div style=\"color:' + color + '\">' + esc(text) + '</div>';"
            "};"
            "const refresh = async () => {"
            "try {"
            "const res = await fetch('/text', {cache:'no-store'});"
            "if (!res.ok) return;"
            "const data = await res.json();"
            "const lines = Array.isArray(data.lines) ? data.lines : [];"
            "telemetryEl.innerHTML = lines.map(renderLine).join('');"
            "} catch (e) { /* ignore */ }"
            "};"
            "setInterval(refresh, 100);"
            "refresh();"
            "</script>"
            f"{controls_script}"
            "</body></html>"
        )

    def _generate_frames(self):
        frame_interval = 1.0 / self.fps
        last_sent = 0.0
        while not self._stop.is_set():
            now = time.time()
            elapsed = now - last_sent
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            try:
                frame = self.frame_provider()
            except Exception:
                frame = None

            if frame is None:
                frame = self._placeholder_frame()
            else:
                frame = frame.copy()

            if self.sharpen:
                frame = self._apply_sharpen(frame)

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if not ok:
                continue
            last_sent = time.time()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + bytearray(encoded)
                + b"\r\n"
            )

    @staticmethod
    def _placeholder_frame():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "WAITING FOR CAMERA...",
            (120, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        return frame

    @staticmethod
    def _apply_sharpen(frame):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)

    @staticmethod
    def _coerce_bool(value, default=True):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    def _coerce_vision_mode(self, value):
        if not self._vision_mode_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._vision_mode_allowed:
            return candidate
        return None

    def _coerce_markerless_profile(self, value):
        if not self._markerless_profile_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._markerless_profile_allowed:
            return candidate
        return None

    @staticmethod
    def _normalize_options(options):
        if not options:
            return []
        normalized = []
        seen = set()
        for item in options:
            value = None
            label = None
            if isinstance(item, dict):
                value = item.get("value")
                label = item.get("label")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                value = item[0]
                label = item[1]
            if value is None:
                continue
            value_norm = str(value).strip().lower()
            if not value_norm or value_norm in seen:
                continue
            label_norm = str(label).strip() if label is not None else value_norm
            if not label_norm:
                label_norm = value_norm
            normalized.append((value_norm, label_norm))
            seen.add(value_norm)
        return normalized
