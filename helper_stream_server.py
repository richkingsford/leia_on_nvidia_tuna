import threading
import time
import socket
import logging
import html
import json
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from flask import cli as flask_cli

DEFAULT_STREAM_HOST = "127.0.0.1"
DEFAULT_STREAM_FPS = 10
DEFAULT_JPEG_QUALITY = 90
DEFAULT_GONG_FILE = Path(__file__).resolve().parent / "gong.mp3"
DEFAULT_BRICK_MODEL_FILE = Path(__file__).resolve().parent / "world_model_brick.json"


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
        text_provider: Optional[Callable[[], object]] = None,
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
        cyan_profile_getter: Optional[Callable[[], str]] = None,
        cyan_profile_setter: Optional[Callable[[str], None]] = None,
        cyan_profile_options: Optional[list] = None,
        cyan_visibility_getter: Optional[Callable[[], str]] = None,
        cyan_visibility_setter: Optional[Callable[[str], None]] = None,
        cyan_visibility_options: Optional[list] = None,
        markerless_profile_getter: Optional[Callable[[], str]] = None,
        markerless_profile_setter: Optional[Callable[[str], None]] = None,
        markerless_profile_options: Optional[list] = None,
        markerless_visibility_getter: Optional[Callable[[], str]] = None,
        markerless_visibility_setter: Optional[Callable[[str], None]] = None,
        markerless_visibility_options: Optional[list] = None,
        success_gate_step_getter: Optional[Callable[[], str]] = None,
        success_gate_step_setter: Optional[Callable[[str], None]] = None,
        success_gate_step_options: Optional[list] = None,
        depth_source_getter: Optional[Callable[[], str]] = None,
        depth_source_setter: Optional[Callable[[str], None]] = None,
        depth_source_options: Optional[list] = None,
        stereo_config_getter: Optional[Callable[[], str]] = None,
        stereo_config_setter: Optional[Callable[[str], None]] = None,
        stereo_config_options: Optional[list] = None,
        gong_file_path: Optional[str] = None,
        xyz_workspace_getter: Optional[Callable[[], Optional[dict]]] = None,
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
            self.vision_mode_options = [("cyan", "Crown Bricks")]
        self._vision_mode_allowed = {value for value, _label in self.vision_mode_options}
        if cyan_profile_getter is None:
            cyan_profile_getter = markerless_profile_getter
        if cyan_profile_setter is None:
            cyan_profile_setter = markerless_profile_setter
        if cyan_profile_options is None:
            cyan_profile_options = markerless_profile_options
        if cyan_visibility_getter is None:
            cyan_visibility_getter = markerless_visibility_getter
        if cyan_visibility_setter is None:
            cyan_visibility_setter = markerless_visibility_setter
        if cyan_visibility_options is None:
            cyan_visibility_options = markerless_visibility_options
        self.cyan_profile_getter = cyan_profile_getter
        self.cyan_profile_setter = cyan_profile_setter
        self.cyan_profile_options = self._normalize_options(cyan_profile_options)
        self._cyan_profile_allowed = {value for value, _label in self.cyan_profile_options}
        self.cyan_visibility_getter = cyan_visibility_getter
        self.cyan_visibility_setter = cyan_visibility_setter
        self.cyan_visibility_options = self._normalize_options(cyan_visibility_options)
        self._cyan_visibility_allowed = {value for value, _label in self.cyan_visibility_options}
        self.success_gate_step_getter = success_gate_step_getter
        self.success_gate_step_setter = success_gate_step_setter
        self.success_gate_step_options = self._normalize_options(success_gate_step_options)
        self._success_gate_step_allowed = {value for value, _label in self.success_gate_step_options}
        self.depth_source_getter = depth_source_getter
        self.depth_source_setter = depth_source_setter
        self.depth_source_options = self._normalize_options(depth_source_options)
        self._depth_source_allowed = {value for value, _label in self.depth_source_options}
        self.stereo_config_getter = stereo_config_getter
        self.stereo_config_setter = stereo_config_setter
        self.stereo_config_options = self._normalize_options(stereo_config_options)
        self._stereo_config_allowed = {value for value, _label in self.stereo_config_options}
        self.gong_file_path = Path(gong_file_path) if gong_file_path is not None else Path(DEFAULT_GONG_FILE)
        self.xyz_workspace_getter = xyz_workspace_getter
        self._stop = threading.Event()
        self._thread = None
        self._startup_error = None
        self._instance_id = f"{int(time.time() * 1000)}-{id(self):x}"

        self.app = Flask(__name__)
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        self.app.logger.disabled = True

        @self.app.after_request
        def _disable_dynamic_response_cache(response):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        @self.app.route("/")
        def index():
            return self._index_html()

        @self.app.route("/video_feed")
        def video_feed():
            resp = Response(
                self._generate_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
            resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
            return resp

        @self.app.route("/text")
        def text_feed():
            lines = []
            step_success_payload = None
            if self.text_provider is not None:
                try:
                    payload = self.text_provider()
                except Exception:
                    payload = None
                if isinstance(payload, list):
                    lines = payload
                elif isinstance(payload, dict):
                    raw_lines = payload.get("lines")
                    if isinstance(raw_lines, list):
                        lines = raw_lines
                    raw_step_success = payload.get("step_success")
                    if isinstance(raw_step_success, dict):
                        seq_val = raw_step_success.get("seq")
                        step_val = raw_step_success.get("step")
                        at_val = raw_step_success.get("at")
                        seq_out = None
                        if seq_val is not None:
                            try:
                                seq_out = int(seq_val)
                            except (TypeError, ValueError):
                                seq_out = None
                        step_out = None
                        if step_val is not None:
                            step_txt = str(step_val).strip()
                            if step_txt:
                                step_out = step_txt
                        at_out = None
                        if at_val is not None:
                            try:
                                at_out = float(at_val)
                            except (TypeError, ValueError):
                                at_out = None
                        if seq_out is not None:
                            step_success_payload = {
                                "seq": int(seq_out),
                                "step": step_out,
                                "at": at_out,
                            }
            return jsonify(
                {
                    "lines": lines,
                    "server_id": self._instance_id,
                    "step_success": step_success_payload,
                }
            )

        @self.app.route("/gong.mp3")
        def gong_audio():
            if self.gong_file_path.exists() and self.gong_file_path.is_file():
                return send_file(str(self.gong_file_path), mimetype="audio/mpeg")
            return ("", 404)

        @self.app.route("/xyz_workspace_live.svg")
        def xyz_workspace_live():
            if self.xyz_workspace_getter:
                try:
                    state = self.xyz_workspace_getter()
                except Exception:
                    state = None
                from helper_xyz_coords import render_workspace_svg
                svg_data = render_workspace_svg(state)
                response = Response(svg_data, mimetype="image/svg+xml")
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response
            
            # fallback local lookup
            svg_path = Path(__file__).resolve().parent / "xyz layout" / "xyz_workspace_live.svg"
            if svg_path.exists() and svg_path.is_file():
                response = send_file(str(svg_path), mimetype="image/svg+xml")
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response
            return ("", 404)

        @self.app.route("/xyz_mast_live.svg")
        def xyz_mast_live():
            if self.xyz_workspace_getter:
                try:
                    state = self.xyz_workspace_getter()
                except Exception:
                    state = None
                from helper_xyz_coords import render_mast_svg
                svg_data = render_mast_svg(state)
                response = Response(svg_data, mimetype="image/svg+xml")
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response

            mast_svg_path = Path(__file__).resolve().parent / "xyz layout" / "mast_view.svg"
            if mast_svg_path.exists() and mast_svg_path.is_file():
                response = send_file(str(mast_svg_path), mimetype="image/svg+xml")
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response
            return ("", 404)

        @self.app.route("/__shutdown_stream_server__", methods=["POST"])
        def _shutdown_stream_server():
            # Best-effort local shutdown for debugger restarts / clean script exits.
            self._stop.set()
            shutdown_fn = request.environ.get("werkzeug.server.shutdown")
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception:
                    pass
            return ("", 204)

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
                profile_payload = None
                if "cyan_profile" in payload:
                    profile_payload = payload.get("cyan_profile")
                elif "markerless_profile" in payload:
                    profile_payload = payload.get("markerless_profile")
                if self.cyan_profile_setter is not None and profile_payload is not None:
                    profile = self._coerce_cyan_profile(profile_payload)
                    if profile is not None:
                        try:
                            self.cyan_profile_setter(profile)
                        except Exception:
                            pass
                visibility_payload = None
                if "cyan_visibility" in payload:
                    visibility_payload = payload.get("cyan_visibility")
                elif "markerless_visibility" in payload:
                    visibility_payload = payload.get("markerless_visibility")
                if self.cyan_visibility_setter is not None and visibility_payload is not None:
                    mode = self._coerce_cyan_visibility(visibility_payload)
                    if mode is not None:
                        try:
                            self.cyan_visibility_setter(mode)
                        except Exception:
                            pass
                if self.success_gate_step_setter is not None and "success_gate_step" in payload:
                    step = self._coerce_success_gate_step(payload.get("success_gate_step"))
                    if step is not None:
                        try:
                            self.success_gate_step_setter(step)
                        except Exception:
                            pass
                if self.depth_source_setter is not None and "depth_source" in payload:
                    val = self._coerce_depth_source(payload.get("depth_source"))
                    if val is not None:
                        try:
                            self.depth_source_setter(val)
                        except Exception:
                            pass
                if self.stereo_config_setter is not None and "stereo_config" in payload:
                    val = self._coerce_stereo_config(payload.get("stereo_config"))
                    if val is not None:
                        try:
                            self.stereo_config_setter(val)
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
            cyan_profile = self.cyan_profile_options[0][0] if self.cyan_profile_options else None
            if self.cyan_profile_getter is not None:
                try:
                    profile = self._coerce_cyan_profile(self.cyan_profile_getter())
                    if profile is not None:
                        cyan_profile = profile
                except Exception:
                    pass
            cyan_profile_options_payload = [
                {"value": value, "label": label}
                for value, label in self.cyan_profile_options
            ]
            cyan_visibility = self.cyan_visibility_options[0][0] if self.cyan_visibility_options else None
            if self.cyan_visibility_getter is not None:
                try:
                    mode = self._coerce_cyan_visibility(self.cyan_visibility_getter())
                    if mode is not None:
                        cyan_visibility = mode
                except Exception:
                    pass
            cyan_visibility_options_payload = [
                {"value": value, "label": label}
                for value, label in self.cyan_visibility_options
            ]
            success_gate_step = self.success_gate_step_options[0][0] if self.success_gate_step_options else None
            if self.success_gate_step_getter is not None:
                try:
                    step = self._coerce_success_gate_step(self.success_gate_step_getter())
                    if step is not None:
                        success_gate_step = step
                except Exception:
                    pass
            success_gate_step_options_payload = [
                {"value": value, "label": label}
                for value, label in self.success_gate_step_options
            ]
            depth_source = self.depth_source_options[0][0] if self.depth_source_options else None
            if self.depth_source_getter is not None:
                try:
                    val = self._coerce_depth_source(self.depth_source_getter())
                    if val is not None:
                        depth_source = val
                except Exception:
                    pass
            depth_source_options_payload = [
                {"value": value, "label": label}
                for value, label in self.depth_source_options
            ]
            stereo_config = self.stereo_config_options[0][0] if self.stereo_config_options else None
            if self.stereo_config_getter is not None:
                try:
                    val = self._coerce_stereo_config(self.stereo_config_getter())
                    if val is not None:
                        stereo_config = val
                except Exception:
                    pass
            stereo_config_options_payload = [
                {"value": value, "label": label}
                for value, label in self.stereo_config_options
            ]
            return jsonify(
                {
                    "show_center_line": bool(show_center_line),
                    "editable": self.show_center_line_setter is not None,
                    "show_center_line_editable": self.show_center_line_setter is not None,
                    "vision_mode": vision_mode,
                    "vision_mode_editable": self.vision_mode_setter is not None,
                    "vision_mode_options": vision_mode_options_payload,
                    "cyan_profile": cyan_profile,
                    "cyan_profile_editable": self.cyan_profile_setter is not None,
                    "cyan_profile_options": cyan_profile_options_payload,
                    "cyan_visibility": cyan_visibility,
                    "cyan_visibility_editable": self.cyan_visibility_setter is not None,
                    "cyan_visibility_options": cyan_visibility_options_payload,
                    "success_gate_step": success_gate_step,
                    "success_gate_step_editable": self.success_gate_step_setter is not None,
                    "success_gate_step_options": success_gate_step_options_payload,
                    "depth_source": depth_source,
                    "depth_source_editable": self.depth_source_setter is not None,
                    "depth_source_options": depth_source_options_payload,
                    "stereo_config": stereo_config,
                    "stereo_config_editable": self.stereo_config_setter is not None,
                    "stereo_config_options": stereo_config_options_payload,
                    # Backward-compatible payload keys for old UI clients.
                    "markerless_profile": cyan_profile,
                    "markerless_profile_editable": self.cyan_profile_setter is not None,
                    "markerless_profile_options": cyan_profile_options_payload,
                    "markerless_visibility": cyan_visibility,
                    "markerless_visibility_editable": self.cyan_visibility_setter is not None,
                    "markerless_visibility_options": cyan_visibility_options_payload,
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
        # Best-effort shutdown of the local Flask dev server to release the port.
        try:
            shutdown_url = format_stream_url(self.host, self.port).rstrip("/") + "/__shutdown_stream_server__"
            req = urllib.request.Request(shutdown_url, data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=0.4):
                pass
        except Exception:
            pass
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass

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
        xyz_refresh_ms = 250
        width_attr = f' width="{self.img_width}"' if self.img_width else ""
        xyz_width_attr = width_attr
        camera_width_attr = width_attr
        layout_xyz_img_width_attr = ""
        layout_camera_img_width_attr = camera_width_attr
        if self.img_width:
            try:
                xyz_width = max(160, int(round(float(self.img_width) * 0.85)))
            except (TypeError, ValueError):
                xyz_width = None
            if xyz_width:
                xyz_width_attr = f' width="{int(xyz_width)}"'
            try:
                camera_width = max(160, int(round(float(self.img_width) * 0.5)))
            except (TypeError, ValueError):
                camera_width = None
            if camera_width:
                camera_width_attr = f' width="{int(camera_width)}"'
            try:
                # Keep the live views readable while the left sidebar holds all
                # operator-facing telemetry and vision reference details.
                layout_width = max(160, int(round(float(self.img_width) * 0.52)))
            except (TypeError, ValueError):
                layout_width = None
            if layout_width:
                layout_xyz_img_width_attr = f' width="{max(160, int(round(float(layout_width) * 0.85)))}"'
                layout_camera_width = max(120, int(round(float(layout_width) * 0.5)))
                layout_camera_img_width_attr = f' width="{int(layout_camera_width)}"'
        header_html = f"<h1>{self.header}</h1>" if self.header else ""
        brick_shape_panel_html = self._brick_shape_panel_html()
        footer_panel_html = f"<div class='footer-panel'>{self.footer}</div>" if self.footer else ""
        footer_below_html = (
            f"<div class='footer-wrap footer-wrap-standalone'>{footer_panel_html}</div>"
            if footer_panel_html
            else ""
        )
        controls_html = ""
        controls_script = ""
        has_center_line_control = self.show_center_line_getter is not None or self.show_center_line_setter is not None
        has_vision_mode_control = bool(self.vision_mode_options) and (
            self.vision_mode_getter is not None or self.vision_mode_setter is not None
        )
        has_cyan_profile_control = len(self.cyan_profile_options) > 1 and (
            self.cyan_profile_getter is not None or self.cyan_profile_setter is not None
        )
        has_cyan_visibility_control = bool(self.cyan_visibility_options) and (
            self.cyan_visibility_getter is not None or self.cyan_visibility_setter is not None
        )
        has_success_gate_step_control = bool(self.success_gate_step_options) and (
            self.success_gate_step_getter is not None or self.success_gate_step_setter is not None
        )
        has_depth_source_control = bool(self.depth_source_options) and (
            self.depth_source_getter is not None or self.depth_source_setter is not None
        )
        has_stereo_config_control = bool(self.stereo_config_options) and (
            self.stereo_config_getter is not None or self.stereo_config_setter is not None
        )
        has_top_controls = (
            has_center_line_control
            or has_vision_mode_control
            or has_cyan_profile_control
            or has_cyan_visibility_control
            or has_depth_source_control
            or has_stereo_config_control
        )
        body_controls_class = " class='with-bottom-controls'" if has_top_controls else ""
        if has_top_controls or has_success_gate_step_control:
            controls_parts = []
            if has_top_controls:
                controls_parts.append("<div class='controls'>")
            if has_center_line_control:
                controls_parts.append(
                    "<label class='control-item'><input type='checkbox' id='showCenterLine' checked> Show center lines</label>"
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
            if has_cyan_profile_control:
                option_parts = []
                for value, label in self.cyan_profile_options:
                    value_escaped = html.escape(str(value), quote=True)
                    label_escaped = html.escape(str(label))
                    option_parts.append(f"<option value='{value_escaped}'>{label_escaped}</option>")
                controls_parts.append(
                    "<label class='control-item'>"
                    "Tri Brick Config: "
                    "<select id='cyanProfile' class='control-select'>"
                    + "".join(option_parts)
                    + "</select>"
                    "</label>"
                )
            if has_cyan_visibility_control:
                option_parts = []
                for value, label in self.cyan_visibility_options:
                    value_escaped = html.escape(str(value), quote=True)
                    label_escaped = html.escape(str(label))
                    option_parts.append(f"<option value='{value_escaped}'>{label_escaped}</option>")
                controls_parts.append(
                    "<label class='control-item'>"
                    "Brick Visibility: "
                    "<select id='cyanVisibility' class='control-select'>"
                    + "".join(option_parts)
                    + "</select>"
                    "</label>"
                )
            if has_depth_source_control:
                option_parts = []
                for value, label in self.depth_source_options:
                    value_escaped = html.escape(str(value), quote=True)
                    label_escaped = html.escape(str(label))
                    option_parts.append(f"<option value='{value_escaped}'>{label_escaped}</option>")
                controls_parts.append(
                    "<label class='control-item'>"
                    "Depth Source: "
                    "<select id='depthSource' class='control-select'>"
                    + "".join(option_parts)
                    + "</select>"
                    "</label>"
                )
            if has_stereo_config_control:
                option_parts = []
                for value, label in self.stereo_config_options:
                    value_escaped = html.escape(str(value), quote=True)
                    label_escaped = html.escape(str(label))
                    option_parts.append(f"<option value='{value_escaped}'>{label_escaped}</option>")
                controls_parts.append(
                    "<label class='control-item'>"
                    "Stereo Config: "
                    "<select id='stereoConfig' class='control-select'>"
                    + "".join(option_parts)
                    + "</select>"
                    "</label>"
                )
            if has_top_controls:
                controls_parts.append("</div>")
                controls_html = "".join(controls_parts)

            controls_script = (
                "<script>"
                "const centerLineToggle = document.getElementById('showCenterLine');"
                "const visionModeInputs = Array.from(document.querySelectorAll(\"input[name='visionMode']\"));"
                "const cyanProfileSelect = document.getElementById('cyanProfile');"
                "const cyanVisibilitySelect = document.getElementById('cyanVisibility');"
                "const depthSourceSelect = document.getElementById('depthSource');"
                "const stereoConfigSelect = document.getElementById('stereoConfig');"
                "const telemetryHostEl = document.getElementById('telemetry');"
                "let successGateStep = null;"
                "let successGateStepEditable = false;"
                "let successGateStepOptions = [];"
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
                "const setCyanProfile = (profile, editable) => {"
                "if (!cyanProfileSelect) return;"
                "if (profile !== null && profile !== undefined) {"
                "cyanProfileSelect.value = String(profile);"
                "}"
                "cyanProfileSelect.disabled = !editable;"
                "};"
                "const setCyanVisibility = (mode, editable) => {"
                "if (!cyanVisibilitySelect) return;"
                "if (mode !== null && mode !== undefined) {"
                "cyanVisibilitySelect.value = String(mode);"
                "}"
                "cyanVisibilitySelect.disabled = !editable;"
                "};"
                "const setDepthSource = (val, editable) => {"
                "if (!depthSourceSelect) return;"
                "if (val !== null && val !== undefined) {"
                "depthSourceSelect.value = String(val);"
                "}"
                "depthSourceSelect.disabled = !editable;"
                "};"
                "const setStereoConfig = (val, editable) => {"
                "if (!stereoConfigSelect) return;"
                "if (val !== null && val !== undefined) {"
                "stereoConfigSelect.value = String(val);"
                "}"
                "stereoConfigSelect.disabled = !editable;"
                "};"
                "const setSuccessGateStepState = (step, editable, options) => {"
                "if (Array.isArray(options)) {"
                "successGateStepOptions = options.map((opt) => {"
                "if (!opt) return null;"
                "const value = (opt.value !== undefined && opt.value !== null) ? String(opt.value) : '';"
                "if (!value) return null;"
                "const label = (opt.label !== undefined && opt.label !== null) ? String(opt.label) : value;"
                "return {value, label};"
                "}).filter((opt) => !!opt);"
                "}"
                "if (step !== null && step !== undefined) {"
                "successGateStep = String(step);"
                "} else if (successGateStepOptions.length) {"
                "successGateStep = String(successGateStepOptions[0].value);"
                "} else {"
                "successGateStep = null;"
                "}"
                "successGateStepEditable = !!editable;"
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
                "const lineText = (line) => {"
                "if (line && Array.isArray(line.segments)) {"
                "return line.segments.map((seg) => String((seg && seg.text) || '')).join('');"
                "}"
                "if (line && line.text !== undefined && line.text !== null) {"
                "return String(line.text);"
                "}"
                "return '';"
                "};"
                "const renderSuccessGateStepControl = () => {"
                "if (!successGateStepOptions.length) return '';"
                "const optionsHtml = successGateStepOptions.map((opt) => {"
                "const value = String(opt.value);"
                "const label = String(opt.label || opt.value);"
                "const selected = successGateStep !== null && value === String(successGateStep) ? ' selected' : '';"
                "return '<option value=\"' + esc(value) + '\"' + selected + '>' + esc(label) + '</option>';"
                "}).join('');"
                "const disabledAttr = successGateStepEditable ? '' : ' disabled';"
                "return '<div class=\"gate-step-inline\"><label class=\"gate-step-label\">Step: <select id=\"successGateStepSelect\" class=\"control-select gate-step-select\"' + disabledAttr + '>' + optionsHtml + '</select></label></div>';"
                "};"
                "window.injectSuccessGateStepControl = (lines, renderedLines) => {"
                "if (!telemetryHostEl || !Array.isArray(renderedLines)) {"
                "return Array.isArray(renderedLines) ? renderedLines.join('') : '';"
                "}"
                "if (!successGateStepOptions.length || !Array.isArray(lines)) {"
                "return renderedLines.join('');"
                "}"
                "let successTitleIdx = -1;"
                "for (let i = 0; i < lines.length; i += 1) {"
                "if (lineText(lines[i]).trim() === '--- SUCCESS GATES ---') {"
                "successTitleIdx = i;"
                "break;"
                "}"
                "}"
                "if (successTitleIdx < 0) {"
                "return renderedLines.join('');"
                "}"
                "const merged = renderedLines.slice();"
                "merged.splice(successTitleIdx + 1, 0, renderSuccessGateStepControl());"
                "return merged.join('');"
                "};"
                "let prefsSyncInFlight = false;"
                "const syncPrefs = async () => {"
                "if (prefsSyncInFlight) return;"
                "prefsSyncInFlight = true;"
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
                "if (cyanProfileSelect) {"
                "setCyanProfile(data.cyan_profile, !!data.cyan_profile_editable);"
                "}"
                "if (cyanVisibilitySelect) {"
                "setCyanVisibility(data.cyan_visibility, !!data.cyan_visibility_editable);"
                "}"
                "if (depthSourceSelect) {"
                "setDepthSource(data.depth_source, !!data.depth_source_editable);"
                "}"
                "if (stereoConfigSelect) {"
                "setStereoConfig(data.stereo_config, !!data.stereo_config_editable);"
                "}"
                "setSuccessGateStepState("
                "data.success_gate_step,"
                "!!data.success_gate_step_editable,"
                "Array.isArray(data.success_gate_step_options) ? data.success_gate_step_options : null"
                ");"
                "} catch (e) { /* ignore */ }"
                "finally { prefsSyncInFlight = false; }"
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
                "if (cyanProfileSelect) {"
                "cyanProfileSelect.addEventListener('change', async () => {"
                "await postPrefs({cyan_profile:cyanProfileSelect.value});"
                "syncPrefs();"
                "});"
                "}"
                "if (cyanVisibilitySelect) {"
                "cyanVisibilitySelect.addEventListener('change', async () => {"
                "await postPrefs({cyan_visibility:cyanVisibilitySelect.value});"
                "syncPrefs();"
                "});"
                "}"
                "if (depthSourceSelect) {"
                "depthSourceSelect.addEventListener('change', async () => {"
                "await postPrefs({depth_source:depthSourceSelect.value});"
                "syncPrefs();"
                "});"
                "}"
                "if (stereoConfigSelect) {"
                "stereoConfigSelect.addEventListener('change', async () => {"
                "await postPrefs({stereo_config:stereoConfigSelect.value});"
                "syncPrefs();"
                "});"
                "}"
                "const handleSuccessGateStepSelection = async (target) => {"
                "if (!target || target.id !== 'successGateStepSelect') return;"
                "await postPrefs({success_gate_step:target.value});"
                "try { target.blur(); } catch (e) { /* ignore */ }"
                "syncPrefs();"
                "};"
                "if (telemetryHostEl) {"
                "telemetryHostEl.addEventListener('input', async (event) => {"
                "const target = event && event.target;"
                "await handleSuccessGateStepSelection(target);"
                "});"
                "telemetryHostEl.addEventListener('change', async (event) => {"
                "const target = event && event.target;"
                "await handleSuccessGateStepSelection(target);"
                "});"
                "}"
                "syncPrefs();"
                "setInterval(syncPrefs, 350);"
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
                ".with-bottom-controls{padding-bottom:94px;}"
                ".stream{display:inline-block;border:4px solid #333;border-radius:8px;"
                "overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.35);}" 
                "img{image-rendering:auto;max-width:100%;height:auto;display:block;}"
                "h1{color:#f0ad4e;}"
                ".controls{position:fixed;left:50%;bottom:12px;transform:translateX(-50%);"
                "margin:0;display:inline-flex;gap:12px;align-items:center;justify-content:center;"
                "flex-wrap:wrap;max-width:calc(100vw - 20px);padding:8px 12px;background:rgba(17,17,17,0.92);"
                "border:1px solid #333;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.35);z-index:1200;}"
                ".control-item{white-space:nowrap;}"
                ".vision-mode{display:inline-flex;gap:12px;align-items:center;}"
                ".control-select{margin-left:6px;}"
                ".gate-step-inline{padding:4px 0 6px 0;border-bottom:1px solid #242424;}"
                ".gate-step-label{display:flex;align-items:center;gap:6px;color:#ddd;}"
                ".gate-step-select{margin-left:0;flex:1;min-width:0;}"
                ".footer-wrap{margin:18px auto 0 auto;display:flex;justify-content:center;}"
                ".footer-wrap-standalone{max-width:min(1100px, calc(100vw - 40px));}"
                ".footer-panel{background:#111;border:1px solid #333;border-radius:8px;"
                "padding:12px;text-align:left;box-shadow:0 6px 20px rgba(0,0,0,0.35);"
                "font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;"
                "font-size:13px;line-height:1.35;}"
                ".footer-section + .footer-section{margin-top:12px;padding-top:10px;border-top:3px double #444;}"
                ".footer-title{color:#f0ad4e;font-weight:700;text-transform:uppercase;font-size:12px;"
                "letter-spacing:0.04em;margin-bottom:6px;}"
                ".footer-line{padding:4px 0;border-bottom:1px solid #242424;}"
                ".footer-line:last-child{border-bottom:none;}"
                "@media (max-width: 760px){"
                ".with-bottom-controls{padding-bottom:122px;}"
                ".controls{bottom:8px;gap:8px;padding:8px 10px;}"
                ".vision-mode{gap:8px;}"
                "}"
                "</style>"
                f"</head><body{body_controls_class}>"
                f"{header_html}"
                f"{controls_html}"
                f"<div class='xyz-layout-container' style='margin-top: 12px; display: flex; justify-content: center;'><div class='stream' style='background: white;'><img id='xyzLayout' src='/xyz_workspace_live.svg'{xyz_width_attr}></div></div>"
                f"<div class='xyz-layout-container' style='margin-top: 12px; display: flex; justify-content: center;'><div class='stream' style='background: white;'><img id='mastLayout' src='/xyz_mast_live.svg'{xyz_width_attr}></div></div>"
                f"<div class='stream' style='margin-top: 12px;'><img id='videoFeed' src='/video_feed'{camera_width_attr}></div>"
                f"{footer_below_html}"
                f"{controls_script}"
                f"<script>"
                "const _vf=document.getElementById('videoFeed');if(_vf){const _b=_vf.src.split('?')[0];_vf.src=_b+'?t='+Date.now();}"
                "const _svgRefreshState = { xyz: false, mast: false };"
                "const refreshSvgImage = (id, path, stateKey) => {"
                "const el = document.getElementById(id);"
                "if (!el) return;"
                "if (_svgRefreshState[stateKey]) return;"
                "_svgRefreshState[stateKey] = true;"
                "const nextSrc = path + '?t=' + Date.now();"
                "const probe = new Image();"
                "probe.onload = () => { el.src = nextSrc; _svgRefreshState[stateKey] = false; };"
                "probe.onerror = () => { _svgRefreshState[stateKey] = false; };"
                "probe.src = nextSrc;"
                "};"
                f"setInterval(() => {{ refreshSvgImage('xyzLayout', '/xyz_workspace_live.svg', 'xyz'); refreshSvgImage('mastLayout', '/xyz_mast_live.svg', 'mast'); }}, {int(xyz_refresh_ms)});"
                "</script>"
                "</body></html>"
            )
        return (
            "<html><head><title>"
            f"{self.title}"
            "</title>"
            "<style>"
            "body{background:#1a1a1a;color:#eee;font-family:sans-serif;"
            "text-align:center;margin-top:40px;}"
            ".with-bottom-controls{padding-bottom:94px;}"
            ".layout{display:flex;justify-content:center;gap:10px;align-items:flex-start;"
            "margin:20px auto 0 auto;flex-wrap:nowrap;overflow-x:auto;padding:0 8px;"
            "max-width:calc(100vw - 8px);}"
            ".sidebar{min-width:180px;max-width:220px;background:#111;border:1px solid #333;"
            "border-radius:8px;padding:10px;text-align:left;box-shadow:0 6px 20px rgba(0,0,0,0.35);}"
            ".info-sidebar{min-width:180px;max-width:240px;}"
            ".telemetry{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;"
            "font-size:12px;line-height:1.28;white-space:pre-wrap;overflow-wrap:anywhere;"
            "word-break:break-word;user-select:text;}"
            ".telemetry > div{padding:3px 0;border-bottom:1px solid #242424;}"
            ".telemetry > div:last-child{border-bottom:none;}"
            ".sidebar-stack{display:flex;flex-direction:column;gap:10px;}"
            ".shape-panel{background:#0f0f0f;border:1px solid #2f2f2f;border-radius:8px;padding:8px;text-align:left;}"
            ".shape-wrap{display:flex;justify-content:center;background:#141414;border:1px solid #2b2b2b;border-radius:6px;padding:6px;}"
            ".shape-svg{max-width:100%;height:auto;display:block;}"
            ".stream{display:inline-block;border:4px solid #333;border-radius:8px;"
            "overflow:hidden;box-shadow:0 6px 20px rgba(0,0,0,0.35);}" 
            ".layout .stream{flex:0 0 auto;}"
            ".layout .stream img{image-rendering:auto;max-width:100%;height:auto;display:block;}"
            "img{image-rendering:auto;max-width:100%;height:auto;display:block;}"
            "h1{color:#f0ad4e;}"
            ".controls{position:fixed;left:50%;bottom:12px;transform:translateX(-50%);"
            "margin:0;display:inline-flex;gap:12px;align-items:center;justify-content:center;"
            "flex-wrap:wrap;max-width:calc(100vw - 20px);padding:8px 12px;background:rgba(17,17,17,0.92);"
            "border:1px solid #333;border-radius:10px;box-shadow:0 6px 18px rgba(0,0,0,0.35);z-index:1200;}"
            ".control-item{white-space:nowrap;}"
            ".vision-mode{display:inline-flex;gap:12px;align-items:center;}"
            ".control-select{margin-left:6px;}"
            ".gate-step-inline{padding:4px 0 6px 0;border-bottom:1px solid #242424;}"
            ".gate-step-label{display:flex;align-items:center;gap:6px;color:#ddd;}"
            ".gate-step-select{margin-left:0;flex:1;min-width:0;}"
            ".footer-panel{font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;"
            "font-size:12px;line-height:1.28;}"
            ".footer-section + .footer-section{margin-top:12px;padding-top:10px;border-top:3px double #444;}"
            ".footer-title{color:#f0ad4e;font-weight:700;text-transform:uppercase;font-size:12px;"
            "letter-spacing:0.04em;margin-bottom:6px;}"
            ".footer-line{padding:4px 0;border-bottom:1px solid #242424;}"
            ".footer-line:last-child{border-bottom:none;}"
            ".celebration-flash{position:fixed;inset:0;pointer-events:none;z-index:9999;opacity:0;display:flex;align-items:center;justify-content:center;}"
            ".celebration-flash.active{animation:celebrationFlashPulse 850ms ease-out forwards;}"
            ".celebration-flash-label{font-weight:900;font-size:clamp(24px,5vw,56px);letter-spacing:0.06em;text-transform:uppercase;color:#111;"
            "text-shadow:0 2px 0 rgba(255,255,255,0.75),0 10px 30px rgba(0,0,0,0.45);padding:14px 22px;border-radius:12px;background:rgba(255,255,255,0.72);"
            "border:2px solid rgba(17,17,17,0.25);opacity:0;transform:scale(0.98);}"
            ".celebration-flash.active .celebration-flash-label{animation:celebrationFlashLabel 850ms ease-out forwards;}"
            "@keyframes celebrationFlashPulse{"
            "0%{opacity:0;background:#ff1493;}"
            "10%{opacity:0.98;background:#ff1493;}"
            "45%{opacity:0.95;background:#ff8c00;}"
            "75%{opacity:0.95;background:#ffffff;}"
            "100%{opacity:0;background:#ffffff;}"
            "}"
            "@keyframes celebrationFlashLabel{"
            "0%{opacity:0;transform:scale(0.98);}"
            "14%{opacity:1;transform:scale(1.0);}"
            "80%{opacity:1;transform:scale(1.0);}"
            "100%{opacity:0;transform:scale(1.02);}"
            "}"
            "@media (max-width: 760px){"
            ".layout{flex-direction:column;align-items:center;}"
            ".with-bottom-controls{padding-bottom:122px;}"
            ".controls{bottom:8px;gap:8px;padding:8px 10px;}"
            ".vision-mode{gap:8px;}"
            ".sidebar,.info-sidebar{min-width:min(520px, calc(100vw - 24px));max-width:min(520px, calc(100vw - 24px));}"
            "}"
            "</style>"
            f"</head><body{body_controls_class}>"
            f"{header_html}"
            f"{controls_html}"
            "<div id='celebrationFlash' class='celebration-flash'><div id='celebrationFlashLabel' class='celebration-flash-label'>STEP ACHIEVED</div></div>"
            "<div class='layout'>"
            "<div class='sidebar sidebar-stack'>"
            "<div id='telemetry' class='telemetry'></div>"
            f"{brick_shape_panel_html}"
            f"{footer_panel_html}"
            "</div>"
            f"<div style='display: flex; flex-direction: column; gap: 12px; align-items: center;'>"
            f"<div class='stream' style='background: white;'><img id='xyzLayout' src='/xyz_workspace_live.svg'{layout_xyz_img_width_attr or xyz_width_attr}></div>"
            f"<div class='stream' style='background: white;'><img id='mastLayout' src='/xyz_mast_live.svg'{layout_xyz_img_width_attr or xyz_width_attr}></div>"
            f"<div class='stream'><img id='videoFeed' src='/video_feed?sid={html.escape(self._instance_id, quote=True)}'{layout_camera_img_width_attr or camera_width_attr}></div>"
            f"</div>"
            "</div>"
            "<script>"
            "const videoFeedEl = document.getElementById('videoFeed');"
            "if(videoFeedEl){const base=videoFeedEl.src.split('?')[0];videoFeedEl.src=base+'?t='+Date.now();}"
            "const telemetryEl = document.getElementById('telemetry');"
            "const celebrationFlashEl = document.getElementById('celebrationFlash');"
            "const celebrationFlashLabelEl = document.getElementById('celebrationFlashLabel');"
            "const _svgRefreshState = { xyz: false, mast: false };"
            "const gongAudio = (() => {"
            "try {"
            "const a = new Audio('/gong.mp3');"
            "a.preload = 'auto';"
            "return a;"
            "} catch (e) {"
            "return null;"
            "}"
            "})();"
            "let lastStepSuccessSeq = null;"
            "let stepSuccessInitialized = false;"
            f"const pageServerId = {self._js_string_literal(self._instance_id)};"
            "let refreshFailures = 0;"
            "let videoFeedErrored = false;"
            "let reloadScheduled = false;"
            "let celebrationClearTimer = null;"
            "const _clearCelebrationFlash = () => {"
            "if (!celebrationFlashEl) return;"
            "celebrationFlashEl.classList.remove('active');"
            "if (celebrationFlashLabelEl) celebrationFlashLabelEl.textContent = 'STEP ACHIEVED';"
            "if (celebrationClearTimer) { clearTimeout(celebrationClearTimer); celebrationClearTimer = null; }"
            "};"
            "const _playGong = () => {"
            "if (!gongAudio) return;"
            "try {"
            "gongAudio.pause();"
            "gongAudio.currentTime = 0;"
            "const p = gongAudio.play();"
            "if (p && typeof p.catch === 'function') { p.catch(() => {}); }"
            "} catch (e) { /* ignore */ }"
            "};"
            "const triggerStepSuccessCelebration = (stepLabel) => {"
            "const label = (stepLabel !== null && stepLabel !== undefined) ? String(stepLabel).trim() : '';"
            "if (celebrationFlashLabelEl) {"
            "celebrationFlashLabelEl.textContent = label ? ('STEP ACHIEVED: ' + label) : 'STEP ACHIEVED';"
            "}"
            "if (celebrationFlashEl) {"
            "celebrationFlashEl.classList.remove('active');"
            "void celebrationFlashEl.offsetWidth;"
            "celebrationFlashEl.classList.add('active');"
            "if (celebrationClearTimer) clearTimeout(celebrationClearTimer);"
            "celebrationClearTimer = setTimeout(_clearCelebrationFlash, 950);"
            "}"
            "_playGong();"
            "};"
            "const scheduleReload = () => {"
            "if (reloadScheduled) return;"
            "reloadScheduled = true;"
            "setTimeout(() => { try { window.location.reload(); } catch (e) {} }, 120);"
            "};"
            "if (videoFeedEl) {"
            "videoFeedEl.addEventListener('error', () => { videoFeedErrored = true; });"
            "videoFeedEl.addEventListener('load', () => { videoFeedErrored = false; });"
            "}"
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
            "let lastTelemetryHtml = '';"
            "const refreshSvgImage = (id, path, stateKey) => {"
            "const el = document.getElementById(id);"
            "if (!el) return;"
            "if (_svgRefreshState[stateKey]) return;"
            "_svgRefreshState[stateKey] = true;"
            "const nextSrc = path + '?t=' + Date.now();"
            "const probe = new Image();"
            "probe.onload = () => { el.src = nextSrc; _svgRefreshState[stateKey] = false; };"
            "probe.onerror = () => { _svgRefreshState[stateKey] = false; };"
            "probe.src = nextSrc;"
            "};"
            "const refresh = async () => {"
            "try {"
            "const res = await fetch('/text', {cache:'no-store'});"
            "if (!res.ok) {"
            "refreshFailures += 1;"
            "return;"
            "}"
            "const data = await res.json();"
            "const serverId = (data && data.server_id) ? String(data.server_id) : '';"
            "if (serverId && pageServerId && serverId !== pageServerId) {"
            "scheduleReload();"
            "return;"
            "}"
            "if (videoFeedErrored && refreshFailures > 0) {"
            "scheduleReload();"
            "return;"
                "}"
                "const lines = Array.isArray(data.lines) ? data.lines : [];"
                "const stepSuccess = (data && typeof data.step_success === 'object' && data.step_success) ? data.step_success : null;"
                "if (stepSuccess && stepSuccess.seq !== undefined && stepSuccess.seq !== null) {"
                "let seqVal = null;"
                "try { seqVal = parseInt(stepSuccess.seq, 10); } catch (e) { seqVal = null; }"
                "if (Number.isFinite(seqVal)) {"
                "if (!stepSuccessInitialized) {"
                "lastStepSuccessSeq = seqVal;"
                "stepSuccessInitialized = true;"
                "let atVal = null;"
                "if (stepSuccess.at !== undefined && stepSuccess.at !== null) {"
                "try { atVal = parseFloat(stepSuccess.at); } catch (e) { atVal = null; }"
                "}"
                "const nowSec = Date.now() / 1000.0;"
                "const recentWindowS = 2.5;"
                "if (Number.isFinite(atVal) && (nowSec - atVal) >= 0 && (nowSec - atVal) <= recentWindowS) {"
                "triggerStepSuccessCelebration(stepSuccess.step);"
                "}"
                "} else if (lastStepSuccessSeq === null || seqVal > lastStepSuccessSeq) {"
                "lastStepSuccessSeq = seqVal;"
                "triggerStepSuccessCelebration(stepSuccess.step);"
                "}"
                "}"
                "} else if (!stepSuccessInitialized) {"
                "stepSuccessInitialized = true;"
                "}"
                "const renderedLines = lines.map(renderLine);"
                "const activeEl = document.activeElement;"
                "const successGateSelectFocused = !!(activeEl && activeEl.id === 'successGateStepSelect');"
                "if (successGateSelectFocused) {"
                "refreshFailures = 0;"
                "return;"
                "}"
                "let nextTelemetryHtml = '';"
                "if (typeof window.injectSuccessGateStepControl === 'function') {"
                "nextTelemetryHtml = window.injectSuccessGateStepControl(lines, renderedLines);"
                "} else {"
                "nextTelemetryHtml = renderedLines.join('');"
                "}"
                "if (nextTelemetryHtml !== lastTelemetryHtml) {"
                "telemetryEl.innerHTML = nextTelemetryHtml;"
                "lastTelemetryHtml = nextTelemetryHtml;"
                "}"
            "refreshFailures = 0;"
            "} catch (e) {"
            "refreshFailures += 1;"
            "}"
            "};"
            "setInterval(refresh, 150);"
            "refresh();"
            f"setInterval(() => {{ refreshSvgImage('xyzLayout', '/xyz_workspace_live.svg', 'xyz'); refreshSvgImage('mastLayout', '/xyz_mast_live.svg', 'mast'); }}, {int(xyz_refresh_ms)});"
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
        candidates = [candidate]
        if candidate in ("markerless", "yolo"):
            candidates.extend(("cyan", "yolo", "markerless"))
        elif candidate == "cyan":
            candidates.extend(("yolo", "markerless"))
        for mode in candidates:
            if mode in self._vision_mode_allowed:
                return mode
        return None

    def _coerce_cyan_profile(self, value):
        if not self._cyan_profile_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._cyan_profile_allowed:
            return candidate
        return None

    def _coerce_cyan_visibility(self, value):
        if not self._cyan_visibility_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._cyan_visibility_allowed:
            return candidate
        return None

    def _coerce_success_gate_step(self, value):
        if not self._success_gate_step_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._success_gate_step_allowed:
            return candidate
        return None

    def _coerce_depth_source(self, value):
        if not self._depth_source_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._depth_source_allowed:
            return candidate
        return None

    def _coerce_stereo_config(self, value):
        if not self._stereo_config_allowed:
            return None
        if value is None:
            return None
        candidate = str(value).strip().lower()
        if candidate in self._stereo_config_allowed:
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

    @staticmethod
    def _js_string_literal(value):
        s = "" if value is None else str(value)
        return "'" + (
            s.replace("\\", "\\\\")
            .replace("'", "\\'")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
        ) + "'"

    def _brick_shape_panel_html(self):
        fallback_html = (
            "<div class='shape-panel'>"
            "<div class='shape-wrap'><div style='color:#bbb;font-size:11px;'>shape coords unavailable</div></div>"
            "</div>"
        )

        try:
            data = json.loads(Path(DEFAULT_BRICK_MODEL_FILE).read_text())
        except Exception:
            return fallback_html

        brick = data.get("brick") if isinstance(data, dict) else None
        if not isinstance(brick, dict):
            return fallback_html

        face_cutouts_raw = brick.get("faceCutouts")
        face_polygon_raw = brick.get("facePolygon")
        face_lines_raw = brick.get("faceLines")
        shape_gate = brick.get("shapeGate") if isinstance(brick.get("shapeGate"), dict) else {}
        shape_gate_mode = str(shape_gate.get("mode") or "").strip().lower()

        face_points = []
        if isinstance(face_polygon_raw, list):
            for row in face_polygon_raw:
                if not isinstance(row, dict):
                    continue
                try:
                    x = float(row.get("x"))
                    y = float(row.get("y"))
                except (TypeError, ValueError):
                    continue
                face_points.append((x, y))

        cutout_polygons = []
        if isinstance(face_cutouts_raw, list):
            for cutout in face_cutouts_raw:
                if not isinstance(cutout, list):
                    continue
                cutout_points = []
                for row in cutout:
                    if not isinstance(row, dict):
                        continue
                    try:
                        cutout_points.append((float(row.get("x")), float(row.get("y"))))
                    except (TypeError, ValueError):
                        continue
                if len(cutout_points) >= 3:
                    cutout_polygons.append(cutout_points)

        display_cutout_indicator = bool(
            shape_gate_mode == "negative_cutouts" and cutout_polygons
        )
        extent_points = (
            [pt for poly in cutout_polygons for pt in poly]
            if display_cutout_indicator
            else list(face_points)
        )
        if len(extent_points) < 3:
            return fallback_html

        min_x = min(p[0] for p in extent_points)
        max_x = max(p[0] for p in extent_points)
        min_y = min(p[1] for p in extent_points)
        max_y = max(p[1] for p in extent_points)
        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        if display_cutout_indicator:
            min_x -= max(2.0, span_x * 0.40)
            max_x += max(2.0, span_x * 0.40)
            min_y -= max(2.0, span_y * 0.55)
            max_y += max(2.0, span_y * 0.55)
            span_x = max(1.0, max_x - min_x)
            span_y = max(1.0, max_y - min_y)

        svg_w = 260.0
        svg_h = 140.0
        pad = 16.0
        shape_offset_y = 5.0
        usable_w = svg_w - (2.0 * pad)
        usable_h = svg_h - (2.0 * pad) - shape_offset_y
        scale = min(usable_w / span_x, usable_h / span_y)

        def map_x(x_val):
            return pad + ((float(x_val) - min_x) * scale)

        def map_y(y_val):
            # World-model Y is positive-up; SVG Y is positive-down.
            return pad + shape_offset_y + ((max_y - float(y_val)) * scale)

        face_polygon_svg = ""
        if not display_cutout_indicator and len(face_points) >= 3:
            polygon_points = " ".join(
                f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in face_points
            )
            face_polygon_svg = (
                f"<polygon points='{polygon_points}' fill='#1f9db1' stroke='#c7f6ff' stroke-width='2.0'/>"
            )

        indicator_backing_svg = ""
        if display_cutout_indicator:
            cutout_points = [pt for poly in cutout_polygons for pt in poly]
            raw_min_x = min(p[0] for p in cutout_points)
            raw_max_x = max(p[0] for p in cutout_points)
            raw_min_y = min(p[1] for p in cutout_points)
            raw_max_y = max(p[1] for p in cutout_points)
            back_pad_x = max(2.0, (raw_max_x - raw_min_x) * 0.32)
            back_pad_y = max(2.0, (raw_max_y - raw_min_y) * 0.45)
            backing_points = [
                (raw_min_x - back_pad_x, raw_min_y - back_pad_y),
                (raw_min_x - back_pad_x, raw_max_y + back_pad_y),
                (raw_max_x + back_pad_x, raw_max_y + back_pad_y),
                (raw_max_x + back_pad_x, raw_min_y - back_pad_y),
            ]
            backing_svg_points = " ".join(
                f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in backing_points
            )
            indicator_backing_svg = (
                f"<polygon points='{backing_svg_points}' fill='#1f9db1' stroke='#c7f6ff' stroke-width='2.0'/>"
            )

        face_cutout_svg = ""
        if cutout_polygons:
            cutout_parts = []
            for cutout_points in cutout_polygons:
                cutout_svg_points = " ".join(
                    f"{map_x(x):.2f},{map_y(y):.2f}" for x, y in cutout_points
                )
                cutout_parts.append(
                    f"<polygon points='{cutout_svg_points}' fill='#11181d' stroke='#def7ff' stroke-width='1.6'/>"
                )
            face_cutout_svg = "".join(cutout_parts)

        pink_dot_svg = ""
        pink_dot_raw = brick.get("pinkDot")
        if isinstance(pink_dot_raw, dict):
            try:
                pd_x = float(pink_dot_raw.get("x", 0.0))
                pd_y = float(pink_dot_raw.get("y", 9.0))
                pd_r = float(pink_dot_raw.get("radius_mm", 2.5))
                pd_r_svg = max(3.0, pd_r * scale)
                pink_dot_svg = (
                    f"<circle cx='{map_x(pd_x):.2f}' cy='{map_y(pd_y):.2f}' r='{pd_r_svg:.1f}' "
                    "fill='#BE2646' stroke='#ffd0d8' stroke-width='1.2'/>"
                )
            except (TypeError, ValueError):
                pass

        face_line_svg = ""
        if isinstance(face_lines_raw, list):
            line_parts = []
            for seg in face_lines_raw:
                if not isinstance(seg, dict):
                    continue
                p1 = seg.get("p1") if isinstance(seg.get("p1"), dict) else None
                p2 = seg.get("p2") if isinstance(seg.get("p2"), dict) else None
                if p1 is None or p2 is None:
                    continue
                try:
                    x1 = float(p1.get("x"))
                    y1 = float(p1.get("y"))
                    x2 = float(p2.get("x"))
                    y2 = float(p2.get("y"))
                except (TypeError, ValueError):
                    continue
                line_parts.append(
                    f"<line x1='{map_x(x1):.2f}' y1='{map_y(y1):.2f}' x2='{map_x(x2):.2f}' y2='{map_y(y2):.2f}' "
                    "stroke='#def7ff' stroke-width='1.4' stroke-dasharray='4 3'/>"
                )
            face_line_svg = "".join(line_parts)

        svg_html = (
            "<svg class='shape-svg' viewBox='0 0 260 140' width='250' height='134' xmlns='http://www.w3.org/2000/svg' "
            "role='img' aria-label='Brick face shape reference from world model coordinates'>"
            "<rect x='1' y='1' width='258' height='138' rx='8' fill='#11181d' stroke='#2f4a54' stroke-width='1.2'/>"
            f"{indicator_backing_svg}"
            f"{face_polygon_svg}"
            f"{face_cutout_svg}"
            f"{pink_dot_svg}"
            f"{face_line_svg}"
            "</svg>"
        )

        return (
            "<div class='shape-panel'>"
            f"<div class='shape-wrap'>{svg_html}</div>"
            "</div>"
        )
