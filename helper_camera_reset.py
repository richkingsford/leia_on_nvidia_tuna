#!/usr/bin/env python3
"""Gracefully reset the OAK/DepthAI camera over USB when it gets stuck.

The Luxonis OAK (Intel Movidius MyriadX, USB vendor id 0x03e7) occasionally
wedges in its bootloader state (pid 0x2485) after a failed open, so DepthAI can
no longer boot it and no /dev/video* node appears. A USB-level reset
(USBDEVFS_RESET ioctl) re-enumerates the device cleanly without a physical
replug, which lets the next DepthAI open boot it normally (pid 0x2486/0xf63b).

This needs only write access to the /dev/bus/usb node (which is world-writable
here), not sudo. Importable as ``reset_oak_camera`` and runnable directly.
"""

from __future__ import annotations

import fcntl
import glob
import os
import time

OAK_VENDOR_ID = "03e7"
# USBDEVFS_RESET == _IO('U', 20) == (ord('U') << 8) | 20
USBDEVFS_RESET = (ord("U") << 8) | 20


def _read_sysfs(path: str) -> str:
    try:
        with open(path, "r", encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def find_oak_devices(vendor_id: str = OAK_VENDOR_ID) -> list[dict]:
    """Return [{sysfs, busnum, devnum, product_id, node}] for each OAK on USB."""
    found: list[dict] = []
    for vendor_path in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        if _read_sysfs(vendor_path).lower() != vendor_id.lower():
            continue
        dev_dir = os.path.dirname(vendor_path)
        busnum = _read_sysfs(os.path.join(dev_dir, "busnum"))
        devnum = _read_sysfs(os.path.join(dev_dir, "devnum"))
        if not (busnum and devnum):
            continue
        node = f"/dev/bus/usb/{int(busnum):03d}/{int(devnum):03d}"
        found.append(
            {
                "sysfs": dev_dir,
                "busnum": int(busnum),
                "devnum": int(devnum),
                "product_id": _read_sysfs(os.path.join(dev_dir, "idProduct")),
                "node": node,
            }
        )
    return found


def _usb_reset_node(node: str) -> None:
    fd = os.open(node, os.O_WRONLY)
    try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
        os.close(fd)


def reset_oak_camera(
    *,
    vendor_id: str = OAK_VENDOR_ID,
    settle_s: float = 3.0,
    reappear_timeout_s: float = 10.0,
    log=print,
) -> dict:
    """Reset every OAK on the USB bus and wait for it to re-enumerate.

    Returns a dict with ``reset`` (count reset), ``reappeared`` (bool) and the
    device lists before/after. Never raises on a missing device; it just
    reports ``reset == 0`` so callers can decide what to do.
    """
    before = find_oak_devices(vendor_id)
    if not before:
        log("[CAM-RESET] No OAK (vid %s) found on USB; nothing to reset." % vendor_id)
        return {"reset": 0, "reappeared": False, "before": [], "after": []}

    reset_count = 0
    for dev in before:
        try:
            log(
                f"[CAM-RESET] Resetting OAK at {dev['node']} "
                f"(pid={dev['product_id']}, bus {dev['busnum']} dev {dev['devnum']})."
            )
            _usb_reset_node(dev["node"])
            reset_count += 1
        except OSError as exc:
            log(f"[CAM-RESET] Reset ioctl failed for {dev['node']}: {exc!r}")

    # After USBDEVFS_RESET the kernel re-enumerates with a NEW devnum, so poll
    # by vendor id rather than the old node path.
    log(f"[CAM-RESET] Settling {settle_s:.1f}s, then waiting up to {reappear_timeout_s:.1f}s for re-enumeration.")
    time.sleep(max(0.0, float(settle_s)))
    deadline = time.monotonic() + max(0.0, float(reappear_timeout_s))
    after: list[dict] = []
    while time.monotonic() < deadline:
        after = find_oak_devices(vendor_id)
        if after:
            break
        time.sleep(0.4)

    reappeared = bool(after)
    if reappeared:
        dev = after[0]
        log(
            f"[CAM-RESET] OAK re-enumerated at {dev['node']} "
            f"(pid={dev['product_id']}, bus {dev['busnum']} dev {dev['devnum']})."
        )
    else:
        log("[CAM-RESET] OAK did not re-enumerate within the timeout; a physical replug may be needed.")
    return {"reset": reset_count, "reappeared": reappeared, "before": before, "after": after}


def main() -> int:
    result = reset_oak_camera()
    if result["reset"] == 0:
        return 2
    return 0 if result["reappeared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
