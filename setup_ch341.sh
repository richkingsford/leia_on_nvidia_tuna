#!/usr/bin/env bash
# Rebuild and load the CH341 USB serial driver for Jetson (5.15.148-tegra).
# Run with: sudo bash setup_ch341.sh
set -e

BUILD_DIR=/tmp/ch341-build
KERNEL=$(uname -r)
KBUILD=/lib/modules/$KERNEL/build
MODULE_INSTALL_DIR=/lib/modules/$KERNEL/extra
MODULE_INSTALL_PATH=$MODULE_INSTALL_DIR/ch341.ko
UDEV_RULE=/etc/udev/rules.d/99-leia-uno.rules
MODULE_LOAD_CONF=/etc/modules-load.d/leia-ch341.conf

find_leia_dev() {
    for dev in /dev/leia-uno /dev/ttyCH341USB* /dev/ttyUSB*; do
        if [ -e "$dev" ]; then
            echo "$dev"
            return 0
        fi
    done
    return 1
}

echo "[ch341] Kernel: $KERNEL"
echo "[ch341] Build dir: $KBUILD"

echo "[ch341] Installing persistent Leia Uno udev rule..."
cat > "$UDEV_RULE" <<'EOF'
# Leia Uno CH340/CH341 USB serial adapter.
SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", GROUP="dialout", MODE="0666", SYMLINK+="leia-uno"
EOF
udevadm control --reload-rules || true
udevadm trigger --subsystem-match=tty || true

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "[ch341] Downloading source from WCHSoftGroup..."
curl -sL "https://github.com/WCHSoftGroup/ch341ser_linux/archive/refs/heads/main.tar.gz" \
    | tar -xz --strip-components=2 'ch341ser_linux-main/driver/'

echo "[ch341] Building module..."
make -C "$KBUILD" M="$BUILD_DIR" modules

echo "[ch341] Installing module for future boots..."
install -D -m 0644 "$BUILD_DIR/ch341.ko" "$MODULE_INSTALL_PATH"
depmod -a "$KERNEL"
cat > "$MODULE_LOAD_CONF" <<'EOF'
usbserial
ch341
EOF

echo "[ch341] Loading module..."
modprobe usbserial
if modprobe ch341; then
    echo "[ch341] Module loaded via modprobe."
elif lsmod | grep -q '^ch341 '; then
    echo "[ch341] Module already loaded."
else
    insmod "$BUILD_DIR/ch341.ko" || {
        if lsmod | grep -q '^ch341 '; then
            echo "[ch341] Module loaded by the kernel while we were checking."
        else
            exit 1
        fi
    }
fi

# Give the device a moment to enumerate
sleep 1

DEV=$(find_leia_dev || true)
if [ -z "$DEV" ]; then
    echo "[ch341] WARNING: no Leia serial device appeared — trying manual bind..."
    IFACE=$(find /sys/bus/usb/devices -name "idVendor" \
            | xargs grep -l "1a86" 2>/dev/null \
            | head -1 | xargs dirname)/*/
    for i in /sys/bus/usb/devices/*/idVendor; do
        v=$(cat "$i" 2>/dev/null)
        if [ "$v" = "1a86" ]; then
            dir=$(dirname "$i")
            for iface_path in "$dir":*/; do
                iface=$(basename "$iface_path")
                echo -n "$iface" > /sys/bus/usb/drivers/ch341/bind 2>/dev/null && echo "[ch341] Bound $iface" || true
            done
        fi
    done
    sleep 1
    DEV=$(find_leia_dev || true)
fi

if [ -n "$DEV" ]; then
    chmod a+rw "$DEV"
    echo "[ch341] Ready: $DEV"
    echo "[ch341] Future reconnects should be readable automatically via $UDEV_RULE"
else
    echo "[ch341] ERROR: device still not found. Check: lsusb, dmesg | tail -20"
    exit 1
fi
