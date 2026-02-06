"""USB transport layer for the Google Coral Edge TPU.

Handles device detection, firmware download, register access,
and framed bulk data transfer.

When the optional ``_usb_accel`` C extension is compiled (requires
``libusb-1.0-dev``), bulk transfers bypass pyusb entirely for lower
latency.  The extension is auto-detected at import time; if absent,
the pure-Python pyusb path is used.
"""

import hashlib
import os
import ssl
import struct
import time
import urllib.request

import usb.core
import usb.util

from .registers import REGISTER_MAP

# Optional C extension for fast USB transfers
try:
    from ._usb_accel import UsbDevice as _CUsbDevice
    _HAS_C_ACCEL = True
except ImportError:
    _HAS_C_ACCEL = False

__all__ = [
    "USBTransport",
    "TAG_INSTRUCTIONS",
    "TAG_INPUT_ACTIVATIONS",
    "TAG_PARAMETERS",
    "TAG_OUTPUT_ACTIVATIONS",
]

# Vendor/product IDs
_CORAL_VID = 0x18D1
_CORAL_PID = 0x9302
_CORAL_BOOT_VID = 0x1A6E
_CORAL_BOOT_PID = 0x089A

# USB endpoints
EP_WRITE = 0x01
EP_OUTPUT = 0x81
EP_STATUS = 0x82

# Chunk size for bulk writes (1 MB)
_CHUNK_SIZE = 0x100000

# USB descriptor tags
TAG_INSTRUCTIONS = 0
TAG_INPUT_ACTIVATIONS = 1
TAG_PARAMETERS = 2
TAG_OUTPUT_ACTIVATIONS = 3

# Firmware download URL and filename
_FIRMWARE_FILENAME = "apex_latest_single_ep.bin"
_FIRMWARE_URL = (
    "https://github.com/ricardodeazambuja/libedgetpu-rpi0"
    "/raw/refs/heads/master/driver/usb/" + _FIRMWARE_FILENAME
)
_FIRMWARE_SHA256 = "3b07311b174b81fd14fd2c887f40d646c0ebf2f140cc5ee5cf8befa0f3e3dc7f"

# --- USB timing and retry constants ---
# After a USB reset the device re-enumerates on the bus.  Empirically
# 0.5 s is usually enough, but 0.6 s gives margin on slower hubs.
_RESET_SETTLE_S = 0.6

# When re-opening the device right after reset, the re-enumeration may
# take longer than _RESET_SETTLE_S (e.g. after many consecutive
# open/close cycles in a test suite).  Retry a few times before giving up.
_REOPEN_MAX_RETRIES = 3
_REOPEN_RETRY_S = 1.0

# After firmware download the device reboots from bootloader to runtime
# mode.  The 3 s delay covers the full reboot (measured ~2 s typical).
_FW_BOOT_S = 3.0

# Firmware upload can be disrupted by USB glitches.  Retry a few times
# before asking the user to power-cycle the device.
_FW_UPLOAD_MAX_RETRIES = 3
_FW_UPLOAD_RETRY_S = 0.5

# Register polling interval — tight loop for low-latency status checks.
_POLL_INTERVAL_S = 0.001


class USBTransport:
    """Low-level USB communication with the Coral Edge TPU.

    When the ``_usb_accel`` C extension is available, bulk transfers
    (send/read_output/read_status) use direct libusb-1.0 calls.
    Register access (ctrl_transfer) goes through the same device handle.
    Firmware download always uses pyusb (targets different VID:PID).
    """

    def __init__(self, firmware_path=None):
        self._firmware_path = firmware_path
        self._dev = None
        self._cdev = None  # C extension device (when available)
        self._regs = REGISTER_MAP

    def open(self) -> None:
        """Detect device, download firmware if needed, claim interface."""
        did_firmware = False
        dev = None
        cdev = None
        try:
            dev = usb.core.find(idVendor=_CORAL_VID, idProduct=_CORAL_PID)
            if dev is None:
                dev = usb.core.find(idVendor=_CORAL_BOOT_VID, idProduct=_CORAL_BOOT_PID)
                if dev is None:
                    raise RuntimeError("No Google Coral Edge TPU found")
                self._download_firmware(dev)
                did_firmware = True
                dev = None  # bootloader device is gone after firmware download

            if _HAS_C_ACCEL:
                # Use C extension for the runtime device.
                # Release pyusb handle first — C extension opens its own via libusb.
                if dev is not None:
                    try:
                        usb.util.dispose_resources(dev)
                    except Exception:
                        pass
                    dev = None

                cdev = _CUsbDevice(_CORAL_VID, _CORAL_PID)
                if not did_firmware:
                    try:
                        cdev.reset()
                    except OSError:
                        pass
                    time.sleep(_RESET_SETTLE_S)
                    # Re-open after reset (device re-enumerates)
                    cdev.close()
                    for _attempt in range(_REOPEN_MAX_RETRIES):
                        try:
                            cdev = _CUsbDevice(_CORAL_VID, _CORAL_PID)
                            break
                        except RuntimeError:
                            if _attempt == _REOPEN_MAX_RETRIES - 1:
                                raise
                            time.sleep(_REOPEN_RETRY_S)
                    del _attempt
                self._cdev = cdev
                self._dev = cdev  # unified reference for _ensure_open()
            else:
                # Pure pyusb path
                if dev is None:
                    dev = usb.core.find(idVendor=_CORAL_VID, idProduct=_CORAL_PID)
                    if dev is None:
                        raise RuntimeError("Device not found after firmware download")

                if not did_firmware:
                    try:
                        dev.reset()
                    except usb.core.USBError:
                        time.sleep(_REOPEN_RETRY_S)
                        dev = usb.core.find(idVendor=_CORAL_VID, idProduct=_CORAL_PID)
                        if dev is None:
                            raise RuntimeError("Device lost after reset")
                    time.sleep(_RESET_SETTLE_S)

                try:
                    if dev.is_kernel_driver_active(0):
                        dev.detach_kernel_driver(0)
                except (usb.core.USBError, NotImplementedError):
                    pass

                dev.set_configuration(1)
                usb.util.claim_interface(dev, 0)
                self._dev = dev
        except Exception:
            # Clean up any partially-acquired resources on failure
            if cdev is not None:
                try:
                    cdev.close()
                except Exception:
                    pass
            if dev is not None:
                try:
                    usb.util.dispose_resources(dev)
                except Exception:
                    pass
            self._dev = None
            self._cdev = None
            raise

    def _download_firmware(self, dev):
        """Download firmware to a device in bootloader mode."""
        fw_path = self._firmware_path
        if fw_path is None:
            pkg_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(pkg_dir, _FIRMWARE_FILENAME),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    fw_path = c
                    break
        if fw_path is None:
            # Auto-download firmware from GitHub
            pkg_dir = os.path.dirname(__file__)
            fw_path = os.path.join(pkg_dir, _FIRMWARE_FILENAME)
            print(f"Firmware not found locally. Downloading from GitHub...")
            try:
                ctx = ssl.create_default_context()
                resp = urllib.request.urlopen(_FIRMWARE_URL, timeout=30, context=ctx)
                fw_data = resp.read()
                # Verify firmware integrity before writing
                actual_hash = hashlib.sha256(fw_data).hexdigest()
                if actual_hash != _FIRMWARE_SHA256:
                    raise RuntimeError(
                        f"Firmware SHA256 mismatch: expected {_FIRMWARE_SHA256}, "
                        f"got {actual_hash}. Download may be corrupted."
                    )
                with open(fw_path, "wb") as fh:
                    fh.write(fw_data)
                print(f"Firmware saved to {fw_path}")
            except Exception as e:
                # Clean up partial downloads
                if os.path.isfile(fw_path):
                    try:
                        os.remove(fw_path)
                    except OSError:
                        pass
                raise FileNotFoundError(
                    f"Cannot find or download {_FIRMWARE_FILENAME}: {e}\n"
                    f"URL: {_FIRMWARE_URL}\n"
                    "You can manually download it and pass firmware_path= to the constructor."
                ) from e

        with open(fw_path, "rb") as f:
            fw = f.read()
        total_chunks = (len(fw) + 0xFF) // 0x100
        for attempt in range(_FW_UPLOAD_MAX_RETRIES):
            cnt = 0
            try:
                for i in range(0, len(fw), 0x100):
                    if cnt > 0xFFFF:
                        raise RuntimeError(
                            f"Firmware too large ({len(fw)} bytes): chunk count "
                            f"exceeds USB 16-bit wValue limit"
                        )
                    dev.ctrl_transfer(0x21, 1, cnt, 0, fw[i : i + 0x100])
                    dev.ctrl_transfer(0xA1, 3, 0, 0, 6)
                    cnt += 1
                if cnt > 0xFFFF:
                    raise RuntimeError(
                        f"Firmware too large ({len(fw)} bytes): final chunk count "
                        f"exceeds USB 16-bit wValue limit"
                    )
                dev.ctrl_transfer(0x21, 1, cnt, 0, b"")
                dev.ctrl_transfer(0xA1, 3, 0, 0, 6)
                break  # success
            except usb.core.USBError as e:
                if attempt < _FW_UPLOAD_MAX_RETRIES - 1:
                    time.sleep(_FW_UPLOAD_RETRY_S)
                    dev = usb.core.find(idVendor=_CORAL_BOOT_VID,
                                        idProduct=_CORAL_BOOT_PID)
                    if dev is None:
                        raise RuntimeError(
                            f"Device disappeared during firmware upload attempt "
                            f"{attempt + 1}/{_FW_UPLOAD_MAX_RETRIES}"
                        ) from e
                    continue
                raise RuntimeError(
                    f"Firmware upload failed after {_FW_UPLOAD_MAX_RETRIES} "
                    f"attempts at chunk {cnt}/{total_chunks}. Device needs "
                    f"power cycle. Error: {e}"
                ) from e

        for i in range(0x81):
            dev.ctrl_transfer(0xA1, 2, i, 0, 0x100)
        try:
            dev.reset()
        except usb.core.USBError:
            pass
        time.sleep(_FW_BOOT_S)

    def _ensure_open(self):
        """Raise if device is not open."""
        if self._dev is None:
            raise RuntimeError(
                "Device not open — call open() first or check USB connection"
            )

    def send(self, data: bytes, tag: int) -> None:
        """Send framed bulk data. *tag* is one of TAG_* constants.

        The 8-byte header is coalesced with the first data chunk to
        eliminate one USB round-trip per call.
        """
        self._ensure_open()
        if self._cdev is not None:
            self._cdev.send(data, tag)
            return
        dev = self._dev
        ll = len(data)
        header = struct.pack("<II", ll, tag)
        if ll <= _CHUNK_SIZE - 8:
            # Common fast path: header + all data in a single write
            dev.write(EP_WRITE, header + data)
        else:
            first = _CHUNK_SIZE - 8
            dev.write(EP_WRITE, header + data[:first])
            off = first
            remaining = ll - first
            while remaining > _CHUNK_SIZE:
                dev.write(EP_WRITE, data[off : off + _CHUNK_SIZE])
                off += _CHUNK_SIZE
                remaining -= _CHUNK_SIZE
            if remaining > 0:
                dev.write(EP_WRITE, data[off : off + remaining])

    def read_output(self, max_size: int = 0x400, timeout_ms: int = 6000) -> bytes:
        """Read output tensor data from EP 0x81.

        For small outputs (≤ 32 KB), a single read suffices.
        For larger outputs, uses a pre-allocated buffer to avoid
        repeated bytearray growth.
        """
        self._ensure_open()
        if self._cdev is not None:
            return self._cdev.read_output(max_size, timeout_ms)
        dev = self._dev
        if max_size <= 32768:
            return bytes(dev.read(EP_OUTPUT, max_size, timeout=timeout_ms))

        buf = bytearray(max_size)
        offset = 0
        while offset < max_size:
            try:
                # Always request 32768 bytes — requesting less can trigger
                # overflow if the device sends a full USB packet.
                chunk = dev.read(EP_OUTPUT, 32768, timeout=timeout_ms)
                n = len(chunk)
                buf[offset:offset + n] = chunk
                offset += n
            except usb.core.USBTimeoutError:
                break
        return bytes(buf[:offset])

    def read_status(self, timeout_ms: int = 6000) -> bytes:
        """Read status packet from EP 0x82."""
        self._ensure_open()
        if self._cdev is not None:
            return self._cdev.read_status(timeout_ms)
        return bytes(self._dev.read(EP_STATUS, 0x10, timeout=timeout_ms))

    def write_register(self, name: str, data: bytes) -> None:
        """Write to a named hardware register."""
        self._ensure_open()
        regnum = self._regs[name]
        bReq = int(regnum >> 16 == 1)
        wIndex = (regnum >> 16) & 0xFFFF
        if self._cdev is not None:
            self._cdev.ctrl_transfer(0x40, bReq, regnum & 0xFFFF, wIndex, data)
        else:
            self._dev.ctrl_transfer(0x40, bReq, regnum & 0xFFFF, wIndex, data)

    def read_register(self, name: str, length: int, offset: int = 0) -> bytes:
        """Read from a named hardware register."""
        self._ensure_open()
        regnum = self._regs[name] + offset
        bReq = int(regnum >> 16 == 1)
        wIndex = (regnum >> 16) & 0xFFFF
        if self._cdev is not None:
            return self._cdev.ctrl_transfer(0xC0, bReq, regnum & 0xFFFF, wIndex, length)
        return bytes(self._dev.ctrl_transfer(0xC0, bReq, regnum & 0xFFFF, wIndex, length))

    def poll_register(self, name, expected_value, mask=None, timeout_s=5.0, read_len=8):
        """Poll register until (value & mask) == expected_value, with timeout.

        Args:
            name: Register name from REGISTER_MAP.
            expected_value: Integer value to wait for.
            mask: Optional bitmask applied before comparison. If None, compare full value.
            timeout_s: Timeout in seconds (default 5).
            read_len: Number of bytes to read (default 8).

        Returns:
            The final register value (as integer).

        Raises:
            TimeoutError: If the register doesn't reach expected_value within timeout.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            raw = self.read_register(name, read_len)
            val = int.from_bytes(raw, "little")
            check = (val & mask) if mask is not None else val
            if check == expected_value:
                return val
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Register {name!r} didn't reach expected value "
                    f"0x{expected_value:X} (mask={mask!r}), last=0x{val:X}"
                )
            time.sleep(_POLL_INTERVAL_S)

    def close(self) -> None:
        """Release the device."""
        if self._cdev is not None:
            try:
                self._cdev.close()
            except Exception:
                pass
            self._cdev = None
            self._dev = None
        elif self._dev is not None:
            try:
                usb.util.release_interface(self._dev, 0)
            except Exception:
                pass  # best-effort cleanup during teardown
            self._dev = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
