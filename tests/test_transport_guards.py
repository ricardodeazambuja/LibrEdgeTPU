#!/usr/bin/env python3
"""Tests for USBTransport device-not-open guards.

Verifies that USB methods raise RuntimeError when called before open().
"""

import pytest

from libredgetpu.transport import USBTransport


@pytest.fixture
def disconnected_transport():
    """Create a USBTransport with _dev = None (simulates disconnect)."""
    t = USBTransport.__new__(USBTransport)
    t._firmware_path = None
    t._dev = None
    t._regs = {}
    return t


def test_send_raises_when_not_open(disconnected_transport):
    with pytest.raises(RuntimeError, match="Device not open"):
        disconnected_transport.send(b"\x00", 0)


def test_read_output_raises_when_not_open(disconnected_transport):
    with pytest.raises(RuntimeError, match="Device not open"):
        disconnected_transport.read_output()


def test_read_status_raises_when_not_open(disconnected_transport):
    with pytest.raises(RuntimeError, match="Device not open"):
        disconnected_transport.read_status()


def test_write_register_raises_when_not_open(disconnected_transport):
    with pytest.raises(RuntimeError, match="Device not open"):
        disconnected_transport.write_register("some_reg", b"\x00")


def test_read_register_raises_when_not_open(disconnected_transport):
    with pytest.raises(RuntimeError, match="Device not open"):
        disconnected_transport.read_register("some_reg", 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
