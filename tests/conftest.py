"""Pytest configuration and shared fixtures for libredgetpu tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-hardware", action="store_true", default=False,
        help="Run tests that require a USB Edge TPU device",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: requires USB Edge TPU device")
    config.addinivalue_line("markers", "validated: hardware + post-processing validation")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-hardware"):
        skip_hw = pytest.mark.skip(reason="needs --run-hardware option to run")
        for item in items:
            if "hardware" in item.keywords or "validated" in item.keywords:
                item.add_marker(skip_hw)
