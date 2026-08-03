"""Shared pytest configuration."""

from __future__ import annotations

import os

# Prefer system/config defaults before any optional astropy import.
os.environ.setdefault("ASTROPY_USE_SYSTEM_CONFIG", "1")
