# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""
Vendor-specific backend implementations.

This package contains hardware vendor-specific backend implementations.
Each vendor subdirectory should contain its own backend implementation.

Nothing is imported or gated here. Whether a vendor backend is usable is
decided at dispatch-registration time by its own ``is_available()`` (see
``vendor/registration.py``), which probes for the hardware and the extension;
``builtin_ops`` imports each vendor's ``register_ops`` directly, so an eager
import in this module would not change what loads.
"""

from __future__ import annotations
