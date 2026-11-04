# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Biren (SUPA) vendor backend operator registrations."""

from __future__ import annotations

from ..op_names import ALL_OPS, OPTIONAL_OPS
from ..registration import register_vendor_ops


def register_builtins(registry) -> None:
    """Register all Biren (VENDOR) operator implementations."""
    from .biren import BirenBackend

    register_vendor_ops(
        registry,
        BirenBackend(),
        ALL_OPS,
        impl_id="vendor.biren",
        vendor="BIREN",
        optional_ops=OPTIONAL_OPS,
    )
