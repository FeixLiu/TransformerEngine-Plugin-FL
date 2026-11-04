# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

"""Guard the FlashAttention contract shared with TransformerEngine main.

Two failure modes are checked here, both silent in normal use:

1. TE main's ``DotProductAttention`` picks up the plugin's implementation via
   ``getattr(tex, "<name>", <native fallback>)``. A missing stub attribute makes
   the getattr fall back to native TE: the plugin's attention code never runs,
   with no error and no line in the dispatch log.

2. TE main uses a plugin class in place of its own ``FlashAttention``, calling it
   with TE main's full argument list. A backend whose signature has drifted only
   fails when attention is first called, so signatures are compared statically
   against TE main here instead.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import pytest

from transformer_engine_plugin_fl import get_default_manager, load_plugins
from transformer_engine_plugin_fl.ops import (
    FlashAttentionBase,
    is_default_load_balancing_strategy,
    resolve_load_balancing_strategy,
)

# Names read with getattr() in
# transformer_engine/pytorch/attention/dot_product_attention/dot_product_attention.py
# under `if os.environ.get("NVTE_PLUGIN")`.
TE_MAIN_GETATTR_NAMES = ["flash_attention", "get_attention_backend"]

# Every FlashAttention implementation in the plugin.
FA_IMPLEMENTATIONS = [
    ("reference.flash_attention", "FlashAttentionTorch"),
    ("flagos.attention.dot_product_attention.backends", "FlashAttentionFL"),
    ("vendor.biren.flash_attention", "FlashAttentionBiren"),
    ("vendor.cuda.flash_attention", "FlashAttentionCUDA"),
    ("vendor.enflame.flash_attention", "FlashAttentionENFLAME"),
    ("vendor.hygon.flash_attention", "FlashAttentionHYGON"),
    ("vendor.metax.flash_attention", "FlashAttentionMETAX"),
    ("vendor.musa.flash_attention", "FlashAttentionMUSA"),
    ("vendor.npu.flash_attention", "NPUFlashAttention"),
    ("vendor.kunlunxin.flash_attention", "FlashAttentionTorch"),
]


@pytest.fixture(scope="module", autouse=True)
def te_pytorch():
    """Import TE the way training does, before touching the stub.

    With NVTE_PLUGIN set, importing transformer_engine.pytorch installs the stub
    itself. Importing it first also avoids re-importing the pybind extension
    later, which fails with "type DType is already registered".
    """
    return pytest.importorskip("transformer_engine.pytorch")


@pytest.fixture(scope="module")
def stub():
    load_plugins()
    return sys.modules["transformer_engine_torch"]


@pytest.fixture(scope="module")
def native_flash_attention():
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        FlashAttention,
    )

    return FlashAttention


def _params(fn):
    """Parameter names, order and kind — the part a caller depends on.

    Annotations and defaults are excluded: a backend narrowing ``Optional[Any]``
    to a concrete type documents its own extension and cannot break a call,
    whereas a renamed, reordered or dropped parameter does.
    """
    return tuple((p.name, p.kind) for p in inspect.signature(fn).parameters.values())


def _load(path: str, cls_name: str):
    module = importlib.import_module(f"transformer_engine_plugin_fl.backends.{path}")
    return getattr(module, cls_name)


@pytest.mark.parametrize("name", TE_MAIN_GETATTR_NAMES)
def test_te_main_mount_point_present(stub, name):
    """TE main's getattr must find a callable, or it silently uses native TE."""
    assert hasattr(stub, name), (
        f"stub is missing '{name}'; TE main's getattr(tex, '{name}', ...) will"
        " silently fall back to the native implementation"
    )
    assert callable(getattr(stub, name))


def test_flash_attention_accepts_te_main_call_shape(stub):
    """TE main passes softmax_scale positionally and the rest by keyword."""
    instance = stub.flash_attention(
        0.125,
        attention_dropout=0.0,
        attention_dropout_ctx=None,
        attention_type="self",
        layer_number=1,
        deterministic=False,
    )

    # Must be a plugin implementation, not TE's native FlashAttention.
    assert type(instance).__module__.startswith("transformer_engine_plugin_fl")
    assert instance.softmax_scale == 0.125


def test_flash_attention_dispatches_through_manager(stub):
    """The factory must route through OpManager so selection policy applies."""
    manager = get_default_manager()
    expected = manager.call("get_flash_attention_class")

    instance = stub.flash_attention(0.125)

    assert type(instance) is expected


def test_stub_factory_matches_native_init(stub, native_flash_attention):
    """The stub attribute stands in for the class, so it takes the class's args."""
    native = _params(native_flash_attention.__init__)[1:]  # drop self
    assert _params(stub.flash_attention) == native


def test_base_forward_matches_native(native_flash_attention):
    """The base declares exactly TE main's forward contract."""
    assert _params(FlashAttentionBase.forward) == _params(native_flash_attention.forward)


def test_load_balancing_strategy_default_resolves_to_te_member(native_flash_attention):
    """The plugin's ``None`` default must mean TE main's own default member.

    The plugin signatures cannot name the member (importing TE at module scope
    there is circular), so the equivalence is asserted here. TE main checks
    ``isinstance`` against its own enum, so the resolved value must be the real
    member and not a copy.
    """
    from transformer_engine.pytorch.constants import CPLoadBalancingStrategy

    native_default = (
        inspect.signature(native_flash_attention.forward)
        .parameters["load_balancing_strategy"]
        .default
    )
    assert native_default is CPLoadBalancingStrategy.DUAL_CHUNK_SWAP

    assert resolve_load_balancing_strategy(None) is native_default
    assert is_default_load_balancing_strategy(None)
    assert not is_default_load_balancing_strategy(CPLoadBalancingStrategy.NO_LOAD_BALANCE)


@pytest.mark.parametrize(
    "path,cls_name", FA_IMPLEMENTATIONS, ids=[p for p, _ in FA_IMPLEMENTATIONS]
)
def test_load_balancing_strategy_default_is_none(path, cls_name):
    """Backends spell that default as ``None``; see the test above for why."""
    default = (
        inspect.signature(_load(path, cls_name).forward)
        .parameters["load_balancing_strategy"]
        .default
    )
    assert default is None


def test_base_init_matches_native(native_flash_attention):
    assert _params(FlashAttentionBase.__init__) == _params(native_flash_attention.__init__)


@pytest.mark.parametrize(
    "path,cls_name", FA_IMPLEMENTATIONS, ids=[p for p, _ in FA_IMPLEMENTATIONS]
)
def test_backend_forward_matches_native(path, cls_name, native_flash_attention):
    """Every backend implements TE main's signature exactly.

    No argument filtering happens at dispatch, so a backend that renames, drops
    or reorders a parameter raises TypeError on the first attention call.
    """
    cls = _load(path, cls_name)
    assert _params(cls.forward) == _params(native_flash_attention.forward), (
        f"{cls_name}.forward drifted from TE main:\n"
        f"  native: {_params(native_flash_attention.forward)}\n"
        f"  impl:   {_params(cls.forward)}"
    )


@pytest.mark.parametrize(
    "path,cls_name", FA_IMPLEMENTATIONS, ids=[p for p, _ in FA_IMPLEMENTATIONS]
)
def test_backend_init_matches_native(path, cls_name, native_flash_attention):
    """``make_flash_attention`` constructs every backend with the same arguments."""
    cls = _load(path, cls_name)
    assert _params(cls.__init__) == _params(native_flash_attention.__init__), (
        f"{cls_name}.__init__ drifted from TE main:\n"
        f"  native: {_params(native_flash_attention.__init__)}\n"
        f"  impl:   {_params(cls.__init__)}"
    )


@pytest.mark.parametrize(
    "path,cls_name", FA_IMPLEMENTATIONS, ids=[p for p, _ in FA_IMPLEMENTATIONS]
)
def test_backend_implements_forward(path, cls_name):
    """``forward`` is abstract, so every backend must define its own."""
    cls = _load(path, cls_name)
    assert not cls.__abstractmethods__, f"{cls_name} leaves {cls.__abstractmethods__} abstract"
    assert "forward" in cls.__dict__, f"{cls_name} does not define forward()"


def test_base_forward_is_abstract():
    """A backend that forgets forward() must fail at construction, not at runtime."""

    class Incomplete(FlashAttentionBase):
        pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete(softmax_scale=0.125)
