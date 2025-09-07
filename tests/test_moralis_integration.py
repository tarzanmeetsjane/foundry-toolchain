# Test framework detected: pytest
# ruff: noqa: S101
# Focus: utils/moralis_integration.py
# Scenarios covered: env present, missing, empty string, independence of instances, varied constructor inputs.

import os
import pathlib
import importlib.util
import pytest

def _import_moralis_symbols_pytest():
    """
    Robust importer for utils/moralis_integration without assuming package layout.
    """
    try:
        from utils.moralis_integration import MoralisProvider, get_moralis_provider  # type: ignore
    except ModuleNotFoundError:
        root = pathlib.Path(__file__).resolve().parents[1]
        module_path = root / "utils" / "moralis_integration.py"
        if not module_path.exists():
            raise
        spec = importlib.util.spec_from_file_location("moralis_integration", module_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        if not (spec and spec.loader):
            msg = "Failed to load moralis_integration module"
            raise AssertionError(msg) from None
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module.MoralisProvider, module.get_moralis_provider  # type: ignore[attr-defined]
    else:
        return MoralisProvider, get_moralis_provider

MoralisProvider, get_moralis_provider = _import_moralis_symbols_pytest()

def test_get_moralis_provider_reads_env(monkeypatch):
    monkeypatch.setenv("MORALIS_API_KEY", "test_key_123")
    provider = get_moralis_provider()
    assert isinstance(provider, MoralisProvider)
    assert provider.api_key == "test_key_123"

def test_get_moralis_provider_missing_env_returns_none(monkeypatch):
    monkeypatch.delenv("MORALIS_API_KEY", raising=False)
    provider = get_moralis_provider()
    assert isinstance(provider, MoralisProvider)
    assert provider.api_key is None

def test_get_moralis_provider_empty_string_env(monkeypatch):
    monkeypatch.setenv("MORALIS_API_KEY", "")
    provider = get_moralis_provider()
    assert provider.api_key == ""

def test_providers_are_independent_instances(monkeypatch):
    monkeypatch.setenv("MORALIS_API_KEY", "A")
    p1 = get_moralis_provider()
    monkeypatch.setenv("MORALIS_API_KEY", "B")
    p2 = get_moralis_provider()
    assert p1 is not p2
    assert p1.api_key == "A"
    assert p2.api_key == "B"

@pytest.mark.parametrize("value", [None, "", "abc123", "  ", "x" * 4096])
def test_moralis_provider_init_accepts_varied_values(value):
    provider = MoralisProvider(value)
    assert provider.api_key == value