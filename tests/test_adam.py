# pytest-based tests for ADAM_System
import json
import math
import types
from pathlib import Path

import builtins
import pytest

# Try typical import paths. If module lives in a different path, adjust if needed.
# We prefer importing from adam or src.adam if available; else fall back to module under test in repository.
MODULE_IMPORT_ERRORS = []
ADAM = None
for mod_name in ("adam", "src.adam", "app.adam", "core.adam", "tests.test_adam"):  # last fallback if code is in tests (as per PR diff)
    try:
        ADAM = __import__(mod_name, fromlist=["ADAM_System", "run_adam_system"])
        if hasattr(ADAM, "ADAM_System"):
            break
    except Exception as e:
        MODULE_IMPORT_ERRORS.append((mod_name, repr(e)))

if ADAM is None or not hasattr(ADAM, "ADAM_System"):
    raise ImportError(f"Could not import ADAM_System from known locations. Tried: {MODULE_IMPORT_ERRORS}")

ADAM_System = ADAM.ADAM_System

@pytest.fixture
def fake_moralis(monkeypatch):
    # Replace get_moralis_provider used in ADAM module with controllable stub
    class StubMoralis:
        def __init__(self, api_key=""):
            self.api_key = api_key

    def provider_factory(api_key=""):
        return StubMoralis(api_key=api_key)

    # Patch the symbol where ADAM module looks it up
    monkeypatch.setattr(ADAM, "get_moralis_provider", lambda: provider_factory(api_key="DUMMY_KEY"), raising=True)
    return StubMoralis("DUMMY_KEY")

@pytest.fixture
def tmp_logfile(tmp_path, monkeypatch):
    # Force deterministic log file path and ensure directory exists
    lf = tmp_path / "adam_test.log"
    # Patch datetime.now().strftime used when ADAM_System is constructed to avoid dynamic filename usage.
    class FixedDatetime:
        @classmethod
        def now(cls):
            class DT:
                def strftime(self_inner, fmt):
                    # Fixed date for log filename generation
                    return "20250101" if "%Y%m%d" in fmt else "2025-01-01 00:00:00"
                def isoformat(self_inner):
                    return "2025-01-01T00:00:00"
            return DT()
    # Patch at module level
    monkeypatch.setattr(ADAM, "datetime", FixedDatetime, raising=True)

    # Build instance then override its log_file
    adam = ADAM_System("TESTADDR")
    adam.log_file = str(lf)
    return lf, adam

def test_get_trading_action_thresholds():
    # Validate boundary behavior
    # >0.8 -> STRONG_BUY
    # >0.6 -> BUY
    # >0.4 -> HOLD
    # >0.2 -> SELL
    # else STRONG_SELL
    class Dummy(ADAM_System):
        pass

    d = Dummy()
    f = d.get_trading_action
    assert f(0.81) == "STRONG_BUY"
    assert f(0.8000001) == "STRONG_BUY"
    assert f(0.8) != "STRONG_BUY"
    assert f(0.61) == "BUY"
    assert f(0.6) != "BUY"
    assert f(0.41) == "HOLD"
    assert f(0.4) != "HOLD"
    assert f(0.21) == "SELL"
    assert f(0.2) != "SELL"
    assert f(0.2) == "STRONG_SELL"
    assert f(0.0) == "STRONG_SELL"

def test_calculate_harmonic_signal_deterministic(monkeypatch, fake_moralis):
    # Fix current time to 01:28 (88 minutes) so minutes_since_midnight % 88.8 == close to 88.0
    # We'll make now() return a structure with hour/minute attributes and isoformat/strftime needed elsewhere.
    class FixedNow:
        hour = 1
        minute = 28  # 88 minutes
        def isoformat(self): return "2025-01-01T01:28:00"
        def strftime(self, fmt): return "2025-01-01 01:28:00"

    class DTClass:
        @classmethod
        def now(cls): return FixedNow()

    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    adam = ADAM_System("ADDR")
    # Use multiplier=3.0, signal calc:
    # harmonic_position = (88 % 88.8)/88.8 = 88/88.8 ≈ 0.991
    # base = sin(0.991*2π)*0.5+0.5 -> compute and then min(1, *3)
    res = adam.calculate_harmonic_signal()
    assert res["timestamp"] == "2025-01-01T01:28:00"
    assert 0.0 <= res["harmonic_position"] <= 1.0
    assert 0.0 <= res["signal_strength"] <= 1.0
    assert res["trading_action"] in {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}

def test_analyze_portfolio_performance_with_moralis(monkeypatch, fake_moralis, tmp_path):
    # Fix now for logfile naming
    class FixedNow:
        def strftime(self, fmt): return "2025-01-01 00:00:00" if "%H" in fmt else "20250101"
        def isoformat(self): return "2025-01-01T00:00:00"
        hour = 0
        minute = 0
    class DTClass:
        @classmethod
        def now(cls): return FixedNow()
    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    adam = ADAM_System("ADDR")
    adam.log_file = str(tmp_path / "log.txt")
    perf = adam.analyze_portfolio_performance()
    assert perf is not None
    assert perf["regal_address"] == "ADDR"
    assert perf["harmonic_frequency"] == 888.0
    assert "current_signal" in perf
    assert perf["current_signal"]["trading_action"] in {"STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"}
    # performance_data appended
    assert len(adam.performance_data) == 1

def test_analyze_portfolio_performance_handles_exception(monkeypatch, tmp_path):
    # Patch calculate_harmonic_signal to raise to exercise except path
    class FakeMoralis:
        api_key = "KEY"
    monkeypatch.setattr(ADAM, "get_moralis_provider", lambda: FakeMoralis(), raising=True)

    class DTClass:
        @classmethod
        def now(cls):
            class N:
                def strftime(self, fmt): return "20250101" if "%Y%m%d" in fmt else "2025-01-01 00:00:00"
                def isoformat(self): return "2025-01-01T00:00:00"
                hour = 0
                minute = 0
            return N()
    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    adam = ADAM_System("ADDR")
    adam.log_file = str(tmp_path / "err.log")
    def boom(): raise RuntimeError("boom")
    monkeypatch.setattr(adam, "calculate_harmonic_signal", boom, raising=True)
    perf = adam.analyze_portfolio_performance()
    assert perf is None
    # Ensure log file captured error message
    contents = Path(adam.log_file).read_text()
    assert "Portfolio analysis error" in contents

@pytest.mark.parametrize(
    "action,strength,expected_msgs",
    [
        ("STRONG_BUY", 0.86, ["STRONG BUY SIGNAL", "Recommended allocation"]),
        ("BUY", 0.7, ["BUY SIGNAL", "Conservative allocation"]),
        ("HOLD", 0.5, ["HOLD SIGNAL"]),
        ("SELL", 0.3, ["SELL SIGNAL", "risk management"]),
        ("STRONG_SELL", 0.1, ["STRONG_SELL SIGNAL", "risk management"]),
    ],
)
def test_execute_autonomous_trading_logging(monkeypatch, tmp_path, action, strength, expected_msgs, fake_moralis):
    # Patch analyze_portfolio_performance to return crafted signal
    class DTClass:
        @classmethod
        def now(cls):
            class N:
                def strftime(self, fmt): return "20250101" if "%Y%m%d" in fmt else "2025-01-01 00:00:00"
                def isoformat(self): return "2025-01-01T00:00:00"
                hour = 0
                minute = 0
            return N()
    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    adam = ADAM_System("ADDR")
    adam.log_file = str(tmp_path / "trading.log")

    perf = {
        "current_signal": {"trading_action": action, "signal_strength": strength},
        "regal_address": "ADDR",
    }
    monkeypatch.setattr(adam, "analyze_portfolio_performance", lambda: perf, raising=True)

    adam.execute_autonomous_trading()

    text = Path(adam.log_file).read_text()
    for msg in expected_msgs:
        assert msg.split()[0] in text  # coarse check due to emojis and formatting

def test_run_continuous_monitoring_cycles(monkeypatch, tmp_path, fake_moralis):
    logs = tmp_path / "monitor.log"

    # Patch datetime for deterministic filename/time
    class DTClass:
        @classmethod
        def now(cls):
            class N:
                def strftime(self, fmt): return "20250101" if "%Y%m%d" in fmt else "2025-01-01 00:00:00"
                def isoformat(self): return "2025-01-01T00:00:00"
                hour = 0
                minute = 0
            return N()
    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    a = ADAM_System("ADDR")
    a.log_file = str(logs)

    # Prevent sleep and shorten cycles: cycles=2 for speed
    monkeypatch.setattr(ADAM.time, "sleep", lambda *_: None, raising=False)

    # Make analyze_portfolio_performance deterministic
    perf_template = {
        "regal_address": "ADDR",
        "harmonic_frequency": 888.0,
        "current_signal": {"trading_action": "HOLD", "signal_strength": 0.5},
        "resonance_quality": 1.0,
        "performance_multiplier": 3.0,
        "recommendation": "HOLD",
    }
    monkeypatch.setattr(a, "analyze_portfolio_performance", lambda: dict(perf_template), raising=True)

    a.run_continuous_monitoring(cycles=2)

    text = logs.read_text()
    # Validate key milestones logged
    assert "autonomous monitoring started" in text
    assert "CYCLE 1/2" in text
    assert "CYCLE 2/2" in text
    assert "Next analysis in" in text
    assert "monitoring cycle complete" in text

def test_generate_performance_report_creates_json(monkeypatch, tmp_path, fake_moralis):
    # Patch datetime for deterministic file names and contents
    class Now:
        def __init__(self, iso="2025-01-01T00:00:00"): self._iso = iso
        def isoformat(self): return self._iso
        def strftime(self, fmt):
            if fmt == "%Y%m%d_%H%M%S":
                return "20250101_000000"
            if fmt == "%Y%m%d":
                return "20250101"
            return "2025-01-01 00:00:00"
        hour = 0
        minute = 0
    class DTClass:
        @classmethod
        def now(cls): return Now()
    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    a = ADAM_System("ADDR")
    a.log_file = str(tmp_path / "report.log")
    # Seed performance_data with multiple actions
    a.performance_data = [
        {
            "regal_address": "ADDR",
            "harmonic_frequency": 888.0,
            "current_signal": {"trading_action": "BUY", "signal_strength": 0.7},
            "resonance_quality": 1.0,
            "performance_multiplier": 3.0,
            "recommendation": "BUY",
        },
        {
            "regal_address": "ADDR",
            "harmonic_frequency": 888.0,
            "current_signal": {"trading_action": "HOLD", "signal_strength": 0.4},
            "resonance_quality": 1.0,
            "performance_multiplier": 3.0,
            "recommendation": "HOLD",
        },
    ]
    a.generate_performance_report()

    # Validate JSON written
    files = list(tmp_path.parent.glob("adam_report_20250101_000000.json"))
    # If repo writes to CWD, search in repo root; otherwise try tmp_path
    candidates = list(Path(".").glob("adam_report_20250101_000000.json"))
    if files:
        report_path = files[0]
    elif candidates:
        report_path = candidates[0]
    else:
        # Fallback: search anywhere under cwd for the specific name
        found = list(Path(".").rglob("adam_report_20250101_000000.json"))
        assert found, "Expected report file not found"
        report_path = found[0]

    data = json.loads(Path(report_path).read_text())
    assert data["regal_address"] == "ADDR"
    assert data["harmonic_frequency"] == 888.0
    assert "performance_data" in data and len(data["performance_data"]) == 2
    assert data["summary"]["total_cycles"] == 2
    assert "avg_signal_strength" in data["summary"]
    assert set(data["summary"]["action_counts"].keys()) >= {"BUY", "HOLD"}

def test_generate_performance_report_no_data(monkeypatch, tmp_path, fake_moralis, capsys):
    # Patch datetime for log filename determinism
    class DTClass:
        @classmethod
        def now(cls):
            class N:
                def strftime(self, fmt): return "20250101" if "%Y%m%d" in fmt else "2025-01-01 00:00:00"
                def isoformat(self): return "2025-01-01T00:00:00"
                hour = 0
                minute = 0
            return N()
    monkeypatch.setattr(ADAM, "datetime", DTClass, raising=True)

    a = ADAM_System("ADDR")
    a.log_file = str(tmp_path / "empty.log")
    a.performance_data = []
    a.generate_performance_report()
    # Ensure it logs "No performance data available"
    text = Path(a.log_file).read_text()
    assert "No performance data available" in text

# Note: Tests assume pytest as the testing framework.