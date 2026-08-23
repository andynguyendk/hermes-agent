"""Tests for skills/autonomous-ai-agents/loop-bounce/scripts/loop_detector.py"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "skills" / "autonomous-ai-agents" / "loop-bounce" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import loop_detector as ld


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalizeToolName:
    def test_docker_compose_variants(self):
        assert ld.normalize_tool_name("docker-compose") == "docker_compose"
        assert ld.normalize_tool_name("docker compose") == "docker_compose"
        assert ld.normalize_tool_name("docker_compose") == "docker_compose"

    def test_ssh_wrapped(self):
        result = ld.strip_ssh_prefix("ssh nd@10.0.0.1 ls -la")
        assert result == "ls -la"

    def test_no_ssh_prefix(self):
        assert ld.strip_ssh_prefix("ls -la") == "ls -la"

    def test_path_normalization(self):
        assert ld.normalize_path("foo/bar/") == "foo/bar"
        assert ld.normalize_path("./foo/../bar") == "bar"
        assert ld.normalize_path("") == "."

    def test_command_normalization(self):
        cmd = ld.normalize_command("docker-compose up -d")
        assert "docker_compose" in cmd
        assert "docker-compose" not in cmd

    def test_home_normalization(self):
        cmd = ld.normalize_command("cat ~/config.yaml")
        assert "/home" in cmd


# ---------------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------------

class TestFingerprinting:
    def test_same_input_same_hash(self):
        args = {"command": "echo hello", "timeout": 30}
        h1 = ld.hash_input(args)
        h2 = ld.hash_input(args)
        assert h1 == h2

    def test_different_input_different_hash(self):
        h1 = ld.hash_input({"command": "echo hello"})
        h2 = ld.hash_input({"command": "echo world"})
        assert h1 != h2

    def test_docker_compose_normalization_affects_hash(self):
        """docker compose ≡ docker-compose should produce same hash."""
        h1 = ld.hash_input({"command": "docker-compose up -d"})
        h2 = ld.hash_input({"command": "docker compose up -d"})
        assert h1 == h2

    def test_ssh_normalization_affects_hash(self):
        """ssh-wrapped and bare commands should produce same hash."""
        h1 = ld.hash_input({"command": "ssh nd@host ls -la"})
        h2 = ld.hash_input({"command": "ls -la"})
        assert h1 == h2

    def test_fingerprint_key_format(self):
        fp = ld.Fingerprint(tool="terminal", input_hash="abc123", error_class="network")
        assert fp.key() == "terminal|abc123|network"

    def test_fingerprint_equality(self):
        fp1 = ld.Fingerprint(tool="a", input_hash="b", error_class="c")
        fp2 = ld.Fingerprint(tool="a", input_hash="b", error_class="c")
        assert fp1 == fp2

    def test_fingerprint_inequality(self):
        fp1 = ld.Fingerprint(tool="a", input_hash="b")
        fp2 = ld.Fingerprint(tool="a", input_hash="c")
        assert fp1 != fp2


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

class TestErrorClassification:
    def test_rate_limit(self):
        cat, ecls = ld.classify_error("HTTP 429: Too Many Requests")
        assert cat == "policy"
        assert ecls == "rate_limit"

    def test_auth_failure(self):
        cat, ecls = ld.classify_error("401 Unauthorized")
        assert cat == "policy"
        assert ecls == "auth"

    def test_network_error(self):
        cat, ecls = ld.classify_error("Connection refused: ENOTFOUND")
        assert cat == "environment"
        assert ecls == "network"

    def test_disk_full(self):
        cat, ecls = ld.classify_error("No space left on device ENOSPC")
        assert cat == "environment"
        assert ecls == "disk"

    def test_oom(self):
        cat, ecls = ld.classify_error("Killed: SIGKILL OOM")
        assert cat == "environment"
        assert ecls == "oom"

    def test_unknown_error(self):
        cat, ecls = ld.classify_error("Something went wrong")
        assert cat == "execution"
        assert ecls == "unknown"

    def test_permission_denied(self):
        cat, ecls = ld.classify_error("Permission denied EACCES")
        assert cat == "environment"
        assert ecls == "permission"


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

class TestParseSession:
    def _write_jsonl(self, messages: list[dict]) -> str:
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with open(path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg) + "\n")
        return path

    def test_basic_tool_calls(self):
        path = self._write_jsonl([
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "terminal", "arguments": json.dumps({"command": "echo hi"})}}
            ]},
            {"role": "tool", "content": "hi", "tool_call_id": "call_1"},
        ])
        calls = ld.parse_session(path)
        assert len(calls) == 1
        assert calls[0].tool_name == "terminal"
        assert calls[0].args["command"] == "echo hi"
        assert calls[0].result == "hi"

    def test_error_detection(self):
        path = self._write_jsonl([
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "function": {"name": "terminal", "arguments": json.dumps({"command": "curl example.com"})}}
            ]},
            {"role": "tool", "content": "Connection refused: ENOTFOUND", "tool_call_id": "call_1"},
        ])
        calls = ld.parse_session(path)
        assert len(calls) == 1
        assert calls[0].error != ""
        assert calls[0].category == "environment"

    def test_empty_session(self):
        path = self._write_jsonl([])
        calls = ld.parse_session(path)
        assert calls == []


# ---------------------------------------------------------------------------
# Sliding window detection
# ---------------------------------------------------------------------------

class TestDetectLoops:
    def _make_calls(self, tool: str, args: dict, n: int, error: str = "") -> list[ld.ToolCall]:
        return [
            ld.ToolCall(turn=i + 1, tool_name=tool, args=args, error=error)
            for i in range(n)
        ]

    def test_no_loop_below_threshold(self):
        calls = self._make_calls("terminal", {"command": "echo hi"}, 2)
        assert ld.detect_loops(calls, threshold=3) == []

    def test_loop_at_threshold(self):
        calls = self._make_calls("terminal", {"command": "echo hi"}, 3)
        detections = ld.detect_loops(calls, threshold=3)
        assert len(detections) == 1
        assert detections[0].category == "execution"

    def test_loop_requires_same_fingerprint(self):
        calls = (
            self._make_calls("terminal", {"command": "echo a"}, 2)
            + self._make_calls("terminal", {"command": "echo b"}, 2)
        )
        # 2 calls to same tool but different input — below threshold of 3
        assert ld.detect_loops(calls, threshold=3) == []

    def test_docker_compose_normalization_deduplicates(self):
        calls = [
            ld.ToolCall(turn=1, tool_name="terminal", args={"command": "docker-compose up"}),
            ld.ToolCall(turn=2, tool_name="terminal", args={"command": "docker compose up"}),
            ld.ToolCall(turn=3, tool_name="terminal", args={"command": "docker_compose up"}),
        ]
        detections = ld.detect_loops(calls, threshold=3)
        assert len(detections) == 1

    def test_sliding_window_not_consecutive(self):
        """A non-matching call between repeats should NOT reset the window."""
        calls = [
            ld.ToolCall(turn=1, tool_name="terminal", args={"command": "echo loop"}),
            ld.ToolCall(turn=2, tool_name="terminal", args={"command": "pwd"}),
            ld.ToolCall(turn=3, tool_name="terminal", args={"command": "echo loop"}),
            ld.ToolCall(turn=4, tool_name="terminal", args={"command": "echo loop"}),
        ]
        detections = ld.detect_loops(calls, threshold=3, window_size=5)
        assert len(detections) == 1

    def test_window_trims_old(self):
        """Fingerprints outside the window should not count."""
        calls = [
            ld.ToolCall(turn=1, tool_name="terminal", args={"command": "echo loop"}),
            # 10 non-matching calls to push it out of window
            *[ld.ToolCall(turn=i + 2, tool_name="terminal", args={"command": f"cmd_{i}"}) for i in range(10)],
            ld.ToolCall(turn=12, tool_name="terminal", args={"command": "echo loop"}),
            ld.ToolCall(turn=13, tool_name="terminal", args={"command": "echo loop"}),
        ]
        # Only 3 occurrences but first is outside window of 10
        detections = ld.detect_loops(calls, threshold=3, window_size=10)
        assert len(detections) == 0

    def test_policy_category_stops_immediately(self):
        calls = self._make_calls("terminal", {"command": "curl api.example.com"}, 2, error="HTTP 429: rate limited")
        detections = ld.detect_loops(calls, threshold=1)
        assert len(detections) >= 1
        assert any(d.severity == "stop" for d in detections)


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------

class TestSeverity:
    def test_policy_is_stop(self):
        assert ld._severity("policy", 1, 3) == "stop"

    def test_environment_is_warning(self):
        assert ld._severity("environment", 10, 3) == "warning"

    def test_repeated_approach_critical(self):
        assert ld._severity("repeated_approach", 4, 3) == "critical"

    def test_repeated_approach_stop_when_high(self):
        assert ld._severity("repeated_approach", 6, 3) == "stop"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_empty_session(self):
        stats = ld.session_stats([])
        assert stats["total_tool_calls"] == 0

    def test_basic_stats(self):
        calls = [
            ld.ToolCall(turn=1, tool_name="terminal", args={"command": "ls"}),
            ld.ToolCall(turn=2, tool_name="terminal", args={"command": "ls"}),
            ld.ToolCall(turn=3, tool_name="read_file", args={"path": "/a"}),
        ]
        stats = ld.session_stats(calls)
        assert stats["total_tool_calls"] == 3
        assert stats["tool_counts"]["terminal"] == 2
        assert stats["tool_counts"]["read_file"] == 1


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

class TestFormatting:
    def test_no_loops_message(self):
        assert ld.format_detections([]) == "No loops detected."

    def test_detections_reported(self):
        det = ld.LoopDetection(
            fingerprint=ld.Fingerprint(tool="terminal", input_hash="abc"),
            occurrences=[1, 2, 3],
            category="execution",
            severity="warning",
            message="test loop",
        )
        output = ld.format_detections([det])
        assert "test loop" in output
        assert "execution" in output
