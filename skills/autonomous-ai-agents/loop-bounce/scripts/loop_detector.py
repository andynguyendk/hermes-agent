#!/usr/bin/env python3
"""Loop detector for Hermes Agent sessions.

Analyzes session transcripts (JSONL) to detect agent loops: repeated tool
calls, identical errors, and reasoning cycles. Uses fingerprint normalization
and sliding-window detection instead of naive consecutive counting.

Stdlib only — no third-party dependencies.

Usage:
    python loop_detector.py analyze <session.jsonl> [--window N] [--threshold N]
    python loop_detector.py stats <session.jsonl>
    python loop_detector.py analyze - < <session.jsonl>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WINDOW = 10
DEFAULT_THRESHOLD = 3
DEFAULT_MAX_BUDGET = 3

# Failure categories and their default weights (higher = more urgent)
CATEGORY_WEIGHTS: dict[str, int] = {
    "execution": 3,
    "diagnostic": 3,
    "verification": 3,
    "repeated_approach": 4,
    "environment": 1,
    "policy": 4,
}

# Error patterns → category
# Order matters: environment patterns checked before policy for overlaps
# (e.g. "Permission denied EACCES" → environment, not auth)
ERROR_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"429|rate.?limit|too.?many.?requests", re.I), "policy", "rate_limit"),
    (re.compile(r"401|403|unauthorized|forbidden", re.I), "policy", "auth"),
    (re.compile(r"network.?unreachable|connection.?refused|ENOTFOUND|ETIMEDOUT", re.I), "environment", "network"),
    (re.compile(r"disk.?full|no.?space.?left|ENOSPC", re.I), "environment", "disk"),
    (re.compile(r"permission.?denied|EACCES", re.I), "environment", "permission"),
    (re.compile(r"503|504|service.?unavailable|gateway.?timeout", re.I), "environment", "transient"),
    (re.compile(r"SIGKILL|OOM|out.?of.?memory", re.I), "environment", "oom"),
]

# Tool name normalization aliases
TOOL_ALIASES: dict[str, str] = {
    "docker-compose": "docker_compose",
    "docker compose": "docker_compose",
    "docker_compose": "docker_compose",
}

SSH_PREFIX_RE = re.compile(r"^ssh\s+\S+@\S+\s+")

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Fingerprint:
    """Normalized representation of a tool call for loop detection."""
    tool: str
    input_hash: str
    error_class: str = ""
    category: str = ""
    raw_tool: str = ""
    raw_input: str = ""

    def key(self) -> str:
        return f"{self.tool}|{self.input_hash}|{self.error_class}"

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fingerprint):
            return NotImplemented
        return self.key() == other.key()


@dataclass
class ToolCall:
    """A single tool invocation from the session."""
    turn: int
    tool_name: str
    args: dict[str, Any]
    result: str = ""
    error: str = ""
    category: str = ""
    error_class: str = ""
    timestamp: float = 0.0


@dataclass
class LoopDetection:
    """A detected loop."""
    fingerprint: Fingerprint
    occurrences: list[int]
    category: str
    severity: str  # "warning" | "critical" | "stop"
    message: str


@dataclass
class DetectionState:
    """Sliding-window state for a single fingerprint."""
    window: list[int] = field(default_factory=list)
    last_seen: int = 0
    category: str = ""
    budget_used: int = 0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_tool_name(name: str) -> str:
    """Normalize tool name to canonical form."""
    lower = name.lower().strip()
    if lower in TOOL_ALIASES:
        return TOOL_ALIASES[lower]
    return re.sub(r"[^a-z0-9]+", "_", lower).strip("_")


def normalize_path(path: str) -> str:
    """Normalize a path string for fingerprinting."""
    # Strip trailing slashes, resolve ./ and ../
    path = path.rstrip("/")
    parts: list[str] = []
    for part in path.split("/"):
        if part == ".":
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts) or "."


def strip_ssh_prefix(cmd: str) -> str:
    """Remove SSH wrapper from a command, returning the inner command."""
    return SSH_PREFIX_RE.sub("", cmd.strip())


def normalize_command(cmd: str) -> str:
    """Normalize a shell command for fingerprinting."""
    cmd = strip_ssh_prefix(cmd)
    # Normalize docker compose variants
    cmd = re.sub(r"docker[\s-]+compose", "docker_compose", cmd)
    # Normalize paths within the command
    cmd = re.sub(r"(?:~|\$HOME)", "/home", cmd)
    cmd = re.sub(r"\./", "/", cmd)
    # Collapse whitespace
    cmd = re.sub(r"\s+", " ", cmd).strip()
    return cmd


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize tool arguments for fingerprinting."""
    normalized: dict[str, Any] = {}
    for key in sorted(args.keys()):
        val = args[key]
        if isinstance(val, str):
            # Normalize command strings
            if key in ("command", "cmd", "script", "query"):
                val = normalize_command(val)
            elif key in ("path", "file", "directory", "dir"):
                val = normalize_path(val)
            # Normalize docker compose in any string value
            val = re.sub(r"docker[\s-]+compose", "docker_compose", val)
        elif isinstance(val, dict):
            val = normalize_args(val)
        elif isinstance(val, list):
            val = [normalize_command(v) if isinstance(v, str) else v for v in val]
        normalized[key] = val
    return normalized


def hash_input(args: dict[str, Any]) -> str:
    """Hash normalized tool arguments for fingerprinting."""
    normalized = normalize_args(args)
    raw = json.dumps(normalized, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def classify_error(error_text: str) -> tuple[str, str]:
    """Classify an error message into (category, error_class)."""
    for pattern, category, error_class in ERROR_PATTERNS:
        if pattern.search(error_text):
            return category, error_class
    return "execution", "unknown"


# Broad error indicators for tool results (catches things ERROR_PATTERNS misses)
_ERROR_INDICATORS = re.compile(
    r"error|traceback|exception|failed|refused|denied|killed|timeout|fatal|panic|abort",
    re.I,
)


def _has_error_indicator(text: str) -> bool:
    """Check if text contains common error indicators."""
    return bool(_ERROR_INDICATORS.search(text))


def fingerprint_tool_call(call: ToolCall) -> Fingerprint:
    """Create a normalized fingerprint from a tool call."""
    tool = normalize_tool_name(call.tool_name)
    h = hash_input(call.args)
    category = call.category
    error_class = call.error_class

    if call.error and not category:
        category, error_class = classify_error(call.error)

    return Fingerprint(
        tool=tool,
        input_hash=h,
        error_class=error_class,
        category=category or "execution",
        raw_tool=call.tool_name,
        raw_input=json.dumps(call.args, sort_keys=True)[:200],
    )


# ---------------------------------------------------------------------------
# Session parsing
# ---------------------------------------------------------------------------

def parse_session(filepath: str) -> list[ToolCall]:
    """Parse a Hermes session JSONL file into tool calls.

    Supports both the standard Hermes JSONL format and the expanded
    format with explicit 'tool_calls' arrays.
    """
    calls: list[ToolCall] = []
    turn = 0

    def _open(path: str):
        if path == "-":
            return sys.stdin
        return open(path, "r", encoding="utf-8")

    with _open(filepath) as fh:
        for line_no, raw_line in enumerate(fh, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue

            role = msg.get("role", "")
            if role == "assistant":
                turn += 1
                # Check for tool_calls in the message
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                    except json.JSONDecodeError:
                        args = {}
                    calls.append(ToolCall(
                        turn=turn,
                        tool_name=name,
                        args=args if isinstance(args, dict) else {},
                        timestamp=msg.get("created_at", msg.get("timestamp", 0)),
                    ))

            elif role == "tool":
                # Match tool result back to the last unmatched call
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Some formats have content as list of parts
                    content = " ".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if isinstance(content, dict):
                    content = json.dumps(content)

                content_str = str(content)
                # Try to find matching call by tool_call_id
                tool_call_id = msg.get("tool_call_id", "")
                matched = False
                for c in reversed(calls):
                    if c.result:
                        continue
                    if tool_call_id:
                        # Match by ID if available
                        c.result = content_str
                        cat, ecls = classify_error(content_str)
                        if ecls != "unknown" or _has_error_indicator(content_str):
                            c.error = content_str[:500]
                            c.category = cat
                            c.error_class = ecls
                        matched = True
                        break
                if not matched and calls:
                    # Fallback: match by position
                    for c in reversed(calls):
                        if not c.result:
                            c.result = content_str
                            cat, ecls = classify_error(content_str)
                            if ecls != "unknown" or _has_error_indicator(content_str):
                                c.error = content_str[:500]
                                c.category = cat
                                c.error_class = ecls
                            break

    return calls


# ---------------------------------------------------------------------------
# Sliding window detection
# ---------------------------------------------------------------------------

def detect_loops(
    calls: list[ToolCall],
    window_size: int = DEFAULT_WINDOW,
    threshold: int = DEFAULT_THRESHOLD,
) -> list[LoopDetection]:
    """Detect loops using sliding-window fingerprint analysis."""
    detections: list[LoopDetection] = []
    states: dict[str, DetectionState] = {}

    for call in calls:
        fp = fingerprint_tool_call(call)
        key = fp.key()

        if key not in states:
            states[key] = DetectionState(category=fp.category)

        state = states[key]
        state.window.append(call.turn)
        state.last_seen = call.turn

        # Trim window: keep only occurrences within [current - window_size, current]
        window_start = call.turn - window_size
        state.window = [t for t in state.window if t > window_start]

        # Check threshold
        if len(state.window) >= threshold:
            severity = _severity(fp.category, len(state.window), threshold)
            msg = (
                f"Loop detected: {fp.raw_tool} called {len(state.window)} times "
                f"in last {window_size} turns (category={fp.category})"
            )
            detections.append(LoopDetection(
                fingerprint=fp,
                occurrences=list(state.window),
                category=fp.category,
                severity=severity,
                message=msg,
            ))

    return detections


def _severity(category: str, count: int, threshold: int) -> str:
    """Determine severity based on category and repetition count."""
    if category == "policy":
        return "stop"
    if category in ("repeated_approach", "diagnostic", "verification"):
        if count >= threshold + 2:
            return "stop"
        return "critical"
    if category == "environment":
        return "warning"
    # execution
    if count >= threshold + 2:
        return "critical"
    return "warning"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def session_stats(calls: list[ToolCall]) -> dict[str, Any]:
    """Compute statistics for a session."""
    tool_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    fps: dict[str, int] = defaultdict(int)

    for call in calls:
        tool_counts[normalize_tool_name(call.tool_name)] += 1
        cat = call.category or ("execution" if not call.error else classify_error(call.error)[0])
        category_counts[cat] += 1
        if call.error:
            _, ecls = classify_error(call.error)
            error_counts[ecls] += 1
        fp = fingerprint_tool_call(call)
        fps[fp.key()] += 1

    # Find most repeated fingerprints
    top_fps = sorted(fps.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_turns": calls[-1].turn if calls else 0,
        "total_tool_calls": len(calls),
        "unique_fingerprints": len(fps),
        "tool_counts": dict(tool_counts.most_common(20)),
        "category_counts": dict(category_counts),
        "error_counts": dict(error_counts),
        "top_repeated": [
            {"fingerprint": k, "count": v}
            for k, v in top_fps
            if v > 1
        ],
    }


# ---------------------------------------------------------------------------
# Persistence (SQLite cache for tracking across runs)
# ---------------------------------------------------------------------------

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    fingerprint TEXT,
    category TEXT,
    severity TEXT,
    occurrences TEXT,
    detected_at REAL,
    UNIQUE(session_id, fingerprint)
);

CREATE TABLE IF NOT EXISTS fingerprints (
    fingerprint TEXT PRIMARY KEY,
    tool TEXT,
    input_hash TEXT,
    error_class TEXT,
    category TEXT,
    first_seen REAL,
    last_seen REAL,
    hit_count INTEGER DEFAULT 1
);
"""


def open_db(db_path: str | None = None) -> sqlite3.Connection:
    """Open (or create) the detection database."""
    if db_path is None:
        db_path = os.environ.get(
            "LOOP_DETECTOR_DB",
            str(Path.home() / ".hermes" / "loop_detector.db"),
        )
    conn = sqlite3.connect(db_path)
    conn.executescript(DB_SCHEMA)
    return conn


def persist_detection(conn: sqlite3.Connection, session_id: str, detection: LoopDetection) -> None:
    """Persist a detection to the database."""
    conn.execute(
        """INSERT OR REPLACE INTO detections
           (session_id, fingerprint, category, severity, occurrences, detected_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            detection.fingerprint.key(),
            detection.category,
            detection.severity,
            json.dumps(detection.occurrences),
            time.time(),
        ),
    )
    conn.commit()


def persist_fingerprint(conn: sqlite3.Connection, fp: Fingerprint) -> None:
    """Update fingerprint tracking."""
    now = time.time()
    conn.execute(
        """INSERT INTO fingerprints (fingerprint, tool, input_hash, error_class, category, first_seen, last_seen, hit_count)
           VALUES (?, ?, ?, ?, ?, ?, ?, 1)
           ON CONFLICT(fingerprint) DO UPDATE SET
               last_seen = excluded.last_seen,
               hit_count = hit_count + 1""",
        (fp.key(), fp.tool, fp.input_hash, fp.error_class, fp.category, now, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_detections(detections: list[LoopDetection]) -> str:
    """Format detections as human-readable output."""
    if not detections:
        return "No loops detected."

    lines: list[str] = []
    lines.append(f"=== Loop Detection Report ({len(detections)} loops found) ===\n")

    for i, det in enumerate(detections, 1):
        lines.append(f"[{det.severity.upper()}] Loop #{i}: {det.message}")
        lines.append(f"  Tool: {det.fingerprint.raw_tool}")
        lines.append(f"  Category: {det.category}")
        lines.append(f"  Occurrences at turns: {det.occurrences}")
        if det.fingerprint.error_class:
            lines.append(f"  Error class: {det.fingerprint.error_class}")
        lines.append("")

    # Summary
    cats = Counter(d.category for d in detections)
    lines.append("--- Summary ---")
    for cat, count in cats.most_common():
        lines.append(f"  {cat}: {count}")
    stops = sum(1 for d in detections if d.severity == "stop")
    if stops:
        lines.append(f"\n  *** {stops} loop(s) require immediate stop ***")

    return "\n".join(lines)


def format_stats(stats: dict[str, Any]) -> str:
    """Format statistics as human-readable output."""
    lines: list[str] = []
    lines.append("=== Session Statistics ===\n")
    lines.append(f"Total turns: {stats['total_turns']}")
    lines.append(f"Total tool calls: {stats['total_tool_calls']}")
    lines.append(f"Unique fingerprints: {stats['unique_fingerprints']}\n")

    if stats["tool_counts"]:
        lines.append("Tool usage:")
        for tool, count in sorted(stats["tool_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"  {tool}: {count}")

    if stats["category_counts"]:
        lines.append("\nFailure categories:")
        for cat, count in sorted(stats["category_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"  {cat}: {count}")

    if stats["error_counts"]:
        lines.append("\nError classes:")
        for ecls, count in sorted(stats["error_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"  {ecls}: {count}")

    if stats["top_repeated"]:
        lines.append("\nMost repeated fingerprints:")
        for item in stats["top_repeated"]:
            lines.append(f"  {item['count']}x: {item['fingerprint'][:60]}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect agent loops in Hermes session transcripts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_p = sub.add_parser(
        "analyze",
        help="Analyze a session file for loops",
    )
    analyze_p.add_argument("session", help="Path to session JSONL (or - for stdin)")
    analyze_p.add_argument("--window", type=int, default=DEFAULT_WINDOW, help=f"Sliding window size (default: {DEFAULT_WINDOW})")
    analyze_p.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help=f"Repeat threshold (default: {DEFAULT_THRESHOLD})")
    analyze_p.add_argument("--db", type=str, default=None, help="SQLite DB path (default: ~/.hermes/loop_detector.db)")
    analyze_p.add_argument("--session-id", type=str, default=None, help="Session ID for persistence")

    stats_p = sub.add_parser(
        "stats",
        help="Show statistics for a session file",
    )
    stats_p.add_argument("session", help="Path to session JSONL (or - for stdin)")

    args = parser.parse_args()

    if args.command == "analyze":
        calls = parse_session(args.session)
        if not calls:
            print("No tool calls found in session.")
            sys.exit(0)

        detections = detect_loops(calls, window_size=args.window, threshold=args.threshold)
        print(format_detections(detections))

        # Persist if DB available
        if args.db or os.environ.get("LOOP_DETECTOR_DB") or Path.home().joinpath(".hermes").exists():
            try:
                conn = open_db(args.db)
                session_id = args.session_id or Path(args.session).stem if args.session != "-" else "stdin"
                for det in detections:
                    persist_detection(conn, session_id, det)
                    persist_fingerprint(conn, det.fingerprint)
                conn.close()
            except Exception as e:
                print(f"Warning: could not persist to DB: {e}", file=sys.stderr)

        # Exit code: 0=no loops, 1=loops found, 2=stop-level loops
        if any(d.severity == "stop" for d in detections):
            sys.exit(2)
        elif detections:
            sys.exit(1)
        sys.exit(0)

    elif args.command == "stats":
        calls = parse_session(args.session)
        stats = session_stats(calls)
        print(format_stats(stats))


if __name__ == "__main__":
    main()
