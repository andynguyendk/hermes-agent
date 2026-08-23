---
name: loop-bounce
description: "Use when the agent repeats identical tool calls in a loop."
version: 1.0.0
author: "andynguyendk (Andy Nguyen) + Hermes Agent"
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [loop-detection, error-classification, model-escalation, retry-budget]
    related_skills: [hermes-agent]
---

# Loop Bounce

Detect and classify agent loops, then escalate or stop with a bounded budget. The detector distinguishes "keep failing at the same thing" from "should never retry this" from "this is a known transient hiccup" — so it escalates when escalation helps and stops when it doesn't.

## Failure Taxonomy

Not all loops are equal. Classify first, then decide:

| Category | Weight | Action | Examples |
|----------|--------|--------|----------|
| **execution** | high | Retry with variation, then escalate | Tool returns same error 3x |
| **diagnostic** | high | Retry with different approach, then escalate | Same reasoning trace repeating |
| **verification** | high | Retry once with different verification, then stop | `read_file` returning same content repeatedly |
| **repeated_approach** | immediate | Escalate or stop now | Identical tool call + identical input 4x |
| **environment** | report | Report to user, do not retry | Network unreachable, permission denied, disk full |
| **policy** | stop | STOP immediately, do not escalate | Rate limit (429), auth failure, user prompt issue |

## Fingerprinting

A loop fingerprint normalizes the tool call to detect repeats across superficial variations:

- **Tool name** (canonical)
- **Input hash** (JSON-serialized args, keys sorted)
- **Error class** (HTTP status, exception type, exit code — not the raw message)

Normalization rules:
- `docker compose` ≡ `docker-compose` ≡ `docker compose` (all map to `docker_compose`)
- SSH-wrapped commands: strip `ssh user@host` prefix, hash the inner command
- Paths: normalize trailing slashes, resolve `./` and `../` but preserve absolute vs relative
- Environment variables: `$HOME` and `~` are equivalent; `$USER` is not

## Sliding Window Detection

Instead of requiring N consecutive identical failures (a `pwd` between failures resets the counter), use a sliding window:

- **Window size:** configurable, default 10 turns
- **Threshold:** N fingerprints of the same class within the window triggers detection
- **Deduplication:** only the most recent occurrence of each unique fingerprint counts toward the threshold

This kills the classic blind spot where a non-failing tool call between failures resets the consecutive counter.

## Escalation Ladder

The escalation ladder has bounded steps. Each step consumes budget:

| Step | Budget Used | Action |
|------|-------------|--------|
| 1 | 1 | Continue with variation (different approach, not same command) |
| 2 | 1 | Try one alternative (different tool, different strategy) |
| 3 | 1 | Report to user with diagnosis and options |
| STOP | — | Max budget (default: 3) reached. Do NOT continue. |

**Rule:** After escalation step 3, STOP. Do not keep retrying. The agent must not auto-escalate beyond the user's choice — present options and wait.

## When NOT to Trigger

- **First failure:** the model sometimes recovers next turn
- **Transient HTTP errors:** 429 (rate limit), 503 (service unavailable), 504 (gateway timeout) — these are environment, not execution loops
- **Prompt-level issues:** if the instruction is wrong, fix the prompt — this is not a loop
- **Expected retries:** HTTP 429 backoff, network blip recovery — use exponential backoff, not escalation

## Usage (Skill-Driven)

When the agent self-detects a loop:

1. **Classify** the failure using the taxonomy above
2. **Fingerprint** the tool call(s) involved
3. **Check the sliding window** — is this a repeated fingerprint within the window?
4. **If yes:** follow the escalation ladder (steps 1 → 2 → 3 → STOP)
5. **If no:** continue normally, the failure is within tolerance

### Agent Self-Detection Signals

| Signal | Threshold | Category |
|--------|-----------|----------|
| Same error message in tool output ≥3 turns | 3x | execution |
| Same tool call with same input repeated ≥4 times | 4x | repeated_approach |
| "Let me try again" / "Let me retry" ≥3x on same task | 3x | diagnostic |
| Tool call returns identical result across 3 calls | 3x | verification |
| Reasoning trace shows same thought cycling | 2x repetition | diagnostic |
| Network/permission/disk error | 1x | environment (report, don't retry) |
| HTTP 429, auth failure, prompt issue | 1x | policy (STOP) |

## Model Escalation Options

When escalation step 3 is reached, present the user with options. Do NOT auto-switch:

```
1. Try a different approach on the same model
2. Switch model (if available)
3. Stop and summarize what was attempted
```

If the user picks option 2, they can use `/model` or `hermes model` to switch.

## Script Reference

The `scripts/loop_detector.py` in this skill directory provides a standalone detection engine:

```bash
# Analyze a session file
python scripts/loop_detector.py analyze <session.jsonl>

# Analyze from stdin (pipe from hermes sessions export)
hermes sessions export <id> | python scripts/loop_detector.py analyze -

# Show fingerprint stats for a session
python scripts/loop_detector.py stats <session.jsonl>
```

## Common Pitfalls

- Do NOT trigger on the first failure — the model sometimes recovers next turn
- Do NOT trigger for expected retries (HTTP 429 rate limits, network blips)
- Do NOT trigger for prompt-level issues (bad instruction) — fix the prompt
- Only trigger for reasoning loops / tool-call loops / repeated identical errors
- After escalation, do NOT keep retrying the same model — commit to the chosen model for this task
- Do NOT auto-escalate past step 3 — the user decides what happens next

## Verification Checklist

- [ ] Failure classified using the taxonomy (execution/diagnostic/verification/repeated_approach/environment/policy)
- [ ] Fingerprint normalized (docker, SSH, path normalization applied)
- [ ] Sliding window checked (not just consecutive counter)
- [ ] Escalation budget tracked and enforced
- [ ] User notified at step 3 with options (not auto-escalated)
