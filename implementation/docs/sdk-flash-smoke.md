# Agent SDK / Flash compatibility smoke — 2026-09-10

This is an actual Agent SDK + paid LLM compatibility test on synthetic engineering data. It is not a clinical report workflow or a desktop Agent entry point.

## Verified runtime

- `claude-agent-sdk==0.2.152`, bundled Claude Code CLI `2.1.259`, Python 3.12 on macOS arm64.
- Official Anthropic-compatible endpoint: `https://api.deepseek.com/anthropic/v1/messages`.
- All 8 real requests and response model IDs: `deepseek-flash`. The current official pricing page identifies it as V4.1 Flash, serving the retired V4 Flash and vision-exp aliases. No Pro or vision request was made. [Official pricing](https://api-docs.deepseek.com/quick_start/pricing/), [protocol compatibility](https://api-docs.deepseek.com/guides/anthropic_api/).
- SDK init exposes the subtask tool as `Task` for this provider; the permission hook reports `Agent`. Both names are constrained to the named read-only `evidence_checker` subagent.

## Enforcement

The host creates a separate synthetic case/run in `runtime/sdk-smoke/harness.sqlite3`. A loopback gateway owns the real provider credential and a host callback to Core. The worker receives only a random, session-scoped gateway capability. Tools have empty argument schemas; a model cannot supply another patient/run, an artifact, a verdict, a path, or a command.

The SDK offers only the subtask builtin and three MCP tools: synthetic prediction, release-gate probe, and evidence-status inspection. PreToolUse allows only these operations; child agents can only inspect. Shell, filesystem, network browsing, user/project settings, skills, arbitrary agents, resumption and background delegation are unavailable to the model. The release probe calls the existing Core release gate. The explicit `ok:false` tool body matters because the provider does not enforce Anthropic `is_error`. Model text is not registered as a report or sent to the desktop.

The gateway accepts only the exact Flash ID, streams only after checking each complete SSE frame, rejects a mismatched response model, fixes the upstream URL, disables redirects and environment proxies, caps output at 384 tokens, and disables thinking. Its serialized requests and tool callbacks do not depend on a provider's parallel-tool flag. SDK retries pass through the same admission budget. The script holds an exclusive process lock; `budget.json` persists a maximum of 12 requests and USD 0.05 conservative reservation across invocations. Reservations use doubled request UTF-8 bytes plus protocol allowance at peak cache-miss pricing, not an invoice or a guaranteed billing settlement. No automatic reset or refund on errors/cancellation.

The real key is at `runtime/private/deepseek.key` (0600 in a 0700 directory, Git ignored). Runtime logs contain only controlled status and usage metadata. The key is neither given to the SDK nor included in command-line arguments. SDK configuration/cache files stay under project `runtime/`; persistence of SDK conversation transcripts is disabled. Only synthetic status and engineering tool descriptions are sent upstream.

The SDK profile is separate from fixture defaults: one worker, 90 seconds wall clock, CPU soft 15/hard 16 seconds, 1 GiB process-group RSS monitoring, 8 MiB per file, 256 descriptors per process, 256 KiB IPC output. Ordinary SDK/CLI descendants inherit the worker process group. This is tool permission control plus trusted process limits, **not an OS filesystem/network sandbox against malicious same-UID code**. The real key remains readable to a compromised same-UID process; no such isolation is claimed. See [worker limits](worker-isolation.md).

## Evidence and limitations

| Scenario | Real result | Evidence |
| --- | --- | --- |
| SDK tools and business failure | 3 requests, prediction fixture accepted, release rejected with REVIEW_REQUIRED, SDK completed; 1 artifact, no drafts/reviews/releases | `runtime/sdk-smoke/tools-result.json` |
| Read-only SDK subagent | 4 requests including child model calls, SubagentStart/Stop and child tool attribution, only evidence status read; no artifacts/reports | `runtime/sdk-smoke/subagent-result.json` |
| Streaming interrupt | 1 request, streamed delta observed, SDK interrupt acknowledged with aborted_streaming | `runtime/sdk-smoke/interrupt-result.json` |
| Real SDK/CLI stalls | Offline fake provider, deterministic wall timeout and cancellation, actual bundled CLI observed in worker group; SIGKILL and no remaining group RSS | `tests/test_sdk.py` |
| Permission/cost attacks | Offline wrong model, forged Bash call, child mutation/Pro/resume requests, bad auth/Origin/path/arguments, budget exhaustion, upstream error redaction | `tests/test_sdk.py` |

The three live workers all exited zero and were reaped; peak group RSS was approximately 300, 342, and 296 MiB. The interrupted upstream request has no final output usage, so its bill cannot be read exactly from the stream. Using peak prices for reported tokens and the full 384-token output cap for that request gives an estimated upper bound of **USD 0.00323** for these 8 requests; provider billing is authoritative. The persistent admission reservation is USD 0.047514 and is deliberately much more conservative. Do not confuse that reservation with money spent.

The SDK is proven in this bounded smoke entry point. Real prediction/RL, clinical rules, true report/reviewer integration, durable production Agent session management, desktop Agent entry, multimodal validation and Observation integration remain separate TODOs. SDK hooks are local smoke evidence, not proof of an Observation connection. Network timeouts use a controlled fake upstream to avoid wasting paid requests; the SDK and its bundled CLI in those tests are real.

## Reproduce

From `implementation/`, install the pinned dependencies using the project environment. Offline `pytest` never reads the real key or contacts DeepSeek:

```bash
.venv/bin/pytest -q tests/test_sdk.py
```

An explicit paid run uses the existing private key and persistent budget:

```bash
.venv/bin/python scripts/sdk_smoke.py --live --scenario tools
.venv/bin/python scripts/sdk_smoke.py --live --scenario subagent
.venv/bin/python scripts/sdk_smoke.py --live --scenario interrupt
```

The current reservation will block further full smoke runs; it is not silently reset. Each invocation adds a synthetic run while preserving old ledger rows; the corresponding scenario result file is the latest export. No App database is used.
