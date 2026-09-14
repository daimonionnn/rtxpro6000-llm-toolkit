# Chat UI

A single-page chat client for the local server that reports the numbers you
actually want while using the model: time-to-first-token, decode speed, prefill
speed and token counts, per turn and for the session.

```bash
scripts/start-ui.sh          # in the background on http://127.0.0.1:5173
PORT=8080 scripts/start-ui.sh
scripts/stop-ui.sh

python3 ui/serve.py          # or in the foreground; opens a browser
```

Point it at another endpoint with a query parameter:

```
http://127.0.0.1:5173/?api=http://<host>:8090
```

## Why it is served over http

Opening `ui/index.html` with `file://` does **not** work. A `file://` page sends
`Origin: null` and the request is refused. SGLang reflects whatever `Origin` it
is given, and vLLM and TabbyAPI allow any origin by default, so any real http
origin is accepted and no proxy is needed — `ui/serve.py` is just `http.server`
with caching disabled.

## What it shows

**Per turn**, under each reply:

| Field | Meaning |
|---|---|
| `TTFT` | Time to the first streamed chunk — the prefill wait |
| `decode` | Output tokens ÷ (total time − TTFT) |
| `total` | Wall time for the request |
| `in` / `out` | `prompt_tokens` / `completion_tokens` from the `usage` field |
| `reasoning` | `reasoning_tokens`, shown only when thinking produced any |
| `cached` | `prompt_tokens_details.cached_tokens`, when the server reports it |
| `prefill` | `prompt_tokens ÷ TTFT` |

The decode counter updates live while the response streams, then is recomputed
from the server's own `usage` numbers when the stream ends.

**Per session**, in the strip under the header: turn count, average TTFT, average
and best decode speed, total output tokens, and the current context size.

## Notes

- **Thinking is off by default in this UI**, deliberately — the model defaults it
  on, and with a small `max_tokens` the whole budget goes to reasoning and
  `content` comes back empty. The toggle is in the header; when a reply comes
  back empty for that reason, the UI says so instead of showing a blank bubble.
- Reasoning output, when present, appears in a collapsible block above the
  answer. It expands while streaming and collapses when the turn finishes.
- The conversation is re-sent in full each turn, which is exactly the pattern
  that benefits from the prefix cache. Watch TTFT drop after the first turn — see
  the cold-versus-cached table in [benchmarks.md](../qwen3.8-flash-next/docs/benchmarks.md).
- State lives in the page only. Reloading clears the conversation.
