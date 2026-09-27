# local-first-llm-gateway

An OpenAI-compatible endpoint that uses **your own GPU when it's up** and **a cloud model when it isn't**.

Point any OpenAI client at it. Requests go to Ollama on your home machine, over Tailscale or a LAN. If that machine is off, asleep, or crashing, the same request goes to any OpenAI-compatible provider (OpenRouter, OpenAI, Together, and so on). Prompts marked `local_only` never leave your hardware.

Zero dependencies. About 300 lines. 29 tests, no network needed to run them.

```bash
LOCAL_OLLAMA_URL=http://100.64.0.7:11434 \
FALLBACK_BASE_URL=https://openrouter.ai/api/v1 \
FALLBACK_API_KEY=sk-... \
bun src/server.ts
# -> http://127.0.0.1:8643/v1
```

## Why this exists

My automation runs on a cloud server, but I have a GPU sitting in a desktop at home. Local inference is free and private, but a desktop is not a data center. It sleeps, reboots for updates, and sometimes Ollama wedges. Every pipeline that talked to it directly had to handle all of that itself.

This gateway puts that handling in one place. Callers see one stable OpenAI endpoint and never need to know where the answer came from, unless they ask (every response carries `gateway.route`).

## Decisions worth looking at

**Health-check before generating, with a short timeout.** A dead machine costs 4 seconds (`HEALTH_TIMEOUT_MS`), not a 120-second generation timeout. The gateway probes `/api/tags` first and only sends the real request if the machine answers.

**`local_only` fails closed.** Some prompts contain data that should never reach a third party. With `"local_only": true`, an unavailable local machine returns `503` and the cloud is never contacted. A test asserts the fallback endpoint receives zero requests.

**`/health` reports *idle*, not just *reachable*.** It checks Ollama's `/api/ps` for a loaded model. A batch job can wait for `idle: true` before starting heavy work on a machine someone is actively using. A machine that can't be reached is never reported as idle.

**`num_gpu: 0` survives translation.** Ollama's `num_gpu: 0` keeps a run entirely off the GPU that draws your display. An easy bug is `...(body.num_gpu ? { num_gpu } : {})`, a truthiness check that silently drops exactly that value. The code uses `!== undefined`, and a test pins it.

**SOCKS5 for Tailscale userspace networking.** In a container, Tailscale often runs in userspace mode, so `100.x.y.z` addresses aren't routable. They're only reachable through tailscaled's local SOCKS5 proxy, and `fetch()` can't speak SOCKS. Set `TAILSCALE_SOCKS5=127.0.0.1:1055` and the gateway switches to a curl transport that sends only tailnet addresses through the proxy. Everything else connects directly.

**The request timeout is clamped.** Callers can shorten the local deadline per request (`local_timeout_ms`), but never below 15 seconds, which would turn every call into a fallback, and never above the configured ceiling.

**Streaming is rejected, not faked.** `stream: true` returns `400`. Answering a streaming client with a non-streamed body breaks it in a confusing way, so the gateway gives a clear error.

**All I/O goes through one injected function.** `createGateway(config, http)` takes the transport as a parameter. The tests pass a fake that records every call, so they check both what came back and which endpoints were contacted.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LOCAL_OLLAMA_URL` | *(none)* | Ollama base URL. Unset means cloud only. |
| `LOCAL_MODEL` | `llama3.1:8b` | Local model when the request doesn't name an Ollama tag |
| `LOCAL_TIMEOUT_MS` | `120000` | Ceiling for one local generation |
| `LOCAL_NUM_CTX` | `4096` | Default context window |
| `HEALTH_TIMEOUT_MS` | `4000` | Reachability probe timeout |
| `FALLBACK_BASE_URL` | *(none)* | OpenAI-compatible base URL. Unset means local only. |
| `FALLBACK_API_KEY` | *(none)* | Sent as a bearer token. Never logged or returned. |
| `FALLBACK_MODEL` | `google/gemini-2.5-flash` | Cloud model when the request doesn't name a `vendor/model` slug |
| `TAILSCALE_SOCKS5` | *(none)* | e.g. `127.0.0.1:1055`, to use the SOCKS-aware curl transport |
| `HOST` / `PORT` | `127.0.0.1` / `8643` | Listen address. It binds to localhost by default. |

**Model routing:** a request for `qwen2.5:7b` (an Ollama tag) runs locally and uses `FALLBACK_MODEL` if it falls back. A request for `openai/gpt-4o-mini` (a provider slug) uses `LOCAL_MODEL` locally and that exact slug in the cloud.

## Endpoints

- `POST /v1/chat/completions`: OpenAI chat format, plus the optional Ollama options `num_ctx`, `num_gpu`, `num_thread`, `num_batch`, `top_k`, `repeat_penalty`, and `keep_alive`, and the gateway options `local_only` and `local_timeout_ms`. These extras are stripped before a cloud request.
- `GET /v1/models`
- `GET /health`: `{ local: { reachable, idle }, fallback }`

## Test

```bash
bun install
bun test           # 29 pass
bun run typecheck
```

## Related

[local-llm-guard](https://github.com/scottshamel-ctrl/local-llm-guard) is the client-side half. It paces calls into a local model so a batch job doesn't make the machine unusable.

## License

MIT
