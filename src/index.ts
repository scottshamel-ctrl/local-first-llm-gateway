import { spawn } from "node:child_process";

export type ChatMessage = { role: string; content: unknown };

export type ChatRequest = {
  model?: string;
  messages?: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  repeat_penalty?: number;
  num_ctx?: number;
  num_batch?: number;
  /** GPU layers. 0 keeps the run entirely off the GPU that is drawing your display. */
  num_gpu?: number;
  /** CPU thread cap, so a CPU-only run still leaves the machine responsive. */
  num_thread?: number;
  keep_alive?: string | number;
  response_format?: { type?: string };
  stream?: boolean;
  /** Per-request local deadline, clamped to [MIN_LOCAL_TIMEOUT_MS, localTimeoutMs]. */
  local_timeout_ms?: number;
  /** Fail with 503 instead of sending this prompt to the cloud. */
  local_only?: boolean;
};

export type HttpResult = { ok: boolean; status: number; text: string };

/** All network I/O goes through this, so the gateway is testable without a network. */
export type Http = (
  url: string,
  init: { body?: unknown; headers?: Record<string, string>; timeoutMs: number },
) => Promise<HttpResult>;

export type GatewayConfig = {
  /** Ollama base URL, e.g. http://100.64.0.7:11434. Null means "no local machine configured". */
  localUrl: string | null;
  localModel: string;
  localTimeoutMs: number;
  /** Short on purpose: a dead machine should cost seconds, not a full generation timeout. */
  healthTimeoutMs: number;
  numCtx: number;
  /** Any OpenAI-compatible base URL (OpenRouter, OpenAI, Together...). Null disables fallback. */
  fallbackUrl: string | null;
  fallbackApiKey?: string;
  fallbackModel: string;
  fallbackTimeoutMs: number;
};

// ponytail: a 15 s floor stops a caller from setting a deadline shorter than a warm model's
// first-token latency, which would turn every local call into a guaranteed fallback.
export const MIN_LOCAL_TIMEOUT_MS = 15_000;

const OLLAMA_TAG = /^[a-z0-9._-]+:[a-z0-9._-]+$/i;

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json" } });
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** `llama3.1:8b` is a local tag; `google/gemini-2.5-flash` is a provider slug. */
export function pickModels(requested: string | undefined, cfg: GatewayConfig) {
  const r = requested && requested !== "default" ? requested : undefined;
  return {
    local: r && OLLAMA_TAG.test(r) ? r : cfg.localModel,
    fallback: r && r.includes("/") ? r : cfg.fallbackModel,
  };
}

export function toOllamaChat(body: ChatRequest, model: string, numCtx: number) {
  const options: Record<string, unknown> = {
    temperature: body.temperature,
    num_predict: body.max_tokens,
    top_p: body.top_p,
    top_k: body.top_k,
    repeat_penalty: body.repeat_penalty,
    num_ctx: body.num_ctx ?? numCtx,
  };
  // `!== undefined`, not truthiness: num_gpu: 0 is exactly the value that moves a run off the
  // display GPU, and a truthy check silently drops it.
  for (const key of ["num_batch", "num_gpu", "num_thread"] as const) {
    if (body[key] !== undefined) options[key] = body[key];
  }
  return {
    model,
    messages: body.messages ?? [],
    stream: false,
    ...(body.response_format?.type === "json_object" ? { format: "json" } : {}),
    ...(body.keep_alive !== undefined ? { keep_alive: body.keep_alive } : {}),
    options,
  };
}

export function createGateway(cfg: GatewayConfig, http: Http) {
  async function localReachable(): Promise<boolean> {
    if (!cfg.localUrl) return false;
    try {
      return (await http(`${cfg.localUrl}/api/tags`, { timeoutMs: cfg.healthTimeoutMs })).ok;
    } catch {
      return false;
    }
  }

  /** Idle = reachable and no model loaded, i.e. safe to start a heavy job on someone's desktop. */
  async function localIdle(): Promise<boolean> {
    if (!cfg.localUrl) return false;
    try {
      const r = await http(`${cfg.localUrl}/api/ps`, { timeoutMs: cfg.healthTimeoutMs });
      if (!r.ok) return false;
      const models = (JSON.parse(r.text) as { models?: unknown[] }).models;
      return Array.isArray(models) && models.length === 0;
    } catch {
      return false;
    }
  }

  async function callLocal(body: ChatRequest): Promise<Response> {
    const model = pickModels(body.model, cfg).local;
    const timeoutMs = Math.max(
      MIN_LOCAL_TIMEOUT_MS,
      Math.min(cfg.localTimeoutMs, body.local_timeout_ms ?? cfg.localTimeoutMs),
    );
    const started = Date.now();
    const r = await http(`${cfg.localUrl}/api/chat`, { body: toOllamaChat(body, model, cfg.numCtx), timeoutMs });
    if (!r.ok) throw new Error(`local ${r.status}: ${r.text.slice(0, 400)}`);
    const data = JSON.parse(r.text);
    const prompt = data?.prompt_eval_count ?? 0;
    const completion = data?.eval_count ?? 0;
    return json({
      id: `chatcmpl-${started}`,
      object: "chat.completion",
      created: Math.floor(started / 1000),
      model,
      choices: [{
        index: 0,
        message: { role: "assistant", content: String(data?.message?.content ?? "") },
        finish_reason: data?.done_reason || "stop",
      }],
      usage: { prompt_tokens: prompt, completion_tokens: completion, total_tokens: prompt + completion },
      gateway: { route: "local", model, latencyMs: Date.now() - started },
    });
  }

  async function callFallback(body: ChatRequest): Promise<Response> {
    const model = pickModels(body.model, cfg).fallback;
    const started = Date.now();
    const { local_only, local_timeout_ms, num_ctx, num_batch, num_gpu, num_thread, keep_alive, top_k, repeat_penalty, ...openai } = body;
    const r = await http(`${cfg.fallbackUrl}/chat/completions`, {
      body: { ...openai, model, stream: false },
      headers: cfg.fallbackApiKey ? { authorization: `Bearer ${cfg.fallbackApiKey}` } : {},
      timeoutMs: cfg.fallbackTimeoutMs,
    });
    if (!r.ok) throw new Error(`fallback ${r.status}: ${r.text.slice(0, 400)}`);
    const data = JSON.parse(r.text);
    return json({ ...data, gateway: { route: "fallback", model, latencyMs: Date.now() - started } });
  }

  async function chatCompletions(req: Request): Promise<Response> {
    let body: ChatRequest;
    try {
      body = await req.json();
    } catch {
      return json({ error: { message: "Invalid JSON body", type: "invalid_request_error" } }, 400);
    }
    if (body.stream) {
      return json({ error: { message: "stream: true is not supported", type: "invalid_request_error" } }, 400);
    }

    const errors: string[] = [];
    if (!cfg.localUrl) {
      errors.push("local: not configured");
    } else if (!(await localReachable())) {
      errors.push(`local: unreachable at ${cfg.localUrl}`);
    } else {
      try {
        return await callLocal(body);
      } catch (e) {
        errors.push(`local: ${errorMessage(e)}`);
      }
    }

    if (body.local_only === true) {
      return json({ error: { message: errors.join(" | "), type: "local_only_unavailable" } }, 503);
    }
    if (!cfg.fallbackUrl) {
      return json({ error: { message: [...errors, "fallback: not configured"].join(" | "), type: "no_route" } }, 503);
    }
    try {
      return await callFallback(body);
    } catch (e) {
      errors.push(`fallback: ${errorMessage(e)}`);
      return json({ error: { message: errors.join(" | "), type: "gateway_error" } }, 502);
    }
  }

  return async function handle(req: Request): Promise<Response> {
    const { pathname } = new URL(req.url);
    if (req.method === "GET" && pathname === "/health") {
      const [reachable, idle] = await Promise.all([localReachable(), localIdle()]);
      return json({
        ok: true,
        local: cfg.localUrl ? { url: cfg.localUrl, model: cfg.localModel, reachable, idle } : null,
        fallback: cfg.fallbackUrl ? { url: cfg.fallbackUrl, model: cfg.fallbackModel } : null,
      });
    }
    if (req.method === "POST" && pathname === "/v1/chat/completions") return chatCompletions(req);
    if (req.method === "GET" && pathname === "/v1/models") {
      const data = [{ id: cfg.localModel, object: "model" }];
      if (cfg.fallbackUrl) data.push({ id: cfg.fallbackModel, object: "model" });
      return json({ object: "list", data });
    }
    return json({ error: { message: "Not found" } }, 404);
  };
}

/** Plain fetch. Use this unless you need SOCKS (see curlHttp). */
export const fetchHttp: Http = async (url, { body, headers, timeoutMs }) => {
  const res = await fetch(url, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? headers : { "content-type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  return { ok: res.ok, status: res.status, text: await res.text() };
};

const TAILSCALE_IP = /^100\.\d{1,3}\.\d{1,3}\.\d{1,3}$/;

/**
 * curl arguments for one request. Tailscale addresses (100.64.0.0/10) go through the SOCKS5
 * proxy when one is given; everything else connects directly.
 */
export function curlArgs(url: string, body: unknown, timeoutMs: number, headers: Record<string, string> = {}, socks5?: string): string[] {
  const args = ["-sS", "--max-time", String(Math.ceil(timeoutMs / 1000))];
  if (socks5 && TAILSCALE_IP.test(new URL(url).hostname)) args.push("--socks5-hostname", socks5);
  for (const [k, v] of Object.entries(headers)) args.push("-H", `${k}: ${v}`);
  if (body !== undefined) {
    args.push("-X", "POST", "-H", "content-type: application/json", "--data-binary", JSON.stringify(body));
  }
  args.push("-w", "\n%{http_code}", url);
  return args;
}

/**
 * For containers running Tailscale in userspace-networking mode: tailnet IPs are not routable
 * from the container, only through tailscaled's local SOCKS5 proxy, and fetch() cannot speak SOCKS.
 */
export function curlHttp(socks5?: string): Http {
  return (url, { body, headers, timeoutMs }) =>
    new Promise((resolve, reject) => {
      const child = spawn("curl", curlArgs(url, body, timeoutMs, headers, socks5), { stdio: ["ignore", "pipe", "pipe"] });
      let out = "";
      let err = "";
      child.stdout.on("data", (c) => (out += c));
      child.stderr.on("data", (c) => (err += c));
      child.on("error", reject);
      child.on("close", (code) => {
        if (code !== 0) return reject(new Error((err || `curl exited with ${code}`).trim()));
        const split = out.lastIndexOf("\n");
        const status = Number(out.slice(split + 1).trim());
        resolve({ ok: status >= 200 && status < 300, status, text: out.slice(0, split) });
      });
    });
}
