import { describe, expect, test } from "bun:test";
import {
  createGateway,
  curlArgs,
  MIN_LOCAL_TIMEOUT_MS,
  pickModels,
  toOllamaChat,
  type GatewayConfig,
  type Http,
  type HttpResult,
} from "./index";

const cfg: GatewayConfig = {
  localUrl: "http://100.64.0.7:11434",
  localModel: "llama3.1:8b",
  localTimeoutMs: 120_000,
  healthTimeoutMs: 4_000,
  numCtx: 4096,
  fallbackUrl: "https://cloud.example/api/v1",
  fallbackApiKey: "test-key",
  fallbackModel: "vendor/cheap-model",
  fallbackTimeoutMs: 60_000,
};

type Call = { url: string; body?: any; headers?: Record<string, string>; timeoutMs: number };

const ok = (data: unknown): HttpResult => ({ ok: true, status: 200, text: JSON.stringify(data) });
const localReply = ok({ message: { content: "from local" }, done_reason: "stop", prompt_eval_count: 12, eval_count: 5 });
const cloudReply = ok({ id: "c1", object: "chat.completion", choices: [{ index: 0, message: { role: "assistant", content: "from cloud" } }] });

/** Routes by path suffix; a value of Error makes that endpoint throw like a dead socket. */
function fakeHttp(routes: Record<string, HttpResult | Error>) {
  const calls: Call[] = [];
  const http: Http = async (url, init) => {
    calls.push({ url, ...init });
    const key = Object.keys(routes).find((k) => url.endsWith(k));
    if (!key) throw new Error(`unexpected request ${url}`);
    const r = routes[key];
    if (r instanceof Error) throw r;
    return r;
  };
  return { http, calls, hit: (suffix: string) => calls.filter((c) => c.url.endsWith(suffix)) };
}

const chat = (body: unknown) =>
  new Request("http://gw/v1/chat/completions", { method: "POST", body: typeof body === "string" ? body : JSON.stringify(body) });

const messages = [{ role: "user", content: "hi" }];

describe("routing", () => {
  test("healthy local machine answers, cloud is never contacted", async () => {
    const f = fakeHttp({ "/api/tags": ok({}), "/api/chat": localReply, "/chat/completions": cloudReply });
    const res = await createGateway(cfg, f.http)(chat({ messages }));
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.choices[0].message.content).toBe("from local");
    expect(data.usage).toEqual({ prompt_tokens: 12, completion_tokens: 5, total_tokens: 17 });
    expect(data.gateway.route).toBe("local");
    expect(f.hit("/chat/completions")).toHaveLength(0);
  });

  test("health check uses the short timeout, not the generation timeout", async () => {
    const f = fakeHttp({ "/api/tags": ok({}), "/api/chat": localReply });
    await createGateway(cfg, f.http)(chat({ messages }));
    expect(f.hit("/api/tags")[0].timeoutMs).toBe(cfg.healthTimeoutMs);
  });

  test("unreachable local machine falls back without attempting generation", async () => {
    const f = fakeHttp({ "/api/tags": new Error("connect ETIMEDOUT"), "/chat/completions": cloudReply });
    const data = await (await createGateway(cfg, f.http)(chat({ messages }))).json();
    expect(data.choices[0].message.content).toBe("from cloud");
    expect(data.gateway.route).toBe("fallback");
    expect(f.hit("/api/chat")).toHaveLength(0);
  });

  test("local generation failure falls back", async () => {
    const f = fakeHttp({
      "/api/tags": ok({}),
      "/api/chat": { ok: false, status: 500, text: "model runner crashed" },
      "/chat/completions": cloudReply,
    });
    const data = await (await createGateway(cfg, f.http)(chat({ messages }))).json();
    expect(data.gateway.route).toBe("fallback");
  });

  test("no local URL configured goes straight to fallback", async () => {
    const f = fakeHttp({ "/chat/completions": cloudReply });
    const data = await (await createGateway({ ...cfg, localUrl: null }, f.http)(chat({ messages }))).json();
    expect(data.gateway.route).toBe("fallback");
    expect(f.calls).toHaveLength(1);
  });
});

describe("local_only", () => {
  test("fails closed with 503 and never sends the prompt to the cloud", async () => {
    const f = fakeHttp({ "/api/tags": new Error("down"), "/chat/completions": cloudReply });
    const res = await createGateway(cfg, f.http)(chat({ messages, local_only: true }));
    expect(res.status).toBe(503);
    expect((await res.json()).error.type).toBe("local_only_unavailable");
    expect(f.hit("/chat/completions")).toHaveLength(0);
  });

  test("is still served locally when the machine is up", async () => {
    const f = fakeHttp({ "/api/tags": ok({}), "/api/chat": localReply });
    const res = await createGateway(cfg, f.http)(chat({ messages, local_only: true }));
    expect(res.status).toBe(200);
  });
});

describe("errors", () => {
  test("both routes failing returns 502 naming both causes", async () => {
    const f = fakeHttp({ "/api/tags": new Error("down"), "/chat/completions": { ok: false, status: 429, text: "rate limited" } });
    const res = await createGateway(cfg, f.http)(chat({ messages }));
    const msg = (await res.json()).error.message;
    expect(res.status).toBe(502);
    expect(msg).toContain("local: unreachable");
    expect(msg).toContain("fallback 429: rate limited");
  });

  test("no fallback configured returns 503 instead of hanging or 500", async () => {
    const f = fakeHttp({ "/api/tags": new Error("down") });
    const res = await createGateway({ ...cfg, fallbackUrl: null }, f.http)(chat({ messages }));
    expect(res.status).toBe(503);
    expect((await res.json()).error.type).toBe("no_route");
  });

  test("invalid JSON is a 400", async () => {
    const res = await createGateway(cfg, fakeHttp({}).http)(chat("{not json"));
    expect(res.status).toBe(400);
  });

  test("stream: true is rejected explicitly rather than answered in the wrong shape", async () => {
    const f = fakeHttp({});
    const res = await createGateway(cfg, f.http)(chat({ messages, stream: true }));
    expect(res.status).toBe(400);
    expect(f.calls).toHaveLength(0);
  });

  test("unknown path is a 404", async () => {
    const res = await createGateway(cfg, fakeHttp({}).http)(new Request("http://gw/v1/embeddings"));
    expect(res.status).toBe(404);
  });
});

describe("request translation", () => {
  test("num_gpu: 0 survives translation (a truthiness check would drop it)", () => {
    expect(toOllamaChat({ messages, num_gpu: 0 }, "m:1", 4096).options.num_gpu).toBe(0);
  });

  test("unset tuning options are omitted, not sent as undefined overrides", () => {
    const opts = toOllamaChat({ messages }, "m:1", 4096).options;
    expect("num_gpu" in opts).toBe(false);
    expect("num_thread" in opts).toBe(false);
    expect(opts.num_ctx).toBe(4096);
  });

  test("max_tokens maps to num_predict and json_object maps to format: json", () => {
    const out = toOllamaChat({ messages, max_tokens: 300, response_format: { type: "json_object" } }, "m:1", 4096);
    expect(out.options.num_predict).toBe(300);
    expect(out.format).toBe("json");
  });

  test("local-only fields are stripped before the cloud request", async () => {
    const f = fakeHttp({ "/api/tags": new Error("down"), "/chat/completions": cloudReply });
    await createGateway(cfg, f.http)(chat({ messages, num_gpu: 0, keep_alive: "5m", local_timeout_ms: 20_000, temperature: 0.2 }));
    const sent = f.hit("/chat/completions")[0];
    expect(sent.body).toEqual({ messages, temperature: 0.2, model: "vendor/cheap-model", stream: false });
    expect(sent.headers).toEqual({ authorization: "Bearer test-key" });
  });
});

describe("model selection", () => {
  test("an Ollama tag stays local; a provider slug goes to the cloud", () => {
    expect(pickModels("qwen2.5:7b", cfg)).toEqual({ local: "qwen2.5:7b", fallback: "vendor/cheap-model" });
    expect(pickModels("openai/gpt-4o-mini", cfg)).toEqual({ local: "llama3.1:8b", fallback: "openai/gpt-4o-mini" });
  });

  test("missing or 'default' model uses both configured defaults", () => {
    expect(pickModels(undefined, cfg)).toEqual({ local: "llama3.1:8b", fallback: "vendor/cheap-model" });
    expect(pickModels("default", cfg)).toEqual({ local: "llama3.1:8b", fallback: "vendor/cheap-model" });
  });
});

describe("local timeout clamp", () => {
  const timeoutFor = async (local_timeout_ms?: number) => {
    const f = fakeHttp({ "/api/tags": ok({}), "/api/chat": localReply });
    await createGateway(cfg, f.http)(chat({ messages, local_timeout_ms }));
    return f.hit("/api/chat")[0].timeoutMs;
  };

  test("defaults to the configured ceiling", async () => expect(await timeoutFor()).toBe(120_000));
  test("a shorter request deadline is honored", async () => expect(await timeoutFor(30_000)).toBe(30_000));
  test("cannot go below the floor", async () => expect(await timeoutFor(1_000)).toBe(MIN_LOCAL_TIMEOUT_MS));
  test("cannot exceed the configured ceiling", async () => expect(await timeoutFor(10 * 60_000)).toBe(120_000));
});

describe("/health", () => {
  const health = async (routes: Record<string, HttpResult | Error>) =>
    (await createGateway(cfg, fakeHttp(routes).http)(new Request("http://gw/health"))).json();

  test("reachable and idle when no model is loaded", async () => {
    const d = await health({ "/api/tags": ok({}), "/api/ps": ok({ models: [] }) });
    expect(d.local).toMatchObject({ reachable: true, idle: true });
  });

  test("reachable but busy when a model is resident", async () => {
    const d = await health({ "/api/tags": ok({}), "/api/ps": ok({ models: [{ name: "llama3.1:8b" }] }) });
    expect(d.local).toMatchObject({ reachable: true, idle: false });
  });

  test("never reports idle for a machine it cannot reach", async () => {
    const d = await health({ "/api/tags": new Error("down"), "/api/ps": new Error("down") });
    expect(d.local).toMatchObject({ reachable: false, idle: false });
  });

  test("does not leak the fallback API key", async () => {
    const d = await health({ "/api/tags": ok({}), "/api/ps": ok({ models: [] }) });
    expect(JSON.stringify(d)).not.toContain("test-key");
  });
});

describe("curlArgs", () => {
  test("tailnet addresses go through the SOCKS5 proxy", () => {
    const args = curlArgs("http://100.64.0.7:11434/api/tags", undefined, 4000, {}, "127.0.0.1:1055");
    expect(args).toContain("--socks5-hostname");
    expect(args).toContain("127.0.0.1:1055");
  });

  test("public addresses connect directly even when a proxy is set", () => {
    const args = curlArgs("https://cloud.example/api/v1/chat/completions", {}, 4000, {}, "127.0.0.1:1055");
    expect(args).not.toContain("--socks5-hostname");
  });

  test("timeout rounds up to whole seconds so a 4.5 s budget is not cut to 4", () => {
    const args = curlArgs("http://x/", undefined, 4500);
    expect(args[args.indexOf("--max-time") + 1]).toBe("5");
  });
});
