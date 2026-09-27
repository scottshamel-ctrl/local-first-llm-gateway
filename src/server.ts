import { createGateway, curlHttp, fetchHttp } from "./index";

const env = process.env;
const trimSlash = (s?: string) => (s?.trim() ? s.trim().replace(/\/+$/, "") : null);

const handle = createGateway(
  {
    localUrl: trimSlash(env.LOCAL_OLLAMA_URL),
    localModel: env.LOCAL_MODEL || "llama3.1:8b",
    localTimeoutMs: Number(env.LOCAL_TIMEOUT_MS || 120_000),
    healthTimeoutMs: Number(env.HEALTH_TIMEOUT_MS || 4_000),
    numCtx: Number(env.LOCAL_NUM_CTX || 4096),
    fallbackUrl: trimSlash(env.FALLBACK_BASE_URL),
    fallbackApiKey: env.FALLBACK_API_KEY,
    fallbackModel: env.FALLBACK_MODEL || "google/gemini-2.5-flash",
    fallbackTimeoutMs: Number(env.FALLBACK_TIMEOUT_MS || 120_000),
  },
  env.TAILSCALE_SOCKS5 ? curlHttp(env.TAILSCALE_SOCKS5) : fetchHttp,
);

const server = Bun.serve({
  hostname: env.HOST || "127.0.0.1",
  port: Number(env.PORT || 8643),
  fetch: handle,
});

console.log(`local-first-llm-gateway listening on http://${server.hostname}:${server.port}/v1`);
