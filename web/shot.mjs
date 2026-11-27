// Headless Chromium screenshot of the web demo with WebGPU, no npm packages:
// starts Chromium with remote debugging, drives it over the DevTools
// protocol (Node's built-in WebSocket), prints the page's console and
// WebGPU adapter info, and saves a PNG.
//   node web/shot.mjs [url] [out.png] [seconds] [width] [height]
// Needs `chromium` on PATH and the page served (sh web/serve.sh).
import { spawn } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const url = process.argv[2] ?? "http://localhost:8080/";
const out = process.argv[3] ?? "web-shot.png";
const seconds = Number(process.argv[4] ?? 8);
const width = Number(process.argv[5] ?? 1280);
const height = Number(process.argv[6] ?? 720);
const port = 9300 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), "fx-web-chrome-"));

const chrome = spawn(process.env.CHROMIUM ?? "chromium", [
  ...(process.env.DISPLAY && !process.env.FX_SHOT_HEADLESS ? [] : ["--headless=new"]),
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  `--window-size=${width},${height}`,
  "--enable-unsafe-webgpu",
  "--enable-features=Vulkan,WebGPU",
  "--use-angle=vulkan",
  "--ignore-gpu-blocklist",
  "--no-first-run",
  "--no-default-browser-check",
  "about:blank",
], { stdio: ["ignore", "ignore", "pipe"] });
let chromeErr = "";
chrome.stderr.on("data", (d) => { chromeErr += d; });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let target;
for (let i = 0; i < 100 && !target; i++) {
  await sleep(200);
  try {
    const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
    target = list.find((t) => t.type === "page");
  } catch {}
}
if (!target) {
  console.error("chromium did not start\n" + chromeErr.slice(-2000));
  process.exit(1);
}

const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
let id = 0;
const pending = new Map();
ws.onmessage = (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  } else if (m.method === "Runtime.consoleAPICalled") {
    console.log(`[console.${m.params.type}]`, m.params.args.map((a) => a.value ?? a.description ?? "").join(" "));
  } else if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    console.log("[exception]", d.exception?.description ?? d.text);
  } else if (m.method === "Log.entryAdded") {
    console.log(`[log.${m.params.entry.level}]`, m.params.entry.text);
  }
};
const send = (method, params = {}) => new Promise((r) => {
  const i = ++id;
  pending.set(i, r);
  ws.send(JSON.stringify({ id: i, method, params }));
});

await send("Runtime.enable");
await send("Log.enable");
await send("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
await send("Page.enable");
// Warm up the GPU process first: an adapter request racing its start-up fails.
await send("Runtime.evaluate", { awaitPromise: true, expression: "navigator.gpu?.requestAdapter().then((a) => !!a)" });
await send("Page.navigate", { url });
await sleep(1500);
const info = await send("Runtime.evaluate", {
  awaitPromise: true,
  returnByValue: true,
  expression: `(async () => {
    if (!navigator.gpu) return "navigator.gpu missing";
    const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!a) return "no WebGPU adapter";
    const i = a.info ?? {};
    return JSON.stringify({ vendor: i.vendor, arch: i.architecture, desc: i.description,
      maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage,
      maxStorageBuffersInVertexStage: a.limits.maxStorageBuffersInVertexStage,
      maxComputeWorkgroupSizeX: a.limits.maxComputeWorkgroupSizeX,
      features: [...a.features] });
  })()`,
});
console.log("[webgpu]", info.result?.result?.value);
await sleep(seconds * 1000);
const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.result.data, "base64"));
console.log("saved", out);
ws.close();
chrome.kill();
await sleep(300);
rmSync(profile, { recursive: true, force: true });
process.exit(0);
