//! `fx_mcp`: a small MCP server (JSON-RPC 2.0 over stdio, one message per
//! line) that lets Claude validate effects, render previews, look up modules
//! and drive the viewer. Local only: no network access.
//!
//! Tools: `validate`, `render_preview`, `list_modules`, `module_doc`,
//! `open_in_viewer`, `stats`. Module docs come from `fx_core::docs`, which
//! is generated from the Rust module definitions.
//!
//! `render_preview` and `open_in_viewer` run the `fxpreview` and `fxviewer`
//! binaries. They are looked up next to this executable (same cargo profile)
//! and built on first use with `cargo build` from the workspace (set
//! `FX_MCP_NO_BUILD=1` to skip that, or `FXPREVIEW_BIN` / `FXVIEWER_BIN` to
//! point at binaries). Run the server inside the Bevy fork's nix dev shell so
//! cargo, ffmpeg and the Vulkan loader are found.

mod tools;

use std::io::{BufRead, Write};

use serde_json::{Value, json};

const SUPPORTED_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

fn main() {
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(Value::Array(batch)) => {
                let out: Vec<Value> = batch.iter().filter_map(handle).collect();
                (!out.is_empty()).then(|| Value::Array(out))
            }
            Ok(msg) => handle(&msg),
            Err(e) => Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
        };
        if let Some(r) = reply {
            let _ = writeln!(stdout, "{r}");
            let _ = stdout.flush();
        }
    }
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
}

/// Handles one request or notification; notifications get no reply.
fn handle(msg: &Value) -> Option<Value> {
    let method = msg.get("method").and_then(Value::as_str);
    let id = msg.get("id").cloned();
    let Some(method) = method else {
        // A response from the client (we send no requests): ignore.
        return None;
    };
    let Some(id) = id else {
        // Notification (initialized, cancelled, ...).
        return None;
    };
    let params = msg.get("params").cloned().unwrap_or(json!({}));
    let result = match method {
        "initialize" => {
            let asked = params
                .get("protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(SUPPORTED_VERSIONS[0]);
            let version = if SUPPORTED_VERSIONS.contains(&asked) {
                asked
            } else {
                SUPPORTED_VERSIONS[0]
            };
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {"tools": {"listChanged": false}},
                "serverInfo": {"name": "fx_mcp", "version": env!("CARGO_PKG_VERSION")},
                "instructions": "Tools for authoring animatey particle effects (.fx.ron, Niagara-style). \
                    Validate after every edit, look modules up with list_modules/module_doc, and check the look \
                    with render_preview (returns a PNG frame strip) or open_in_viewer (live window that hot-reloads)."
            }))
        }
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tools::list()})),
        "tools/call" => {
            let name = params.get("name").and_then(Value::as_str).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            match tools::call(name, &args) {
                Some(r) => Ok(r),
                None => Err((-32602, format!("unknown tool \"{name}\""))),
            }
        }
        _ => Err((-32601, format!("method not found: {method}"))),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err((code, m)) => error(id, code, &m),
    })
}
