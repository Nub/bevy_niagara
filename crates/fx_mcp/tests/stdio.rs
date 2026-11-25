//! Drives the fx_mcp server over stdio the way Claude Code does: initialize,
//! list tools, validate a good and a bad file, fetch a module doc, render a
//! preview (needs the GPU; builds fxpreview on first use).

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};

struct Server {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next: u64,
}

impl Server {
    fn start() -> Server {
        let mut child = Command::new(env!("CARGO_BIN_EXE_fx_mcp"))
            .current_dir(root())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("start fx_mcp");
        let stdin = child.stdin.take().unwrap();
        let stdout = BufReader::new(child.stdout.take().unwrap());
        Server {
            child,
            stdin,
            stdout,
            next: 1,
        }
    }

    fn send(&mut self, msg: &Value) {
        writeln!(self.stdin, "{msg}").unwrap();
        self.stdin.flush().unwrap();
    }

    fn request(&mut self, method: &str, params: Value) -> Value {
        let id = self.next;
        self.next += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        let mut line = String::new();
        self.stdout.read_line(&mut line).unwrap();
        let reply: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
        assert_eq!(reply["jsonrpc"], "2.0");
        assert_eq!(reply["id"], id, "{reply}");
        reply
    }

    fn call(&mut self, tool: &str, args: Value) -> Value {
        let r = self.request("tools/call", json!({"name": tool, "arguments": args}));
        assert!(r.get("error").is_none(), "{r}");
        r["result"].clone()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn text(result: &Value) -> String {
    result["content"][0]["text"].as_str().unwrap().to_string()
}

#[test]
fn mcp_over_stdio() {
    let mut s = Server::start();

    let init = s.request(
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}),
    );
    assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
    assert_eq!(init["result"]["serverInfo"]["name"], "fx_mcp");
    s.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));

    let tools = s.request("tools/list", json!({}));
    let names: Vec<&str> = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for want in [
        "validate",
        "render_preview",
        "list_modules",
        "module_doc",
        "open_in_viewer",
        "stats",
    ] {
        assert!(names.contains(&want), "{want} missing from {names:?}");
    }

    // A good file.
    let good = s.call("validate", json!({"path": "effects/samples/fountain.fx.ron"}));
    assert_eq!(good["isError"], false);
    assert_eq!(good["structuredContent"]["ok"], true);
    assert!(text(&good).contains(": ok"), "{}", text(&good));

    // A bad file: the error names the line and suggests the fix.
    let bad = s.call("validate", json!({"path": "effects/invalid/unknown_module.fx.ron"}));
    assert_eq!(bad["structuredContent"]["ok"], false);
    let t = text(&bad);
    assert!(t.contains(":7:") && t.contains("unknown module \"Spawn Rat\""), "{t}");
    assert!(t.contains("did you mean \"Spawn Rate\""), "{t}");
    assert_eq!(bad["structuredContent"]["diagnostics"][0]["line"], 7);

    // Module docs, generated from the module table.
    let doc = s.call("module_doc", json!({"name": "gravity force"}));
    let t = text(&doc);
    assert!(t.starts_with("# Gravity Force"), "{t}");
    assert!(t.contains("| Gravity | Vec3 | `Vec3((0.0, -9.81, 0.0))`"), "{t}");
    assert!(t.contains("Niagara equivalent"), "{t}");
    let unknown = s.call("module_doc", json!({"name": "Gravity Forse"}));
    assert_eq!(unknown["isError"], true);
    let list = s.call("list_modules", json!({}));
    assert!(text(&list).contains("\"Spawn Burst Instantaneous\""));

    // Stats from the CPU reference sim.
    let st = s.call("stats", json!({"path": "effects/samples/burst.fx.ron", "seconds": 1}));
    assert_eq!(st["isError"], false, "{}", text(&st));
    assert!(text(&st).contains("peak"), "{}", text(&st));

    // A preview with the real renderer.
    let out = root().join("target/fx_mcp_test");
    let _ = std::fs::remove_dir_all(&out);
    let pv = s.call(
        "render_preview",
        json!({"path": "effects/samples/fountain.fx.ron", "times": [0.5, 1.0], "size": [160, 120], "out": out}),
    );
    assert_eq!(pv["isError"], false, "{}", text(&pv));
    let strip = PathBuf::from(pv["structuredContent"]["strip"].as_str().unwrap());
    let png = std::fs::read(&strip).expect("strip written");
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let img = &pv["content"][1];
    assert_eq!(img["type"], "image");
    assert_eq!(img["mimeType"], "image/png");
    assert!(img["data"].as_str().unwrap().len() > 100);

    // A bad file is not rendered.
    let pv = s.call(
        "render_preview",
        json!({"path": "effects/invalid/unknown_module.fx.ron"}),
    );
    assert_eq!(pv["isError"], true);

    // Unknown method.
    let r = s.request("nope", json!({}));
    assert_eq!(r["error"]["code"], -32601);
}
