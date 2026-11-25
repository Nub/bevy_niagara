//! The tools and their implementations.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

use base64::Engine;
use fx_core::sim::Simulation;
use fx_core::{check_file, docs, modules};
use serde_json::{Value, json};

pub fn list() -> Value {
    let path = json!({"type": "string", "description": "Path to a .fx.ron effect file (absolute, or relative to the server's working directory)"});
    let vec3 = |d: &str| json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": d});
    json!([
        {
            "name": "validate",
            "description": "Parse, validate and cook an effect file. Returns ok, or every error and warning as file:line:col with a help line saying how to fix it.",
            "inputSchema": {"type": "object", "properties": {"path": path}, "required": ["path"]}
        },
        {
            "name": "render_preview",
            "description": "Render the effect headless with the real GPU renderer and return a PNG frame strip (frames at the given times, left to right) plus file paths. Deterministic: the same file and options give the same pixels. mode \"window\" also opens or reuses the live viewer window.",
            "inputSchema": {"type": "object", "properties": {
                "path": path,
                "times": {"type": "array", "items": {"type": "number"}, "description": "Seconds to capture (default [0, 0.25, 0.5, 1, 1.5, 2, 3])"},
                "duration": {"type": "number", "description": "Seconds simulated (default: last time)"},
                "camera": vec3("Camera position X,Y,Z in metres (default 0,1.6,6.5)"),
                "target": vec3("Point the camera looks at (default 0,1.3,0)"),
                "backdrop": {"type": "string", "enum": ["dark", "grey"], "description": "Default dark"},
                "size": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2, "description": "Frame width and height in pixels (default 480x360)"},
                "seed": {"type": "integer", "description": "Random seed (default 1)"},
                "mp4": {"type": "boolean", "description": "Also write an MP4 of the whole duration at 30 fps (default false)"},
                "mode": {"type": "string", "enum": ["headless", "window"], "description": "Default headless"},
                "inline": {"type": "boolean", "description": "Return the strip as an image in the result (default true)"},
                "out": {"type": "string", "description": "Output directory (default target/fx_preview/<name> in the workspace)"}
            }, "required": ["path"]}
        },
        {
            "name": "list_modules",
            "description": "List every module (name, stages it may be placed in, what it does) and renderer.",
            "inputSchema": {"type": "object", "properties": {}}
        },
        {
            "name": "module_doc",
            "description": "Reference for one module or renderer: inputs with types and defaults, stages, the Niagara equivalent and an example, generated from the Rust module definitions. Name is case-insensitive (e.g. \"Gravity Force\", \"Sprite\").",
            "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        },
        {
            "name": "open_in_viewer",
            "description": "Open the effect in the interactive viewer window (fxviewer), reusing a running viewer if there is one. The viewer hot-reloads the file on every save, so the user can watch edits live.",
            "inputSchema": {"type": "object", "properties": {
                "path": path,
                "backdrop": {"type": "string", "enum": ["dark", "grey", "room"]}
            }, "required": ["path"]}
        },
        {
            "name": "stats",
            "description": "Effect statistics: emitters, capacities, modules per stage, renderers and GPU memory, plus living particle counts over time from the CPU reference simulation (the GPU sim matches it). If the viewer is showing this file, adds its live GPU ms per pass.",
            "inputSchema": {"type": "object", "properties": {
                "path": path,
                "seconds": {"type": "number", "description": "Seconds to simulate (default 3)"}
            }, "required": ["path"]}
        }
    ])
}

fn text(s: impl Into<String>, is_error: bool) -> Value {
    json!({"content": [{"type": "text", "text": s.into()}], "isError": is_error})
}

pub fn call(name: &str, args: &Value) -> Option<Value> {
    let r = match name {
        "validate" => validate(args),
        "render_preview" => render_preview(args),
        "list_modules" => Ok(text(docs::module_list(), false)),
        "module_doc" => {
            let n = args.get("name").and_then(Value::as_str).unwrap_or("");
            Ok(match docs::module_doc(n) {
                Ok(d) => text(d, false),
                Err(e) => text(e, true),
            })
        }
        "open_in_viewer" => open_in_viewer(args),
        "stats" => stats(args),
        _ => return None,
    };
    Some(r.unwrap_or_else(|e| text(e, true)))
}

fn arg_path(args: &Value) -> Result<PathBuf, String> {
    let p = args
        .get("path")
        .and_then(Value::as_str)
        .ok_or("missing \"path\"")?;
    let p = PathBuf::from(p);
    if !p.exists() {
        return Err(format!("{}: no such file", p.display()));
    }
    std::path::absolute(&p).map_err(|e| e.to_string())
}

fn validate(args: &Value) -> Result<Value, String> {
    let path = arg_path(args)?;
    let c = check_file(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut s = String::new();
    for d in &c.diagnostics {
        let _ = writeln!(s, "{d}");
    }
    match &c.cooked {
        Some(sys) => {
            let cap: u32 = sys.emitters.iter().map(|e| e.max_particles).sum();
            let _ = writeln!(
                s,
                "{}: ok (\"{}\", {} emitter(s), {} particles max)",
                path.display(),
                sys.name,
                sys.emitters.len(),
                cap
            );
        }
        None => {
            let n = c.errors().count();
            let _ = writeln!(s, "{}: INVALID, {n} error(s)", path.display());
        }
    }
    // A file with errors is a successful validation, not a tool failure.
    Ok(json!({
        "content": [{"type": "text", "text": s}],
        "structuredContent": {
            "ok": c.ok(),
            "diagnostics": c.diagnostics.iter().map(|d| json!({
                "severity": if d.is_error() { "error" } else { "warning" },
                "line": d.span.line, "col": d.span.col,
                "message": d.message, "help": d.help,
            })).collect::<Vec<_>>()
        },
        "isError": false
    }))
}

// ---------------------------------------------------------------- binaries

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from("."))
}

static BUILT: Mutex<Option<Result<(), String>>> = Mutex::new(None);

/// The `fxpreview` or `fxviewer` binary, built once per session.
fn tool_binary(name: &str) -> Result<PathBuf, String> {
    let env = format!("{}_BIN", name.to_uppercase());
    if let Some(p) = std::env::var_os(&env) {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut dir = exe.parent().unwrap().to_path_buf();
    if dir.ends_with("deps") {
        dir.pop();
    }
    let bin = dir.join(name);
    if std::env::var_os("FX_MCP_NO_BUILD").is_some() {
        return if bin.exists() {
            Ok(bin)
        } else {
            Err(format!("{} not built (FX_MCP_NO_BUILD is set)", bin.display()))
        };
    }
    let mut built = BUILT.lock().unwrap();
    if built.is_none() {
        let release = dir.file_name().is_some_and(|n| n == "release");
        let mut cmd = Command::new("cargo");
        cmd.current_dir(workspace_root())
            .args(["build", "--quiet", "-p", "fx_preview", "--bin", "fxpreview"])
            .args(["-p", "fx_viewer", "--bin", "fxviewer"])
            .stdin(Stdio::null());
        if release {
            cmd.arg("--release");
        }
        eprintln!("fx_mcp: building fxpreview and fxviewer ({})", if release { "release" } else { "debug" });
        *built = Some(match cmd.output() {
            Ok(o) if o.status.success() => Ok(()),
            Ok(o) => Err(format!(
                "cargo build failed:\n{}",
                String::from_utf8_lossy(&o.stderr)
            )),
            Err(e) if bin.exists() => {
                eprintln!("fx_mcp: cannot run cargo ({e}); using the existing {name}");
                Ok(())
            }
            Err(e) => Err(format!(
                "cannot run cargo to build {name}: {e} (run the server in the nix dev shell)"
            )),
        });
    }
    match built.as_ref().unwrap() {
        Ok(()) => Ok(bin),
        Err(e) => Err(e.clone()),
    }
}

fn effect_stem(path: &Path) -> String {
    path.file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.trim_end_matches(".fx.ron").trim_end_matches(".ron").to_string())
        .unwrap_or_else(|| "effect".into())
}

fn nums(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

fn render_preview(args: &Value) -> Result<Value, String> {
    let path = arg_path(args)?;
    // Fail fast with the diagnostics instead of a GPU run.
    let c = check_file(&path).map_err(|e| e.to_string())?;
    if !c.ok() {
        let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
        return Ok(text(
            format!("{} has errors; nothing rendered\n{}", path.display(), errs.join("\n")),
            true,
        ));
    }
    let stem = effect_stem(&path);
    let out = match args.get("out").and_then(Value::as_str) {
        Some(o) => PathBuf::from(o),
        None => workspace_root().join("target/fx_preview").join(&stem),
    };
    let bin = tool_binary("fxpreview")?;
    let mut cmd = Command::new(bin);
    cmd.arg(&path).arg("--out").arg(&out);
    if let Some(t) = args.get("times").and_then(nums) {
        let t: Vec<String> = t.iter().map(|x| x.to_string()).collect();
        cmd.args(["--times", &t.join(",")]);
    }
    if let Some(d) = args.get("duration").and_then(Value::as_f64) {
        cmd.args(["--duration", &d.to_string()]);
    }
    for key in ["camera", "target"] {
        if let Some(v) = args.get(key).and_then(nums) {
            if v.len() != 3 {
                return Err(format!("\"{key}\" needs 3 numbers"));
            }
            cmd.arg(format!("--{key}"))
                .arg(format!("{},{},{}", v[0], v[1], v[2]));
        }
    }
    if let Some(b) = args.get("backdrop").and_then(Value::as_str) {
        cmd.args(["--backdrop", b]);
    }
    if let Some(s) = args.get("size").and_then(nums) {
        if s.len() != 2 {
            return Err("\"size\" needs [width, height]".into());
        }
        cmd.args(["--size", &format!("{}x{}", s[0] as u32, s[1] as u32)]);
    }
    if let Some(s) = args.get("seed").and_then(Value::as_u64) {
        cmd.args(["--seed", &s.to_string()]);
    }
    if !args.get("mp4").and_then(Value::as_bool).unwrap_or(false) {
        cmd.arg("--no-mp4");
    }
    let o = cmd
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run fxpreview: {e}"))?;
    let stdout = String::from_utf8_lossy(&o.stdout);
    let stderr = String::from_utf8_lossy(&o.stderr);
    if !o.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(30).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return Ok(text(
            format!("fxpreview failed ({}):\n{}", o.status, tail.join("\n")),
            true,
        ));
    }
    let strip = out.join(format!("{stem}_strip.png"));
    let mut msg = format!("{}\n", stdout.trim());
    if args.get("mode").and_then(Value::as_str) == Some("window") {
        match open_viewer(&path, None) {
            Ok(m) => msg.push_str(&m),
            Err(e) => msg.push_str(&format!("viewer: {e}")),
        }
    }
    let mut content = vec![json!({"type": "text", "text": msg})];
    if args.get("inline").and_then(Value::as_bool).unwrap_or(true) {
        let png = std::fs::read(&strip).map_err(|e| format!("{}: {e}", strip.display()))?;
        content.push(json!({
            "type": "image",
            "mimeType": "image/png",
            "data": base64::engine::general_purpose::STANDARD.encode(png),
        }));
    }
    Ok(json!({
        "content": content,
        "structuredContent": {"strip": strip.display().to_string(), "out": out.display().to_string()},
        "isError": false
    }))
}

// ------------------------------------------------------------------ viewer

fn viewer_dir() -> PathBuf {
    std::env::var_os("FXVIEWER_DIR")
        .or_else(|| std::env::var_os("XDG_RUNTIME_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

/// The running viewer's pid, if any.
fn viewer_pid() -> Option<u32> {
    let pid: u32 = std::fs::read_to_string(viewer_dir().join("fxviewer.pid"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let comm = std::fs::read_to_string(format!("/proc/{pid}/comm")).ok()?;
    comm.trim().starts_with("fxviewer").then_some(pid)
}

fn send_viewer(lines: &[String]) -> Result<(), String> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(viewer_dir().join("fxviewer.cmd"))
        .map_err(|e| e.to_string())?;
    for l in lines {
        writeln!(f, "{l}").map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn open_viewer(path: &Path, backdrop: Option<&str>) -> Result<String, String> {
    if let Some(pid) = viewer_pid() {
        let mut lines = vec![format!("open {}", path.display())];
        if let Some(b) = backdrop {
            lines.push(format!("backdrop {b}"));
        }
        send_viewer(&lines)?;
        return Ok(format!(
            "opened {} in the running viewer (pid {pid}); it reloads on every save",
            path.display()
        ));
    }
    if std::env::var_os("DISPLAY").is_none() && std::env::var_os("WAYLAND_DISPLAY").is_none() {
        return Err("no display (DISPLAY / WAYLAND_DISPLAY unset): the viewer needs a desktop session; use render_preview instead".into());
    }
    let bin = tool_binary("fxviewer")?;
    let mut cmd = Command::new(bin);
    cmd.arg(path);
    if let Some(b) = backdrop {
        cmd.args(["--backdrop", b]);
    }
    let log = std::fs::File::create(viewer_dir().join("fxviewer.log")).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(log)
        .spawn()
        .map_err(|e| format!("cannot start fxviewer: {e}"))?;
    Ok(format!(
        "started the viewer (pid {}) on {}; it reloads on every save. Log: {}",
        child.id(),
        path.display(),
        viewer_dir().join("fxviewer.log").display()
    ))
}

fn open_in_viewer(args: &Value) -> Result<Value, String> {
    let path = arg_path(args)?;
    let backdrop = args.get("backdrop").and_then(Value::as_str);
    Ok(text(open_viewer(&path, backdrop)?, false))
}

// ------------------------------------------------------------------- stats

fn renderer_name(r: &impl std::fmt::Debug) -> String {
    let s = format!("{r:?}");
    s.split(['(', ' ', '{']).next().unwrap_or("?").to_string()
}

fn stats(args: &Value) -> Result<Value, String> {
    let path = arg_path(args)?;
    let seconds = args
        .get("seconds")
        .and_then(Value::as_f64)
        .unwrap_or(3.0)
        .clamp(0.0, 60.0) as f32;
    let c = check_file(&path).map_err(|e| e.to_string())?;
    let Some(sys) = c.cooked else {
        let errs: Vec<String> = c.errors().map(|d| d.to_string()).collect();
        return Ok(text(format!("invalid effect:\n{}", errs.join("\n")), true));
    };
    let mut s = format!("\"{}\" ({})\n", sys.name, path.display());
    if !sys.user_params.is_empty() {
        let names: Vec<String> = sys
            .user_params
            .iter()
            .map(|p| format!("{} = {}", p.name, docs::ron_value(&p.value)))
            .collect();
        let _ = writeln!(s, "user parameters: {}", names.join(", "));
    }
    let mut gpu_bytes = 0u64;
    for e in &sys.emitters {
        // Two ping-pong buffers of 28 words per particle, plus state.
        let bytes = e.max_particles as u64 * 28 * 4 * 2 + 128;
        gpu_bytes += bytes;
        let _ = writeln!(
            s,
            "emitter \"{}\": {} sim, max {} particles, {:.2} MB GPU",
            e.name,
            if e.gpu { "GPU" } else { "CPU" },
            e.max_particles,
            bytes as f64 / 1048576.0
        );
        for (stage, list) in [
            ("emitter_spawn", &e.emitter_spawn),
            ("emitter_update", &e.emitter_update),
            ("particle_spawn", &e.particle_spawn),
            ("particle_update", &e.particle_update),
        ] {
            if list.is_empty() {
                continue;
            }
            let names: Vec<&str> = list
                .iter()
                .map(|m| {
                    modules::from_id(m.module)
                        .map(|k| modules::by_kind(k).name)
                        .unwrap_or("?")
                })
                .collect();
            let _ = writeln!(s, "  {stage}: {}", names.join(", "));
        }
        let rs: Vec<String> = e.renderers.iter().map(renderer_name).collect();
        let _ = writeln!(s, "  renderers: {}", if rs.is_empty() { "none".into() } else { rs.join(", ") });
    }
    let _ = writeln!(s, "GPU memory: {:.2} MB", gpu_bytes as f64 / 1048576.0);

    // Counts over time from the CPU reference simulation (seed 1, 60 Hz).
    let n_em = sys.emitters.len();
    let names: Vec<String> = sys.emitters.iter().map(|e| e.name.clone()).collect();
    let mut sim = Simulation::new(sys, 1);
    let steps = (seconds * 60.0).round() as u32;
    let mut peak = vec![0usize; n_em];
    let mut samples = Vec::new();
    let marks: Vec<u32> = [0.25f32, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0]
        .iter()
        .map(|t| (t * 60.0).round() as u32)
        .filter(|&m| m < steps)
        .chain([steps])
        .collect();
    let t0 = std::time::Instant::now();
    for step in 1..=steps {
        sim.step(1.0 / 60.0);
        for (ei, p) in peak.iter_mut().enumerate() {
            *p = (*p).max(sim.particles(ei).len());
        }
        if marks.contains(&step) {
            samples.push((step as f32 / 60.0, (0..n_em).map(|e| sim.particles(e).len()).collect::<Vec<_>>()));
        }
    }
    let _ = writeln!(
        s,
        "\nLiving particles (CPU reference sim, seed 1, {steps} steps in {:.0} ms):",
        t0.elapsed().as_secs_f64() * 1000.0
    );
    let _ = writeln!(s, "  t (s)  {}", names.join("  "));
    for (t, counts) in &samples {
        let c: Vec<String> = counts.iter().map(|n| n.to_string()).collect();
        let _ = writeln!(s, "  {t:5.2}  {}", c.join("  "));
    }
    let p: Vec<String> = peak.iter().map(|n| n.to_string()).collect();
    let _ = writeln!(s, "  peak   {}", p.join("  "));

    // Live numbers from the viewer, if it shows this file.
    if viewer_pid().is_some()
        && let Ok(st) = std::fs::read_to_string(viewer_dir().join("fxviewer.status.json"))
        && let Ok(v) = serde_json::from_str::<Value>(&st)
        && v.get("file").and_then(Value::as_str) == Some(&path.display().to_string())
    {
        let _ = writeln!(s, "\nViewer (live, t = {:.2} s):", v["time"].as_f64().unwrap_or(0.0));
        if let Some(ems) = v["emitters"].as_array() {
            for e in ems {
                let _ = writeln!(s, "  {}: {} / {}", e["name"].as_str().unwrap_or("?"), e["alive"], e["capacity"]);
            }
        }
        if let Some(g) = v["gpu_ms"].as_object() {
            for (k, ms) in g {
                let _ = writeln!(s, "  {k}: {ms} ms GPU");
            }
        }
    }
    Ok(text(s, false))
}
