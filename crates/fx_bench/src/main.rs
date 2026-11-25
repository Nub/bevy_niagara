//! `fx_bench`: fixed headless benchmark scenes on the real renderer, GPU ms
//! per pass from timestamp queries plus CPU ms, written as JSON per commit to
//! `bench/results/`, compared against the previous commit (a regression over
//! 10% is a bug), and turned into the Performance dashboard's tile data.
//!
//! ```text
//! fx_bench [run] [--scenes a,b] [--frames N] [--warm N] [--size WxH]
//!                [--out FILE] [--base FILE] [--no-compare]
//! fx_bench compare BASE.json NEW.json
//! fx_bench dashboard [--dir bench/results]
//! fx_bench list
//! ```
//!
//! `run` starts each scene in its own process (`fx_bench scene NAME ...`) so
//! every scene gets a fresh device and no state leaks between them. Exit
//! codes: 0 ok, 1 error, 3 regressions found (run and compare).

mod scene;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

use serde::{Deserialize, Serialize};

/// A fixed benchmark scene.
pub struct SceneDef {
    pub name: &'static str,
    /// Effect file, relative to the repository root.
    pub file: &'static str,
    pub camera: [f32; 3],
    pub target: [f32; 3],
    /// Budget for the scene's total GPU time (sim + sort + draw + RT build).
    pub budget_gpu_ms: Option<f64>,
}

/// Starting budgets on the 3080 at 1080p (Spec: Performance budgets).
pub const SCENES: &[SceneDef] = &[
    SceneDef {
        name: "sprites_1m",
        file: "bench/scenes/sprites_1m.fx.ron",
        camera: [0.0, 2.0, 7.0],
        target: [0.0, 2.0, 0.0],
        budget_gpu_ms: Some(2.0),
    },
    SceneDef {
        name: "sprites_1m_sorted",
        file: "bench/scenes/sprites_1m_sorted.fx.ron",
        camera: [0.0, 2.0, 7.0],
        target: [0.0, 2.0, 0.0],
        budget_gpu_ms: None,
    },
    SceneDef {
        name: "fire",
        file: "bench/scenes/fire.fx.ron",
        camera: [0.0, 1.0, 3.5],
        target: [0.0, 0.8, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "sparks",
        file: "bench/scenes/sparks.fx.ron",
        camera: [0.0, 1.0, 4.0],
        target: [0.5, 0.8, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    // Reference effect library (effects/library): 0.2 ms each.
    SceneDef {
        name: "lib_campfire",
        file: "effects/library/campfire.fx.ron",
        camera: [0.0, 1.0, 3.2],
        target: [0.0, 0.7, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_sparks_burst",
        file: "effects/library/sparks_burst.fx.ron",
        camera: [0.0, 1.2, 4.0],
        target: [0.0, 0.9, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_rising_smoke",
        file: "effects/library/rising_smoke.fx.ron",
        camera: [0.0, 1.6, 5.5],
        target: [0.0, 1.4, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_magic_swirl",
        file: "effects/library/magic_swirl.fx.ron",
        camera: [0.0, 1.2, 3.5],
        target: [0.0, 0.9, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_sword_trail",
        file: "effects/library/sword_trail.fx.ron",
        camera: [0.0, 1.3, 3.5],
        target: [0.0, 1.2, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_debris",
        file: "effects/library/debris.fx.ron",
        camera: [0.0, 1.3, 5.0],
        target: [0.0, 1.0, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    // Tutorial batch 2 (effects/library), 0.2 ms each.
    SceneDef {
        name: "lib_shockwave_ring",
        file: "effects/library/shockwave_ring.fx.ron",
        camera: [0.0, 1.0, 4.5],
        target: [0.0, 1.0, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_lightning_strike",
        file: "effects/library/lightning_strike.fx.ron",
        camera: [0.0, 1.7, 6.0],
        target: [0.0, 1.6, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_portal_vortex",
        file: "effects/library/portal_vortex.fx.ron",
        camera: [0.0, 1.3, 4.5],
        target: [0.0, 1.3, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_stylized_tornado",
        file: "effects/library/stylized_tornado.fx.ron",
        camera: [0.0, 2.3, 9.0],
        target: [0.0, 2.1, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_rain_splash",
        file: "effects/library/rain_splash.fx.ron",
        camera: [0.0, 1.7, 6.0],
        target: [0.0, 1.0, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_muzzle_flash",
        file: "effects/library/muzzle_flash.fx.ron",
        camera: [0.0, 1.5, 2.6],
        target: [0.0, 1.35, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_footstep_dust",
        file: "effects/library/footstep_dust.fx.ron",
        camera: [-0.6, 1.1, 3.0],
        target: [-0.6, 0.15, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_heal_aura",
        file: "effects/library/heal_aura.fx.ron",
        camera: [0.0, 1.6, 4.0],
        target: [0.0, 0.6, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_fireworks_show",
        file: "effects/library/fireworks_show.fx.ron",
        camera: [0.0, 6.0, 18.0],
        target: [0.0, 6.0, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "lib_meteor_rain",
        file: "effects/library/meteor_rain.fx.ron",
        camera: [0.0, 2.5, 9.0],
        target: [0.0, 1.5, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    // Events and collisions.
    SceneDef {
        name: "fireworks",
        file: "bench/scenes/fireworks.fx.ron",
        camera: [0.0, 5.0, 15.0],
        target: [0.0, 5.5, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "collision_100k",
        file: "bench/scenes/collision_100k.fx.ron",
        camera: [0.0, 2.0, 6.0],
        target: [0.0, 0.5, 0.0],
        budget_gpu_ms: None,
    },
    SceneDef {
        name: "collision_depth_100k",
        file: "bench/scenes/collision_depth_100k.fx.ron",
        camera: [0.0, 2.0, 6.0],
        target: [0.0, 0.5, 0.0],
        budget_gpu_ms: None,
    },
    // Custom WGSL modules and simulation stages.
    SceneDef {
        name: "wgsl_swirl",
        file: "effects/samples/wgsl_swirl.fx.ron",
        camera: [0.0, 2.5, 4.0],
        target: [0.0, 0.6, 0.0],
        budget_gpu_ms: Some(0.2),
    },
    SceneDef {
        name: "grid_smoke",
        file: "effects/samples/grid_smoke.fx.ron",
        camera: [0.0, 1.5, 4.5],
        target: [0.0, 1.5, 0.0],
        // Sim with 28 grid passes is ~0.08 ms; the rest sorts and draws 12k alpha sprites.
        budget_gpu_ms: Some(0.4),
    },
];

/// Relative slowdown that counts as a regression.
pub const REGRESSION: f64 = 0.10;

pub const SCHEMA: u32 = 1;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PassTiming {
    pub gpu_ms: Option<f64>,
    pub cpu_ms: Option<f64>,
}

/// The renderer's passes, mapped from Bevy's render diagnostic spans.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct GpuMs {
    /// `fx_sim`: spawn, update, compaction.
    pub sim: Option<f64>,
    /// `fx_cull_sort`: frustum cull, depth sort, indirect args.
    pub sort: Option<f64>,
    /// `main_transparent_pass_3d`: the particle draw.
    pub draw: Option<f64>,
    /// Ray-tracing acceleration-structure build for particles (not yet
    /// rendered: null until the Solari path lands).
    pub rt_build: Option<f64>,
    /// Sum of the above.
    pub total: f64,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct CpuMs {
    /// Median wall time of a whole frame (main + render world, no pipelining).
    pub frame: f64,
    /// CPU time recording the fx passes (sim, cull/sort and draw spans).
    pub fx_encode: Option<f64>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SceneResult {
    pub name: String,
    pub file: String,
    pub max_particles: u64,
    pub budget_gpu_ms: Option<f64>,
    pub within_budget: Option<bool>,
    pub gpu_ms: GpuMs,
    pub cpu_ms: CpuMs,
    /// Heap allocations per frame once warm, whole process (Bevy + wgpu + fx).
    pub allocs_per_frame: f64,
    pub frames: u32,
    pub adapter: Option<String>,
    /// Every render diagnostic span: medians over the measured frames.
    pub passes: BTreeMap<String, PassTiming>,
}

const SIM_SPAN: &str = "fx_sim";
const SORT_SPAN: &str = "fx_cull_sort";
const DRAW_SPAN: &str = "main_transparent_pass_3d";
const RT_SPAN: &str = "fx_rt_build";

impl SceneResult {
    pub fn new(
        def: &SceneDef,
        max_particles: u64,
        adapter: Option<String>,
        passes: BTreeMap<String, PassTiming>,
        frame_ms: f64,
        allocs_per_frame: f64,
        frames: u32,
    ) -> Self {
        let find = |key: &str| {
            passes
                .iter()
                .filter(|(p, _)| p.rsplit('/').next() == Some(key))
                .map(|(_, t)| t.clone())
                .next()
        };
        let (sim, sort, draw, rt) = (find(SIM_SPAN), find(SORT_SPAN), find(DRAW_SPAN), find(RT_SPAN));
        let gpu = |t: &Option<PassTiming>| t.as_ref().and_then(|t| t.gpu_ms);
        let total = [&sim, &sort, &draw, &rt].iter().filter_map(|t| gpu(t)).sum();
        let cpu: Vec<f64> = [&sim, &sort, &draw]
            .iter()
            .filter_map(|t| t.as_ref().and_then(|t| t.cpu_ms))
            .collect();
        SceneResult {
            name: def.name.into(),
            file: def.file.into(),
            max_particles,
            budget_gpu_ms: def.budget_gpu_ms,
            within_budget: def.budget_gpu_ms.map(|b| total <= b),
            gpu_ms: GpuMs {
                sim: gpu(&sim),
                sort: gpu(&sort),
                draw: gpu(&draw),
                rt_build: gpu(&rt),
                total,
            },
            cpu_ms: CpuMs {
                frame: frame_ms,
                fx_encode: (!cpu.is_empty()).then(|| cpu.iter().sum()),
            },
            allocs_per_frame,
            frames,
            adapter,
            passes,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RunResult {
    pub schema: u32,
    /// Short commit hash of the code measured.
    pub commit: String,
    /// Commit time (unix seconds), orders the history.
    pub commit_time: i64,
    pub subject: String,
    /// Uncommitted changes outside bench/results were present.
    pub dirty: bool,
    /// When the run happened (unix seconds).
    pub run_time: u64,
    pub host: String,
    pub resolution: [u32; 2],
    pub warm_frames: u32,
    pub scenes: Vec<SceneResult>,
}

pub fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut v = v.to_vec();
    v.sort_by(f64::total_cmp);
    Some(v[v.len() / 2])
}

pub fn bench_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../bench")
}

fn results_dir() -> PathBuf {
    bench_dir().join("results")
}

fn git(args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .current_dir(bench_dir().join(".."))
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

const RESULT_TAG: &str = "FXBENCH_RESULT ";

struct RunOpts {
    scenes: Vec<String>,
    frames: u32,
    warm: u32,
    size: (u32, u32),
    out: Option<PathBuf>,
    base: Option<PathBuf>,
    compare: bool,
}

fn parse_size(v: &str) -> Result<(u32, u32), String> {
    let (w, h) = v.split_once('x').ok_or("--size WxH")?;
    Ok((
        w.parse().map_err(|e| format!("{e}"))?,
        h.parse().map_err(|e| format!("{e}"))?,
    ))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let res = match args.first().map(String::as_str) {
        Some("scene") => run_scene(&args[1..]),
        Some("compare") => match &args[1..] {
            [a, b] => load(Path::new(a)).and_then(|base| {
                let new = load(Path::new(b))?;
                Ok(if compare(&base, &new) { 3 } else { 0 })
            }),
            _ => Err("usage: fx_bench compare BASE.json NEW.json".into()),
        },
        Some("dashboard") => {
            let dir = match &args[1..] {
                [f, d] if f == "--dir" => PathBuf::from(d),
                [] => results_dir(),
                _ => return fail("usage: fx_bench dashboard [--dir DIR]".into()),
            };
            dashboard(&dir).map(|_| 0)
        }
        Some("list") => {
            for s in SCENES {
                println!("{:20} {:40} budget {:?} ms", s.name, s.file, s.budget_gpu_ms);
            }
            Ok(0)
        }
        Some("run") => run(&args[1..]),
        Some(a) if a.starts_with("--") => run(&args),
        None => run(&[]),
        Some(a) => Err(format!("unknown command {a}")),
    };
    match res {
        Ok(0) => ExitCode::SUCCESS,
        Ok(c) => ExitCode::from(c),
        Err(e) => fail(e),
    }
}

fn fail(e: String) -> ExitCode {
    eprintln!("fx_bench: {e}");
    ExitCode::from(1)
}

fn scene_def(name: &str) -> Result<&'static SceneDef, String> {
    SCENES
        .iter()
        .find(|s| s.name == name)
        .ok_or_else(|| format!("unknown scene {name} (fx_bench list)"))
}

/// Child process: one scene, result on stdout after [`RESULT_TAG`].
fn run_scene(args: &[String]) -> Result<u8, String> {
    let name = args.first().ok_or("scene NAME")?;
    let def = scene_def(name)?;
    let (mut frames, mut warm, mut size) = (300, 180, (1920, 1080));
    let mut it = args[1..].iter();
    while let Some(a) = it.next() {
        let v = it.next().ok_or(format!("{a} needs a value"))?;
        match a.as_str() {
            "--frames" => frames = v.parse().map_err(|e| format!("{e}"))?,
            "--warm" => warm = v.parse().map_err(|e| format!("{e}"))?,
            "--size" => size = parse_size(v)?,
            _ => return Err(format!("unknown option {a}")),
        }
    }
    let r = scene::run(def, frames, warm, size);
    println!("{RESULT_TAG}{}", serde_json::to_string(&r).unwrap());
    Ok(0)
}

fn run(args: &[String]) -> Result<u8, String> {
    let mut o = RunOpts {
        scenes: SCENES.iter().map(|s| s.name.to_string()).collect(),
        frames: 300,
        warm: 180,
        size: (1920, 1080),
        out: None,
        base: None,
        compare: true,
    };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        if a == "--no-compare" {
            o.compare = false;
            continue;
        }
        let v = it.next().ok_or(format!("{a} needs a value"))?;
        match a.as_str() {
            "--scenes" => o.scenes = v.split(',').map(str::to_string).collect(),
            "--frames" => o.frames = v.parse().map_err(|e| format!("{e}"))?,
            "--warm" => o.warm = v.parse().map_err(|e| format!("{e}"))?,
            "--size" => o.size = parse_size(v)?,
            "--out" => o.out = Some(v.into()),
            "--base" => o.base = Some(v.into()),
            _ => return Err(format!("unknown option {a}")),
        }
    }
    for s in &o.scenes {
        scene_def(s)?;
    }

    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let mut scenes = Vec::new();
    for name in &o.scenes {
        eprintln!("fx_bench: running {name} ...");
        let out = Command::new(&exe)
            .args(["scene", name, "--frames", &o.frames.to_string(), "--warm", &o.warm.to_string()])
            .args(["--size", &format!("{}x{}", o.size.0, o.size.1)])
            .stderr(Stdio::inherit())
            .output()
            .map_err(|e| format!("starting scene {name}: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        let line = stdout
            .lines()
            .find_map(|l| l.strip_prefix(RESULT_TAG))
            .ok_or_else(|| format!("scene {name} failed ({}) with no result:\n{stdout}", out.status))?;
        let r: SceneResult = serde_json::from_str(line).map_err(|e| e.to_string())?;
        scenes.push(r);
    }

    let commit = git(&["rev-parse", "--short=10", "HEAD"]);
    let dirty = !git(&["status", "--porcelain", "--", ".", ":!bench/results"]).is_empty();
    let result = RunResult {
        schema: SCHEMA,
        commit_time: git(&["show", "-s", "--format=%ct", "HEAD"]).parse().unwrap_or(0),
        subject: git(&["show", "-s", "--format=%s", "HEAD"]),
        commit: commit.clone(),
        dirty,
        run_time: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        host: std::fs::read_to_string("/etc/hostname")
            .map(|h| h.trim().to_string())
            .unwrap_or_default(),
        resolution: [o.size.0, o.size.1],
        warm_frames: o.warm,
        scenes,
    };

    print_summary(&result);
    let out = o.out.clone().unwrap_or_else(|| {
        let suffix = if dirty { "-dirty" } else { "" };
        results_dir().join(format!("{commit}{suffix}.json"))
    });
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&out, serde_json::to_string_pretty(&result).unwrap() + "\n")
        .map_err(|e| format!("writing {}: {e}", out.display()))?;
    println!("wrote {}", out.display());

    if !o.compare {
        return Ok(0);
    }
    let base = match &o.base {
        Some(b) => Some(load(b)?),
        None => history(&results_dir())?
            .into_iter()
            .filter(|r| r.commit != result.commit)
            .last(),
    };
    match base {
        Some(base) => Ok(if compare(&base, &result) { 3 } else { 0 }),
        None => {
            println!("no earlier result to compare with");
            Ok(0)
        }
    }
}

fn fmt(v: Option<f64>) -> String {
    v.map_or("-".into(), |v| format!("{v:.3}"))
}

fn print_summary(r: &RunResult) {
    println!(
        "fx_bench {} {}{} at {}x{}",
        r.commit,
        r.subject,
        if r.dirty { " (dirty)" } else { "" },
        r.resolution[0],
        r.resolution[1]
    );
    println!(
        "{:18} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7}  budget",
        "scene", "particles", "sim", "sort", "draw", "gpu", "cpu fr", "fx enc", "allocs"
    );
    for s in &r.scenes {
        let budget = match (s.budget_gpu_ms, s.within_budget) {
            (Some(b), Some(true)) => format!("ok (< {b} ms)"),
            (Some(b), _) => format!("OVER ({b} ms)"),
            _ => "-".into(),
        };
        println!(
            "{:18} {:>9} {:>8} {:>8} {:>8} {:>8.3} {:>8.3} {:>9} {:>7.0}  {budget}",
            s.name,
            s.max_particles,
            fmt(s.gpu_ms.sim),
            fmt(s.gpu_ms.sort),
            fmt(s.gpu_ms.draw),
            s.gpu_ms.total,
            s.cpu_ms.frame,
            fmt(s.cpu_ms.fx_encode),
            s.allocs_per_frame,
        );
    }
}

fn load(p: &Path) -> Result<RunResult, String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("reading {}: {e}", p.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
}

/// Committed (non-dirty) results, oldest commit first.
fn history(dir: &Path) -> Result<Vec<RunResult>, String> {
    let mut v = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(v);
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.extension().is_some_and(|x| x == "json") {
            let r = load(&p)?;
            if !r.dirty {
                v.push(r);
            }
        }
    }
    v.sort_by_key(|r| (r.commit_time, r.run_time));
    Ok(v)
}

/// Metrics compared between runs: (name, value, noise floor). A change is a
/// regression when it is over [`REGRESSION`] relative AND over the floor
/// absolute, so microsecond jitter on tiny passes is not flagged.
fn metrics(s: &SceneResult) -> Vec<(&'static str, Option<f64>, f64)> {
    vec![
        ("gpu sim ms", s.gpu_ms.sim, 0.005),
        ("gpu sort ms", s.gpu_ms.sort, 0.005),
        ("gpu draw ms", s.gpu_ms.draw, 0.005),
        ("gpu rt_build ms", s.gpu_ms.rt_build, 0.005),
        ("gpu total ms", Some(s.gpu_ms.total), 0.005),
        ("cpu fx encode ms", s.cpu_ms.fx_encode, 0.01),
        ("allocs/frame", Some(s.allocs_per_frame), 2.0),
    ]
}

/// Prints the comparison; true when anything regressed.
fn compare(base: &RunResult, new: &RunResult) -> bool {
    println!(
        "\ncompare {} -> {}{} (regression: > {:.0}% and over the noise floor)",
        base.commit,
        new.commit,
        if new.dirty { " (dirty)" } else { "" },
        REGRESSION * 100.0
    );
    let mut regressions = Vec::new();
    for s in &new.scenes {
        let Some(b) = base.scenes.iter().find(|b| b.name == s.name) else {
            println!("  {:18} new scene", s.name);
            continue;
        };
        for ((name, nv, floor), (_, bv, _)) in metrics(s).into_iter().zip(metrics(b)) {
            let (Some(nv), Some(bv)) = (nv, bv) else { continue };
            let delta = nv - bv;
            let rel = if bv > 0.0 { delta / bv } else { 0.0 };
            let flag = rel > REGRESSION && delta > floor;
            let mark = if flag {
                "REGRESSION"
            } else if -rel > REGRESSION && -delta > floor {
                "faster"
            } else {
                ""
            };
            println!(
                "  {:18} {:16} {:>9.3} -> {:>9.3}  {:>+7.1}%  {mark}",
                s.name,
                name,
                bv,
                nv,
                rel * 100.0
            );
            if flag {
                regressions.push(format!("{} {name} +{:.1}%", s.name, rel * 100.0));
            }
        }
    }
    if regressions.is_empty() {
        println!("no regressions");
        false
    } else {
        println!("REGRESSIONS ({}): {}", regressions.len(), regressions.join(", "));
        true
    }
}

/// Tile data for the Performance dashboard, one JSON object keyed by tile.
fn dashboard(dir: &Path) -> Result<(), String> {
    use serde_json::{Value, json};
    let hist = history(dir)?;
    let latest = hist.last().ok_or("no results yet")?;
    let mut out = serde_json::Map::new();
    for s in &latest.scenes {
        if let Some(b) = s.budget_gpu_ms {
            out.insert(
                format!("budget_{}", s.name),
                json!({
                    "value": (s.gpu_ms.total * 1000.0).round() / 1000.0,
                    "unit": "ms",
                    "target": b,
                    "better": "lower",
                    "min": 0,
                    "max": b * 2.0,
                    "label": format!("GPU ms at {}", latest.commit),
                }),
            );
        }
    }
    let labels: Vec<Value> = hist.iter().map(|r| json!(r.commit)).collect();
    let series = |scenes: &[&str], f: &dyn Fn(&SceneResult) -> Option<f64>| -> Vec<Value> {
        scenes
            .iter()
            .map(|n| {
                let values: Vec<Value> = hist
                    .iter()
                    .map(|r| {
                        r.scenes
                            .iter()
                            .find(|s| s.name == *n)
                            .and_then(f)
                            .map_or(Value::Null, |v| json!((v * 1000.0).round() / 1000.0))
                    })
                    .collect();
                json!({ "name": n, "values": values })
            })
            .collect()
    };
    let total = |s: &SceneResult| Some(s.gpu_ms.total);
    out.insert(
        "history_gpu_1m".into(),
        json!({ "type": "line", "labels": labels, "unit": "ms",
                "series": series(&["sprites_1m", "sprites_1m_sorted"], &total) }),
    );
    out.insert(
        "history_gpu_typical".into(),
        json!({ "type": "line", "labels": labels, "unit": "ms",
                "series": series(&["fire", "sparks"], &total) }),
    );
    let library: Vec<&str> = SCENES.iter().map(|s| s.name).filter(|n| n.starts_with("lib_")).collect();
    out.insert(
        "history_gpu_library".into(),
        json!({ "type": "line", "labels": labels, "unit": "ms",
                "series": series(&library, &total) }),
    );
    let all: Vec<&str> = SCENES.iter().map(|s| s.name).collect();
    out.insert(
        "history_cpu_frame".into(),
        json!({ "type": "line", "labels": labels, "unit": "ms",
                "series": series(&all, &|s: &SceneResult| Some(s.cpu_ms.frame)) }),
    );
    let r3 = |v: Option<f64>| v.map_or(Value::Null, |v| json!((v * 1000.0).round() / 1000.0));
    let rows: Vec<Value> = latest
        .scenes
        .iter()
        .map(|s| {
            json!([
                s.name,
                s.max_particles,
                r3(s.gpu_ms.sim),
                r3(s.gpu_ms.sort),
                r3(s.gpu_ms.draw),
                r3(Some(s.gpu_ms.total)),
                r3(s.cpu_ms.fx_encode),
                r3(Some(s.cpu_ms.frame)),
                s.allocs_per_frame,
            ])
        })
        .collect();
    out.insert(
        "passes_latest".into(),
        json!({
            "columns": ["Scene", "Max particles", "Sim GPU ms", "Sort GPU ms", "Draw GPU ms",
                        "Total GPU ms", "FX encode CPU ms", "Frame CPU ms", "Allocs/frame"],
            "rows": rows,
        }),
    );
    println!("{}", serde_json::to_string_pretty(&Value::Object(out)).unwrap());
    Ok(())
}
