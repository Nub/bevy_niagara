//! fxpreview must stay headless whatever else is in the build: the workspace
//! enables bevy_winit (for fx_viewer), so fxpreview disables WinitPlugin.
//! Runs it with no display at all (needs the GPU).

use std::path::PathBuf;
use std::process::Command;

#[test]
fn fxpreview_runs_without_a_display() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = root.join("target/fxpreview_headless_test");
    let _ = std::fs::remove_dir_all(&out);
    let o = Command::new(env!("CARGO_BIN_EXE_fxpreview"))
        .arg(root.join("effects/samples/burst.fx.ron"))
        .args(["--out"])
        .arg(&out)
        .args(["--times", "0,0.5", "--size", "96x64", "--no-mp4"])
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .env_remove("WAYLAND_SOCKET")
        .output()
        .expect("run fxpreview");
    assert!(
        o.status.success(),
        "fxpreview failed without a display:\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(out.join("burst_strip.png").exists());
}
