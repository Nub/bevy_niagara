//! Every emitter of a system draws (GPU: runs fxpreview headless).
//!
//! Regression: renderers of one effect shared one sorted-phase key, so only
//! the last emitter's renderer drew. Two emitters here draw a red and a
//! green sprite cloud left and right of the centre; both must show.

use std::path::PathBuf;
use std::process::Command;

const EFFECT: &str = r#"(
    name: "Two Emitters",
    emitters: [
        (
            name: "Red",
            max_particles: 200,
            emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })],
            particle_spawn: [(module: "Initialize Particle", inputs: {
                "Lifetime": Float(100.0),
                "Position": RandomRange(Vec3((-1.6, 0.9, -0.2)), Vec3((-0.6, 1.7, 0.2))),
                "Color": Color((1.0, 0.0, 0.0, 1.0)),
                "Sprite Size": Vec2((0.25, 0.25)),
            })],
            particle_update: [(module: "Particle State")],
            renderers: [Sprite((blend_mode: Additive))],
        ),
        (
            name: "Green",
            max_particles: 200,
            emitter_update: [(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(200) })],
            particle_spawn: [(module: "Initialize Particle", inputs: {
                "Lifetime": Float(100.0),
                "Position": RandomRange(Vec3((0.6, 0.9, -0.2)), Vec3((1.6, 1.7, 0.2))),
                "Color": Color((0.0, 1.0, 0.0, 1.0)),
                "Sprite Size": Vec2((0.25, 0.25)),
            })],
            particle_update: [(module: "Particle State")],
            renderers: [Sprite((blend_mode: Additive))],
        ),
    ],
)"#;

#[test]
fn both_emitters_draw() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("multi_emitter");
    std::fs::create_dir_all(&dir).unwrap();
    let effect = dir.join("two.fx.ron");
    std::fs::write(&effect, EFFECT).unwrap();
    let status = Command::new(env!("CARGO_BIN_EXE_fxpreview"))
        .arg(&effect)
        .args(["--out"])
        .arg(&dir)
        .args(["--times", "0.5", "--duration", "0.5", "--no-mp4"])
        .status()
        .expect("runs fxpreview");
    assert!(status.success());

    let file = std::fs::File::open(dir.join("two_strip.png")).unwrap();
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .unwrap();
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    let (w, h) = (info.width as usize, info.height as usize);
    // Summed red in the left half and green in the right half.
    let (mut red_left, mut green_right) = (0u64, 0u64);
    for y in 0..h {
        for x in 0..w {
            let p = &buf[(y * w + x) * 4..];
            if x < w / 2 {
                red_left += p[0].saturating_sub(p[1]) as u64;
            } else {
                green_right += p[1].saturating_sub(p[0]) as u64;
            }
        }
    }
    eprintln!("red left {red_left}, green right {green_right}");
    assert!(red_left > 10_000, "the first emitter did not draw");
    assert!(green_right > 10_000, "the second emitter did not draw");
}
