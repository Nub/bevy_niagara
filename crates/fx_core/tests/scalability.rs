//! System lifecycle (`duration`, emitter spawn end) and scalability settings.

mod common;

use common::compile;
use fx_core::cook;
use fx_core::scalability::{CookedQuality, CookedScalability, QualityLevel};

const FULL: &str = r#"(
    name: "Impact",
    duration: 0.6,
    scalability: (
        max_distance: 40.0,
        spawn_count_scale_by_distance: [(10.0, 1.0), (40.0, 0.25)],
        cull_when_not_visible: true,
        visibility_grace: 0.25,
        bounds_radius: 3.0,
        max_instances: 32,
        quality: (
            Low: (spawn_count_scale: 0.25, max_distance: 20.0, max_instances: 8),
            Medium: (spawn_count_scale: 0.5),
            High: (),
            Epic: (),
            Cinematic: (max_instances: 0),
        ),
    ),
    emitters: [(
        name: "E",
        emitter_update: [(module: "Spawn Rate")],
    )],
)"#;

fn system(top: &str, emitters: &[&str]) -> String {
    let es: Vec<String> = emitters
        .iter()
        .enumerate()
        .map(|(i, u)| format!("(name: \"E{i}\", emitter_update: [ {u} ])"))
        .collect();
    format!("(name: \"T\", {top} emitters: [ {} ])", es.join(", "))
}

const BURST_0: &str = r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Count": Int(5) })"#;
const BURST_1_5: &str = r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Time": Float(1.5) })"#;
const RATE: &str = r#"(module: "Spawn Rate")"#;

#[test]
fn block_cooks_to_its_values() {
    let sys = compile(FULL);
    assert_eq!(sys.duration, Some(0.6));
    let s = &sys.scalability;
    assert_eq!(s.max_distance, 40.0);
    assert_eq!(s.spawn_count_scale_by_distance, vec![(10.0, 1.0), (40.0, 0.25)]);
    assert!(s.cull_when_not_visible);
    assert_eq!(s.visibility_grace, 0.25);
    assert_eq!(s.bounds_radius, 3.0);
    assert_eq!(s.max_instances, 32);
    assert_eq!(
        s.quality[QualityLevel::Low.index()],
        CookedQuality {
            spawn_count_scale: 0.25,
            max_distance: Some(20.0),
            max_instances: Some(8)
        }
    );
    assert_eq!(s.quality[QualityLevel::High.index()], CookedQuality::default());
    assert_eq!(s.quality[QualityLevel::Cinematic.index()].max_instances, Some(0));
    assert!(!s.is_default());
    // Survives the binary.
    assert_eq!(cook::from_bytes(&cook::to_bytes(&sys)).unwrap(), sys);
}

#[test]
fn defaults_when_absent() {
    let sys = compile(&system("", &[RATE]));
    assert_eq!(sys.scalability, CookedScalability::default());
    assert!(sys.scalability.is_default());
    let s = &sys.scalability;
    assert!(s.max_distance.is_infinite());
    assert!(!s.cull_when_not_visible);
    assert_eq!((s.visibility_grace, s.bounds_radius, s.max_instances), (0.5, 2.0, 0));
    for l in QualityLevel::ALL {
        let r = s.resolve(l);
        assert_eq!(r.spawn_count_scale, 1.0);
        assert!(r.max_distance.is_infinite());
        assert_eq!(r.max_instances, 0);
    }
    assert_eq!(s.spawn_scale_at(1000.0), 1.0);
    // An empty block is the default too.
    let sys = compile(&system("scalability: (),", &[RATE]));
    assert!(sys.scalability.is_default());
}

#[test]
fn quality_levels_override_the_base() {
    let s = compile(FULL).scalability;
    let low = s.resolve(QualityLevel::Low);
    assert_eq!((low.max_distance, low.spawn_count_scale, low.max_instances), (20.0, 0.25, 8));
    let med = s.resolve(QualityLevel::Medium);
    assert_eq!((med.max_distance, med.spawn_count_scale, med.max_instances), (40.0, 0.5, 32));
    let high = s.resolve(QualityLevel::High);
    assert_eq!((high.max_distance, high.spawn_count_scale, high.max_instances), (40.0, 1.0, 32));
    assert_eq!(s.resolve(QualityLevel::Cinematic).max_instances, 0);
    assert_eq!(QualityLevel::default(), QualityLevel::High);
    for (i, l) in QualityLevel::ALL.into_iter().enumerate() {
        assert_eq!(l.index(), i);
        assert_eq!(QualityLevel::from_name(l.name()), Some(l));
    }
}

#[test]
fn spawn_scale_interpolates_and_clamps() {
    let s = CookedScalability {
        spawn_count_scale_by_distance: vec![(10.0, 1.0), (40.0, 0.25), (50.0, 0.0)],
        ..Default::default()
    };
    assert_eq!(s.spawn_scale_at(0.0), 1.0);
    assert_eq!(s.spawn_scale_at(10.0), 1.0);
    assert!((s.spawn_scale_at(25.0) - 0.625).abs() < 1e-6);
    assert_eq!(s.spawn_scale_at(40.0), 0.25);
    assert!((s.spawn_scale_at(45.0) - 0.125).abs() < 1e-6);
    assert_eq!(s.spawn_scale_at(500.0), 0.0);
    let one = CookedScalability {
        spawn_count_scale_by_distance: vec![(5.0, 0.5)],
        ..Default::default()
    };
    assert_eq!(one.spawn_scale_at(0.0), 0.5);
    assert_eq!(one.spawn_scale_at(9.0), 0.5);
}

#[test]
fn spawn_end_and_duration_from_bursts() {
    let sys = compile(&system("", &[&format!("{BURST_0}, {BURST_1_5}"), BURST_0]));
    assert_eq!(sys.emitters[0].spawn_end, Some(1.5));
    assert_eq!(sys.emitters[1].spawn_end, Some(0.0));
    assert_eq!(sys.duration, Some(1.5));
}

#[test]
fn spawn_rate_loops() {
    let sys = compile(&system("", &[BURST_1_5, RATE]));
    assert_eq!(sys.emitters[0].spawn_end, Some(1.5));
    assert_eq!(sys.emitters[1].spawn_end, None);
    assert_eq!(sys.duration, None);
}

#[test]
fn non_constant_spawn_time_loops() {
    let burst = r#"(module: "Spawn Burst Instantaneous", inputs: { "Spawn Time": RandomRange(Float(0.0), Float(1.0)) })"#;
    let sys = compile(&system("", &[burst]));
    assert_eq!(sys.emitters[0].spawn_end, None);
    assert_eq!(sys.duration, None);
}

#[test]
fn no_spawn_modules_end_at_once() {
    let sys = compile(&system("", &[""]));
    assert_eq!(sys.emitters[0].spawn_end, Some(0.0));
    assert_eq!(sys.duration, Some(0.0));
}

#[test]
fn explicit_duration_wins() {
    let sys = compile(&system("duration: 4.0,", &[RATE]));
    assert_eq!(sys.duration, Some(4.0));
    assert_eq!(sys.emitters[0].spawn_end, None);
    let sys = compile(&system("duration: 0.25,", &[BURST_1_5]));
    assert_eq!(sys.duration, Some(0.25));
}

fn errors(src: &str) -> Vec<(u32, String)> {
    let c = fx_core::check_source("t.fx.ron", src);
    assert!(c.cooked.is_none(), "expected errors:\n{src}");
    c.errors().map(|d| (d.span.line, d.message.clone())).collect()
}

#[test]
fn bad_values_are_reported_at_their_line() {
    let src = "(
    name: \"T\",
    scalability: (
        max_distance: -3.0,
        bounds_radius: inf,
        visibility_grace: 0.5,
        max_instances: 2.5,
        spawn_count_scale_by_distance: [(10.0, -1.0)],
        quality: (Ultra: (), Low: (spawn_count_scale: -0.5, max_distnce: 3.0)),
        wobble: true,
    ),
    emitters: [(name: \"E\")],
)";
    let e = errors(src);
    let has = |line: u32, text: &str| {
        assert!(
            e.iter().any(|(l, m)| *l == line && m.contains(text)),
            "no error on line {line} containing {text:?}: {e:?}"
        )
    };
    has(4, "`max_distance` must be a finite number greater than 0");
    has(5, "`bounds_radius` must be a finite number of at least 0");
    has(7, "`max_instances` must be a whole number");
    has(8, "scale must be a finite number of at least 0");
    has(9, "unknown quality level `Ultra`");
    has(9, "`spawn_count_scale` must be");
    has(9, "unknown field `max_distnce` in a quality level");
    has(10, "unknown field `wobble` in `scalability`");
    let help = fx_core::check_source("t.fx.ron", src)
        .diagnostics
        .into_iter()
        .find(|d| d.message.contains("Ultra"))
        .and_then(|d| d.help)
        .unwrap();
    assert!(help.contains("Low, Medium, High, Epic, Cinematic"), "{help}");
}

#[test]
fn zero_duration_is_an_error() {
    let e = errors(&system("\n duration: 0.0,", &[RATE]));
    assert!(e.iter().any(|(l, m)| *l == 2 && m.contains("`duration`")), "{e:?}");
}

#[test]
fn source_round_trips_through_ron() {
    let c = fx_core::check_source("t.fx.ron", FULL);
    let model = c.source.unwrap();
    let text = ron::ser::to_string(&model).unwrap();
    let again = fx_core::check_source("t.fx.ron", &text);
    assert!(again.ok(), "{text}\n{:?}", again.diagnostics);
    assert_eq!(again.cooked.unwrap(), c.cooked.unwrap());
}
