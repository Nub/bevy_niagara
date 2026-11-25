//! Custom WGSL modules, grids and simulation stages: parsing, cooking and
//! file:line errors from naga.

mod common;
use common::compile;

fn errors(src: &str) -> Vec<String> {
    fx_core::check_source("test.fx.ron", src)
        .errors()
        .map(|d| d.to_string())
        .collect()
}

fn with_module(wgsl: &str, extra_emitter: &str) -> String {
    format!(
        r##"(
    name: "T",
    custom_modules: [
        (
            name: "Push",
            inputs: {{ "Amount": Float(1.0) }},
            outputs: ["Particles.Velocity"],
            wgsl: r#"{wgsl}"#,
        ),
    ],
    emitters: [(
        name: "E",
        emitter_update: [(module: "Spawn Rate")],
        particle_update: [(module: "Particle State"), (module: "Push", inputs: {{ "Amount": Float(2.0) }})],
        {extra_emitter}
    )],
)"##
    )
}

#[test]
fn a_valid_body_cooks_to_a_custom_call() {
    let sys = compile(&with_module("Velocity.y += Amount * DeltaTime;", ""));
    assert_eq!(sys.custom_modules.len(), 1);
    assert_eq!(sys.custom_modules[0].outputs, vec![fx_core::params::attr::VELOCITY]);
    let m = &sys.emitters[0].particle_update[1];
    assert_eq!(fx_core::custom::custom_index(m.module), Some(0));
    assert_eq!(m.inputs, vec![fx_core::cook::CookedInput::Const(fx_core::types::Value::Float(2.0))]);
}

#[test]
fn a_bad_body_reports_the_file_line_and_column() {
    // The body starts on line 8 (after `r#"`), the error is on its third line.
    let src = with_module("\n    let up = Amount;\n    Velocity += Velocty * DeltaTime;", "");
    let line = src.lines().position(|l| l.contains("Velocty")).unwrap() + 1;
    let col = src.lines().nth(line - 1).unwrap().find("Velocty").unwrap() + 1;
    let errs = errors(&src);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(
        errs[0].starts_with(&format!("test.fx.ron:{line}:{col}: error: WGSL error in custom module \"Push\"")),
        "{}",
        errs[0]
    );
    assert!(errs[0].contains("Velocty"), "{}", errs[0]);
    assert!(errs[0].contains("help: the body sees its inputs (Amount)"), "{}", errs[0]);
}

#[test]
fn type_errors_point_into_the_body() {
    let src = with_module("Velocity = Amount;", "");
    let errs = errors(&src);
    assert_eq!(errs.len(), 1, "{errs:?}");
    let line = src.lines().position(|l| l.contains("Velocity = Amount")).unwrap() + 1;
    assert!(errs[0].starts_with(&format!("test.fx.ron:{line}:")), "{}", errs[0]);
}

#[test]
fn unbalanced_braces_are_reported_at_the_module() {
    let errs = errors(&with_module("if (true) { Velocity.y = 1.0;", ""));
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert!(errs[0].contains("does not compile"), "{}", errs[0]);
}

#[test]
fn declaration_errors_name_the_fix() {
    let bad_output = with_module("", "").replace("Particles.Velocity", "Particles.Velocit");
    assert!(errors(&bad_output).iter().any(|e| e.contains("did you mean \"Particles.Velocity\"")));
    let bad_input = with_module("", "").replace("\"Amount\": Float(1.0)", "\"Position\": Float(1.0)");
    assert!(errors(&bad_input).iter().any(|e| e.contains("has the name of a value the body already sees")));
    let cpu = with_module("", "sim_target: Cpu,");
    assert!(errors(&cpu).iter().any(|e| e.contains("run only on the GPU")));
    let unknown_in = with_module("", "").replace("{ \"Amount\": Float(2.0) }", "{ \"Amt\": Float(2.0) }");
    assert!(errors(&unknown_in).iter().any(|e| e.contains("has no input \"Amt\"")));
}

const GRID: &str = r#"grids: [Grid2D(name: "G", cells: (8, 4), min: (0.0, 0.0), max: (2.0, 1.0), attributes: { "V": Vec2, "D": Float })],"#;

#[test]
fn grids_and_stages_cook() {
    let src = format!(
        r##"(
    name: "T",
    custom_modules: [
        (name: "Blur", inputs: {{ "K": Float(0.5) }}, wgsl: r#"D = mix(D, 0.25 * (G_D(Cell + vec2<i32>(1, 0)) + G_D(Cell - vec2<i32>(1, 0)) + G_D(Cell + vec2<i32>(0, 1)) + G_D(Cell - vec2<i32>(0, 1))), K); V = G_sample_V(CellPosition);"#),
        (name: "Read", outputs: ["Particles.Velocity"], wgsl: r#"Velocity = vec3<f32>(G_sample_V(Position.xy), 0.0) + vec3<f32>(rand(0u));"#),
    ],
    emitters: [(
        name: "E",
        {GRID}
        emitter_update: [(module: "Spawn Rate")],
        particle_update: [(module: "Read")],
        simulation_stages: [
            (name: "Blur", iterate: Grid("G"), iterations: 3, modules: [(module: "Blur")]),
            (name: "Again", iterate: Particles, modules: [(module: "Read"), (module: "Drag")]),
        ],
    )],
)"##
    );
    let sys = compile(&src);
    let e = &sys.emitters[0];
    assert_eq!(e.grids[0].cells, [8, 4, 1]);
    assert_eq!(e.grids[0].words, 3);
    assert_eq!(e.grids[0].attributes[1].word, 2);
    assert_eq!(e.stages[0].grid, Some(0));
    assert_eq!(e.stages[0].iterations, 3);
    assert_eq!(e.stages[1].grid, None);
    assert_eq!(e.stages[1].modules.len(), 2);
    // The cooked binary keeps it all.
    let bytes = fx_core::cook::to_bytes(&sys);
    assert_eq!(fx_core::cook::from_bytes(&bytes).unwrap(), sys);
}

#[test]
fn stage_errors_name_the_fix() {
    let src = |stages: &str| {
        format!(
            r##"(
    name: "T",
    custom_modules: [(name: "M", wgsl: r#"D += 1.0;"#)],
    emitters: [(
        name: "E",
        {GRID}
        emitter_update: [(module: "Spawn Rate")],
        simulation_stages: [{stages}],
    )],
)"##
        )
    };
    let e = errors(&src(r#"(name: "S", iterate: Grid("H"), modules: [(module: "M")])"#));
    assert!(e.iter().any(|x| x.contains("did you mean \"G\"")), "{e:?}");
    let e = errors(&src(r#"(name: "S", iterate: Grid("G"), modules: [(module: "Drag")])"#));
    assert!(e.iter().any(|x| x.contains("grid stages run only custom WGSL modules")), "{e:?}");
    let e = errors(&src(r#"(name: "S", iterate: Particles, modules: [(module: "Kill Particles")])"#));
    assert!(e.iter().any(|x| x.contains("cannot run in a simulation stage")), "{e:?}");
    let e = errors(&src(r#"(name: "S", iterate: Grid("G"), iterations: 0, modules: [(module: "M")])"#));
    assert!(e.iter().any(|x| x.contains("`iterations` must be a whole number from 1 to 64")), "{e:?}");
    // The grid body may only use the grid's names: a particle attribute is an error in the body.
    let e = errors(&src(r#"(name: "S", iterate: Grid("G"), modules: [(module: "M")])"#).replace("D += 1.0;", "D += Position.x;"));
    assert!(e.iter().any(|x| x.contains("in grid stages over \"G\"") && x.contains("Position")), "{e:?}");
}

#[test]
fn raw_strings_parse_over_lines() {
    let v = fx_core::ron_parse::parse("(a: r#\"x \"q\"\ny\"#, b: r\"z\")").unwrap();
    let fx_core::ron_parse::SKind::Struct(_, f) = v.kind else { panic!() };
    assert_eq!(f[0].value.kind, fx_core::ron_parse::SKind::RawStr("x \"q\"\ny".into(), fx_core::model::Span { line: 1, col: 8 }));
    assert!(matches!(&f[1].value.kind, fx_core::ron_parse::SKind::RawStr(s, _) if s == "z"));
}
