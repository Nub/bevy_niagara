//! Niagara conversion on synthetic packages: a small system written by
//! `fx_import::synth` with a real stack graph (output, function call,
//! map-set and input nodes wired through their pins, rapid-iteration
//! constants, a user parameter, a curve data interface and a dynamic
//! input), plus the curve and HLSL helpers on their own.

use fx_import::convert::{convert, curve, eval_curve, Status, Unit};
use fx_import::fx::Ty;
use fx_import::niagara::{read_system, CurveKey, HlslNode, NCurve, NInput, NSystem, NValue};
use fx_import::synth::{fstring, Synth, SynthPin};
use fx_import::Package;

/// Export indices, in creation order (1-based).
mod ix {
    pub const SYSTEM: i32 = 1;
    pub const EMITTER: i32 = 2;
    pub const EU_SCRIPT: i32 = 3;
    pub const GPU_SCRIPT: i32 = 4;
    pub const SOURCE: i32 = 5;
    pub const GRAPH: i32 = 6;
    pub const CURVE: i32 = 7;
    pub const SPRITE: i32 = 8;
    pub const OUT_EU: i32 = 9;
    pub const SPAWN_RATE: i32 = 10;
    pub const IN_EU: i32 = 11;
    pub const OUT_PS: i32 = 12;
    pub const INIT: i32 = 13;
    pub const SET_PS: i32 = 14;
    pub const IN_PS: i32 = 15;
    pub const GET_USER: i32 = 16;
    pub const OUT_PU: i32 = 17;
    pub const GRAVITY: i32 = 18;
    pub const SET_PU: i32 = 19;
    pub const SCALE_COLOR: i32 = 20;
    pub const IN_PU: i32 = 21;
    pub const FROM_CURVE: i32 = 22;
    pub const IN_CURVE: i32 = 23;
    pub const UNKNOWN: i32 = 24;
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn system_package(ue5: i32) -> Vec<u8> {
    use ix::*;
    let mut s = Synth::new(ue5).with_graph_versions();
    let niagara = s.import("/Script/CoreUObject", "Package", 0, "/Script/Niagara");
    let editor = s.import("/Script/CoreUObject", "Package", 0, "/Script/NiagaraEditor");
    let class = |s: &mut Synth, pkg: i32, n: &str| s.import("/Script/CoreUObject", "Class", pkg, n);
    let c_system = class(&mut s, niagara, "NiagaraSystem");
    let c_emitter = class(&mut s, niagara, "NiagaraEmitter");
    let c_script = class(&mut s, niagara, "NiagaraScript");
    let c_curve = class(&mut s, niagara, "NiagaraDataInterfaceCurve");
    let c_sprite = class(&mut s, niagara, "NiagaraSpriteRendererProperties");
    let c_source = class(&mut s, editor, "NiagaraScriptSource");
    let c_graph = class(&mut s, editor, "NiagaraGraph");
    let c_out = class(&mut s, editor, "NiagaraNodeOutput");
    let c_call = class(&mut s, editor, "NiagaraNodeFunctionCall");
    let c_set = class(&mut s, editor, "NiagaraNodeParameterMapSet");
    let c_get = class(&mut s, editor, "NiagaraNodeParameterMapGet");
    let c_in = class(&mut s, editor, "NiagaraNodeInput");
    let st = |s: &mut Synth, n: &str| s.import("/Script/CoreUObject", "ScriptStruct", niagara, n);
    let t_map = st(&mut s, "NiagaraParameterMap");
    let t_float = st(&mut s, "NiagaraFloat");
    let t_vec3 = st(&mut s, "Vector3f");
    let module = |s: &mut Synth, path: &str| {
        let short = path.rsplit('/').next().unwrap().to_string();
        let p = s.import("/Script/CoreUObject", "Package", 0, path);
        s.import("/Script/Niagara", "NiagaraScript", p, &short)
    };
    let m_rate = module(&mut s, "/Niagara/Modules/Emitter/SpawnRate");
    let m_init = module(&mut s, "/Niagara/Modules/Spawn/Initialization/V2/InitializeParticle");
    let m_gravity = module(&mut s, "/Niagara/Modules/Update/Forces/GravityForce");
    let m_scale = module(&mut s, "/Niagara/Modules/Update/Color/ScaleColor");
    let m_curve = module(&mut s, "/Niagara/DynamicInputs/ValueFromCurve/FloatFromCurve");
    let m_unknown = module(&mut s, "/Niagara/Modules/Weird/Wobble");
    let mat_pkg = s.import("/Script/CoreUObject", "Package", 0, "/Game/FX/M_Spark");
    let mat = s.import("/Script/Engine", "Material", mat_pkg, "M_Spark");

    // A parameter store: (name, type, offset) + data.
    let store = |s: &mut Synth, vars: &[(&str, i32, i32)], data: &[u8]| -> Vec<u8> {
        let mut p = s.props();
        let elems: Vec<Vec<u8>> = vars
            .iter()
            .map(|(n, t, o)| {
                let mut e = p.fname(n);
                e.extend(p.nested(|q| {
                    q.object("ClassStructOrEnum", *t);
                }));
                e.extend_from_slice(&o.to_le_bytes());
                e
            })
            .collect();
        p.struct_array("SortedParameterOffsets", "NiagaraVariableWithOffset", &elems);
        p.byte_array("ParameterData", data);
        p.finish()
    };

    // 1 System.
    let user = store(&mut s, &[("User.Size", t_float, 0)], &f32s(&[50.0]));
    let mut p = s.props();
    let handle = p.nested(|q| {
        q.name("Name", "Sparks").bool("bIsEnabled", true).tagged_struct("VersionedInstance", "VersionedNiagaraEmitter", |r| {
            r.object("Emitter", EMITTER);
        });
    });
    p.struct_array("EmitterHandles", "NiagaraEmitterHandle", &[handle]);
    p.raw("ExposedParameters", &fx_import::synth::T::Struct("NiagaraUserRedirectionParameterStore"), &user, false, false);
    let props = p.finish();
    assert_eq!(s.export(c_system, 0, "NS_Synth", 0, props, Vec::new()), SYSTEM);

    // 2 Emitter (GPU: particle constants come from the GPU compute script).
    let mut p = s.props();
    let vd = p.nested(|q| {
        q.byte_enum("SimTarget", "ENiagaraSimTarget", "ENiagaraSimTarget::GPUComputeSim")
            .bool("bLocalSpace", true)
            .tagged_struct("EmitterUpdateScriptProps", "NiagaraEmitterScriptProperties", |r| {
                r.object("Script", EU_SCRIPT);
            })
            .object("GPUComputeScript", GPU_SCRIPT)
            .object("GraphSource", SOURCE)
            .object_array("RendererProperties", &[SPRITE]);
    });
    p.struct_array("VersionData", "VersionedNiagaraEmitterData", &[vd]).str("UniqueEmitterName", "Sparks");
    let props = p.finish();
    assert_eq!(s.export(c_emitter, SYSTEM, "Sparks", 0, props, Vec::new()), EMITTER);

    // 3, 4 Scripts with rapid-iteration constants.
    for (idx, vars, data) in [
        (EU_SCRIPT, vec![("Constants.Sparks.SpawnRate.SpawnRate", t_float, 0)], f32s(&[25.0])),
        (
            GPU_SCRIPT,
            vec![
                ("Constants.Sparks.InitializeParticle.Lifetime Min", t_float, 0),
                ("Constants.Sparks.InitializeParticle.Lifetime Max", t_float, 4),
                ("Constants.Sparks.GravityForce.Gravity", t_vec3, 8),
            ],
            f32s(&[1.0, 2.0, 0.0, 0.0, -980.0]),
        ),
    ] {
        let ri = store(&mut s, &vars, &data);
        let mut p = s.props();
        p.raw("RapidIterationParameters", &fx_import::synth::T::Struct("NiagaraParameterStore"), &ri, false, false);
        let props = p.finish();
        assert_eq!(s.export(c_script, EMITTER, &format!("Script{idx}"), 0, props, Vec::new()), idx);
    }
    // 5 Source, 6 Graph.
    let mut p = s.props();
    p.object("NodeGraph", GRAPH);
    let props = p.finish();
    assert_eq!(s.export(c_source, EMITTER, "NiagaraScriptSource_0", 0, props, Vec::new()), SOURCE);
    let props = s.props().finish();
    assert_eq!(s.export(c_graph, SOURCE, "NiagaraGraph_0", 0, props, Vec::new()), GRAPH);
    // 7 Curve: 1 -> 0 (linear keys).
    let mut p = s.props();
    // FRichCurveKey is saved natively: modes, then time, value and tangents.
    let key = |t: f32, v: f32| {
        let mut b = vec![0u8, 0, 0];
        b.extend(f32s(&[t, v, 0.0, 0.0, 0.0, 0.0]));
        b
    };
    let (k0, k1) = (key(0.0, 1.0), key(1.0, 0.0));
    p.tagged_struct("Curve", "RichCurve", |q| {
        q.struct_array("Keys", "RichCurveKey", &[k0, k1]);
    });
    let props = p.finish();
    assert_eq!(s.export(c_curve, GRAPH, "FloatCurve", 0, props, Vec::new()), CURVE);
    // 8 Sprite renderer.
    let mut p = s.props();
    p.object("Material", mat);
    let props = p.finish();
    assert_eq!(s.export(c_sprite, EMITTER, "Sprite", 0, props, Vec::new()), SPRITE);

    // Graph nodes. Pin ids are unique numbers; links go from input pins
    // back to the pin they read.
    let usage = |s: &mut Synth, u: &str| {
        let mut p = s.props();
        p.byte_enum("ScriptType", "ENiagaraScriptUsage", &format!("ENiagaraScriptUsage::{u}"));
        p.finish()
    };
    let call = |s: &mut Synth, script: i32, name: &str| {
        let mut p = s.props();
        p.object("FunctionScript", script).str("FunctionDisplayName", name);
        p.finish()
    };
    let map = |id, name: &str, out| SynthPin::new(id, name, out, t_map);
    let node = |s: &mut Synth, idx: i32, class: i32, name: &str, props: Vec<u8>, pins: Vec<SynthPin>| {
        let native = s.pins(idx, &pins);
        assert_eq!(s.export(class, GRAPH, name, 0, props, native), idx, "{name}");
    };
    // Emitter update: In -> SpawnRate -> Out.
    let pr = usage(&mut s, "EmitterUpdateScript");
    node(&mut s, OUT_EU, c_out, "Out_EU", pr, vec![map(1, "Out", false).link(SPAWN_RATE, 2)]);
    let pr = call(&mut s, m_rate, "SpawnRate");
    node(&mut s, SPAWN_RATE, c_call, "Call_SpawnRate", pr, vec![map(3, "InputMap", false).link(IN_EU, 4), map(2, "OutputMap", true)]);
    let pr = s.props().finish();
    node(&mut s, IN_EU, c_in, "In_EU", pr, vec![map(4, "Input", true)]);
    // Particle spawn: In -> MapSet (Uniform Sprite Size <- User.Size) -> InitializeParticle -> Out.
    let pr = usage(&mut s, "ParticleSpawnScript");
    node(&mut s, OUT_PS, c_out, "Out_PS", pr, vec![map(5, "Out", false).link(INIT, 6)]);
    let pr = call(&mut s, m_init, "InitializeParticle");
    node(
        &mut s,
        INIT,
        c_call,
        "Call_Init",
        pr,
        vec![
            map(7, "InputMap", false).link(SET_PS, 8),
            map(6, "OutputMap", true),
            SynthPin::new(30, "Lifetime Mode", false, 0).default("NewEnumerator1"),
            SynthPin::new(31, "Sprite Size Mode", false, 0).default("NewEnumerator3"),
        ],
    );
    let pr = s.props().finish();
    node(
        &mut s,
        SET_PS,
        c_set,
        "MapSet_PS",
        pr,
        vec![
            map(9, "Source", false).link(IN_PS, 10),
            SynthPin::new(11, "InitializeParticle.Uniform Sprite Size", false, t_float).link(GET_USER, 12),
            map(8, "Dest", true),
        ],
    );
    let pr = s.props().finish();
    node(&mut s, IN_PS, c_in, "In_PS", pr, vec![map(10, "Input", true)]);
    let pr = s.props().finish();
    node(&mut s, GET_USER, c_get, "MapGet_User", pr, vec![map(13, "Source", false), SynthPin::new(12, "User.Size", true, t_float)]);
    // Particle update: In -> Gravity -> MapSet (Scale Alpha <- FloatFromCurve(curve)) -> ScaleColor -> Wobble -> Out.
    let pr = usage(&mut s, "ParticleUpdateScript");
    node(&mut s, OUT_PU, c_out, "Out_PU", pr, vec![map(14, "Out", false).link(UNKNOWN, 15)]);
    let pr = call(&mut s, m_gravity, "GravityForce");
    node(&mut s, GRAVITY, c_call, "Call_Gravity", pr, vec![map(16, "InputMap", false).link(IN_PU, 17), map(18, "OutputMap", true)]);
    let pr = s.props().finish();
    node(
        &mut s,
        SET_PU,
        c_set,
        "MapSet_PU",
        pr,
        vec![
            map(19, "Source", false).link(GRAVITY, 18),
            SynthPin::new(20, "ScaleColor.Scale Alpha", false, t_float).link(FROM_CURVE, 21),
            SynthPin::new(22, "FloatFromCurve.FloatCurve", false, 0).link(IN_CURVE, 23),
            map(24, "Dest", true),
        ],
    );
    let pr = call(&mut s, m_scale, "ScaleColor");
    node(&mut s, SCALE_COLOR, c_call, "Call_ScaleColor", pr, vec![map(25, "InputMap", false).link(SET_PU, 24), map(27, "OutputMap", true)]);
    let pr = s.props().finish();
    node(&mut s, IN_PU, c_in, "In_PU", pr, vec![map(17, "Input", true)]);
    let pr = call(&mut s, m_curve, "FloatFromCurve");
    node(
        &mut s,
        FROM_CURVE,
        c_call,
        "Call_FromCurve",
        pr,
        vec![map(26, "InputMap", false).link(SET_PU, 24), SynthPin::new(21, "Value", true, t_float)],
    );
    let mut p = s.props();
    p.object("DataInterface", CURVE);
    let pr = p.finish();
    node(&mut s, IN_CURVE, c_in, "In_Curve", pr, vec![SynthPin::new(23, "Input", true, 0)]);
    let pr = call(&mut s, m_unknown, "Wobble");
    node(&mut s, UNKNOWN, c_call, "Call_Wobble", pr, vec![map(28, "InputMap", false).link(SCALE_COLOR, 27), map(15, "OutputMap", true)]);
    let _ = fstring;
    s.build()
}

#[test]
fn synthetic_system_converts() {
    for &v in &[1008, 1012, 1016] {
        let bytes = system_package(v);
        let pkg = Package::read(&bytes).unwrap();
        assert!(pkg.problems.is_empty(), "UE5 {v}: {:?}", pkg.problems);
        let sys = read_system(&pkg).unwrap();
        assert_eq!(sys.emitters.len(), 1);
        let e = &sys.emitters[0];
        assert!(e.gpu && e.local_space);
        let names: Vec<Vec<&str>> = e.stacks.iter().map(|s| s.modules.iter().map(|m| m.name.as_str()).collect()).collect();
        assert_eq!(names, vec![vec!["SpawnRate"], vec!["InitializeParticle"], vec!["GravityForce", "ScaleColor", "Wobble"]], "UE5 {v}");
        let init = &e.stacks[1].modules[0];
        assert_eq!(init.switch("Lifetime Mode"), Some("NewEnumerator1"));
        assert_eq!(init.input("Uniform Sprite Size"), Some(&NInput::Link("User.Size".into())));
        let scale = &e.stacks[2].modules[1];
        let Some(NInput::Dynamic(d)) = scale.input("Scale Alpha") else { panic!("{scale:?}") };
        assert!(matches!(d.input("FloatCurve"), Some(NInput::Value(NValue::Curve(_)))));

        let c = convert(&sys, "NS_Synth.uasset");
        let ron = &c.ron;
        assert!(c.report.validation.is_empty(), "UE5 {v}: {:?}\n{ron}", c.report.validation);
        for want in [
            "\"Spawn Rate\": Float(25.0)",
            "\"Lifetime\": RandomRange(Float(1.0), Float(2.0))",
            "\"Sprite Size\": Dynamic(\"Multiply Vec2 by Float\", { \"A\": Vec2((1.0, 1.0)), \"B\": Link(Float, \"User.Size\") })",
            // 50 cm used as a size: 0.5 m.
            "\"User.Size\": Float(0.5)",
            // -980 cm/s² along Unreal Z: -9.8 m/s² along Y.
            "\"Gravity\": Vec3((0.0, -9.8, 0.0))",
            "\"Scale Alpha\": CurveOverAge(Float([(0.0, 1.0), (1.0, 0.0)]))",
            "Sprite((blend_mode: Additive))",
            "max_particles: 128",
        ] {
            assert!(ron.contains(want), "UE5 {v}: missing {want}\n{ron}");
        }
        let r = &c.report;
        assert_eq!((r.mapped(), r.total()), (4, 5));
        let wobble = r.modules.iter().find(|m| m.module == "Wobble").unwrap();
        assert!(matches!(&wobble.status, Status::Unmapped(w) if w.contains("Wobble")), "{:?}", wobble.status);
        assert_eq!((r.dynamic_mapped, r.dynamic_total), (1, 1));
        let md = r.markdown();
        assert!(md.contains("Modules mapped: 4/5 (80%)") && md.contains("**Wobble**"), "{md}");
    }
}

#[test]
fn cubic_curves_become_linear_keys() {
    // An ease-in-out from 0 to 1 (flat tangents) is sampled; a straight
    // segment keeps its two keys.
    let keys = vec![
        CurveKey { time: 0.0, value: 0.0, interp: 2, arrive: 0.0, leave: 0.0 },
        CurveKey { time: 1.0, value: 1.0, interp: 2, arrive: 0.0, leave: 0.0 },
    ];
    assert!((eval_curve(&keys, 0.5) - 0.5).abs() < 1e-6);
    assert!(eval_curve(&keys, 0.25) < 0.25);
    let c = curve(&NCurve { class: "NiagaraDataInterfaceCurve".into(), channels: vec![keys] }, Ty::Float, Unit::None);
    assert!(c.keys.len() > 4 && c.keys.len() < 14, "{:?}", c.keys);
    for (t, v) in &c.keys {
        let want = 3.0 * t * t - 2.0 * t * t * t;
        assert!((v[0] - want).abs() < 0.02, "{t}: {v:?}");
    }
    let lin = vec![
        CurveKey { time: 0.0, value: 2.0, interp: 0, arrive: 0.0, leave: 0.0 },
        CurveKey { time: 1.0, value: 4.0, interp: 0, arrive: 0.0, leave: 0.0 },
    ];
    // Vector curves in centimetres: Y and Z swap, values scale to metres.
    let c = curve(
        &NCurve { class: "NiagaraDataInterfaceVectorCurve".into(), channels: vec![lin.clone(), vec![], lin] },
        Ty::Vec3,
        Unit::Len,
    );
    assert_eq!(c.keys, vec![(0.0, vec![0.02, 0.02, 0.0]), (1.0, vec![0.04, 0.04, 0.0])]);
}

#[test]
fn custom_hlsl_translates_where_feasible() {
    // Reads the position and a module input, writes the velocity.
    let node = HlslNode {
        code: "#pragma once\r\nfloat3 Up = float3(0.0f, 0.0f, 1.0f);\r\nfloat h = _Position.z;\r\nif (h < _Floor) OutVelocity = Up * _Lift; else OutVelocity = _Velocity;\r\nOutVelocity.x = h > 100.0 ? 0.0 : OutVelocity.x;\r\n".into(),
        inputs: vec![
            ("_Position".into(), "NiagaraPosition".into(), Some("Particles.Position".into())),
            ("_Velocity".into(), "Vector3f".into(), Some("Particles.Velocity".into())),
            ("_Floor".into(), "NiagaraFloat".into(), Some("Module.Floor".into())),
            ("_Lift".into(), "NiagaraFloat".into(), Some("Module.Lift".into())),
        ],
        outputs: vec![("OutVelocity".into(), "Vector3f".into(), Some("Particles.Velocity".into()))],
    };
    let sys = NSystem {
        name: "S".into(),
        user_parameters: vec![("User.Lift".into(), NValue::Float(300.0))],
        emitters: vec![],
        stacks: vec![],
        warmup_time: 0.0,
        notes: vec![],
    };
    let inputs = vec![("Floor".into(), NInput::Value(NValue::Float(20.0))), ("Lift".into(), NInput::Link("User.Lift".into()))];
    let t = fx_import::hlsl::translate("Lift Below Floor", &[node.clone()], &inputs, &sys).unwrap();
    assert_eq!(t.name, "Lift_Below_Floor");
    assert!(t.source.contains("select(OutVelocity.x, 0.0, h > 100.0)"), "{}", t.source);
    assert!(t.source.contains("vec3<f32>(Position.x, Position.z, Position.y) * 100.0"), "{}", t.source);
    // It validates with naga inside an effect.
    let mut src = String::new();
    for l in t.source.lines() {
        src.push_str("        ");
        src.push_str(l);
        src.push('\n');
    }
    let fx = format!(
        "(\n    name: \"t\",\n    user_parameters: {{ \"User.Lift\": Float(3.0) }},\n    custom_modules: [\n{src}    ],\n    emitters: [(\n        name: \"E\",\n        sim_target: Gpu,\n        max_particles: 16,\n        particle_update: [(module: \"Lift_Below_Floor\", inputs: {{ \"Lift\": Link(Float, \"User.Lift\") }})],\n    )],\n)\n"
    );
    let checked = fx_core::check_source("t.fx.ron", &fx);
    assert!(checked.ok(), "{:?}\n{fx}", checked.errors().map(|d| d.to_string()).collect::<Vec<_>>());

    // Neighbour reads and custom attributes are not feasible.
    let mut reads = node.clone();
    reads.code.push_str("_Reader.GetVectorByIndex<Attribute=\"Position\">(0, ok, p);\n");
    assert!(fx_import::hlsl::translate("X", &[reads], &inputs, &sys).unwrap_err().contains("data interface"));
    let mut custom = node;
    custom.outputs[0].2 = Some("Particles._Force".into());
    assert!(fx_import::hlsl::translate("X", &[custom], &inputs, &sys).unwrap_err().contains("Particles._Force"));
}
