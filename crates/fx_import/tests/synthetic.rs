//! Reader tests on small synthetic packages written by `fx_import::synth`,
//! across the UE5 versions that change the layout.

use fx_import::synth::{fstring, Synth, T};
use fx_import::{dump, find, DumpOptions, Package, PackageIndex, Value};

/// UE5 object versions: 5.1 (old tags), 5.4 preview (tag extensions), 5.4
/// (complete type names), 5.5+, and newest (saved hash, import hierarchies).
const VERSIONS: &[i32] = &[1008, 1011, 1012, 1013, 1016, 1018];

fn guid_bytes(words: [u32; 4]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// A package with one NiagaraSystem export using most property kinds, and
/// an emitter export inside it with trailing native bytes.
fn sample(ue5: i32) -> Vec<u8> {
    let mut s = Synth::new(ue5);
    let script_pkg = s.import("/Script/CoreUObject", "Package", 0, "/Script/Niagara");
    let system_class = s.import("/Script/CoreUObject", "Class", script_pkg, "NiagaraSystem");
    let emitter_class = s.import("/Script/CoreUObject", "Class", script_pkg, "NiagaraEmitter");
    let mat = s.import("/Script/Engine", "Material", 0, "M_Smoke");

    let mut p = s.props();
    let inner_a = p.nested(|q| {
        q.int("Count", 3);
    });
    let inner_b = p.nested(|q| {
        q.int("Count", 4).name("Label", "Second");
    });
    let v1 = p.vector_bytes([1.0, 2.0, 3.0]);
    let v2 = p.vector_bytes([-4.5, 0.0, 9.25]);
    p.int("WarmupTickCount", 12)
        .float("WarmupTime", 0.5)
        .bool("bFixedBounds", true)
        .bool("bDisabled", false)
        .name("Category", "Fire_2")
        .str("Description", "a fire")
        .object("Material", mat)
        .byte_enum("Mode", "ENiagaraMode", "ENiagaraMode::Burst")
        .vector("Offset", [1.5, -2.0, 100.0])
        .linear_color("Tint", [1.0, 0.5, 0.25, 1.0])
        .tagged_struct("Settings", "NiagaraSettings", |q| {
            q.float("Rate", 30.0).int_array("Seeds", &[7, 8, 9]);
        })
        .int_array("Empty", &[])
        .struct_array("Entries", "NiagaraEntry", &[inner_a, inner_b])
        .struct_array("Points", "Vector", &[v1, v2])
        .name_int_map("Offsets", &[("A", 1), ("B", 2)]);
    let props = p.finish();
    let system = s.export(system_class, 0, "NS_Test", 0, props, Vec::new());

    let mut p = s.props();
    p.str("UniqueEmitterName", "Sparks");
    let props = p.finish();
    s.export(emitter_class, system, "Sparks", 0, props, vec![0xAB; 37]);
    s.build()
}

#[test]
fn reads_every_property_kind_across_versions() {
    for &v in VERSIONS {
        let bytes = sample(v);
        let pkg = Package::read(&bytes).unwrap_or_else(|e| panic!("UE5 {v}: {e}"));
        assert!(pkg.problems.is_empty(), "UE5 {v}: {:?}", pkg.problems);
        assert_eq!(pkg.summary.ue5_version, v);
        assert_eq!(pkg.summary.package_name, "/Game/Synth");
        assert_eq!(pkg.imports.len(), 4);
        assert_eq!(pkg.exports.len(), 2);
        assert_eq!(pkg.export_class(0), "NiagaraSystem");
        assert_eq!(pkg.export_class(1), "NiagaraEmitter");
        assert_eq!(pkg.exports[1].outer, PackageIndex(1));
        assert_eq!(pkg.path(PackageIndex(2)), "NS_Test.Sparks");

        let props = &pkg.exports[0].object.properties;
        assert_eq!(find(props, "WarmupTickCount"), Some(&Value::Int(12)), "UE5 {v}");
        assert_eq!(find(props, "WarmupTime"), Some(&Value::Float(0.5)));
        assert_eq!(find(props, "bFixedBounds"), Some(&Value::Bool(true)));
        assert_eq!(find(props, "bDisabled"), Some(&Value::Bool(false)));
        assert_eq!(find(props, "Category"), Some(&Value::Name("Fire_2".into())));
        assert_eq!(find(props, "Description"), Some(&Value::Str("a fire".into())));
        assert_eq!(find(props, "Material"), Some(&Value::Object(PackageIndex(-4))));
        assert_eq!(find(props, "Mode"), Some(&Value::Enum("ENiagaraMode::Burst".into())));

        let off = find(props, "Offset").unwrap();
        let xyz: Vec<f64> = ["X", "Y", "Z"].iter().map(|c| off.field(c).unwrap().as_f64().unwrap()).collect();
        assert_eq!(xyz, vec![1.5, -2.0, 100.0], "UE5 {v}");
        let tint = find(props, "Tint").unwrap();
        assert_eq!(tint.field("G"), Some(&Value::Float(0.5)));

        let settings = find(props, "Settings").unwrap();
        assert_eq!(settings.field("Rate"), Some(&Value::Float(30.0)));
        assert_eq!(
            settings.field("Seeds"),
            Some(&Value::Array(vec![Value::Int(7), Value::Int(8), Value::Int(9)]))
        );
        assert_eq!(find(props, "Empty"), Some(&Value::Array(vec![])));

        let Some(Value::Array(entries)) = find(props, "Entries") else { panic!("Entries") };
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].field("Count"), Some(&Value::Int(4)));
        assert_eq!(entries[1].field("Label"), Some(&Value::Name("Second".into())));

        let Some(Value::Array(points)) = find(props, "Points") else { panic!("Points") };
        assert_eq!(points[1].field("Z").unwrap().as_f64(), Some(9.25));

        let Some(Value::Map(m)) = find(props, "Offsets") else { panic!("Offsets") };
        assert_eq!(m, &vec![(Value::Name("A".into()), Value::Int(1)), (Value::Name("B".into()), Value::Int(2))]);

        let emitter = &pkg.exports[1].object;
        assert_eq!(find(&emitter.properties, "UniqueEmitterName"), Some(&Value::Str("Sparks".into())));
        assert_eq!(emitter.native_size, 37, "UE5 {v}");
        assert_eq!(pkg.exports[0].object.native_size, 0);
    }
}

#[test]
fn dump_shows_tree_and_values() {
    let pkg = Package::read(&sample(1012)).unwrap();
    let text = dump(&pkg, &DumpOptions::default());
    for want in [
        "Package /Game/Synth",
        "[1] NiagaraSystem NS_Test",
        "    [2] NiagaraEmitter Sparks",
        "WarmupTickCount: IntProperty = 12",
        "Offset: StructProperty(Vector) = Vector(X=1.5, Y=-2.0, Z=100.0)",
        "Material: ObjectProperty = Material'M_Smoke'",
        "Mode: ByteProperty(ENiagaraMode) = ENiagaraMode::Burst",
        "<native data: 37 bytes>",
    ] {
        assert!(text.contains(want), "missing {want:?} in\n{text}");
    }
    assert!(!text.contains("Problems"));

    // --only keeps the matching export and its subtree.
    let only = dump(
        &pkg,
        &DumpOptions {
            only: vec!["NiagaraEmitter".into()],
            imports: false,
            ..Default::default()
        },
    );
    assert!(only.contains("NiagaraEmitter Sparks") && !only.contains("WarmupTickCount"), "{only}");
}

#[test]
fn undecodable_value_is_kept_raw_and_reported() {
    for &v in &[1008, 1012] {
        let mut s = Synth::new(v);
        let class = s.import("/Script/CoreUObject", "Class", 0, "NiagaraScript");
        let mut p = s.props();
        p.raw("Weird", &T::Struct("SomethingNative"), &[1, 2, 3], false, true);
        p.int("After", 5);
        let props = p.finish();
        s.export(class, 0, "Script", 0, props, Vec::new());
        let pkg = Package::read(&s.build()).unwrap();
        let props = &pkg.exports[0].object.properties;
        assert!(matches!(find(props, "Weird"), Some(Value::Raw { bytes, .. }) if bytes == &[1, 2, 3]), "UE5 {v}");
        assert_eq!(find(props, "After"), Some(&Value::Int(5)), "UE5 {v}");
        assert_eq!(pkg.problems.len(), 1, "UE5 {v}: {:?}", pkg.problems);
        assert_eq!(pkg.problems[0].path, "Weird");
    }
}

#[test]
fn old_format_struct_maps_are_resolved_by_trying_layouts() {
    // Before UE5 1012 map tags do not name key/value structs.
    let mut s = Synth::new(1008);
    let class = s.import("/Script/CoreUObject", "Class", 0, "NiagaraGraph");
    let mut p = s.props();
    let mut b = Vec::new();
    b.extend_from_slice(&0i32.to_le_bytes());
    b.extend_from_slice(&2i32.to_le_bytes());
    for k in 0..2u32 {
        b.extend(guid_bytes([k, 1, 2, 3]));
        b.extend(guid_bytes([k, 9, 9, 9]));
    }
    let ty = T::Map(Box::new(T::Struct("Guid")), Box::new(T::Struct("Guid")));
    p.raw("PinOutputToPinDefaultPersistentId", &ty, &b, false, false);
    let props = p.finish();
    s.export(class, 0, "Graph", 0, props, Vec::new());
    let pkg = Package::read(&s.build()).unwrap();
    assert!(pkg.problems.is_empty(), "{:?}", pkg.problems);
    let Some(Value::Map(m)) = find(&pkg.exports[0].object.properties, "PinOutputToPinDefaultPersistentId") else {
        panic!()
    };
    assert_eq!(m.len(), 2);
    assert_eq!(m[1].0.field("Guid"), Some(&Value::Guid(fx_import::Guid([1, 1, 2, 3]))));
}

#[test]
fn niagara_variable_is_native() {
    // FNiagaraVariable: FName + tagged NiagaraTypeDefinition + byte array.
    for &v in &[1008, 1012] {
        let mut s = Synth::new(v);
        let class = s.import("/Script/CoreUObject", "Class", 0, "NiagaraScript");
        let float_struct = s.import("/Script/CoreUObject", "ScriptStruct", 0, "NiagaraFloat");
        let mut p = s.props();
        let mut var = p.fname("Particles.Lifetime");
        var.extend(p.nested(|q| {
            q.object("ClassStructOrEnum", float_struct);
        }));
        var.extend_from_slice(&4i32.to_le_bytes());
        var.extend_from_slice(&2.5f32.to_le_bytes());
        p.struct_array("Parameters", "NiagaraVariable", &[var]);
        let props = p.finish();
        s.export(class, 0, "Script", 0, props, Vec::new());
        let pkg = Package::read(&s.build()).unwrap();
        assert!(pkg.problems.is_empty(), "UE5 {v}: {:?}", pkg.problems);
        let Some(Value::Array(a)) = find(&pkg.exports[0].object.properties, "Parameters") else { panic!() };
        assert_eq!(a[0].field("Name"), Some(&Value::Name("Particles.Lifetime".into())));
        let td = a[0].field("TypeDef").unwrap();
        assert_eq!(td.field("ClassStructOrEnum"), Some(&Value::Object(PackageIndex(-2))));
        assert_eq!(a[0].field("VarData"), Some(&Value::Bytes(2.5f32.to_le_bytes().to_vec())));
    }
}

#[test]
fn rejects_what_it_cannot_read() {
    let good = sample(1012);

    let mut bad = good.clone();
    bad[0] = 0;
    assert!(Package::read(&bad).unwrap_err().msg.contains("not an Unreal package"));

    // Unversioned (cooked) header.
    let mut bad = good.clone();
    bad[12..20].fill(0);
    assert!(Package::read(&bad).unwrap_err().msg.contains("unversioned"));

    // UE4 package (legacy version -7).
    let mut bad = good.clone();
    bad[4..8].copy_from_slice(&(-7i32).to_le_bytes());
    assert!(Package::read(&bad).is_err());

    // Truncated anywhere: an error, never a panic.
    for cut in (0..good.len()).step_by(7) {
        let _ = Package::read(&good[..cut]);
    }

    // Garbage in the property area: errors or problems, never a panic.
    let pkg = Package::read(&good).unwrap();
    let start = pkg.exports[0].serial_offset;
    for i in 0..pkg.exports[0].serial_size {
        let mut bad = good.clone();
        bad[start + i] ^= 0x5A;
        let _ = Package::read(&bad);
    }
}

#[test]
fn fstring_utf16() {
    let mut s = Synth::new(1012);
    let class = s.import("/Script/CoreUObject", "Class", 0, "NiagaraSystem");
    let mut p = s.props();
    // UTF-16 FString: negative length.
    let text = "Feu é";
    let units: Vec<u16> = text.encode_utf16().chain([0]).collect();
    let mut b = Vec::new();
    b.extend_from_slice(&(-(units.len() as i32)).to_le_bytes());
    for u in units {
        b.extend_from_slice(&u.to_le_bytes());
    }
    p.raw("Wide", &T::Str, &b, false, false);
    let mut ascii = Vec::new();
    fstring(&mut ascii, "plain");
    p.raw("Narrow", &T::Str, &ascii, false, false);
    let props = p.finish();
    s.export(class, 0, "NS", 0, props, Vec::new());
    let pkg = Package::read(&s.build()).unwrap();
    let props = &pkg.exports[0].object.properties;
    assert_eq!(find(props, "Wide"), Some(&Value::Str(text.into())));
    assert_eq!(find(props, "Narrow"), Some(&Value::Str("plain".into())));
}
