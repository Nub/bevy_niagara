// Shared simulation library, prepended to every generated emitter shader.
// Must match fx_core::sim bit for bit where it can (integer hashing) and
// within float tolerance elsewhere.

const WG: u32 = 256u;

// Words of the emitter state buffer (see fx_gpu::layout).
const S_COUNT: u32 = 0u;      // living particles in the current buffer
const S_NEW: u32 = 1u;        // append counter while writing the next buffer
const S_NEXT_ID: u32 = 2u;    // next UniqueID
const S_SPAWN_N: u32 = 3u;    // particles to spawn this step
const S_BASE_ID: u32 = 4u;    // UniqueID of the first one
const S_REGULAR_N: u32 = 5u;  // of S_SPAWN_N, the emitter stack's request; the rest come from events
const S_HANDLER_END: u32 = 8u; // + h: spawn index where handler h's particles end

struct Frame {
    dt: f32,
    time: f32,
    emitter_age: f32,
    frame: u32,
    spawn_request: u32,
    seed: u32,
    emitter: u32,
    capacity: u32,
    owner_velocity: vec3<f32>,  // Engine.Owner.Velocity; padded to 16 bytes
    user: array<vec4<f32>, 16>,
    looped_age: f32,            // Emitter.LoopedAge
    loop_count: u32,            // Emitter.LoopCount
    normalized_loop_age: f32,   // Emitter.NormalizedLoopAge
    _pad: u32,
    // Where this instance lives in the batch buffers (fx_gpu::GpuBatch), in words.
    src_base: u32,              // particles read (the current copy)
    dst_base: u32,              // particles written (the other copy)
    state_base: u32,
    ev_count_base: u32,
    ev_data_base: u32,
    grid_base: u32,
    slot: u32,                  // instance slot (scene, transforms)
    n_active: u32,              // instances stepped by this dispatch
}

// One Frame per instance stepped (dynamic offset per step); `fr` is this
// invocation's, loaded by `load_frame` at the start of every kernel.
@group(0) @binding(0) var<storage, read> frames: array<Frame>;
var<private> fr: Frame;
// Every instance's particles: two copies per slot, each `region` words (see fx_gpu::GpuBatch).
@group(0) @binding(1) var<storage, read_write> parts: array<f32>;

fn load_frame(j: u32) {
    fr = frames[j];
}
@group(0) @binding(3) var<storage, read_write> state: array<atomic<u32>>;
// Indirect dispatch args: update at words 0..3, spawn at 4..7. Bound only to the
// kernels that write them, never to the indirect dispatches that read them.
@group(1) @binding(0) var<storage, read_write> args: array<atomic<u32>>;
// Events of the whole effect: one counter per (emitter, kind) region, and each
// region's events (EVENT_WORDS each) and sorted order (see fx_gpu::events).
@group(0) @binding(4) var<storage, read_write> event_counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> event_data: array<u32>;
// Grid collections of the emitter (fx_core::custom): each grid twice (ping-pong),
// attribute word w of cell c at w * cells + c within a copy.
@group(0) @binding(6) var<storage, read_write> grid: array<f32>;
// The iteration of the simulation stage being run (dynamic offset per dispatch).
struct Iteration {
    j: u32,
}
@group(0) @binding(7) var<uniform> it: Iteration;

fn grid_ld(w: u32) -> f32 {
    return grid[fr.grid_base + w];
}

fn grid_st(w: u32, v: f32) {
    grid[fr.grid_base + w] = v;
}

// ---- random numbers (fx_core::sim::rng; pcg and to_unit are in fx_core's lib.wgsl) ----

struct Key {
    particle: u32,
    frame: u32,
    module_id: u32,
}

fn rand_unit(k: Key, slot: u32) -> f32 {
    var h = 0x9E3779B9u;
    h = pcg(h ^ fr.seed);
    h = pcg(h ^ fr.emitter);
    h = pcg(h ^ k.particle);
    h = pcg(h ^ k.frame);
    h = pcg(h ^ k.module_id);
    h = pcg(h ^ slot);
    return to_unit(h);
}

// Custom WGSL modules draw with rand(slot): the key of the module call being run.
var<private> cm_key: Key;

fn rand(slot: u32) -> f32 {
    return rand_unit(cm_key, 0x40000000u + slot);
}

// ---- particles: structure of arrays, word w of particle i at w * capacity + i ----
// Words 0..27 are the base attributes every emitter stores. Words 28..66 (Scale,
// MeshOrientation, the Ribbon attributes, LightRadius, DynamicMaterialParameter,
// SubImageIndex, Initial.Scale, Initial.RibbonWidth, CameraOffset, SpriteFacing,
// SpriteAlignment, DynamicMaterialParameter1, RotationalVelocity,
// Physics.RotationalForce, Physics.RotationalDrag)
// exist only when the generated `EXT` is true; otherwise they stay at their
// defaults in registers and the buffers end at word 27.

struct Particle {
    position: vec3<f32>,
    velocity: vec3<f32>,
    color: vec4<f32>,
    sprite_size: vec2<f32>,
    sprite_rotation: f32,
    mass: f32,
    lifetime: f32,
    age: f32,
    normalized_age: f32,
    initial_color: vec4<f32>,
    initial_sprite_size: vec2<f32>,
    force: vec3<f32>,
    drag: f32,
    scale: vec3<f32>,
    mesh_orientation: vec4<f32>,
    ribbon_id: i32,
    ribbon_width: f32,
    ribbon_twist: f32,
    ribbon_link_order: f32,
    light_radius: f32,
    dynamic_material_parameter: vec4<f32>,
    sub_image_index: f32,
    initial_scale: vec3<f32>,
    initial_ribbon_width: f32,
    camera_offset: f32,
    sprite_facing: vec3<f32>,
    sprite_alignment: vec3<f32>,
    dynamic_material_parameter_1: vec4<f32>,
    rotational_velocity: vec3<f32>,
    rotational_force: vec3<f32>,
    rotational_drag: f32,
    id: u32,
    alive: bool,
}

fn new_particle(id: u32) -> Particle {
    var p: Particle;
    p.position = vec3<f32>(0.0);
    p.velocity = vec3<f32>(0.0);
    p.color = vec4<f32>(1.0);
    p.sprite_size = vec2<f32>(0.1);
    p.sprite_rotation = 0.0;
    p.mass = 1.0;
    p.lifetime = 2.0;
    p.age = 0.0;
    p.normalized_age = 0.0;
    p.initial_color = vec4<f32>(1.0);
    p.initial_sprite_size = vec2<f32>(0.1);
    p.force = vec3<f32>(0.0);
    p.drag = 0.0;
    p.scale = vec3<f32>(1.0);
    p.mesh_orientation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    p.ribbon_id = 0;
    p.ribbon_width = 0.1;
    p.ribbon_twist = 0.0;
    p.ribbon_link_order = 0.0;
    p.light_radius = 1.0;
    p.dynamic_material_parameter = vec4<f32>(0.0);
    p.sub_image_index = 0.0;
    p.initial_scale = vec3<f32>(1.0);
    p.initial_ribbon_width = 0.1;
    p.camera_offset = 0.0;
    p.sprite_facing = vec3<f32>(0.0);
    p.sprite_alignment = vec3<f32>(0.0);
    p.dynamic_material_parameter_1 = vec4<f32>(0.0);
    p.rotational_velocity = vec3<f32>(0.0);
    p.rotational_force = vec3<f32>(0.0);
    p.rotational_drag = 0.0;
    p.id = id;
    p.alive = true;
    return p;
}

// Particle simulation stages read the particles in place (in dst).
var<private> read_dst: bool;

fn ld(w: u32, i: u32) -> f32 {
    if (read_dst) {
        return parts[fr.dst_base + w * fr.capacity + i];
    }
    return parts[fr.src_base + w * fr.capacity + i];
}

fn load_particle(i: u32) -> Particle {
    var p = new_particle(bitcast<u32>(ld(27u, i)));
    p.position = vec3<f32>(ld(0u, i), ld(1u, i), ld(2u, i));
    p.velocity = vec3<f32>(ld(3u, i), ld(4u, i), ld(5u, i));
    p.color = vec4<f32>(ld(6u, i), ld(7u, i), ld(8u, i), ld(9u, i));
    p.sprite_size = vec2<f32>(ld(10u, i), ld(11u, i));
    p.sprite_rotation = ld(12u, i);
    p.mass = ld(13u, i);
    p.lifetime = ld(14u, i);
    p.age = ld(15u, i);
    p.normalized_age = ld(16u, i);
    p.initial_color = vec4<f32>(ld(17u, i), ld(18u, i), ld(19u, i), ld(20u, i));
    p.initial_sprite_size = vec2<f32>(ld(21u, i), ld(22u, i));
    p.force = vec3<f32>(ld(23u, i), ld(24u, i), ld(25u, i));
    p.drag = ld(26u, i);
    if (EXT) {
        p.scale = vec3<f32>(ld(28u, i), ld(29u, i), ld(30u, i));
        p.mesh_orientation = vec4<f32>(ld(31u, i), ld(32u, i), ld(33u, i), ld(34u, i));
        p.ribbon_id = bitcast<i32>(ld(35u, i));
        p.ribbon_width = ld(36u, i);
        p.ribbon_twist = ld(37u, i);
        p.ribbon_link_order = ld(38u, i);
        p.light_radius = ld(39u, i);
        p.dynamic_material_parameter = vec4<f32>(ld(40u, i), ld(41u, i), ld(42u, i), ld(43u, i));
        p.sub_image_index = ld(44u, i);
        p.initial_scale = vec3<f32>(ld(45u, i), ld(46u, i), ld(47u, i));
        p.initial_ribbon_width = ld(48u, i);
        p.camera_offset = ld(49u, i);
        p.sprite_facing = vec3<f32>(ld(50u, i), ld(51u, i), ld(52u, i));
        p.sprite_alignment = vec3<f32>(ld(53u, i), ld(54u, i), ld(55u, i));
        p.dynamic_material_parameter_1 = vec4<f32>(ld(56u, i), ld(57u, i), ld(58u, i), ld(59u, i));
        p.rotational_velocity = vec3<f32>(ld(60u, i), ld(61u, i), ld(62u, i));
        p.rotational_force = vec3<f32>(ld(63u, i), ld(64u, i), ld(65u, i));
        p.rotational_drag = ld(66u, i);
    }
    return p;
}

fn st(w: u32, i: u32, v: f32) {
    parts[fr.dst_base + w * fr.capacity + i] = v;
}

fn store_particle(i: u32, p: Particle) {
    st(0u, i, p.position.x); st(1u, i, p.position.y); st(2u, i, p.position.z);
    st(3u, i, p.velocity.x); st(4u, i, p.velocity.y); st(5u, i, p.velocity.z);
    st(6u, i, p.color.x); st(7u, i, p.color.y); st(8u, i, p.color.z); st(9u, i, p.color.w);
    st(10u, i, p.sprite_size.x); st(11u, i, p.sprite_size.y);
    st(12u, i, p.sprite_rotation);
    st(13u, i, p.mass);
    st(14u, i, p.lifetime);
    st(15u, i, p.age);
    st(16u, i, p.normalized_age);
    st(17u, i, p.initial_color.x); st(18u, i, p.initial_color.y); st(19u, i, p.initial_color.z); st(20u, i, p.initial_color.w);
    st(21u, i, p.initial_sprite_size.x); st(22u, i, p.initial_sprite_size.y);
    st(23u, i, p.force.x); st(24u, i, p.force.y); st(25u, i, p.force.z);
    st(26u, i, p.drag);
    st(27u, i, bitcast<f32>(p.id));
    if (EXT) {
        st(28u, i, p.scale.x); st(29u, i, p.scale.y); st(30u, i, p.scale.z);
        st(31u, i, p.mesh_orientation.x); st(32u, i, p.mesh_orientation.y);
        st(33u, i, p.mesh_orientation.z); st(34u, i, p.mesh_orientation.w);
        st(35u, i, bitcast<f32>(p.ribbon_id));
        st(36u, i, p.ribbon_width);
        st(37u, i, p.ribbon_twist);
        st(38u, i, p.ribbon_link_order);
        st(39u, i, p.light_radius);
        st(40u, i, p.dynamic_material_parameter.x); st(41u, i, p.dynamic_material_parameter.y);
        st(42u, i, p.dynamic_material_parameter.z); st(43u, i, p.dynamic_material_parameter.w);
        st(44u, i, p.sub_image_index);
        st(45u, i, p.initial_scale.x); st(46u, i, p.initial_scale.y); st(47u, i, p.initial_scale.z);
        st(48u, i, p.initial_ribbon_width);
        st(49u, i, p.camera_offset);
        st(50u, i, p.sprite_facing.x); st(51u, i, p.sprite_facing.y); st(52u, i, p.sprite_facing.z);
        st(53u, i, p.sprite_alignment.x); st(54u, i, p.sprite_alignment.y); st(55u, i, p.sprite_alignment.z);
        st(56u, i, p.dynamic_material_parameter_1.x); st(57u, i, p.dynamic_material_parameter_1.y);
        st(58u, i, p.dynamic_material_parameter_1.z); st(59u, i, p.dynamic_material_parameter_1.w);
        st(60u, i, p.rotational_velocity.x); st(61u, i, p.rotational_velocity.y); st(62u, i, p.rotational_velocity.z);
        st(63u, i, p.rotational_force.x); st(64u, i, p.rotational_force.y); st(65u, i, p.rotational_force.z);
        st(66u, i, p.rotational_drag);
    }
}

fn groups(n: u32) -> u32 {
    return (n + WG - 1u) / WG;
}

// ---- the step: begin → update (indirect) → spawn_prep → spawn (indirect) → finish ----
//
// Every instance of a batch steps in the same dispatches: the single-thread
// kernels run one thread per instance and raise the indirect arguments to
// the largest instance's (x) times the instances (y); the per-particle
// kernels take their instance from workgroup_id.y. The host clears `args`
// before `begin`.

const WG_INST: u32 = 64u;

fn raise_args(at: u32, x: u32) {
    atomicMax(&args[at], x);
    atomicStore(&args[at + 1u], fr.n_active);
    atomicStore(&args[at + 2u], 1u);
}

@compute @workgroup_size(WG_INST)
fn begin(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frames[0].n_active) {
        return;
    }
    load_frame(gid.x);
    let n = atomicLoad(&state[S_COUNT]);
    raise_args(0u, groups(n));
    atomicStore(&state[S_NEW], 0u);
    reset_events();
}

@compute @workgroup_size(WG)
fn update(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    load_frame(wid.y);
    let i = gid.x;
    if (i >= atomicLoad(&state[S_COUNT])) {
        return;
    }
    grids_at_update();
    var p = load_particle(i);
    ev_reset();
    update_particle(&p);
    every_handlers(&p);
    write_events(p);
    if (p.alive) {
        store_particle(atomicAdd(&state[S_NEW], 1u), p);
    }
}

@compute @workgroup_size(WG_INST)
fn spawn_prep(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frames[0].n_active) {
        return;
    }
    load_frame(gid.x);
    let alive = atomicLoad(&state[S_NEW]);
    let room = fr.capacity - min(alive, fr.capacity);
    let n = spawn_counts(room);
    let base = atomicLoad(&state[S_NEXT_ID]);
    atomicStore(&state[S_BASE_ID], base);
    atomicStore(&state[S_NEXT_ID], base + n);
    atomicStore(&state[S_SPAWN_N], n);
    raise_args(4u, groups(n));
    raise_args(8u, groups(max_event_count()));
}

@compute @workgroup_size(WG)
fn spawn(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    load_frame(wid.y);
    let k = gid.x;
    if (k >= atomicLoad(&state[S_SPAWN_N])) {
        return;
    }
    grids_at_update();
    var p = new_particle(atomicLoad(&state[S_BASE_ID]) + k);
    spawn_particle(&p);
    spawn_event(k, &p);
    if (p.alive) {
        store_particle(atomicAdd(&state[S_NEW], 1u), p);
    }
}

// Sorts this emitter's events by source UniqueID (after update, before spawn).
@compute @workgroup_size(WG)
fn event_rank(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    load_frame(wid.y);
    rank_events(gid.x, lid);
}

@compute @workgroup_size(WG_INST)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frames[0].n_active) {
        return;
    }
    load_frame(gid.x);
    let n = atomicLoad(&state[S_NEW]);
    atomicStore(&state[S_COUNT], n);
    // Particle simulation stages run over the living particles.
    raise_args(12u, groups(n));
}

// ---- collisions (fx_core::sim::collide_plane) ----

const RESPONSE_STOP: i32 = 1;
const RESPONSE_KILL: i32 = 2;

fn collide_plane(p: ptr<function, Particle>, n: vec3<f32>, o: vec3<f32>, radius: f32,
                 response: i32, restitution: f32, friction: f32, rest_speed: f32) -> bool {
    let d = dot((*p).position - o, n) - radius;
    if (!(d < 0.0)) {
        return false;
    }
    (*p).position -= n * d;
    let vn = dot((*p).velocity, n);
    if (vn >= 0.0) {
        return false;
    }
    let speed_in = -vn;
    let impact = speed_in >= rest_speed;
    if (response == RESPONSE_KILL) {
        (*p).alive = false;
        return true;
    }
    if (response == RESPONSE_STOP) {
        (*p).velocity = vec3<f32>(0.0);
        return impact;
    }
    var vt = (*p).velocity - n * vn;
    let t = length(vt);
    if (t > 0.0) {
        vt *= max(1.0 - friction * speed_in / t, 0.0);
    }
    let out = select(0.0, speed_in * restitution, impact);
    (*p).velocity = vt + n * out;
    return impact;
}

// ---- events (fx_core::events) ----

const EVENT_WORDS: u32 = 12u;

struct FxEvent {
    position: vec3<f32>,
    velocity: vec3<f32>,
    normal: vec3<f32>,
    age: f32,
    id: u32,
}

// The event a handler runs for, and the random-key frame of its modules.
var<private> cur_event: FxEvent;
var<private> ev_frame: u32;
// What this particle's update found and will report.
var<private> ev_collided: bool;
var<private> ev_normal: vec3<f32>;
var<private> ev_death: bool;
var<private> ev_has: array<bool, 3>;
var<private> ev_pending: array<FxEvent, 3>;

fn ev_reset() {
    ev_collided = false;
    ev_death = false;
    ev_has = array<bool, 3>(false, false, false);
}

fn make_event(p: Particle, normal: vec3<f32>) -> FxEvent {
    return FxEvent(p.position, p.velocity, normal, p.age, p.id);
}

fn load_event(data: u32, sorted: u32, rank: u32) -> FxEvent {
    let b = data + event_data[sorted + rank] * EVENT_WORDS;
    let w = array<f32, 10>(
        bitcast<f32>(event_data[b]), bitcast<f32>(event_data[b + 1u]), bitcast<f32>(event_data[b + 2u]),
        bitcast<f32>(event_data[b + 3u]), bitcast<f32>(event_data[b + 4u]), bitcast<f32>(event_data[b + 5u]),
        bitcast<f32>(event_data[b + 6u]), bitcast<f32>(event_data[b + 7u]), bitcast<f32>(event_data[b + 8u]),
        bitcast<f32>(event_data[b + 9u]),
    );
    return FxEvent(vec3<f32>(w[0], w[1], w[2]), vec3<f32>(w[3], w[4], w[5]), vec3<f32>(w[6], w[7], w[8]), w[9], event_data[b + 10u]);
}

fn write_event(region: u32, data: u32, cap: u32, e: FxEvent) {
    let slot = atomicAdd(&event_counts[region], 1u);
    if (slot >= cap) {
        return;
    }
    let b = data + slot * EVENT_WORDS;
    event_data[b] = bitcast<u32>(e.position.x);
    event_data[b + 1u] = bitcast<u32>(e.position.y);
    event_data[b + 2u] = bitcast<u32>(e.position.z);
    event_data[b + 3u] = bitcast<u32>(e.velocity.x);
    event_data[b + 4u] = bitcast<u32>(e.velocity.y);
    event_data[b + 5u] = bitcast<u32>(e.velocity.z);
    event_data[b + 6u] = bitcast<u32>(e.normal.x);
    event_data[b + 7u] = bitcast<u32>(e.normal.y);
    event_data[b + 8u] = bitcast<u32>(e.normal.z);
    event_data[b + 9u] = bitcast<u32>(e.age);
    event_data[b + 10u] = e.id;
    event_data[b + 11u] = 0u;
}

// Events a region holds (at most its capacity).
fn event_count(region: u32, cap: u32) -> u32 {
    return min(atomicLoad(&event_counts[region]), cap);
}

// Events a handler takes: all, or the first `max_events` (0 = all).
fn events_taken(region: u32, cap: u32, max_events: u32) -> u32 {
    let n = event_count(region, cap);
    return select(n, min(n, max_events), max_events > 0u);
}

// Sorting a region by source UniqueID: each event counts the events with a
// smaller id (ids are unique within a region), 256 at a time from shared memory.
var<workgroup> rank_tile: array<u32, 256>;
var<workgroup> rank_n: u32;

fn rank_region(i: u32, lid: u32, region: u32, data: u32, sorted: u32, cap: u32) {
    workgroupBarrier();
    if (lid == 0u) {
        rank_n = event_count(region, cap);
    }
    let n = workgroupUniformLoad(&rank_n);
    let valid = i < n;
    var key = 0u;
    if (valid) {
        key = event_data[data + i * EVENT_WORDS + 10u];
    }
    var rank = 0u;
    for (var base = 0u; base < n; base += WG) {
        let j = base + lid;
        if (j < n) {
            rank_tile[lid] = event_data[data + j * EVENT_WORDS + 10u];
        }
        workgroupBarrier();
        let m = min(WG, n - base);
        for (var t = 0u; t < m; t++) {
            rank += select(0u, 1u, rank_tile[t] < key);
        }
        workgroupBarrier();
    }
    if (valid) {
        event_data[sorted + rank] = i;
    }
}

// Spawn-stage handler counts: `n` of `room` particles for a handler taking
// `taken` events of `per` particles each (fx_core::sim's u64 product, clamped).
fn handler_spawns(taken: u32, per: u32, room: u32) -> u32 {
    if (per == 0u) {
        return 0u;
    }
    return select(taken * per, room, taken > room / per);
}
