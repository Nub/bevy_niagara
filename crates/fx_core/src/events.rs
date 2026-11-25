//! Niagara events: what particles report (death, collision, location) and
//! how receiving emitters handle them.
//!
//! A source emitter's particles write events with the Generate ... Event
//! modules in `particle_update`. Each step, an emitter's events are cleared
//! when it starts updating and sorted by the source particle's UniqueID once
//! its update is done, so the order (and which events are kept when a handler
//! caps them) is the same on the CPU and the GPU. A receiving emitter's
//! `event_handlers` read the source's events as they are when the receiver
//! steps: events of an earlier emitter (in `emitters` order) or of itself
//! arrive in the same step, those of a later emitter one step later.

use serde::Serialize;

use crate::model::{ModuleCall, Span};
use crate::render::{IdentEnum, ident_enum};

ident_enum! {
    /// Which event a handler receives (Niagara's event generators).
    EventKind {
        /// A particle died this step (Generate Death Event).
        Death,
        /// A particle hit something this step (Generate Collision Event, after a Collision module).
        Collision,
        /// A particle reports where it is (Generate Location Event), each step.
        Location,
    }
}

impl EventKind {
    pub const ALL: [EventKind; 3] = [EventKind::Death, EventKind::Collision, EventKind::Location];

    /// Index of the kind (0..3), as stored in cooked handlers and event buffers.
    pub fn index(self) -> usize {
        self as usize
    }

    pub fn from_index(i: u8) -> EventKind {
        Self::ALL.get(i as usize).copied().unwrap_or(EventKind::Death)
    }

    /// The module that writes this event.
    pub fn generator(self) -> &'static str {
        match self {
            EventKind::Death => "Generate Death Event",
            EventKind::Collision => "Generate Collision Event",
            EventKind::Location => "Generate Location Event",
        }
    }

    pub fn name(self) -> &'static str {
        Self::VARIANTS[self.index()].0
    }
}

ident_enum! {
    /// What a handler does with each event (Niagara's Execution Mode).
    EventExecutionMode {
        /// Spawns `spawn_number` particles per event; they run `particle_spawn`, then the handler's modules.
        SpawnedParticles,
        /// Runs the handler's modules on every living particle, once per event.
        EveryParticle,
    }
}

/// One entry of an emitter's `event_handlers: [ ... ]`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EventHandlerSource {
    /// Name of the emitter whose events are received (may be this one).
    pub source: String,
    pub event: EventKind,
    pub execution_mode: EventExecutionMode,
    /// Particles spawned per event (SpawnedParticles).
    pub spawn_number: u32,
    /// Events handled per step at most (the first ones by source UniqueID); 0 = all.
    pub max_events_per_frame: u32,
    /// The handler's module stack; reads `Event.` parameters.
    pub modules: Vec<ModuleCall>,
    #[serde(skip)]
    pub span: Span,
}

/// Fields of an event handler, for `fxc modules` and error help: `(name, form, default, meaning)`.
pub const HANDLER_FIELDS: &[(&str, &str, &str, &str)] = &[
    ("source", "\"Emitter Name\"", "(required)", "the emitter whose events are received; may be this one"),
    ("event", "Death, Collision or Location", "Death", "which events"),
    ("execution_mode", "SpawnedParticles or EveryParticle", "SpawnedParticles", "spawn particles per event, or run `modules` on every living particle once per event"),
    ("spawn_number", "whole number", "1", "particles spawned per event (SpawnedParticles)"),
    ("max_events_per_frame", "whole number", "0", "events handled per step at most, lowest source UniqueID first; 0 = all"),
    ("modules", "[ module, ... ]", "[]", "modules run on the spawned (or every) particle; they read Event.Position, Event.Velocity, Event.Normal, Event.Age, Event.ParticleID"),
];

pub fn handler_field_names() -> &'static [&'static str] {
    &[
        "source",
        "event",
        "execution_mode",
        "spawn_number",
        "max_events_per_frame",
        "modules",
    ]
}

/// One event, as the CPU stores it. The GPU stores [`EVENT_WORDS`] words
/// per event in this order: position (3), velocity (3), normal (3), age,
/// source UniqueID (u32 bits), padding.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Event {
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    /// Surface normal for collision events; zero otherwise.
    pub normal: [f32; 3],
    /// The source particle's age (seconds).
    pub age: f32,
    /// The source particle's UniqueID.
    pub particle_id: u32,
}

/// 32-bit words per event on the GPU.
pub const EVENT_WORDS: u32 = 12;

/// Random-key stage of handler `h`'s modules (after the six stage numbers).
pub fn handler_stage(h: usize) -> u32 {
    8 + h as u32
}

/// Random-key frame of the `rank`-th event handled by EveryParticle in step
/// `frame` (the update's `frame + 1`): one draw per event and step.
pub fn every_particle_frame(frame: u32, rank: u32) -> u32 {
    frame.wrapping_add(1).wrapping_add(rank.wrapping_mul(0x0100_0193))
}
