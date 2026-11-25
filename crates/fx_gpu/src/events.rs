//! Where events live on the GPU.
//!
//! One pair of buffers per effect: `event_counts` holds one atomic counter
//! per region, `event_data` each region's events ([`EVENT_WORDS`] words
//! each, unordered as the update appends them) followed by its sorted order
//! (one event index per event, by source UniqueID, written by the source's
//! `event_rank` pass). A region is one (emitter, event kind) the emitter
//! writes; it holds up to the emitter's `max_particles` events, since each
//! particle writes at most one event of each kind per step.

use fx_core::cook::CookedSystem;
use fx_core::events::{EVENT_WORDS, EventKind};

#[derive(Clone, Debug, PartialEq)]
pub struct EventRegion {
    pub emitter: usize,
    pub kind: EventKind,
    /// Counter index in `event_counts`.
    pub index: u32,
    pub capacity: u32,
    /// First word of the events in `event_data`.
    pub data: u32,
    /// First word of the sorted order in `event_data`.
    pub sorted: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct EventLayout {
    pub regions: Vec<EventRegion>,
    /// Words of `event_data`.
    pub data_words: u64,
}

impl EventLayout {
    pub fn new(sys: &CookedSystem) -> Self {
        let mut regions = Vec::new();
        let mut words = 0u64;
        for (ei, e) in sys.emitters.iter().enumerate() {
            for kind in EventKind::ALL {
                if !e.generates(kind) {
                    continue;
                }
                let capacity = e.max_particles.max(1);
                let data = words;
                let sorted = data + capacity as u64 * EVENT_WORDS as u64;
                words = sorted + capacity as u64;
                regions.push(EventRegion {
                    emitter: ei,
                    kind,
                    index: regions.len() as u32,
                    capacity,
                    data: data as u32,
                    sorted: sorted as u32,
                });
            }
        }
        EventLayout {
            regions,
            data_words: words,
        }
    }

    pub fn region(&self, emitter: usize, kind: EventKind) -> Option<&EventRegion> {
        self.regions
            .iter()
            .find(|r| r.emitter == emitter && r.kind == kind)
    }
}
