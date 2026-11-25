//! Emitter life cycle: Niagara's Emitter State module (loops), and when the
//! emitter's spawn modules fire within it.
//!
//! An emitter runs `loops` loops (one for Once, Loop Count for Multiple,
//! without end for Infinite). Each loop waits Loop Delay (only the first one
//! with Delay First Loop Only), then is active for Loop Duration (forever with
//! Loop Duration Mode Infinite). While active, continuous spawners (Spawn
//! Rate, Spawn Per Unit, Spawn Per Frame) spawn and bursts fire at their Spawn
//! Time measured from the loop's start (`Emitter.LoopedAge`). An emitter
//! without an Emitter State module has one endless loop without delay, so
//! `Emitter.LoopedAge` is `Emitter.Age`.
//!
//! The CPU simulation and the GPU host share this code: the emitter stacks
//! run on the CPU for both.

/// Loop Behavior values.
pub const LOOP_ONCE: i32 = 0;
pub const LOOP_MULTIPLE: i32 = 1;
pub const LOOP_INFINITE: i32 = 2;

/// Most loops a single step looks at (a burst fires at most this many times
/// in one step, for very short loops).
const MAX_LOOPS_PER_STEP: u32 = 64;

/// An emitter's loop settings, from its Emitter State module.
#[derive(Clone, Copy, Debug, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct Lifecycle {
    /// Number of loops; 0 = without end.
    pub loops: u32,
    /// Seconds each loop is active; infinite = never ends.
    pub duration: f32,
    /// Seconds before a loop starts.
    pub delay: f32,
    /// Only the first loop waits `delay`.
    pub delay_first_only: bool,
    /// The emitter has an Emitter State module.
    pub explicit: bool,
}

impl Default for Lifecycle {
    fn default() -> Self {
        Lifecycle {
            loops: 0,
            duration: f32::INFINITY,
            delay: 0.0,
            delay_first_only: false,
            explicit: false,
        }
    }
}

/// Where the emitter is in its life at some age.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoopPhase {
    /// Which loop (`Emitter.LoopCount`, from 0).
    pub index: u32,
    /// Seconds since the loop became active; negative during its delay
    /// (`Emitter.LoopedAge`).
    pub looped_age: f32,
    /// The loop is active: not in its delay, not after the last loop.
    pub active: bool,
}

impl LoopPhase {
    /// `Emitter.NormalizedLoopAge`: looped age / duration, 0..1 (0 for endless loops).
    pub fn normalized(&self, lc: &Lifecycle) -> f32 {
        if lc.duration.is_finite() && lc.duration > 0.0 {
            (self.looped_age / lc.duration).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

impl Lifecycle {
    /// Emitter age at which loop `k` becomes active (after its delay).
    pub fn loop_start(&self, k: u32) -> f32 {
        if !self.duration.is_finite() {
            return self.delay;
        }
        let k = k as f32;
        if self.delay_first_only {
            self.delay + k * self.duration
        } else {
            k * (self.delay + self.duration) + self.delay
        }
    }

    /// Length of one whole loop (delay and active part) after the first.
    fn period(&self) -> f32 {
        self.duration + if self.delay_first_only { 0.0 } else { self.delay }
    }

    /// Emitter age at which the last loop ends; `None` if it never does.
    pub fn end(&self) -> Option<f32> {
        (self.loops > 0 && self.duration.is_finite())
            .then(|| self.loop_start(self.loops - 1) + self.duration)
    }

    /// The loop that `age` falls in (or is waiting for).
    pub fn phase(&self, age: f32) -> LoopPhase {
        let finished = self.end().is_some_and(|e| age >= e);
        let index = self.loop_index(age);
        let looped_age = age - self.loop_start(index);
        LoopPhase {
            index,
            looped_age,
            active: !finished && looped_age >= 0.0 && looped_age < self.duration,
        }
    }

    fn loop_index(&self, age: f32) -> u32 {
        if !self.duration.is_finite() {
            return 0;
        }
        let first_end = self.delay + self.duration;
        let k = if age < first_end || self.period() <= 0.0 {
            0
        } else {
            1 + ((age - first_end) / self.period()).floor().max(0.0) as u32
        };
        if self.loops > 0 { k.min(self.loops - 1) } else { k }
    }

    /// How many times something at `t` seconds into each loop happens in
    /// the step from emitter age `age` to `age + dt` (start included, end
    /// excluded). `t` outside a loop's active part never happens.
    pub fn fires(&self, age: f32, dt: f32, t: f32) -> u32 {
        if !(t >= 0.0 && t < self.duration) && !(t == 0.0 && self.duration == 0.0) {
            return 0;
        }
        let first = self.loop_index(age);
        let mut n = 0;
        for k in first..first.saturating_add(MAX_LOOPS_PER_STEP) {
            if self.loops > 0 && k >= self.loops {
                break;
            }
            let at = self.loop_start(k) + t;
            if at >= age + dt {
                break;
            }
            if at >= age {
                n += 1;
            }
            if !self.duration.is_finite() {
                break;
            }
        }
        n
    }

    /// Like [`Self::fires`] for repeating times `t0`, `t0 + interval`, ...
    /// within each loop (`max` of them per loop; 0 = no limit).
    pub fn fires_every(&self, age: f32, dt: f32, t0: f32, interval: f32, max: u32) -> u32 {
        if !(interval > 0.0) {
            return self.fires(age, dt, t0);
        }
        let first = self.loop_index(age);
        let mut n = 0;
        for k in first..first.saturating_add(MAX_LOOPS_PER_STEP) {
            if self.loops > 0 && k >= self.loops {
                break;
            }
            let start = self.loop_start(k) + t0;
            if start >= age + dt {
                break;
            }
            // The first repeat at or after `age`.
            let mut b = ((age - start) / interval).ceil().max(0.0) as u32;
            while b > 0 && self.loop_start(k) + (t0 + (b - 1) as f32 * interval) >= age {
                b -= 1;
            }
            for _ in 0..MAX_LOOPS_PER_STEP {
                let t = t0 + b as f32 * interval;
                let at = self.loop_start(k) + t;
                if at >= age + dt || (max > 0 && b >= max) || !(t < self.duration) {
                    break;
                }
                if at >= age && t >= 0.0 {
                    n += 1;
                }
                b += 1;
            }
            if !self.duration.is_finite() {
                break;
            }
        }
        n
    }
}

/// Steps the warm-up takes: `warmup_time / warmup_tick_delta`, rounded up
/// (at most 100 000).
pub fn warmup_steps(sys: &crate::cook::CookedSystem) -> u32 {
    if sys.warmup_time > 0.0 && sys.warmup_tick_delta > 0.0 {
        (sys.warmup_time / sys.warmup_tick_delta).ceil().min(100_000.0) as u32
    } else {
        0
    }
}

/// The seed an emitter's random draws use: the instance seed, or for an
/// emitter with `deterministic: false` the seed mixed with the instance's
/// random `nonce`.
pub fn emitter_seed(seed: u32, nonce: u32, deterministic: bool) -> u32 {
    if deterministic {
        seed
    } else {
        crate::sim::rng::pcg(seed ^ crate::sim::rng::pcg(nonce ^ 0x9e37_79b9))
    }
}

/// A random value for [`emitter_seed`], different for every call.
pub fn random_nonce() -> u32 {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u32(COUNTER.fetch_add(1, Ordering::Relaxed));
    h.finish() as u32
}
