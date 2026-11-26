//! The title's acceleration-structure builder runs whole or not at all.
//!
//! `XboxBvhBuilder.dll` builds and refits the ray-tracing structures in
//! title memory every frame through some fifty compute programs, one
//! dispatch after another. A dispatch whose host pipeline is missing is
//! dropped on its own (`State::dispatch`), and some of the builder's are
//! missing for a long time or for good: refused (64-lane waves on a host of
//! 32-lane waves), not translated, or still being made. The stages that do
//! run then work on what the missing ones never wrote, and write the result
//! into title memory: on fxceless/victory-fork, garbage bounds that crashed
//! the title. So the builder's dispatches run only in frames where every
//! pipeline it has created has its host pipeline; otherwise none of them
//! run (the structures keep what they held).
//!
//! Its pipelines are known by where they come from: created by its code (a
//! return address in its image on the stack) or from bytecode in its image.
//!
//! The programs that walk the structures (`image_bvh*_intersect_ray`) wait
//! for the builder too: until it has run whole once, the structures are
//! whatever lay in that memory, and a walk of them (the title bounds it at
//! 65,535 steps per wave) outlasts the host's GPU timeout. On a host where
//! the builder cannot run (its 64-lane programs refused on 32-lane waves)
//! they never run, and the device is not lost (`built`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use super::make::{self, Pending};

/// The builder's image, `[start, end)`, once mapped.
static IMAGE: Mutex<Option<(usize, usize)>> = Mutex::new(None);

/// The builder's file name.
const FILE: &str = "XboxBvhBuilder.dll";

/// A title DLL was mapped at `base`, `size` bytes: noted if it is the
/// builder.
pub fn note_title_dll(file: &str, base: usize, size: usize) {
    if file.eq_ignore_ascii_case(FILE) {
        tracing::info!(base = format_args!("{base:#x}"), size, "BVH builder mapped");
        *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = Some((base, base + size));
        if disabled() {
            tracing::info!("BVH building and traversal disabled for this run");
        }
    }
}

fn image() -> Option<(usize, usize)> {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Opt-in fallback for a title whose acceleration structures cannot be
/// built correctly on this host.
pub(crate) fn disabled() -> bool {
    static DISABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DISABLED.get_or_init(|| crate::switch::on("VICTORY_DISABLE_BVH"))
}

/// The builder has run whole in some frame: its structures exist.
static BUILT: AtomicBool = AtomicBool::new(false);

/// Whether programs that walk the structures may run: the builder has run
/// whole once, or the title has no builder (its structures come from
/// elsewhere).
pub(crate) fn built() -> bool {
    (!disabled() && BUILT.load(Ordering::Relaxed)) || image().is_none()
}

/// Whether the pipeline being created is the builder's: `code` (the title's
/// bytecode pointers) lies in its image, or its code is among our callers.
#[inline(never)]
pub(crate) fn created_by_builder(code: &[*const u8]) -> bool {
    let Some((start, end)) = image() else {
        return false;
    };
    let inside = |a: usize| (start..end).contains(&a);
    if code.iter().any(|&c| inside(c as usize)) {
        return true;
    }
    callers().into_iter().any(inside)
}

/// Return addresses on the stack, ours first.
fn callers() -> Vec<usize> {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Diagnostics::Debug::RtlCaptureStackBackTrace;
        let mut frames = [std::ptr::null_mut(); 24];
        // SAFETY: room for that many frames.
        let n = unsafe {
            RtlCaptureStackBackTrace(
                1,
                frames.len() as u32,
                frames.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        frames[..usize::from(n)]
            .iter()
            .map(|&f| f as usize)
            .collect()
    }
    #[cfg(not(windows))]
    Vec::new()
}

/// The builder's pipelines (alive while the title holds them).
static PIPELINES: Mutex<Vec<Weak<Pending>>> = Mutex::new(Vec::new());

/// The builder created pipeline `p`: it counts towards [`runs`], and is
/// made at once (no frame's wait budget holds it back).
pub(crate) fn note(p: &Arc<Pending>) {
    PIPELINES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(Arc::downgrade(p));
    make::begin(p);
}

/// The frame [`runs`] last decided for, and its decision.
static DECIDED: Mutex<(u64, bool)> = Mutex::new((u64::MAX, true));

/// Whether the builder's dispatches run in the current frame: decided once
/// per frame, so that a frame never runs part of the builder.
pub(crate) fn runs() -> bool {
    if disabled() {
        return false;
    }
    static SAID: AtomicBool = AtomicBool::new(false);
    let frame = crate::frame::current();
    let mut decided = DECIDED.lock().unwrap_or_else(|e| e.into_inner());
    if decided.0 == frame {
        return decided.1;
    }
    let states: Vec<Option<bool>> = {
        let mut pipelines = PIPELINES.lock().unwrap_or_else(|e| e.into_inner());
        pipelines.retain(|p| p.strong_count() > 0);
        pipelines
            .iter()
            .filter_map(Weak::upgrade)
            .map(|p| make::state(&p))
            .collect()
    };
    let open = all_made(&states);
    if open != decided.1 || !SAID.swap(true, Ordering::Relaxed) {
        let (made, refused) = (
            states.iter().filter(|s| **s == Some(true)).count(),
            states.iter().filter(|s| **s == Some(false)).count(),
        );
        tracing::info!(
            frame,
            pipelines = states.len(),
            made,
            refused,
            "BVH builder {}",
            if open {
                "runs"
            } else {
                "held back: not all its pipelines are made"
            }
        );
    }
    *decided = (frame, open);
    // Asked at a builder dispatch: with every pipeline made, it runs whole.
    if open && !states.is_empty() && !BUILT.swap(true, Ordering::Relaxed) {
        tracing::info!(frame, "BVH builder ran whole: programs that walk its structures run from here on");
    }
    open
}

/// Whether every pipeline is made (`Some(true)`); refused ones are
/// `Some(false)`, those still being made `None`.
fn all_made(states: &[Option<bool>]) -> bool {
    states.iter().all(|s| *s == Some(true))
}

/// Tests that set the builder's image, one at a time; each starts with
/// no decision made.
#[cfg(test)]
pub(crate) fn test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    *DECIDED.lock().unwrap_or_else(|e| e.into_inner()) = (u64::MAX, true);
    BUILT.store(false, Ordering::Relaxed);
    guard
}

/// No builder mapped (a title without it).
#[cfg(test)]
pub(crate) fn clear_image_for_tests() {
    *IMAGE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The builder runs only when every one of its pipelines is made: one
    /// refused or still being made holds all of them back.
    #[test]
    fn the_builder_runs_only_with_every_pipeline_made() {
        assert!(all_made(&[]), "no builder: nothing to hold back");
        assert!(all_made(&[Some(true), Some(true)]));
        assert!(!all_made(&[Some(true), Some(false)]), "one refused");
        assert!(!all_made(&[Some(true), None]), "one still being made");
    }

    /// Pipelines are the builder's by their bytecode in its image, or by
    /// its code among the callers; not otherwise.
    #[test]
    fn builder_pipelines_are_known_by_their_origin() {
        let _one = test_lock();
        // A range that holds this test's own code (a caller of the check).
        let here = builder_pipelines_are_known_by_their_origin as *const () as usize;
        let elsewhere = [0x7_0000_0000usize as *const u8];
        *IMAGE.lock().unwrap() = None;
        assert!(!created_by_builder(&elsewhere), "no builder mapped");
        // Bytecode inside the image.
        *IMAGE.lock().unwrap() = Some((0x7_0000_0000, 0x7_0001_0000));
        assert!(created_by_builder(&elsewhere));
        assert!(!created_by_builder(&[0x8_0000_0000usize as *const u8]));
        // A caller inside the image: this function's own code.
        *IMAGE.lock().unwrap() = Some((here - 0x10_0000, here + 0x10_0000));
        assert!(created_by_builder(&[]));
        *IMAGE.lock().unwrap() = None;
    }
}
