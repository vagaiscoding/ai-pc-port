//! The dynamic descriptor slots a list holds, and giving back those of lists
//! that already ran when a heap runs short.
//!
//! A list keeps the dynamic slots its draws took until the title resets it
//! (see `draw::State`). Every submission is waited for, so once a list has
//! run the GPU no longer reads its slots; only submitting the same list a
//! second time, without a reset, would read them again. When a heap has no
//! room for a table, a draw would be dropped. Instead the slots of the
//! oldest list that already ran are taken back and that list is marked lost:
//! submitting it again is refused (the same as a draw whose table could not
//! be made), and it works again once the title resets it.
//!
//! shortcut: a list is taken back whole, oldest first, and only when
//! something would otherwise be dropped; the title's frames in flight decide
//! how much that frees.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Dynamic slots: the title heap they belong to (its address), the first
/// and the count.
pub(crate) type Range = (usize, u32, u32);

/// The slots one list holds.
#[derive(Default)]
pub(crate) struct Lease {
    ranges: Mutex<Vec<Range>>,
    /// Active submissions referencing this lease, updated under [`FINISHED`].
    in_flight: AtomicUsize,
    /// The slots were taken back after the list ran.
    lost: AtomicBool,
}

/// The leases of lists that ran, oldest first, and that still hold slots.
/// Its lock also orders [`Lease::start`] against [`reclaim`], so a list is
/// never taken back while it runs.
static FINISHED: Mutex<VecDeque<Arc<Lease>>> = Mutex::new(VecDeque::new());

fn finished_queue() -> std::sync::MutexGuard<'static, VecDeque<Arc<Lease>>> {
    FINISHED.lock().unwrap_or_else(|e| e.into_inner())
}

impl Lease {
    /// Records a range the list took.
    pub(crate) fn add(&self, heap: usize, first: u32, n: u32) {
        self.push((heap, first, n));
    }

    /// As [`Lease::add`], for a range as a tuple: a list's `dynamic` slots
    /// were a `Vec` of them, and code written for that still compiles.
    pub(crate) fn push(&self, range: Range) {
        self.ranges
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(range);
    }

    /// Everything the list still holds, to give back (the list is reset or
    /// released). Empty if the slots were taken back already.
    pub(crate) fn take(&self) -> Vec<Range> {
        std::mem::take(&mut *self.ranges.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Whether the slots were taken back after the list ran.
    pub(crate) fn is_lost(&self) -> bool {
        self.lost.load(Ordering::Acquire)
    }

    fn holds(&self, heap: usize) -> bool {
        self.ranges
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|r| r.0 == heap)
    }

    /// Before the list is submitted: false if its slots were taken back
    /// (it must not run); else it cannot be taken back until every started
    /// submission calls [`Lease::finish`].
    pub(crate) fn start(self: &Arc<Self>) -> bool {
        let mut queue = finished_queue();
        if self.is_lost() {
            return false;
        }
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        queue.retain(|l| !Arc::ptr_eq(l, self));
        true
    }

    /// After the list ran: once every active submission finishes, its slots
    /// may be taken back if a heap runs short.
    pub(crate) fn finish(self: &Arc<Self>) {
        let mut queue = finished_queue();
        let left = self.in_flight.load(Ordering::Relaxed).saturating_sub(1);
        self.in_flight.store(left, Ordering::Relaxed);
        if left > 0
            || self.is_lost()
            || self
                .ranges
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_empty()
        {
            return;
        }
        queue.retain(|l| !Arc::ptr_eq(l, self));
        queue.push_back(self.clone());
        // Lists that were reset hold nothing and only take room here.
        if queue.len() > 4096 {
            queue.retain(|l| {
                !l.ranges
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .is_empty()
            });
        }
    }
}

/// Takes back the slots of the oldest list that ran and holds some in `heap`,
/// handing each of its ranges to `release`. False if there is none.
pub(crate) fn reclaim(heap: usize, mut release: impl FnMut(Range)) -> bool {
    let ranges = {
        let mut queue = finished_queue();
        let Some(at) = queue
            .iter()
            .position(|l| l.in_flight.load(Ordering::Relaxed) == 0 && l.holds(heap))
        else {
            return false;
        };
        let lease = queue.remove(at).expect("a position of the queue");
        lease.lost.store(true, Ordering::Release);
        lease.take()
    };
    for range in ranges {
        release(range);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(heap: usize, ranges: &[(u32, u32)]) -> Arc<Lease> {
        let lease = Arc::new(Lease::default());
        for &(first, n) in ranges {
            lease.add(heap, first, n);
        }
        lease
    }

    /// Oldest first, only leases holding slots of the heap, each once.
    #[test]
    fn the_oldest_list_that_ran_is_taken_back_first() {
        let heap = 0x1000_0001;
        let (a, b) = (lease(heap, &[(10, 4), (20, 2)]), lease(heap, &[(30, 8)]));
        let elsewhere = lease(0x1000_0002, &[(0, 1)]);
        for l in [&a, &elsewhere, &b] {
            l.finish();
        }
        let mut given = Vec::new();
        assert!(reclaim(heap, |r| given.push(r)));
        assert_eq!(given, [(heap, 10, 4), (heap, 20, 2)]);
        assert!(a.is_lost() && !b.is_lost() && !elsewhere.is_lost());
        // Taken back: giving back when the list is reset releases nothing more.
        assert!(a.take().is_empty());
        given.clear();
        assert!(reclaim(heap, |r| given.push(r)));
        assert_eq!(given, [(heap, 30, 8)]);
        assert!(!reclaim(heap, |_| panic!("nothing is left in this heap")));
        assert!(!elsewhere.is_lost());
        // Clean up the other heap's entry for the other tests.
        assert!(reclaim(0x1000_0002, |_| {}));
    }

    /// A list that has not finished, or that runs again, is not taken back,
    /// and one that was taken back must not run.
    #[test]
    fn a_running_list_is_kept_and_a_lost_one_is_refused() {
        let heap = 0x1000_0003;
        let l = lease(heap, &[(1, 1)]);
        assert!(!reclaim(heap, |_| panic!("it has not run")));
        l.finish();
        // Submitted again: it is not takeable while it runs.
        assert!(l.start());
        assert!(!reclaim(heap, |_| panic!("it is running")));
        l.finish();
        assert!(reclaim(heap, |_| {}));
        assert!(l.is_lost());
        assert!(!l.start(), "a list that lost its slots is refused");
        // A lost list is not queued again.
        l.finish();
        assert!(!reclaim(heap, |_| panic!("it holds nothing")));
    }

    /// Finishing twice queues once; a list that was reset holds nothing and
    /// is passed over.
    #[test]
    fn finishing_twice_queues_once_and_reset_lists_are_skipped() {
        let heap = 0x1000_0004;
        let (reset, kept) = (lease(heap, &[(5, 5)]), lease(heap, &[(50, 5)]));
        reset.finish();
        reset.finish();
        kept.finish();
        // The title resets the first: it gives its slots back itself.
        assert_eq!(reset.take(), [(heap, 5, 5)]);
        let mut given = Vec::new();
        assert!(reclaim(heap, |r| given.push(r)));
        assert_eq!(given, [(heap, 50, 5)]);
        assert!(!reset.is_lost() && kept.is_lost());
        assert!(!reclaim(heap, |_| panic!("one entry was kept, once")));
    }

    /// Two overlapping submissions of the same list (concurrent or listed
    /// twice) keep its slots un-reclaimable until both have finished.
    #[test]
    fn overlapping_submissions_keep_slots_until_all_finish() {
        let heap = 0x1000_0005;
        let l = lease(heap, &[(10, 4)]);
        assert!(l.start());
        assert!(l.start());
        // The first submission ended, but the second is still running.
        l.finish();
        assert!(
            !reclaim(heap, |_| panic!("still running in the second submission")),
            "a lease still in flight must not be reclaimed"
        );
        assert!(!l.is_lost());
        // Once the second submission also ends, its slots may be taken back.
        l.finish();
        let mut given = Vec::new();
        assert!(reclaim(heap, |r| given.push(r)));
        assert_eq!(given, [(heap, 10, 4)]);
        assert!(l.is_lost());
    }
}
