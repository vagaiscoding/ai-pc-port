//! GPU breadcrumbs (`VICTORY_GPU_BREADCRUMBS=1`, diagnostics builds): which
//! draw, dispatch, indirect execution, copy or clear the GPU was running
//! when a submission hangs.
//!
//! Every such command recorded on a host list gets an id (one counter for
//! the run), kept on the CPU with what it is (`Entry`). Around the command
//! the list writes the id into its slot of one buffer in host memory, with
//! `WriteBufferImmediate`: "started" before it (`MARKER_IN`: once the
//! commands before it began) and "finished" after it (`MARKER_OUT`: once it
//! and everything before completed). The buffer is a read-back one, mapped
//! once and never unmapped, so its words stay readable while the GPU hangs
//! and after the device is lost. A submission still running after
//! [`STUCK_AFTER`] (or a removed device) logs, per list, the last started
//! and finished ids and the first command started but not finished.
//!
//! vkd3d-proton records `WriteBufferImmediate` as `vkCmdWriteBufferMarkerAMD`
//! (top of pipe for `MARKER_IN`, bottom of pipe for `MARKER_OUT`) where the
//! driver has `VK_AMD_buffer_marker`, which NVIDIA's does; without it the
//! modes may be ignored and the writes are only in order.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

/// Slots in the buffer: a list uses slot `list id % SLOTS`.
pub(crate) const SLOTS: u64 = 16384;
/// Per slot: the last started id, then the last finished id.
const SLOT_BYTES: u64 = 8;
/// A submission running this long is reported (the host resets the device
/// after about 2 s of one command on Windows; NVIDIA's Linux driver later).
pub(crate) const STUCK_AFTER: Duration = Duration::from_secs(3);
/// Wine aborts on log lines over about 1020 bytes.
const LINE: usize = 880;

/// Whether breadcrumbs are on (`VICTORY_GPU_BREADCRUMBS`).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = crate::switch::on("VICTORY_GPU_BREADCRUMBS");
        if on {
            tracing::info!(slots = SLOTS, stuck_after_ms = STUCK_AFTER.as_millis() as u64, "GPU breadcrumbs on");
        }
        on
    })
}

pub(crate) use crate::host::Crumb as Kind;

impl Kind {
    fn text(self) -> &'static str {
        match self {
            Kind::Draw => "draw",
            Kind::Dispatch => "dispatch",
            Kind::Mesh => "mesh",
            Kind::Indirect => "indirect",
            Kind::Copy => "copy",
            Kind::Clear => "clear",
            Kind::Resolve => "resolve",
        }
    }
}

/// One recorded command.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) id: u32,
    pub(crate) kind: Kind,
    /// Frames presented when it was recorded.
    pub(crate) frame: u64,
    /// The bound pipeline's shaders (as host pipelines are named), or what
    /// our own command is.
    pub(crate) what: Option<Arc<str>>,
}

/// A list's breadcrumbs since its reset.
#[derive(Default)]
pub(crate) struct Trail {
    /// The list, for `WriteBufferImmediate`; `None` with breadcrumbs off.
    pub(crate) list: Option<ID3D12GraphicsCommandList2>,
    pub(crate) slot: u64,
    /// The pipeline bound last.
    pub(crate) pipeline: Option<Arc<str>>,
    pub(crate) entries: Vec<Entry>,
    /// The submission that ran the list last.
    pub(crate) submission: u64,
}

impl Trail {
    /// A trail for list `id` (`list` its host list on `device`), empty with
    /// breadcrumbs off or without the buffer.
    pub(crate) fn new(id: u64, list: &ID3D12GraphicsCommandList, device: &ID3D12Device) -> Trail {
        let list = (on() && init(device).is_some()).then(|| list.cast().ok()).flatten();
        Trail { list, slot: id % SLOTS, ..Trail::default() }
    }

    /// Notes the pipeline bound from now on.
    pub(crate) fn bind(&mut self, name: &str) {
        if self.list.is_some() && self.pipeline.as_deref() != Some(name) {
            self.pipeline = Some(name.into());
        }
    }

    /// Records a command of `kind` (running `what`, else the bound
    /// pipeline) about to be recorded: its "started" write. The id, for
    /// [`Trail::finish`], if breadcrumbs are on.
    pub(crate) fn start(&mut self, kind: Kind, what: Option<&str>) -> Option<u32> {
        let (list, buffer) = (self.list.as_ref()?, buffer()?);
        let id = next_id();
        let what = what.map(Arc::from).or_else(|| self.pipeline.clone());
        self.entries.push(Entry { id, kind, frame: crate::queue::presented(), what });
        write(list, buffer.va + self.slot * SLOT_BYTES, id, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_IN);
        Some(id)
    }

    /// The "finished" write of command `id`, recorded after it.
    pub(crate) fn finish(&self, id: u32) {
        if let (Some(list), Some(buffer)) = (&self.list, buffer()) {
            write(list, buffer.va + self.slot * SLOT_BYTES + 4, id, D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT);
        }
    }

    /// After the list's reset (the bound pipeline is kept: a list is
    /// bound again before its next draw, which renames it).
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
    }
}

fn write(list: &ID3D12GraphicsCommandList2, address: u64, value: u32, mode: D3D12_WRITEBUFFERIMMEDIATE_MODE) {
    let parameter = D3D12_WRITEBUFFERIMMEDIATE_PARAMETER { Dest: address, Value: value };
    // SAFETY: an open list; the address is inside our buffer, which stays
    // in the copy destination state (a read-back buffer's only state).
    unsafe { list.WriteBufferImmediate(1, &parameter, Some(&mode)) };
}

/// Ids from 1 (0: nothing written yet); they wrap after 2^32 commands.
fn next_id() -> u32 {
    static NEXT: AtomicU32 = AtomicU32::new(1);
    loop {
        let id = NEXT.fetch_add(1, Ordering::Relaxed);
        if id != 0 {
            return id;
        }
    }
}

/// The buffer the GPU writes, mapped.
pub(crate) struct Buffer {
    _resource: ID3D12Resource,
    va: u64,
    mapped: *const u32,
}

// SAFETY: the mapping is only read (volatile), and lives with the resource.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl Buffer {
    /// (last started, last finished) of `slot`, as the GPU left them.
    pub(crate) fn read(&self, slot: u64) -> (u32, u32) {
        let at = (slot % SLOTS) as usize * 2;
        // SAFETY: inside the mapping, which is never unmapped; the GPU may
        // write it at any time, so each word is read once, volatile.
        unsafe { (self.mapped.add(at).read_volatile(), self.mapped.add(at + 1).read_volatile()) }
    }
}

static BUFFER: OnceLock<Option<Buffer>> = OnceLock::new();

/// The buffer, once [`init`] made it.
pub(crate) fn buffer() -> Option<&'static Buffer> {
    BUFFER.get().and_then(Option::as_ref)
}

/// The buffer, made on `device` on first use; `None` if it could not be
/// made (logged).
pub(crate) fn init(device: &ID3D12Device) -> Option<&'static Buffer> {
    BUFFER
        .get_or_init(|| {
            let made = make(device);
            if let Err(e) = &made {
                tracing::warn!(error = %e, "GPU breadcrumbs: no buffer; breadcrumbs off");
            }
            made.ok()
        })
        .as_ref()
}

fn make(device: &ID3D12Device) -> windows::core::Result<Buffer> {
    let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: SLOTS * SLOT_BYTES,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS3::default();
    // SAFETY: the query writes its own structure of the given size.
    unsafe {
        device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, (&raw mut options).cast(), size_of_val(&options) as u32)?;
    }
    if !options.WriteBufferImmediateSupportFlags.contains(D3D12_COMMAND_LIST_SUPPORT_FLAG_DIRECT) {
        return Err(windows::core::Error::new(
            windows::Win32::Foundation::E_NOTIMPL,
            "no WriteBufferImmediate on direct lists",
        ));
    }
    let mut resource: Option<ID3D12Resource> = None;
    let mut mapped = std::ptr::null_mut();
    // SAFETY: plain creation; the mapping covers the whole buffer, zeroed
    // before any list writes it.
    unsafe {
        device.CreateCommittedResource(
            &readback,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut resource,
        )?;
        let resource = resource.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
        resource.SetName(windows::core::w!("victory GPU breadcrumbs"))?;
        resource.Map(0, Some(&D3D12_RANGE { Begin: 0, End: (SLOTS * SLOT_BYTES) as usize }), Some(&mut mapped))?;
        std::ptr::write_bytes(mapped.cast::<u8>(), 0, (SLOTS * SLOT_BYTES) as usize);
        let va = resource.GetGPUVirtualAddress();
        Ok(Buffer { _resource: resource, va, mapped: mapped.cast_const().cast() })
    }
}

/// Where a list is, from its ids in order and its slot's words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Progress {
    /// No breadcrumbed command.
    Empty,
    /// None of its commands started.
    NotStarted,
    /// Every command finished.
    Done,
    /// `entries[first]` started (or is about to) and did not finish;
    /// `in_flight` commands from it on started.
    Stuck { first: usize, in_flight: usize },
}

/// The progress of a list whose commands have the ids `ids` (ascending),
/// its slot holding `started` and `finished`. Words from before the list's
/// reset are older (smaller) than its ids.
pub(crate) fn progress(ids: &[u32], started: u32, finished: u32) -> Progress {
    let (Some(&first), Some(&last)) = (ids.first(), ids.last()) else {
        return Progress::Empty;
    };
    if finished == last {
        return Progress::Done;
    }
    if started < first || started > last {
        return Progress::NotStarted;
    }
    // The first not finished: after `finished` if it is one of ours.
    let first = if finished >= first { ids.partition_point(|&id| id <= finished) } else { 0 };
    let in_flight = ids[first..].partition_point(|&id| id <= started);
    Progress::Stuck { first, in_flight }
}

/// One command, short.
pub(crate) fn describe(e: &Entry) -> String {
    let what: String = e.what.as_deref().unwrap_or("no pipeline").chars().take(200).collect();
    format!("#{} {} {what} frame {}", e.id, e.kind.text(), e.frame)
}

/// `text` as log lines of at most [`LINE`] bytes.
pub(crate) fn lines(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while rest.len() > LINE {
        let mut cut = LINE;
        while !rest.is_char_boundary(cut) {
            cut -= 1;
        }
        out.push(rest[..cut].to_owned());
        rest = &rest[cut..];
    }
    out.push(rest.to_owned());
    for (i, l) in out.iter_mut().enumerate().skip(1) {
        *l = format!("(cont. {i}) {l}");
    }
    out
}

/// One list's report line(s), or `None` when it is done, empty or never
/// started (those are summed up by the caller).
pub(crate) fn report_list(list_id: u64, at: usize, of: usize, trail: &Trail, why: &str) -> (Progress, Option<String>) {
    let Some(buffer) = buffer() else {
        return (Progress::Empty, None);
    };
    let (started, finished) = buffer.read(trail.slot);
    let ids: Vec<u32> = trail.entries.iter().map(|e| e.id).collect();
    let p = progress(&ids, started, finished);
    let Progress::Stuck { first, in_flight } = p else {
        return (p, None);
    };
    let stuck = &trail.entries[first];
    let before: Vec<String> = trail.entries[first.saturating_sub(3)..first].iter().map(describe).collect();
    let text = format!(
        "GPU breadcrumbs ({why}, submission {}): list {list_id} ({} of {of}) started #{started} finished #{finished} of {} commands; \
         first not finished: {} (command {} of the list, {in_flight} started); before it: [{}]",
        trail.submission,
        at + 1,
        trail.entries.len(),
        describe(stuck),
        first + 1,
        before.join("; ")
    );
    (p, Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_list() {
        assert_eq!(progress(&[], 5, 5), Progress::Empty);
    }

    #[test]
    fn stale_words_mean_not_started() {
        // Words left by the list's previous recording (or zeros).
        assert_eq!(progress(&[10, 12, 15], 0, 0), Progress::NotStarted);
        assert_eq!(progress(&[10, 12, 15], 7, 7), Progress::NotStarted);
    }

    #[test]
    fn all_finished_is_done() {
        assert_eq!(progress(&[10, 12, 15], 15, 15), Progress::Done);
    }

    #[test]
    fn first_command_hung() {
        assert_eq!(progress(&[10, 12, 15], 10, 3), Progress::Stuck { first: 0, in_flight: 1 });
    }

    #[test]
    fn middle_command_hung_with_the_next_started() {
        // 10 finished, 12 and 15 started: 12 is the first not finished.
        assert_eq!(progress(&[10, 12, 15], 15, 10), Progress::Stuck { first: 1, in_flight: 2 });
    }

    #[test]
    fn ids_and_entries_round_trip() {
        let mut trail = Trail::default();
        // No list: breadcrumbs off, nothing recorded.
        assert_eq!(trail.start(Kind::Draw, None), None);
        trail.bind("[compute 72f552f4553fb5f9]");
        let ids: Vec<u32> = (0..4).map(|_| next_id()).collect();
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        for (i, &id) in ids.iter().enumerate() {
            let kind = [Kind::Dispatch, Kind::Draw, Kind::Copy, Kind::Indirect][i];
            trail.entries.push(Entry { id, kind, frame: 7, what: Some("[compute 72f552f4553fb5f9]".into()) });
        }
        let found = trail.entries.iter().find(|e| e.id == ids[2]).unwrap();
        assert_eq!(found.kind, Kind::Copy);
        assert_eq!(describe(found), format!("#{} copy [compute 72f552f4553fb5f9] frame 7", ids[2]));
        trail.clear();
        assert!(trail.entries.is_empty());
    }

    #[test]
    fn long_text_splits_under_the_limit() {
        let text = "x".repeat(2000);
        let out = lines(&text);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|l| l.len() < 900));
        assert!(out[1].starts_with("(cont. 1) "));
    }

    /// A list recorded and run on the host device (or WARP) leaves the ids
    /// of its last command in its slot.
    #[test]
    fn executed_list_writes_its_slot() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU
        };
        let Some(buffer) = init(&host.device) else {
            return; // no WriteBufferImmediate
        };
        let list = host.new_list().unwrap();
        // The list's trail, on even with the switch off.
        let mut trail = Trail { list: list.list.cast().ok(), slot: 4321, ..Trail::default() };
        let a = trail.start(Kind::Clear, Some("first")).unwrap();
        trail.finish(a);
        let b = trail.start(Kind::Copy, Some("second")).unwrap();
        trail.finish(b);
        let c = trail.start(Kind::Dispatch, Some("third")).unwrap();
        list.close();
        host.execute(&[&list]);
        let (started, finished) = buffer.read(4321);
        assert_eq!((started, finished), (c, b));
        let ids: Vec<u32> = trail.entries.iter().map(|e| e.id).collect();
        assert_eq!(progress(&ids, started, finished), Progress::Stuck { first: 2, in_flight: 1 });
        // SAFETY: a plain query.
        assert!(unsafe { host.device.GetDeviceRemovedReason() }.is_ok());
    }
}
