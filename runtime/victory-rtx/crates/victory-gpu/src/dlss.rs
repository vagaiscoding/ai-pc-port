//! The title's FidelityFX upscale calls answered by NVIDIA DLSS on the host
//! GPU, through Streamline: `sl.interposer.dll` and its plugins
//! (`sl.common.dll`, `sl.dlss.dll`, `nvngx_dlss.dll`) from the public
//! Streamline SDK's `bin/x64`, in the folder `VICTORY_DLSS_DIR` names.
//! Nothing of the SDK is in this repository: the structures below are
//! declared from the SDK's MIT-licensed headers (2.x), and their sizes are
//! checked against the ones MSVC gives for those headers.
//!
//! Streamline runs in manual hooking mode on VIctory's own host device
//! (`slSetD3DDevice`), with frame-based resource tagging. Each upscale
//! context the title creates is one Streamline viewport. A dispatch arrives
//! here after `victory_hle`'s FidelityFX module has swapped the title's
//! textures for their host copies and their states for D3D12 states
//! (the output in `UNORDERED_ACCESS`; an RGB9E5 output the host cannot
//! store to is an FP16 texture of `crate::rgb9e5`'s by then) and its command
//! list for the host one, as it does for AMD's upscaler, so this only translates the
//! description: FidelityFX's jitter is DLSS's (both in pixels, the offset
//! applied to the camera); its motion vector scale converts to pixels,
//! DLSS's to [-1, 1], so the scale is divided by the size the vectors are
//! rendered at.
//!
//! The title also asks FidelityFX for its jitter sequence and for render
//! sizes. DLSS has no such calls, so they are answered with FSR's own
//! formulas (`ffx_fsr3upscaler.cpp`): the title renders exactly as it does
//! for FSR.
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Dxgi::IDXGIFactory2;
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::core::{Interface, PCSTR, PCWSTR};

/// FidelityFX's return codes.
pub type ReturnCode = u32;
const OK: ReturnCode = 0;
const ERROR: ReturnCode = 1;
const ERROR_PARAMETER: ReturnCode = 6;

/// The start of every FidelityFX description: its type and the next one.
#[repr(C)]
struct Header {
    kind: u64,
    next: *mut Header,
}

/// `ffxCreateContextDescUpscale`.
#[repr(C)]
struct CreateUpscale {
    kind: u64,
    next: *mut Header,
    flags: u32,
    max_render: [u32; 2],
    max_upscale: [u32; 2],
    message: *const c_void,
}
const _: () = assert!(size_of::<CreateUpscale>() == 48);

/// `FfxApiResource`: the resource, its description, its state.
#[repr(C)]
#[derive(Clone, Copy)]
struct FfxResource {
    resource: *mut c_void,
    description: [u32; 8],
    state: u32,
}
const _: () = assert!(size_of::<FfxResource>() == 48);

/// `ffxDispatchDescUpscale`.
#[repr(C)]
#[derive(Clone, Copy)]
struct DispatchUpscale {
    kind: u64,
    next: *mut Header,
    command_list: *mut c_void,
    /// Color, depth, motion vectors, exposure, reactive, transparency and
    /// composition, output.
    resources: [FfxResource; 7],
    jitter: [f32; 2],
    motion_vector_scale: [f32; 2],
    render_size: [u32; 2],
    upscale_size: [u32; 2],
    sharpening: bool,
    sharpness: f32,
    frame_time_delta: f32,
    pre_exposure: f32,
    reset: bool,
    camera: [f32; 4],
    flags: u32,
}
const _: () = assert!(size_of::<DispatchUpscale>() == 432);
const OUTPUT: usize = 6;

trait WideNul {
    fn encode_wide_nul(&self) -> Vec<u16>;
}

impl WideNul for std::ffi::OsStr {
    fn encode_wide_nul(&self) -> Vec<u16> {
        use std::os::windows::ffi::OsStrExt;
        self.encode_wide().chain(Some(0)).collect()
    }
}

/// Whether DLSS answers the title's FidelityFX calls: `VICTORY_DLSS_DIR`
/// set. It takes precedence over `VICTORY_FFX_DIR`.
pub fn enabled() -> bool {
    std::env::var_os("VICTORY_DLSS_DIR").is_some()
}

// FidelityFX (`ffx_upscale.h`).
const FFX_HDR: u32 = 1 << 0;
const FFX_DISPLAY_RESOLUTION_MOTION_VECTORS: u32 = 1 << 1;
const FFX_MOTION_VECTORS_JITTER_CANCELLATION: u32 = 1 << 2;
const FFX_DEPTH_INVERTED: u32 = 1 << 3;
const FFX_AUTO_EXPOSURE: u32 = 1 << 5;
const QUERY_UPSCALE_RATIO: u64 = 0x0001_0002;
const QUERY_RENDER_RESOLUTION: u64 = 0x0001_0003;
const QUERY_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
const QUERY_JITTER_OFFSET: u64 = 0x0001_0005;
const QUERY_GET_VERSIONS: u64 = 4;

/// FSR's per-dimension ratio of a FidelityFX quality mode.
fn upscale_ratio(quality: u32) -> Option<f32> {
    Some(match quality {
        0 => 1.0,
        1 => 1.5,
        2 => 1.7,
        3 => 2.0,
        4 => 3.0,
        _ => return None,
    })
}

/// FSR's jitter sequence length for a render and a display width.
fn jitter_phase_count(render_width: u32, display_width: u32) -> i32 {
    let ratio = display_width as f32 / render_width as f32;
    (8.0 * ratio.powf(2.0)) as i32
}

/// The Halton sequence FSR jitters with.
fn halton(index: i32, base: i32) -> f32 {
    let (mut f, mut result, mut i) = (1.0f32, 0.0f32, index);
    while i > 0 {
        f /= base as f32;
        result += f * (i % base) as f32;
        i = (i as f32 / base as f32).floor() as i32;
    }
    result
}

/// FSR's jitter offset at `index` of a sequence of `phase_count`.
fn jitter_offset(index: i32, phase_count: i32) -> (f32, f32) {
    let n = index.rem_euclid(phase_count) + 1;
    (halton(n, 2) - 0.5, halton(n, 3) - 0.5)
}

/// Answers the FidelityFX queries that need no upscaler: the jitter
/// sequence, render sizes and the provider list. `None` for others.
///
/// # Safety
///
/// `desc` is a valid description of the type its header names, with valid
/// output pointers.
pub unsafe fn query(desc: *mut c_void) -> Option<ReturnCode> {
    let desc = desc.cast::<Header>();
    // SAFETY: the caller's description.
    let kind = unsafe { (*desc).kind };
    Some(match kind {
        QUERY_JITTER_PHASE_COUNT => {
            #[repr(C)]
            struct Q {
                header: Header,
                render_width: u32,
                display_width: u32,
                out: *mut i32,
            }
            // SAFETY: a description of this type.
            let q = unsafe { &*desc.cast::<Q>() };
            if q.out.is_null() || q.render_width == 0 {
                return Some(ERROR_PARAMETER);
            }
            // SAFETY: the title's output.
            unsafe { *q.out = jitter_phase_count(q.render_width, q.display_width) };
            OK
        }
        QUERY_JITTER_OFFSET => {
            #[repr(C)]
            struct Q {
                header: Header,
                index: i32,
                phase_count: i32,
                x: *mut f32,
                y: *mut f32,
            }
            // SAFETY: a description of this type.
            let q = unsafe { &*desc.cast::<Q>() };
            if q.x.is_null() || q.y.is_null() || q.phase_count <= 0 {
                return Some(ERROR_PARAMETER);
            }
            let (x, y) = jitter_offset(q.index, q.phase_count);
            // SAFETY: the title's outputs.
            unsafe { (*q.x, *q.y) = (x, y) };
            OK
        }
        QUERY_UPSCALE_RATIO => {
            #[repr(C)]
            struct Q {
                header: Header,
                quality: u32,
                out: *mut f32,
            }
            // SAFETY: a description of this type.
            let q = unsafe { &*desc.cast::<Q>() };
            let Some(ratio) = upscale_ratio(q.quality).filter(|_| !q.out.is_null()) else {
                return Some(ERROR_PARAMETER);
            };
            // SAFETY: the title's output.
            unsafe { *q.out = ratio };
            OK
        }
        QUERY_RENDER_RESOLUTION => {
            #[repr(C)]
            struct Q {
                header: Header,
                display_width: u32,
                display_height: u32,
                quality: u32,
                width: *mut u32,
                height: *mut u32,
            }
            // SAFETY: a description of this type.
            let q = unsafe { &*desc.cast::<Q>() };
            let Some(ratio) = upscale_ratio(q.quality) else {
                return Some(ERROR_PARAMETER);
            };
            if q.width.is_null() || q.height.is_null() {
                return Some(ERROR_PARAMETER);
            }
            // SAFETY: the title's outputs.
            unsafe {
                *q.width = (q.display_width as f32 / ratio) as u32;
                *q.height = (q.display_height as f32 / ratio) as u32;
            }
            OK
        }
        QUERY_GET_VERSIONS => {
            #[repr(C)]
            struct Q {
                header: Header,
                create_kind: u64,
                device: *mut c_void,
                count: *mut u64,
                ids: *mut u64,
                names: *mut *const c_char,
            }
            // SAFETY: a description of this type.
            let q = unsafe { &*desc.cast::<Q>() };
            if q.count.is_null() {
                return Some(ERROR_PARAMETER);
            }
            // One provider. With arrays, the title's count is their size.
            // SAFETY: the title's count and arrays of that size.
            unsafe {
                if !q.ids.is_null() && *q.count > 0 {
                    *q.ids = PROVIDER_ID;
                    if !q.names.is_null() {
                        *q.names = c"NVIDIA DLSS (Streamline)".as_ptr();
                    }
                }
                *q.count = 1;
            }
            OK
        }
        _ => return None,
    })
}

/// The provider id answered for version queries.
const PROVIDER_ID: u64 = 0x0D15_5000;

// Streamline (`sl_struct.h`, `sl_core_types.h`, `sl_consts.h`, `sl_dlss.h`).

/// `sl::StructType`: a GUID.
#[repr(C)]
#[derive(Clone, Copy)]
struct StructType(u32, u16, u16, [u8; 8]);

/// `sl::BaseStructure`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Base {
    next: *mut c_void,
    kind: StructType,
    version: usize,
}
const _: () = assert!(size_of::<Base>() == 32);

impl Base {
    const fn new(kind: StructType, version: usize) -> Self {
        Base {
            next: std::ptr::null_mut(),
            kind,
            version,
        }
    }
}

const PREFERENCES: StructType = StructType(
    0x1ca1_0965,
    0xbf8e,
    0x432b,
    [0x8d, 0xa1, 0x67, 0x16, 0xd8, 0x79, 0xfb, 0x14],
);
const RESOURCE: StructType = StructType(
    0x3a9d_70cf,
    0x2418,
    0x4b72,
    [0x83, 0x91, 0x13, 0xf8, 0x72, 0x1c, 0x72, 0x61],
);
const RESOURCE_TAG: StructType = StructType(
    0x4c6a_5aad,
    0xb445,
    0x496c,
    [0x87, 0xff, 0x1a, 0xf3, 0x84, 0x5b, 0xe6, 0x53],
);
const VIEWPORT: StructType = StructType(
    0x171b_6435,
    0x9b3c,
    0x4fc8,
    [0x99, 0x94, 0xfb, 0xe5, 0x25, 0x69, 0xaa, 0xa4],
);
const CONSTANTS: StructType = StructType(
    0xdcd3_5ad7,
    0x4e4a,
    0x4bad,
    [0xa9, 0x0c, 0xe0, 0xc4, 0x9e, 0xb2, 0x3a, 0xfe],
);
const DLSS_OPTIONS: StructType = StructType(
    0x6ac8_26e4,
    0x4c61,
    0x4101,
    [0xa9, 0x2d, 0x63, 0x8d, 0x42, 0x10, 0x57, 0xb8],
);

/// `sl::kSDKVersion` of the headers these structures follow (2.14.16).
const SDK_VERSION: u64 = 2 << 48 | 14 << 32 | 16 << 16 | 0xfedc;
const FEATURE_DLSS: u32 = 0;
// `sl::PreferenceFlags`.
const DISABLE_CL_STATE_TRACKING: u64 = 1 << 0;
const USE_MANUAL_HOOKING: u64 = 1 << 2;
const USE_FRAME_BASED_RESOURCE_TAGGING: u64 = 1 << 7;
// `sl::BufferType`.
const BUFFER_DEPTH: u32 = 0;
const BUFFER_MOTION_VECTORS: u32 = 1;
const BUFFER_INPUT_COLOR: u32 = 3;
const BUFFER_OUTPUT_COLOR: u32 = 4;
const BUFFER_EXPOSURE: u32 = 13;
/// `sl::ResourceLifecycle::eValidUntilEvaluate`.
const VALID_UNTIL_EVALUATE: i32 = 2;
// `sl::Boolean`.
const FALSE: u8 = 0;
const TRUE: u8 = 1;
/// `sl::DLSSMode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum Mode {
    MaxPerformance = 1,
    Balanced = 2,
    MaxQuality = 3,
    UltraPerformance = 4,
    Dlaa = 6,
}

/// `sl::DLSSPreset` values used by this backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
enum Preset {
    Default = 0,
    M = 13,
}
/// `sl::Preferences`.
#[repr(C)]
struct Preferences {
    base: Base,
    show_console: bool,
    log_level: u32,
    paths_to_plugins: *const *const u16,
    num_paths_to_plugins: u32,
    path_to_logs_and_data: *const u16,
    allocate_callback: *const c_void,
    release_callback: *const c_void,
    log_message_callback: Option<extern "C" fn(u32, *const c_char)>,
    flags: u64,
    features_to_load: *const u32,
    num_features_to_load: u32,
    application_id: u32,
    engine: u32,
    engine_version: *const c_char,
    project_id: *const c_char,
    render_api: u32,
}
const _: () = assert!(size_of::<Preferences>() == 144);
const _: () = assert!(std::mem::offset_of!(Preferences, flags) == 88);
const _: () = assert!(std::mem::offset_of!(Preferences, render_api) == 136);

/// `sl::Resource`.
#[repr(C)]
struct Resource {
    base: Base,
    kind: u8,
    native: *mut c_void,
    memory: *mut c_void,
    view: *mut c_void,
    state: u32,
    width: u32,
    height: u32,
    native_format: u32,
    mip_levels: u32,
    array_layers: u32,
    gpu_virtual_address: u64,
    flags: u32,
    usage: u32,
    internal_flags: u16,
    reserved: u16,
}
const _: () = assert!(size_of::<Resource>() == 112);
const _: () = assert!(std::mem::offset_of!(Resource, native) == 40);
const _: () = assert!(std::mem::offset_of!(Resource, state) == 64);
const _: () = assert!(std::mem::offset_of!(Resource, gpu_virtual_address) == 88);

impl Resource {
    /// A 2D texture (`eTex2d`) in a D3D12 state.
    fn texture(native: *mut c_void, state: u32) -> Self {
        Resource {
            base: Base::new(RESOURCE, 1),
            kind: 0,
            native,
            memory: std::ptr::null_mut(),
            view: std::ptr::null_mut(),
            state,
            width: 0,
            height: 0,
            native_format: 0,
            mip_levels: 0,
            array_layers: 0,
            gpu_virtual_address: 0,
            flags: 0,
            usage: 0,
            internal_flags: 0,
            reserved: 0,
        }
    }
}

/// `sl::Extent`: top, left, width, height.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Extent([u32; 4]);

/// `sl::ResourceTag`.
#[repr(C)]
struct ResourceTag {
    base: Base,
    resource: *mut Resource,
    kind: u32,
    lifecycle: i32,
    extent: Extent,
}
const _: () = assert!(size_of::<ResourceTag>() == 64);
const _: () = assert!(std::mem::offset_of!(ResourceTag, extent) == 48);

/// `sl::ViewportHandle`.
#[repr(C)]
struct Viewport {
    base: Base,
    value: u32,
}
const _: () = assert!(size_of::<Viewport>() == 40);

impl Viewport {
    fn new(value: u32) -> Self {
        Viewport {
            base: Base::new(VIEWPORT, 1),
            value,
        }
    }
}

/// `sl::Constants` (version 2): matrices row-major, without jitter.
#[repr(C)]
struct Constants {
    base: Base,
    camera_view_to_clip: [f32; 16],
    clip_to_camera_view: [f32; 16],
    clip_to_lens_clip: [f32; 16],
    clip_to_prev_clip: [f32; 16],
    prev_clip_to_clip: [f32; 16],
    jitter_offset: [f32; 2],
    mvec_scale: [f32; 2],
    camera_pinhole_offset: [f32; 2],
    camera_pos: [f32; 3],
    camera_up: [f32; 3],
    camera_right: [f32; 3],
    camera_fwd: [f32; 3],
    camera_near: f32,
    camera_far: f32,
    camera_fov: f32,
    camera_aspect_ratio: f32,
    motion_vectors_invalid_value: f32,
    depth_inverted: u8,
    camera_motion_included: u8,
    motion_vectors_3d: u8,
    reset: u8,
    orthographic_projection: u8,
    motion_vectors_dilated: u8,
    motion_vectors_jittered: u8,
    min_relative_linear_depth_object_separation: f32,
}
const _: () = assert!(size_of::<Constants>() == 456);
const _: () = assert!(std::mem::offset_of!(Constants, jitter_offset) == 352);
const _: () = assert!(std::mem::offset_of!(Constants, camera_near) == 424);
const _: () = assert!(std::mem::offset_of!(Constants, depth_inverted) == 444);
const _: () =
    assert!(std::mem::offset_of!(Constants, min_relative_linear_depth_object_separation) == 452);

/// `sl::DLSSOptions` (version 3).
#[repr(C)]
struct Options {
    base: Base,
    mode: u32,
    output_width: u32,
    output_height: u32,
    sharpness: f32,
    pre_exposure: f32,
    exposure_scale: f32,
    color_buffers_hdr: u8,
    indicator_invert_x: u8,
    indicator_invert_y: u8,
    /// DLAA, quality, balanced, performance, ultra performance and ultra
    /// quality presets, in Streamline's field order.
    presets: [Preset; 6],
    use_auto_exposure: u8,
    alpha_upscaling_enabled: u8,
}
const _: () = assert!(size_of::<Options>() == 88);
const _: () = assert!(std::mem::offset_of!(Options, presets) == 60);
const _: () = assert!(std::mem::offset_of!(Options, use_auto_exposure) == 84);

type SlResult = i32;
type Init = unsafe extern "C" fn(*const Preferences, u64) -> SlResult;
type SetD3dDevice = unsafe extern "C" fn(*mut c_void) -> SlResult;
type GetNewFrameToken = unsafe extern "C" fn(*mut *mut c_void, *const u32) -> SlResult;
type SetConstants =
    unsafe extern "C" fn(*const Constants, *const c_void, *const Viewport) -> SlResult;
type SetTagForFrame = unsafe extern "C" fn(
    *const c_void,
    *const Viewport,
    *const ResourceTag,
    u32,
    *mut c_void,
) -> SlResult;
type EvaluateFeature =
    unsafe extern "C" fn(u32, *const c_void, *const *const Base, u32, *mut c_void) -> SlResult;
type FreeResources = unsafe extern "C" fn(u32, *const Viewport) -> SlResult;
type GetFeatureFunction = unsafe extern "C" fn(u32, *const c_char, *mut *mut c_void) -> SlResult;
type SetOptions = unsafe extern "C" fn(*const Viewport, *const Options) -> SlResult;
type UpgradeInterface = unsafe extern "C" fn(*mut *mut c_void) -> SlResult;
type Farproc = unsafe extern "system" fn() -> isize;

struct Streamline {
    get_new_frame_token: GetNewFrameToken,
    set_constants: SetConstants,
    set_tag_for_frame: SetTagForFrame,
    evaluate_feature: EvaluateFeature,
    free_resources: FreeResources,
    set_options: SetOptions,
    upgrade_interface: UpgradeInterface,
}

/// Streamline's warnings and errors in VIctory's log.
extern "C" fn log_message(kind: u32, message: *const c_char) {
    if message.is_null() {
        return;
    }
    // SAFETY: Streamline's nul-terminated message.
    let text = unsafe { std::ffi::CStr::from_ptr(message) }.to_string_lossy();
    let text = text.trim_end();
    match kind {
        2 => tracing::error!("dlss: {text}"),
        1 => tracing::warn!("dlss: {text}"),
        _ => tracing::debug!("dlss: {text}"),
    }
}

/// VIctory's Streamline project id (any fixed GUID identifies a custom
/// engine's project).
const PROJECT_ID: &std::ffi::CStr = c"6f1c9d2a-4b7e-4a35-9d0e-5c2b8a71f364";

/// Streamline, initialised on first use for `device` (a host
/// `ID3D12Device`). Later calls return the same, whatever the device.
fn streamline(device: *mut c_void) -> Option<&'static Streamline> {
    static SL: OnceLock<Option<Streamline>> = OnceLock::new();
    SL.get_or_init(|| {
        let dir = std::path::PathBuf::from(std::env::var_os("VICTORY_DLSS_DIR")?);
        let interposer = dir.join("sl.interposer.dll").as_os_str().encode_wide_nul();
        // SAFETY: a nul-terminated path.
        let Ok(module) = (unsafe { LoadLibraryW(PCWSTR(interposer.as_ptr())) }) else {
            tracing::error!(
                ?dir,
                "dlss: sl.interposer.dll not found; set VICTORY_DLSS_DIR to the Streamline SDK's bin/x64"
            );
            return None;
        };
        let get = |name: &[u8]| {
            // SAFETY: a loaded module and a nul-terminated name.
            unsafe { GetProcAddress(module, PCSTR(name.as_ptr())) }
        };
        // SAFETY: Streamline's exports, of these types (`sl_core_api.h`).
        let (init, set_device, get_function) = unsafe {
            (
                std::mem::transmute::<Farproc, Init>(get(b"slInit\0")?),
                std::mem::transmute::<Farproc, SetD3dDevice>(get(b"slSetD3DDevice\0")?),
                std::mem::transmute::<Farproc, GetFeatureFunction>(
                    get(b"slGetFeatureFunction\0")?,
                ),
            )
        };
        let plugins = dir.as_os_str().encode_wide_nul();
        let paths = [plugins.as_ptr()];
        let features = [FEATURE_DLSS];
        let preferences = Preferences {
            base: Base::new(PREFERENCES, 1),
            show_console: false,
            // eDefault
            log_level: 1,
            paths_to_plugins: paths.as_ptr(),
            num_paths_to_plugins: 1,
            path_to_logs_and_data: std::ptr::null(),
            allocate_callback: std::ptr::null(),
            release_callback: std::ptr::null(),
            log_message_callback: Some(log_message),
            flags: USE_MANUAL_HOOKING
                | DISABLE_CL_STATE_TRACKING
                | USE_FRAME_BASED_RESOURCE_TAGGING,
            features_to_load: features.as_ptr(),
            num_features_to_load: 1,
            application_id: 0,
            // eCustom
            engine: 0,
            engine_version: c"victory".as_ptr(),
            project_id: PROJECT_ID.as_ptr(),
            // eD3D12
            render_api: 1,
        };
        // SAFETY: preferences that outlive the call.
        let r = unsafe { init(&preferences, SDK_VERSION) };
        if r != 0 {
            tracing::error!(result = r, "dlss: slInit failed");
            return None;
        }
        // SAFETY: the host device, alive for the process.
        let r = unsafe { set_device(device) };
        if r != 0 {
            tracing::error!(result = r, "dlss: slSetD3DDevice failed");
            return None;
        }
        let feature = |name: &[u8]| {
            let mut f = std::ptr::null_mut();
            // SAFETY: a nul-terminated name and an output pointer.
            let r = unsafe { get_function(FEATURE_DLSS, name.as_ptr().cast(), &mut f) };
            (r == 0 && !f.is_null()).then_some(f)
        };
        let Some(set_options) = feature(b"slDLSSSetOptions\0") else {
            tracing::error!("dlss: DLSS is not available on this GPU or driver");
            return None;
        };
        // SAFETY: Streamline's exports and DLSS's function, of these types.
        let sl = unsafe {
            Streamline {
                get_new_frame_token: std::mem::transmute::<Farproc, GetNewFrameToken>(get(
                    b"slGetNewFrameToken\0",
                )?),
                set_constants: std::mem::transmute::<Farproc, SetConstants>(get(
                    b"slSetConstants\0",
                )?),
                set_tag_for_frame: std::mem::transmute::<Farproc, SetTagForFrame>(get(
                    b"slSetTagForFrame\0",
                )?),
                evaluate_feature: std::mem::transmute::<Farproc, EvaluateFeature>(get(
                    b"slEvaluateFeature\0",
                )?),
                free_resources: std::mem::transmute::<Farproc, FreeResources>(get(
                    b"slFreeResources\0",
                )?),
                set_options: std::mem::transmute::<*mut c_void, SetOptions>(set_options),
                upgrade_interface: std::mem::transmute::<Farproc, UpgradeInterface>(get(
                    b"slUpgradeInterface\0",
                )?),
            }
        };
        tracing::info!(?dir, "dlss: Streamline initialised on the host device");
        Some(sl)
    })
    .as_ref()
}

/// The factory to make the presenter's swap chain with: with DLSS on,
/// Streamline's proxy for `factory` (Streamline started for `device`, the
/// host `ID3D12Device`), so that each present runs Streamline's
/// `presentCommon`, which manual hooking otherwise never calls (its
/// bookkeeping and the release of what upscales left); else `factory`.
pub(crate) fn present_factory(device: *mut c_void, factory: IDXGIFactory2) -> IDXGIFactory2 {
    if !enabled() {
        return factory;
    }
    let Some(sl) = streamline(device) else {
        return factory;
    };
    let mut raw = factory.as_raw();
    // SAFETY: a live factory; Streamline replaces the pointer with its
    // proxy, which takes over this reference (`slUpgradeInterface`).
    let r = unsafe { (sl.upgrade_interface)(&mut raw) };
    if r != 0 || raw.is_null() {
        report("slUpgradeInterface", r);
        return factory;
    }
    std::mem::forget(factory);
    tracing::info!("dlss: presenting through Streamline");
    // SAFETY: Streamline's proxy, an IDXGIFactory2, with our reference.
    unsafe { IDXGIFactory2::from_raw(raw) }
}

/// One upscale context of the title: a Streamline viewport.
pub(super) struct Context {
    viewport: u32,
    flags: u32,
    mode: Mode,
    preset: Preset,
}

/// The DLSS mode for a context's largest upscale.
fn mode_for(max_render: [u32; 2], max_upscale: [u32; 2]) -> Mode {
    let ratio = max_upscale[0] as f32 / max_render[0].max(1) as f32;
    match ratio {
        r if r <= 1.01 => Mode::Dlaa,
        r if r < 1.6 => Mode::MaxQuality,
        r if r < 1.85 => Mode::Balanced,
        r if r < 2.5 => Mode::MaxPerformance,
        _ => Mode::UltraPerformance,
    }
}

/// Makes a context for the title's upscaler description `desc` on the host
/// device: the context pointer the title keeps is a `Context` of ours.
///
/// # Safety
///
/// `context` is null or writable; `desc` is an upscaler context
/// description.
pub unsafe fn create(context: *mut *mut c_void, desc: *const c_void) -> ReturnCode {
    let Some(host) = crate::host::host() else {
        return ERROR;
    };
    if context.is_null() || desc.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: the caller's description.
    create_on(context, host.device.as_raw(), unsafe {
        &*desc.cast::<CreateUpscale>()
    })
}

fn create_on(
    context: *mut *mut c_void,
    device: *mut c_void,
    upscale: &CreateUpscale,
) -> ReturnCode {
    if streamline(device).is_none() {
        return ERROR;
    }
    static NEXT: AtomicU32 = AtomicU32::new(0);
    let ctx = Context {
        viewport: NEXT.fetch_add(1, Ordering::Relaxed),
        flags: upscale.flags,
        mode: mode_for(upscale.max_render, upscale.max_upscale),
        preset: if crate::switch::on("VICTORY_DLSS_PRESET_M") {
            Preset::M
        } else {
            Preset::Default
        },
    };
    tracing::info!(
        viewport = ctx.viewport,
        render = ?upscale.max_render,
        upscale = ?upscale.max_upscale,
        flags = upscale.flags,
        mode = ?ctx.mode,
        preset = ?ctx.preset,
        "dlss: upscale context on the host GPU"
    );
    // SAFETY: the title's context pointer, checked for null by the caller.
    unsafe { *context = Box::into_raw(Box::new(ctx)).cast() };
    OK
}

/// Frees a context and the DLSS resources of its viewport.
///
/// # Safety
///
/// `context` is null, or holds null or a context [`create`] made.
pub unsafe fn destroy(context: *mut *mut c_void) -> ReturnCode {
    // SAFETY: the title's context pointer, null or one `create` filled.
    if context.is_null() || unsafe { (*context).is_null() } {
        return ERROR_PARAMETER;
    }
    // SAFETY: a context `create` made, given back once.
    let ctx = unsafe { Box::from_raw((*context).cast::<Context>()) };
    if let Some(sl) = streamline(std::ptr::null_mut()) {
        // SAFETY: a viewport of ours.
        unsafe { (sl.free_resources)(FEATURE_DLSS, &Viewport::new(ctx.viewport)) };
    }
    // SAFETY: as above.
    unsafe { *context = std::ptr::null_mut() };
    OK
}

/// What DLSS gets for a dispatch, worked out from the FidelityFX one.
struct Frame {
    options: Options,
    constants: Constants,
    /// Color, depth, motion vectors, output and, without auto exposure, the
    /// exposure: (buffer type, slot, extent).
    tags: Vec<(u32, usize, Extent)>,
}

fn frame(ctx: &Context, d: &DispatchUpscale) -> Frame {
    let [rw, rh] = d.render_size;
    let [uw, uh] = d.upscale_size;
    let display_mvs = ctx.flags & FFX_DISPLAY_RESOLUTION_MOTION_VECTORS != 0;
    let [mw, mh] = if display_mvs { [uw, uh] } else { [rw, rh] };
    let auto_exposure = ctx.flags & FFX_AUTO_EXPOSURE != 0 || d.resources[3].resource.is_null();
    let flag = |b: bool| if b { TRUE } else { FALSE };
    let identity = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let [near, far, fov, _meters] = d.camera;
    let constants = Constants {
        base: Base::new(CONSTANTS, 2),
        camera_view_to_clip: identity,
        clip_to_camera_view: identity,
        clip_to_lens_clip: identity,
        clip_to_prev_clip: identity,
        prev_clip_to_clip: identity,
        jitter_offset: d.jitter,
        mvec_scale: [
            d.motion_vector_scale[0] / mw.max(1) as f32,
            d.motion_vector_scale[1] / mh.max(1) as f32,
        ],
        camera_pinhole_offset: [0.0; 2],
        camera_pos: [0.0; 3],
        camera_up: [0.0, 1.0, 0.0],
        camera_right: [1.0, 0.0, 0.0],
        camera_fwd: [0.0, 0.0, 1.0],
        camera_near: near,
        camera_far: far,
        camera_fov: fov,
        camera_aspect_ratio: uw as f32 / uh.max(1) as f32,
        motion_vectors_invalid_value: f32::MAX,
        depth_inverted: flag(ctx.flags & FFX_DEPTH_INVERTED != 0),
        camera_motion_included: TRUE,
        motion_vectors_3d: FALSE,
        reset: flag(d.reset),
        orthographic_projection: FALSE,
        motion_vectors_dilated: FALSE,
        motion_vectors_jittered: flag(ctx.flags & FFX_MOTION_VECTORS_JITTER_CANCELLATION != 0),
        min_relative_linear_depth_object_separation: 40.0,
    };
    let options = Options {
        base: Base::new(DLSS_OPTIONS, 3),
        mode: ctx.mode as u32,
        output_width: uw,
        output_height: uh,
        sharpness: 0.0,
        pre_exposure: if d.pre_exposure > 0.0 {
            d.pre_exposure
        } else {
            1.0
        },
        exposure_scale: 1.0,
        color_buffers_hdr: flag(ctx.flags & FFX_HDR != 0),
        indicator_invert_x: FALSE,
        indicator_invert_y: FALSE,
        presets: [ctx.preset; 6],
        use_auto_exposure: flag(auto_exposure),
        alpha_upscaling_enabled: FALSE,
    };
    let render = Extent([0, 0, rw, rh]);
    let mut tags = vec![
        (BUFFER_INPUT_COLOR, 0, render),
        (BUFFER_DEPTH, 1, render),
        (BUFFER_MOTION_VECTORS, 2, Extent([0, 0, mw, mh])),
        (BUFFER_OUTPUT_COLOR, OUTPUT, Extent([0, 0, uw, uh])),
    ];
    if !auto_exposure {
        tags.push((BUFFER_EXPOSURE, 3, Extent::default()));
    }
    Frame {
        options,
        constants,
        tags,
    }
}

/// Records DLSS for the title's upscaling description `desc` whose
/// resources and command list are already the host's (see the module
/// description), its output in `UNORDERED_ACCESS`.
///
/// # Safety
///
/// `context` is a context [`create`] made; `desc` an upscaling description
/// naming live host resources in the supplied D3D12 states and an open host
/// command list.
pub unsafe fn dispatch(context: *mut *mut c_void, desc: *const c_void) -> ReturnCode {
    if context.is_null() || desc.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: the caller's description.
    upscale(context, unsafe { &*desc.cast::<DispatchUpscale>() })
}

fn upscale(context: *mut *mut c_void, d: &DispatchUpscale) -> ReturnCode {
    // SAFETY: the title's context pointer, checked for null by the caller.
    let ctx = unsafe { *context }.cast::<Context>();
    if ctx.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: a context `create` made.
    let ctx = unsafe { &*ctx };
    let Some(sl) = streamline(std::ptr::null_mut()) else {
        return ERROR;
    };
    if d.resources[1].resource.is_null() || d.resources[2].resource.is_null() {
        return ERROR_PARAMETER;
    }
    let f = frame(ctx, d);
    let viewport = Viewport::new(ctx.viewport);
    // One frame of Streamline's at a time, numbered by dispatch.
    static FRAME: Mutex<u32> = Mutex::new(0);
    let mut index = FRAME.lock().unwrap_or_else(|e| e.into_inner());
    *index = index.wrapping_add(1);
    let mut token = std::ptr::null_mut();
    // SAFETY: Streamline's calls with structures that outlive them, on an
    // open host command list.
    let r = unsafe {
        let r = (sl.get_new_frame_token)(&mut token, &*index);
        if r != 0 {
            return report("slGetNewFrameToken", r);
        }
        let r = (sl.set_options)(&viewport, &f.options);
        if r != 0 {
            return report("slDLSSSetOptions", r);
        }
        let r = (sl.set_constants)(&f.constants, token, &viewport);
        if r != 0 {
            return report("slSetConstants", r);
        }
        let mut resources: Vec<Resource> = f
            .tags
            .iter()
            .map(|&(_, slot, _)| {
                let resource = d.resources[slot];
                Resource::texture(resource.resource, resource.state)
            })
            .collect();
        let tags: Vec<ResourceTag> = f
            .tags
            .iter()
            .zip(resources.iter_mut())
            .map(|(&(kind, _, extent), resource)| ResourceTag {
                base: Base::new(RESOURCE_TAG, 1),
                resource,
                kind,
                lifecycle: VALID_UNTIL_EVALUATE,
                extent,
            })
            .collect();
        let r = (sl.set_tag_for_frame)(
            token,
            &viewport,
            tags.as_ptr(),
            tags.len() as u32,
            d.command_list,
        );
        if r != 0 {
            return report("slSetTagForFrame", r);
        }
        let inputs = [(&raw const viewport).cast::<Base>()];
        (sl.evaluate_feature)(FEATURE_DLSS, token, inputs.as_ptr(), 1, d.command_list)
    };
    if r != 0 {
        return report("slEvaluateFeature", r);
    }
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| {
        tracing::info!(render = ?d.render_size, upscale = ?d.upscale_size, mode = ?ctx.mode, preset = ?ctx.preset, "dlss: first upscale recorded on the host GPU");
    });
    OK
}

/// Logs a failed Streamline call (once per call and result).
fn report(call: &'static str, result: SlResult) -> ReturnCode {
    static SEEN: Mutex<Vec<(&str, SlResult)>> = Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if !seen.contains(&(call, result)) {
        seen.push((call, result));
        tracing::error!(result, "dlss: {call} failed");
    }
    ERROR
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jitter_follows_fsr() {
        // FSR's documented sequence lengths: 8 at 1x, 18 at 1.5x, 32 at 2x,
        // 72 at 3x (`super-resolution-upscaler.md`).
        assert_eq!(jitter_phase_count(3840, 3840), 8);
        assert_eq!(jitter_phase_count(2560, 3840), 18);
        assert_eq!(jitter_phase_count(1920, 3840), 32);
        assert_eq!(jitter_phase_count(1280, 3840), 72);
        // Halton(2, 3) from index 1, centred.
        assert_eq!(jitter_offset(0, 8), (0.0, 1.0 / 3.0 - 0.5));
        assert_eq!(jitter_offset(1, 8), (0.25 - 0.5, 2.0 / 3.0 - 0.5));
        assert_eq!(jitter_offset(8, 8), jitter_offset(0, 8));
        assert_eq!(jitter_offset(-1, 8), jitter_offset(7, 8));
    }

    #[test]
    fn queries_answer_like_fsr() {
        #[repr(C)]
        struct PhaseCount {
            header: Header,
            render_width: u32,
            display_width: u32,
            out: *mut i32,
        }
        let mut count = 0;
        let mut q = PhaseCount {
            header: Header {
                kind: QUERY_JITTER_PHASE_COUNT,
                next: std::ptr::null_mut(),
            },
            render_width: 1920,
            display_width: 3840,
            out: &mut count,
        };
        // SAFETY: a description of that type with a live output.
        assert_eq!(unsafe { query((&raw mut q).cast()) }, Some(OK));
        assert_eq!(count, 32);
        #[repr(C)]
        struct RenderSize {
            header: Header,
            display_width: u32,
            display_height: u32,
            quality: u32,
            width: *mut u32,
            height: *mut u32,
        }
        let (mut w, mut h) = (0, 0);
        let mut q = RenderSize {
            header: Header {
                kind: QUERY_RENDER_RESOLUTION,
                next: std::ptr::null_mut(),
            },
            display_width: 3840,
            display_height: 2160,
            quality: 2,
            width: &mut w,
            height: &mut h,
        };
        // SAFETY: as above.
        assert_eq!(unsafe { query((&raw mut q).cast()) }, Some(OK));
        assert_eq!((w, h), (2258, 1270));
        let mut other = Header {
            kind: 0x0001_0009,
            next: std::ptr::null_mut(),
        };
        // SAFETY: a header only.
        assert_eq!(unsafe { query((&raw mut other).cast()) }, None);
    }

    #[test]
    fn modes_follow_the_largest_upscale() {
        assert_eq!(mode_for([3840, 2160], [3840, 2160]), Mode::Dlaa);
        assert_eq!(mode_for([2560, 1440], [3840, 2160]), Mode::MaxQuality);
        assert_eq!(mode_for([2258, 1270], [3840, 2160]), Mode::Balanced);
        assert_eq!(mode_for([1920, 1080], [3840, 2160]), Mode::MaxPerformance);
        assert_eq!(mode_for([1280, 720], [3840, 2160]), Mode::UltraPerformance);
    }

    #[test]
    fn a_dispatch_becomes_dlss_constants_and_tags() {
        let ctx = Context {
            viewport: 0,
            flags: FFX_HDR | FFX_DEPTH_INVERTED | FFX_AUTO_EXPOSURE,
            mode: Mode::MaxPerformance,
            preset: Preset::M,
        };
        let mut d = dispatch_description();
        d.jitter = [0.25, -0.125];
        d.motion_vector_scale = [1920.0, 1080.0];
        d.render_size = [1920, 1080];
        d.upscale_size = [3840, 2160];
        d.reset = true;
        let f = frame(&ctx, &d);
        assert_eq!(f.options.mode, Mode::MaxPerformance as u32);
        assert_eq!(f.options.presets, [Preset::M; 6]);
        assert_eq!(f.constants.jitter_offset, [0.25, -0.125]);
        // UV-space vectors scaled to pixels by FSR are [-1, 1] for DLSS.
        assert_eq!(f.constants.mvec_scale, [1.0, 1.0]);
        assert_eq!(
            (f.constants.depth_inverted, f.constants.reset),
            (TRUE, TRUE)
        );
        assert_eq!(
            (
                f.options.output_width,
                f.options.color_buffers_hdr,
                f.options.use_auto_exposure
            ),
            (3840, TRUE, TRUE)
        );
        let kinds: Vec<u32> = f.tags.iter().map(|t| t.0).collect();
        assert_eq!(
            kinds,
            [
                BUFFER_INPUT_COLOR,
                BUFFER_DEPTH,
                BUFFER_MOTION_VECTORS,
                BUFFER_OUTPUT_COLOR
            ]
        );
        assert_eq!(f.tags[0].2.0, [0, 0, 1920, 1080]);
        assert_eq!(f.tags[3].2.0, [0, 0, 3840, 2160]);
    }

    fn dispatch_description() -> DispatchUpscale {
        // SAFETY: a plain description; zero is a valid value of every field.
        unsafe { std::mem::zeroed() }
    }

    /// DLSS on the host GPU through Streamline: a constant colour comes out
    /// of the upscale unchanged. Needs `VICTORY_DLSS_DIR` (the Streamline
    /// SDK's `bin/x64`) and an NVIDIA RTX GPU; skipped otherwise.
    #[test]
    fn dlss_upscales_a_constant_colour_on_the_host_gpu() {
        if !enabled() {
            eprintln!("VICTORY_DLSS_DIR unset; skipped");
            return;
        }
        gpu::upscale_constant_colour();
    }

    mod gpu {
        use super::*;
        use windows::Win32::Foundation::HANDLE;
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::*;
        use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
        use windows::core::Interface;

        fn texture(
            device: &ID3D12Device,
            size: [u32; 2],
            format: DXGI_FORMAT,
            flags: D3D12_RESOURCE_FLAGS,
            state: D3D12_RESOURCE_STATES,
            clear: Option<[f32; 4]>,
        ) -> ID3D12Resource {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: size[0].into(),
                Height: size[1],
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Flags: flags,
                ..Default::default()
            };
            let clear = clear.map(|c| D3D12_CLEAR_VALUE {
                Format: format,
                Anonymous: D3D12_CLEAR_VALUE_0 { Color: c },
            });
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: descriptions that outlive the call.
            unsafe {
                device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    state,
                    clear.as_ref().map(|c| c as *const _),
                    &mut r,
                )
            }
            .expect("texture");
            r.expect("texture")
        }

        fn transition(
            r: &ID3D12Resource,
            before: D3D12_RESOURCE_STATES,
            after: D3D12_RESOURCE_STATES,
        ) -> D3D12_RESOURCE_BARRIER {
            D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        // SAFETY: a borrowed pointer the barrier does not release.
                        pResource: unsafe { std::mem::transmute_copy(r) },
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        StateBefore: before,
                        StateAfter: after,
                    }),
                },
            }
        }

        fn half(h: u16) -> f32 {
            let (sign, exp, man) = (h >> 15, (h >> 10) & 31, (h & 1023) as f32);
            let v = if exp == 0 {
                man * 2f32.powi(-24)
            } else {
                (man + 1024.0) * 2f32.powi(exp as i32 - 25)
            };
            if sign == 1 { -v } else { v }
        }

        pub(super) fn upscale_constant_colour() {
            let mut device: Option<ID3D12Device> = None;
            // SAFETY: the default adapter.
            unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device) }
                .expect("device");
            let device = device.expect("device");
            // As VIctory does: the device exists before Streamline starts.
            let render = [960, 540];
            let upscale = [1920, 1080];
            let mut ctx = std::ptr::null_mut();
            let create = CreateUpscale {
                kind: 0x0001_0000,
                next: std::ptr::null_mut(),
                flags: 0,
                max_render: render,
                max_upscale: upscale,
                message: std::ptr::null(),
            };
            assert_eq!(create_context(&mut ctx, device.as_raw(), &create), OK);
            assert!(!ctx.is_null());

            let colour = [0.5, 0.25, 0.75, 1.0];
            let rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            let target = D3D12_RESOURCE_STATE_RENDER_TARGET;
            let inputs = [
                (DXGI_FORMAT_R16G16B16A16_FLOAT, colour),
                (DXGI_FORMAT_R32_FLOAT, [0.5, 0.0, 0.0, 0.0]),
                (DXGI_FORMAT_R16G16_FLOAT, [0.0; 4]),
            ]
            .map(|(f, c)| (texture(&device, render, f, rt, target, Some(c)), c));
            let output = texture(
                &device,
                upscale,
                DXGI_FORMAT_R16G16B16A16_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                None,
            );
            // SAFETY: D3D12 calls with descriptions that outlive them;
            // resources live until the fence says the GPU is done.
            unsafe {
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default())
                    .expect("queue");
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .expect("allocator");
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .expect("list");
                let rtvs: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                        NumDescriptors: 3,
                        ..Default::default()
                    })
                    .expect("rtv heap");
                let step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)
                    as usize;
                for (i, (t, c)) in inputs.iter().enumerate() {
                    let mut at = rtvs.GetCPUDescriptorHandleForHeapStart();
                    at.ptr += i * step;
                    device.CreateRenderTargetView(t, None, at);
                    list.ClearRenderTargetView(at, c, None);
                }
                let barriers = inputs.each_ref().map(|(t, _)| {
                    transition(t, target, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
                });
                list.ResourceBarrier(&barriers);

                let mut d = dispatch_description();
                d.kind = 0x0001_0001;
                d.command_list = list.as_raw();
                d.resources[0].resource = inputs[0].0.as_raw();
                d.resources[1].resource = inputs[1].0.as_raw();
                d.resources[2].resource = inputs[2].0.as_raw();
                d.resources[OUTPUT].resource = output.as_raw();
                for resource in &mut d.resources[..3] {
                    resource.state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0 as u32;
                }
                d.resources[OUTPUT].state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS.0 as u32;
                d.motion_vector_scale = [1.0, 1.0];
                d.render_size = render;
                d.upscale_size = upscale;
                d.pre_exposure = 1.0;
                d.reset = true;
                d.camera = [0.1, 1000.0, 1.0, 1.0];
                assert_eq!(super::super::upscale(&mut ctx, &d), OK);

                let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
                let mut total = 0;
                let od = output.GetDesc();
                device.GetCopyableFootprints(
                    &od,
                    0,
                    1,
                    0,
                    Some(&mut footprint),
                    None,
                    None,
                    Some(&mut total),
                );
                let mut readback: Option<ID3D12Resource> = None;
                device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: D3D12_HEAP_TYPE_READBACK,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        &D3D12_RESOURCE_DESC {
                            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                            Width: total,
                            Height: 1,
                            DepthOrArraySize: 1,
                            MipLevels: 1,
                            SampleDesc: DXGI_SAMPLE_DESC {
                                Count: 1,
                                Quality: 0,
                            },
                            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                            ..Default::default()
                        },
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut readback,
                    )
                    .expect("readback");
                let readback = readback.expect("readback");
                list.ResourceBarrier(&[transition(
                    &output,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                list.CopyTextureRegion(
                    &D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&readback),
                        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            PlacedFootprint: footprint,
                        },
                    },
                    0,
                    0,
                    0,
                    &D3D12_TEXTURE_COPY_LOCATION {
                        pResource: std::mem::transmute_copy(&output),
                        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                            SubresourceIndex: 0,
                        },
                    },
                    None,
                );
                list.Close().expect("close");
                queue.ExecuteCommandLists(&[Some(list.cast().expect("list"))]);
                let fence: ID3D12Fence =
                    device.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("fence");
                let event: HANDLE = CreateEventW(None, false, false, None).expect("event");
                queue.Signal(&fence, 1).expect("signal");
                fence.SetEventOnCompletion(1, event).expect("event");
                WaitForSingleObject(event, 10_000);
                device.GetDeviceRemovedReason().expect("device not removed");

                let mut p = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut p)).expect("map");
                let p = p.cast::<u8>();
                let pitch = footprint.Footprint.RowPitch as usize;
                for (x, y) in [(0, 0), (960, 540), (1919, 1079), (311, 977)] {
                    let px = p
                        .add(footprint.Offset as usize + y * pitch + x * 8)
                        .cast::<u16>();
                    let got: Vec<f32> = (0..4).map(|k| half(*px.add(k))).collect();
                    for k in 0..3 {
                        assert!(
                            (got[k] - colour[k]).abs() < 0.01,
                            "pixel ({x}, {y}): {got:?}, want {colour:?}"
                        );
                    }
                }
                readback.Unmap(0, None);
            }
            // SAFETY: the context made above.
            assert_eq!(unsafe { destroy(&mut ctx) }, OK);
        }

        fn create_context(
            ctx: &mut *mut c_void,
            device: *mut c_void,
            create: &CreateUpscale,
        ) -> ReturnCode {
            super::super::create_on(ctx, device, create)
        }
    }
}
