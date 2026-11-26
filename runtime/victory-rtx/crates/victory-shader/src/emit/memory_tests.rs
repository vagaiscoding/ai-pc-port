//! Title memory at addresses the program computes (decision 0005, the
//! unified-memory item), run on the host GPU and WARP with the test host
//! serving the contract (`emit` "Title memory"): a synthetic GFX10 program
//! takes a 64-bit pointer from root constants, reads through it with a
//! global and a scalar load, follows a pointer it loaded with another
//! scalar load, stores through a pointer it computes (64-bit multiply-add)
//! and does float min/max atomics, reads through the address it takes
//! from a buffer view's descriptor (the host's view addresses), reads
//! pointers into memory no copy holds (zeros, marked missed) and stores and
//! loads through a root UAV (title memory in such a program).

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;
const S_WAITCNT_ALL: u32 = 0xBF8C_0000;
/// Inline constant 0, 64.
const ZERO: u16 = 128;
const SIXTY_FOUR: u16 = 192;
/// The title address of the memory the program reads: above 4 GiB.
const TITLE: u64 = 0x12_3450_0000;

const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SMEM `s_load_dword*` (`op` 0, 1, 2: one, two, four dwords) at
/// `s[base:base+1] + offset`.
const fn s_load(op: u32, sdata: u8, sbase: u8, offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | op << 18 | (sdata as u32) << 6 | (sbase as u32 >> 1),
        125 << 25 | offset,
    ]
}

/// VOP1 `v_mov_b32 vdst, src0`.
const fn v_mov_b32(vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | 0x01 << 9 | (src0 as u32 & 0x1FF)
}

/// GLOBAL `op` with address `s[saddr:saddr+1] + vaddr + offset`.
const fn global(op: u32, vdst: u8, vaddr: u8, data: u8, saddr: u8, offset: u32) -> [u32; 2] {
    [
        0x37 << 26 | op << 18 | 2 << 14 | (offset & 0xFFF),
        vaddr as u32 | (data as u32) << 8 | (saddr as u32) << 16 | (vdst as u32) << 24,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

fn program() -> HostShader {
    let mut w = vec![s_mov_b32(5, 0x80)]; // s[4:5]: the table at s4
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(v_mov_b32(1, ZERO));
    w.extend(global(12, 2, 1, 0, 0, 0)); // global_load_dword v2, v1, s[0:1]
    w.extend(s_load(1, 12, 0, 8)); // s[12:13] = the pointer at +8
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 14, 12, 0)); // s14 = what it points at
    w.push(v_mov_b32(3, SIXTY_FOUR));
    w.push(v_mov_b32(4, 2)); // v4 = s2
    // v_mad_u64_u32 v[8:9], s[20:21], v3, 1, s[0:1]: the pointer plus 0x40,
    // stored through as a VGPR pair.
    w.extend([0xD576_1408, 0x0001_0303]);
    w.extend(global(28, 0, 8, 4, 0x7D, 0)); // global_store_dword v[8:9], v4, off
    w.push(v_mov_b32(6, 3)); // v6 = s3 (a float)
    w.extend(global(63, 0, 3, 6, 0, 0x40)); // global_atomic_fmin at +0x80
    w.extend(global(64, 0, 3, 6, 0, 0x44)); // global_atomic_fmax at +0x84
    // The address of the view at heap index 1 from its descriptor (as the
    // tracers take their BVH's), read through.
    w.extend(s_load(1, 16, 4, 0x20)); // s[16:17] = descriptor 1, dwords 0-1
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 18, 16, 0x100)); // s18 = the dword at +0x100
    w.push(S_WAITCNT_ALL);
    // Pointers into a page no copy holds and into a 256 MiB region with no
    // block, read through: zeros, and both marked missed.
    w.extend(s_load(1, 22, 0, 0x10)); // s[22:23]
    w.extend(s_load(1, 24, 0, 0x18)); // s[24:25]
    w.push(S_WAITCNT_ALL);
    w.extend(s_load(0, 26, 22, 0));
    w.extend(s_load(0, 27, 24, 0));
    w.push(S_WAITCNT_ALL);
    w.push(v_mov_b32(5, 14));
    w.push(v_mov_b32(7, 18));
    w.extend(buffer_store_dword(2, 8, 0));
    w.extend(buffer_store_dword(5, 8, 4));
    w.extend(buffer_store_dword(7, 8, 8));
    w.push(v_mov_b32(5, 26));
    w.push(v_mov_b32(7, 27));
    w.extend(buffer_store_dword(5, 8, 12));
    w.extend(buffer_store_dword(7, 8, 16));
    // A root UAV (s[6:7]), title memory in such a program: a store and a
    // load at its address.
    w.extend(global(28, 0, 1, 4, 6, 0x20)); // global_store_dword v1, v4, s[6:7]
    w.extend(global(12, 9, 1, 0, 6, 0x24)); // global_load_dword v9, v1, s[6:7]
    w.push(S_WAITCNT_ALL);
    w.extend(buffer_store_dword(9, 8, 20));
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 10,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 2,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let memory = res.scalar.iter().filter(|s| s.1.param == resource::MEMORY).count();
    assert_eq!(memory, 11, "every access through the pointer is title memory");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes && l.view_addresses);
    assert!(l.root_uavs.is_empty(), "the root UAV is read as title memory");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    shader
}

/// The page table for one copy of title memory at [`TITLE`] whose views
/// are at heap index `view` (null pair at `null`).
fn page_table(view: u32, null: u32) -> Vec<u32> {
    let mut t = vec![0u32; 4096];
    let (hi, lo) = ((TITLE >> 32) as u32, TITLE as u32);
    t[(hi << 4 | lo >> 28) as usize] = 1;
    for _ in 0..2 * 4096 {
        t.extend([null, 0]);
    }
    let entry = 4096 + 2 * (4096 + (lo >> 16 & 0xFFF) as usize);
    t[entry..entry + 2].copy_from_slice(&[view, lo]);
    t
}

#[test]
fn title_memory_is_read_and_written_at_computed_addresses() {
    let shader = program();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut memory = vec![0u32; 256];
    memory[0] = 0x1111_1111;
    let pointer = TITLE + 0x100;
    memory[2] = pointer as u32;
    memory[3] = (pointer >> 32) as u32;
    // The next page (no copy) and another 256 MiB region (no block).
    let (next, far) = (TITLE + 0x1_0010, 0x55_0000_0040u64);
    memory[4..8].copy_from_slice(&[next as u32, (next >> 32) as u32, far as u32, (far >> 32) as u32]);
    memory[0x40] = 0x2222_2222;
    // The root UAV's address, and what it loads there.
    let root_uav = TITLE + 0x200;
    memory[(0x200 + 0x24) / 4] = 0x3333_3333;
    memory[0x80 / 4] = 1.0f32.to_bits();
    memory[0x84 / 4] = (-1.0f32).to_bits();
    let value = 0xCAFE_F00D;
    // Heap index 1 (the memory's view) has the title address.
    let mut view_addresses = vec![0u32; 2 * 5];
    view_addresses[2..4].copy_from_slice(&[TITLE as u32, (TITLE >> 32) as u32]);
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        // fmin with 1.0 and fmax with -1.0 in memory, against a negative
        // and a positive operand.
        for operand in [-2.0f32, 0.5] {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                let constants = [TITLE as u32, (TITLE >> 32) as u32, value, operand.to_bits()];
                ud[u.index as usize] = match u.root {
                    RootType::Constants => constants[u.dword as usize],
                    RootType::Descriptor => (root_uav >> (32 * u.dword)) as u32,
                    _ => 0,
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: page_table(1, 3) }),
                            (2, Root::Uav(2 * 4096 / 32)),
                            (3, Root::View { srv: true, data: view_addresses.clone() }),
                        ],
                        views: &[
                            View::RawUav(6),
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(
                out[0],
                [0x1111_1111, 0x2222_2222, 0x2222_2222, 0, 0, 0x3333_3333],
                "{label}: reads"
            );
            assert_eq!(out[1][0x40 / 4], value, "{label}: the store");
            assert_eq!(out[1][(0x200 + 0x20) / 4], value, "{label}: the root UAV store");
            let floats = [f32::from_bits(out[1][0x20]), f32::from_bits(out[1][0x21])];
            let want = [operand.min(1.0), operand.max(-1.0)];
            assert_eq!(floats, want, "{label}: float min and max");
            // The far region by its level-1 index, the page written, the
            // page missed after it; nothing else.
            let entry = 4096 + (TITLE as u32 >> 16 & 0xFFF) as usize;
            let marked: Vec<usize> = (0..out[2].len() * 32)
                .filter(|&b| out[2][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, [(far >> 28) as usize, entry, entry + 1], "{label}: pages");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// A root SRV in a program that does not use title memory is read through
/// the host's view of its host copy (`emit`, "Root SRVs and root UAVs"): the
/// view's index and the root address's offset in it come as user data, and
/// the shader adds the offset. Lanes read 64 bytes apart from the root
/// address 16 bytes into a 1 KiB copy: those inside it read the copy, those
/// past its end read 0 (a host root descriptor read whatever lay there).
#[test]
fn root_views_read_their_copy_and_zeros_past_it() {
    const fn vop2(op: u32, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        op << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | src0 as u32
    }
    /// MUBUF `op` (offen) of `vdata` at `vaddr` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
        [0x38 << 26 | op << 18 | 1 << 12, 0x80 << 24 | (srsrc as u32 >> 2) << 16 | (vdata as u32) << 8 | vaddr as u32]
    }
    const V_LSHLREV_B32: u32 = 0x1A;
    let mut w = vec![
        s_mov_b32(5, ZERO as u8), // s[4:5]: the table at s4
        s_mov_b32(0, 6),          // s[0:3]: the root SRV's buffer descriptor
        0x8801_FF07,
        0x0001_0000, // s_or_b32 s1, s7, stride 1
        0xBE82_03C1, // s_mov_b32 s2, -1
        0xBE83_03FF,
        0x1101_4FAC, // s_mov_b32 s3, (the console's buffer word 3)
    ];
    w.extend(s_load(2, 8, 4, 0)); // s[8:11]: the result UAV
    w.push(vop2(V_LSHLREV_B32, 1, 128 + 6, 0)); // v1 = lane << 6
    w.extend(mubuf(0x0C, 2, 1, 0)); // buffer_load_dword v2, v1, s[0:3] offen
    w.push(vop2(V_LSHLREV_B32, 3, 128 + 2, 0)); // v3 = lane << 2
    w.push(S_WAITCNT_ALL);
    w.extend(mubuf(0x1C, 2, 3, 8)); // buffer_store_dword v2, v3, s[8:11] offen
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 4,
        layout: vec![
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
            RootParam {
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                sgprs: 2,
                base_register: 0,
                user_data: 6,
                slot: 2,
            },
        ],
        used: vec![
            UsedResource {
                kind: ResourceType::Uav,
                root: RootType::Table,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 0,
                dwords: 4,
                user_data: 4,
                slot: 1,
            },
            UsedResource {
                kind: ResourceType::Srv,
                root: RootType::Descriptor,
                table_offset_dwords: 0,
                lower_bound: 0,
                stride: 1,
                dwords: 4,
                user_data: 6,
                slot: 2,
            },
        ],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 8 << 1,
            threads: [32, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .expect("lower");
    assert!(l.root_srvs.is_empty() && !l.memory, "no root descriptor, no title memory");
    let marks: Vec<(u32, u32)> = l
        .user_data
        .iter()
        .filter(|u| u.root == RootType::Descriptor)
        .map(|u| (u.index, u.dword))
        .collect();
    assert_eq!(marks, [(6, super::ROOT_SRV_VIEW), (7, super::ROOT_VIEW_OFFSET)]);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let copy: Vec<u32> = (0..256).map(|k| 0x1000 + k).collect();
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        // The table at heap index 0, the copy's view at 1, the root address
        // 16 bytes into it.
        (ud[4], ud[6], ud[7]) = (0, 1, 16);
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[
                        View::RawUav(32),
                        View::Buffer {
                            data: &copy,
                            stride: super::host::Stride::Raw,
                        },
                    ],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let want: Vec<u32> = (0..32).map(|lane| if lane < 16 { 0x1000 + 4 + 16 * lane } else { 0 }).collect();
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

/// Writes through unordered buffer views mark their pages (`emit`, "Write
/// marks"): a dword store and a 16-byte store through view 0 (base bit 100,
/// its first byte 16 bytes before a 64 KiB page ends, so the second store
/// crosses into the next page), a store through view 1 (not marked) and an
/// atomic add through view 2 (base bit 200). The stores land; exactly bits
/// 100, 101 and 200 are set.
#[test]
fn writes_through_views_mark_their_pages() {
    /// MUBUF `op` of `vdata` at `byte_off` through `s[srsrc:srsrc+3]`.
    const fn mubuf(op: u32, vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
        [
            0x38 << 26 | op << 18 | (byte_off as u32 & 0xFFF),
            0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
        ]
    }
    const BUFFER_STORE_DWORDX4: u32 = 0x1E;
    const BUFFER_ATOMIC_ADD: u32 = 0x32;
    let mut w = vec![s_mov_b32(5, ZERO as u8)];
    for (k, sdata) in [8u8, 12, 16].into_iter().enumerate() {
        w.extend(s_load(2, sdata, 4, 32 * k as u32));
    }
    w.push(S_WAITCNT_ALL);
    w.extend((0..4).map(|v| v_mov_b32(v, 0)));
    w.extend(buffer_store_dword(0, 8, 0));
    w.extend(mubuf(BUFFER_STORE_DWORDX4, 0, 8, 12));
    w.extend(buffer_store_dword(0, 12, 4));
    w.extend(mubuf(BUFFER_ATOMIC_ADD, 0, 16, 8));
    w.push(S_ENDPGM);
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let uav = |k: u32| UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: k,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![param(RootType::Constants, 0, 4, 0), param(RootType::Table, 4, 1, 1)],
        used: (0..3).map(uav).collect(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: super::Target::ANY,
        console_cf: false,
        per_lane_only: false,
        view_marks: true,
    })
    .expect("lower");
    assert!(l.view_marks && !l.memory, "marks without the page table");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let value = 0x1234_5678;
    let marks = vec![100, 0xFFF0, super::VIEW_MARK_NONE, 0, 200, 0];
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let mut ud = vec![0u32; shader.user_data_dwords() as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = if u.root == RootType::Constants { value } else { 0 };
        }
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (1, Root::View { srv: true, data: marks.clone() }),
                        (2, Root::Uav(8)),
                    ],
                    views: &[View::RawUav(8), View::RawUav(8), View::RawUav(8)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(out[0], [value, 0, 0, value, value, value, value, 0], "{label}: view 0");
        assert_eq!(out[1][1], value, "{label}: view 1");
        assert_eq!(out[2][2], value, "{label}: the atomic");
        let marked: Vec<usize> = (0..out[3].len() * 32)
            .filter(|&b| out[3][b / 32] >> (b % 32) & 1 != 0)
            .collect();
        assert_eq!(marked, [100, 101, 200], "{label}: pages");
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

#[test]
fn straddling_writes_across_256mib_and_4gib_boundaries_mark_exact_pages() {
    let mut w = vec![v_mov_b32(0, ZERO)];
    for k in 0..4u8 {
        // Inline constants 129..=132 are 1..=4.
        w.push(v_mov_b32(1 + k, 129 + u16::from(k)));
    }
    w.extend(global(30, 0, 0, 1, 0, 0)); // global_store_dwordx4 v0, v[1:4], s[0:1]
    w.push(S_ENDPGM);
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 5,
        layout: vec![RootParam {
            root: RootType::Constants,
            table_offset_dwords: 0,
            sgprs: 2,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: Vec::new(),
        io: StageIo::Cs(ComputeIo {
            rsrc2: 2 << 1,
            threads: [1, 1, 1],
        }),
        code: w.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let l = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
        target: Default::default(),
        console_cf: false,
        per_lane_only: false,
        view_marks: false,
    })
    .expect("lower");
    assert!(l.memory && l.memory_writes);
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: l.dxil,
        user_data: l.user_data,
        samplers: l.samplers,
        static_samplers: Vec::new(),
        root_cbvs: l.root_cbvs,
        root_srvs: l.root_srvs,
        root_uavs: l.root_uavs,
        memory: l.memory,
        memory_writes: l.memory_writes,
        view_addresses: l.view_addresses,
        heap: l.heap,
        heap_writes: l.heap_writes,
        view_marks: l.view_marks,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    }

    // 513 blocks (block 0 null + 512 capacity blocks), 4096 entries each,
    // View::Memory at heap slot 0 (UAV at 1) and View::NullPair at slot 2.
    const BLOCKS: usize = 513;
    const ENTRIES: usize = 4096;
    let mut table = vec![0u32; ENTRIES];
    for _ in 0..BLOCKS * ENTRIES {
        table.extend([2, 0]);
    }
    let set_entry = |table: &mut Vec<u32>, block: usize, page: usize, start_lo: u32| {
        let at = ENTRIES + 2 * (block * ENTRIES + page);
        table[at..at + 2].copy_from_slice(&[0, start_lo]);
    };
    // 4 GiB boundary: 0x1_FFFF_0000 (l1 0x1F -> block 2) -> 0x2_0000_0000 (l1 0x20 -> block 1).
    table[0x1F] = 2;
    table[0x20] = 1;
    set_entry(&mut table, 2, 0xFFF, 0xFFFF_0000);
    set_entry(&mut table, 1, 0x000, 0xFFFF_0000);
    // 256 MiB / capacity edge: 0x1_0FFF_0000 (l1 0x10 -> block 512) -> 0x1_1000_0000 (l1 0x11 -> block 0, unmapped).
    table[0x10] = 512;
    set_entry(&mut table, 512, 0xFFF, 0x0FFF_0000);

    // 128 KiB copy spanning two 64 KiB pages; the store writes at +0xFFF8..+0x10008.
    let memory = vec![0u32; 32 * 1024];
    let bitmap_dwords = BLOCKS * ENTRIES / 32;
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host.compute_pipeline(&shader).expect("pipeline");
        let cases: [(&str, u64, Vec<usize>); 3] = [
            (
                "4 GiB boundary",
                0x1_FFFF_FFF8,
                vec![4096, 2 * ENTRIES + 0xFFF],
            ),
            (
                "256 MiB boundary into unmapped region at block capacity",
                0x1_0FFF_FFF8,
                vec![0x11, 512 * ENTRIES + 0xFFF],
            ),
            (
                "same-page write inside 0x1_FFFF_0000",
                0x1_FFFF_FF00,
                vec![2 * ENTRIES + 0xFFF],
            ),
        ];
        for (case, addr, want_bits) in cases {
            let mut ud = vec![0u32; shader.user_data_dwords() as usize];
            for u in &shader.user_data {
                if u.root == RootType::Constants {
                    ud[u.index as usize] = (addr >> (32 * u.dword)) as u32;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (
                                1,
                                Root::View {
                                    srv: true,
                                    data: table.clone(),
                                },
                            ),
                            (2, Root::Uav(bitmap_dwords as u32)),
                        ],
                        views: &[
                            View::Memory(&memory),
                            View::Pair,
                            View::NullPair,
                            View::Pair,
                        ],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label} ({case}): {e}"));
            let off = ((addr as u32) & 0xFFFF) as usize / 4;
            assert_eq!(
                &out[0][off..off + 4],
                &[1, 2, 3, 4],
                "{label} ({case}): store data"
            );
            let marked: Vec<usize> = (0..out[1].len() * 32)
                .filter(|&b| out[1][b / 32] >> (b % 32) & 1 != 0)
                .collect();
            assert_eq!(marked, want_bits, "{label} ({case}): dirty bits");
        }
        ran += 1;
    }
    assert!(ran > 0, "neither hardware nor WARP SM 6.6 device available");
}

enum BcConst { Int(i64), Agg(Vec<u32>) }
enum BcInst { Bin(u64, u32, u32), Cast(u64, u32), Cmp(u64, u32, u32), Extract(u32, u64), Call(String, Vec<u32>), Other }

/// Decoded instructions and conditional branches of `main` from a DXIL container.
struct Bc {
    consts: std::collections::HashMap<u32, BcConst>,
    insts: std::collections::HashMap<u32, (usize, BcInst)>,
    cond_brs: Vec<(usize, usize, u32)>,
}

impl Bc {
    fn parse(dxil: &[u8]) -> Self {
        use std::collections::HashMap;
        let c = crate::dxbc::Container::parse(dxil).expect("dxbc");
        let bc = &c.part(crate::dxbc::FourCC::DXIL).expect("DXIL").data[24..];
        assert_eq!(&bc[..4], b"BC\xC0\xDE");
        let mut pos = 32usize;
        let read = |p: &mut usize, n: u32| -> u64 {
            (0..n).fold(0u64, |v, i| { let b = (bc[*p / 8] >> (*p % 8)) & 1; *p += 1; v | u64::from(b) << i })
        };
        let vbr = |p: &mut usize, w: u32| -> u64 {
            let (mut v, mut sh) = (0u64, 0u32);
            loop {
                let x = read(p, w);
                v |= (x & ((1 << (w - 1)) - 1)) << sh;
                if x >> (w - 1) == 0 { return v; }
                sh += w - 1;
            }
        };
        let (mut stack, mut void_ty, mut ty_void) = (Vec::<(u64, u32)>::new(), None, HashMap::<u32, bool>::new());
        let (mut ty_n, mut globals, mut funcs, mut const_n, mut next_id, mut blk) = (0u32, 0u32, 0u32, 0u32, 0u32, 0usize);
        let (mut consts, mut names, mut insts, mut cond_brs) = (HashMap::new(), HashMap::<u32, String>::new(), HashMap::new(), Vec::new());
        while pos < bc.len() * 8 {
            match read(&mut pos, stack.last().map_or(2, |s| s.1)) {
                0 => { pos = pos.next_multiple_of(32); stack.pop(); if stack.is_empty() { break; } }
                1 => {
                    let (id, nw) = (vbr(&mut pos, 8), vbr(&mut pos, 4) as u32);
                    pos = pos.next_multiple_of(32) + 32;
                    if id == 12 { (next_id, blk) = (globals + funcs + const_n, 0); }
                    stack.push((id, nw));
                }
                3 => {
                    let (code, n) = (vbr(&mut pos, 6), vbr(&mut pos, 6) as usize);
                    let ops: Vec<u64> = (0..n).map(|_| vbr(&mut pos, 6)).collect();
                    match stack.last().map(|s| s.0) {
                        Some(17) if code != 1 && code != 19 => {
                            if code == 2 { void_ty = Some(ty_n); }
                            else if code == 21 { ty_void.insert(ty_n, Some(ops[1] as u32) == void_ty); }
                            ty_n += 1;
                        }
                        Some(8) if code == 7 || code == 8 => if code == 7 { globals += 1 } else { funcs += 1 },
                        Some(11) if code != 1 => {
                            let id = globals + funcs + const_n;
                            if code == 4 { let m = (ops[0] >> 1) as i64; consts.insert(id, BcConst::Int(if ops[0] & 1 == 0 { m } else { m.wrapping_neg() })); }
                            else if code == 7 { consts.insert(id, BcConst::Agg(ops.iter().map(|&o| o as u32).collect())); }
                            const_n += 1;
                        }
                        Some(14) if code == 1 => { names.insert(ops[0] as u32, ops[1..].iter().map(|&c| c as u8 as char).collect()); }
                        Some(12) if code != 1 => {
                            let rel = |r: u64| next_id - r as u32;
                            match code {
                                10 | 11 => { if code == 11 && ops.len() == 3 { cond_brs.push((blk, ops[0] as usize, rel(ops[2]))); } blk += 1; }
                                44 => {}
                                34 if ty_void.get(&(ops[2] as u32)).copied().unwrap_or(false) => {}
                                _ => {
                                    let inst = match code {
                                        2 => BcInst::Bin(ops[2], rel(ops[0]), rel(ops[1])),
                                        3 => BcInst::Cast(ops[2], rel(ops[0])),
                                        26 => BcInst::Extract(rel(ops[0]), ops[1]),
                                        28 => BcInst::Cmp(ops[2], rel(ops[0]), rel(ops[1])),
                                        34 => BcInst::Call(names.get(&rel(ops[3])).cloned().unwrap_or_default(), ops[4..].iter().map(|&r| rel(r)).collect()),
                                        _ => BcInst::Other,
                                    };
                                    insts.insert(next_id, (blk, inst));
                                    next_id += 1;
                                }
                            }
                        }
                        _ => {}
                    }
                }
                other => panic!("unexpected bitcode abbrev {other}"),
            }
        }
        Self { consts, insts, cond_brs }
    }

    fn int(&self, id: u32) -> Option<i64> { match self.consts.get(&id) { Some(BcConst::Int(v)) => Some(*v), _ => None } }

    fn binding(&self, h: u32) -> Option<(i64, i64, i64)> {
        let (_, BcInst::Call(name, args)) = self.insts.get(&h)? else { return None };
        let inner = if name == "dx.op.annotateHandle" { args[1] } else { h };
        let (_, BcInst::Call(name, args)) = self.insts.get(&inner)? else { return None };
        let BcConst::Agg(f) = (name == "dx.op.createHandleFromBinding").then(|| self.consts.get(&args[1]))?? else { return None };
        Some((self.int(f[0])?, self.int(f[2])?, self.int(f[3])?))
    }

    fn calls_on(&self, op: &str, bind: (i64, i64, i64)) -> Vec<(usize, &[u32])> {
        self.insts.values().filter_map(|(b, i)| match i {
            BcInst::Call(n, a) if n == op && self.binding(a[1]) == Some(bind) => Some((*b, a.as_slice())),
            _ => None,
        }).collect()
    }

    fn assert_guarded_dirty_or(&self, blk: usize, args: &[u32], bitmap: (i64, i64, i64)) -> usize {
        assert_eq!((self.binding(args[1]), self.int(args[2])), (Some(bitmap), Some(2)));
        let (word, bit) = (args[3], args[6]);
        let &(test_blk, _, cond) = self.cond_brs.iter().find(|&&(_, then, _)| then == blk).expect("dirty atomic must be branched");
        let (_, BcInst::Cmp(32, a, b)) = self.insts[&cond] else { panic!("guard must be icmp eq 0") };
        let set = if self.int(b) == Some(0) { a } else { assert_eq!(self.int(a), Some(0)); b };
        let (_, BcInst::Bin(10, x, y)) = self.insts[&set] else { panic!("test must mask word with bit") };
        let old = if y == bit { x } else { assert_eq!(x, bit); y };
        let (_, BcInst::Cast(11, ext)) = self.insts[&old] else { panic!("expected bitcast") };
        let (_, BcInst::Extract(load, 0)) = self.insts[&ext] else { panic!("expected extract 0") };
        let (load_blk, BcInst::Call(ref name, ref la)) = self.insts[&load] else { panic!("expected rawBufferLoad") };
        assert_eq!((load_blk, name.as_str(), self.binding(la[1]), la[2]), (test_blk, "dx.op.rawBufferLoad.f32", Some(bitmap), word));
        assert_eq!((self.int(la[4]), self.int(la[5]), self.insts[&word].0, self.insts[&bit].0), (Some(1), Some(4), test_blk, test_blk));
        test_blk
    }

    fn gated_by_ne(&self, blk: usize, word: u32) -> bool {
        let Some((_, BcInst::Bin(7, sh, _))) = self.insts.get(&word) else { return false };
        let Some((_, BcInst::Bin(8, n, _))) = self.insts.get(sh) else { return false };
        self.cond_brs.iter().any(|&(_, t, c)| t == blk && matches!(self.insts.get(&c), Some((_, BcInst::Cmp(33, a, b))) if a == n || b == n))
    }

    fn verify_dirty_marks(&self, bitmap: (i64, i64, i64), want_atomics: usize, want_ne: Option<usize>) {
        let atomics = self.calls_on("dx.op.atomicBinOp.i32", bitmap);
        assert_eq!(atomics.len(), want_atomics);
        let ne = atomics.iter().filter(|&&(b, a)| self.gated_by_ne(self.assert_guarded_dirty_or(b, a, bitmap), a[3])).count();
        if let Some(expected_ne) = want_ne {
            assert_eq!(ne, expected_ne, "2nd marks must check n != bits[0] before loading bitmap");
        }
    }
}

/// Across 13 title-memory accesses (`program`), `memory_null` is loaded once in
/// block 0, `(view, start)` loads coalesce via `raw_dword2`, every dirty-mark
/// atomic is guarded by load + bit-test, and `WAVE_OPS` is not set.
#[test]
fn memory_entry_caches_null_coalesces_loads_and_guards_dirty_atomics() {
    let shader = program();
    let sfi0 = crate::dxbc::Container::parse(&shader.dxil).expect("dxbc").part(crate::dxbc::FourCC::SFI0).expect("SFI0").data;
    assert_eq!(u64::from_le_bytes(sfi0.try_into().unwrap()) & 0x4000, 0, "no WAVE_OPS");
    let bc = Bc::parse(&shader.dxil);
    let pt_loads = bc.calls_on("dx.op.rawBufferLoad.f32", (0, 2, 0));
    assert_eq!(pt_loads.iter().filter(|(_, a)| bc.int(a[2]) == Some(4096 * 4)).map(|(b, _)| *b).collect::<Vec<_>>(), [0]);
    assert_eq!(pt_loads.iter().filter(|(_, a)| (bc.int(a[4]), bc.int(a[5])) == (Some(3), Some(8))).count(), 13);
    // 1 hoisted memory_null + 13 primary memory_page L1 loads + 4 straddling-write L1 loads.
    assert_eq!(pt_loads.iter().filter(|(_, a)| bc.int(a[4]) == Some(1)).count(), 18);
    bc.verify_dirty_marks((0, 2, 1), 17, None);
}

/// Single-byte writes omit the 2nd mark statically in both `Lower::memory` and
/// `Lower::mark_view_write`; `mark_view_write` coalesces `(base, phase)` and
/// checks `n != bits[0]` before loading the bitmap; and `memory_null` hoists
/// to entry block 0 even from a non-entry branch.
#[test]
fn dirty_mark_width_and_branch_hoisting() {
    let mut val = validator::Validator::load(validator::Validator::default_path().as_deref()).ok();
    for (view_mark, bytes, in_branch) in [(false, 1, false), (false, 4, true), (true, 1, false), (true, 4, false)] {
        let dxil = lower::synthetic_mark_dxil(view_mark, bytes, in_branch);
        if let Some(v) = val.as_mut() { v.validate(&mut dxil.clone()).expect("DXIL validation failed"); }
        let bc = Bc::parse(&dxil);
        let (table, bitmap) = if view_mark { ((2, 2, 0), (1, 2, 1)) } else { ((0, 2, 0), (0, 2, 1)) };
        let t_loads = bc.calls_on("dx.op.rawBufferLoad.f32", table);
        assert_eq!(t_loads.iter().filter(|(_, a)| (bc.int(a[4]), bc.int(a[5])) == (Some(3), Some(8))).count(), 1);
        if view_mark { assert_eq!(t_loads.len(), 1); }
        else { assert_eq!(t_loads.iter().filter(|(_, a)| bc.int(a[2]) == Some(4096 * 4)).map(|(b, _)| *b).collect::<Vec<_>>(), [0]); }
        let want = if bytes <= 1 { 1 } else { 2 };
        assert_eq!(bc.calls_on("dx.op.rawBufferLoad.f32", bitmap).len(), want);
        bc.verify_dirty_marks(bitmap, want, view_mark.then_some(usize::from(bytes > 1)));
    }
}
