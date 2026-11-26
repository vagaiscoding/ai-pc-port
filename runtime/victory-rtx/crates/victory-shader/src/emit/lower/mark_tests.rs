use super::*;

/// Emits a minimal compute shader exercising [`Lower::memory`] (write) or
/// [`Lower::mark_view_write`] for `bytes`, optionally inside a non-entry branch.
pub(in super::super) fn synthetic_mark_dxil(view_mark: bool, bytes: u32, in_branch: bool) -> Vec<u8> {
    let native = NativeProgram {
        stage: HwStage::Cs, wave64: false, vgprs: 1, layout: vec![], used: vec![],
        io: StageIo::Cs(crate::native::ComputeIo { rsrc2: 0, threads: [1, 1, 1] }),
        code: 0xBF81_0000u32.to_le_bytes().to_vec(),
    };
    let p = crate::ir::build(&native.code, false, 1).expect("ir");
    let st = crate::structure::structure(&p).expect("st");
    let (res, io) = (crate::resource::track(&p, &[], HwStage::Cs, Some(&st)).expect("res"), crate::io::raise(&p, &native).expect("io"));
    let parts = Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
        target: Default::default(),
        console_cf: false,
        per_lane_only: false,
        view_marks: view_mark,
    };
    let bad = HashMap::new();
    let mut l = Lower::new(&parts, &bad, false);
    l.cur = l.s.entry;
    let (z, iu) = (l.s.m.i32(0), l.s.m.undef(l.s.t.i32));
    let at = in_branch.then(|| {
        let tid = l.s.op(l.cur, "threadIdInGroup.i32", Attr::ReadNone, l.s.t.i32, 95, &[z]);
        let cond = l.s.m.cmp(l.cur, pred::NE, tid, z);
        l.begin_guarded(Some(cond))
    }).flatten();
    let (h, off) = if view_mark { (l.s.heap_handle(l.cur, z, false, false, (11 | 0x1000, 0)), z) } else { l.memory(z, z, bytes, true, None) };
    let (m, al, v) = (l.s.m.i8(1), l.s.m.i32(4), l.s.t.void);
    l.s.op(l.cur, "rawBufferStore.i32", Attr::NoUnwind, v, 140, &[h, off, iu, z, iu, iu, iu, m, al]);
    if view_mark { l.mark_view_write(h, z, bytes, None); }
    l.end_guarded(at, &[]);
    l.s.m.terminate(l.cur, Term::Ret);
    l.s.finish()
}
