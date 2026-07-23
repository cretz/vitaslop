//! End-to-end coverage for the ScePvf vector-font host module. Loads a real font
//! file through the guest filesystem (the first case to do so), then drives
//! NewLib/OpenUserFile/SetCharSize/IsElement/GetCharInfo/GetFontInfo/
//! GetCharGlyphImage/DoneLib and asserts the printed transcript. Run with:
//!   cargo test -p vitaslop-conformance-harness --test vita_pvf

use std::cell::RefCell;
use std::rc::Rc;

use vitaslop_loader as loader;
use vitaslop_native::{DeterministicWorld, HostAbi, VitaEnv, Vm};

const PVF: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/pvf-src/pvf.velf");
/// The Ahem test font (public domain / CC0), preloaded so the guest can open it.
const AHEM: &[u8] =
    include_bytes!("../../vitaslop-conformance-suite-vita/pvf-src/ahem.ttf");

/// The deterministic transcript pvf.c prints. Ahem's geometry fixes the principled
/// values exactly: the horizontal advance at a 16 px em is 16 px = 1024 in 26.6, and
/// 'X' is present while a CJK codepoint is not. The bitmap dimensions and coverage
/// counts are our rasterizer's output for this font/size, pinned as a regression guard.
const EXPECTED: &str = "\
lib_ok=1 err=0
font_ok=1 err=0
setsize_ret=0
isX=1 isBEL=0
charinfo ret=0 w=16 h=16 adv64=1024 left=0 top=13
fontinfo ret=0 numchars=278 maxadv64=1024
glyph ret=0 filled=256 near=256 center=255
donelib=0
";

#[test]
fn pvf_font_calls_produce_correct_results() {
    let m = loader::load(PVF).expect("load pvf.velf");
    let inputs = m.program_inputs();
    let imports: Vec<(u32, u32)> =
        m.imports.iter().map(|i| (i.library_nid, i.func_nid)).collect();

    let mut env = VitaEnv::new(
        imports,
        inputs.base,
        inputs.mem_bytes,
        Box::new(DeterministicWorld::default()),
    );
    // Preload the font the guest opens with scePvfOpenUserFile.
    env.state.add_file("app0:/ahem.ttf", AHEM.to_vec());
    let env = Rc::new(RefCell::new(env));

    let mut vm = Vm::new(
        &inputs.code,
        inputs.base,
        inputs.thumb_entry,
        &inputs.entries,
        &inputs.externs,
        inputs.mem_bytes,
        &HostAbi::default(),
    )
    .expect("instantiate pvf");
    vm.set_import_env(Box::new(env.clone()));

    vm.call(m.entry & !1).expect("run pvf main");

    let env = env.borrow();
    let cap = &env.state.capture;
    let output = String::from_utf8_lossy(&cap.stdout);
    eprintln!("---output---\n{output}------------");

    assert!(cap.unimplemented.is_empty(), "unimplemented NIDs: {:?}", cap.unimplemented);
    assert_eq!(output, EXPECTED);
}
