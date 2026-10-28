//! Persistence, through the guest-facing API rather than around it: a title writes a
//! save the way a title does (`sceIoOpen`/`sceIoWrite`/`sceIoClose`), the run is
//! exported to a container, a FRESH state imports it, and the title reads its save
//! back. That is the whole product claim, and none of it is provable from the pieces
//! alone - a collect that reads the right map and a restore that writes it can both be
//! right while the guest still sees nothing.
use super::*;
use crate::gamedata::GameData;
use crate::world::DeterministicWorld;

const CREAT_WRITE: u32 = SCE_O_WRONLY | SCE_O_CREAT | SCE_O_TRUNC;

fn state() -> VitaState {
    VitaState::new(0, 64, Box::new(DeterministicWorld::default()))
}

/// Write a whole file the way a guest does.
fn guest_write(st: &mut VitaState, path: &str, bytes: &[u8]) {
    let fd = st.io_open(path, CREAT_WRITE);
    assert!(fd > 0, "open {path} failed: {fd}");
    assert_eq!(st.io_write(fd, bytes), Some(bytes.len()));
    assert_eq!(st.io_close(fd), 0);
}

/// Read a whole file the way a guest does, or `None` if it is not there.
fn guest_read(st: &mut VitaState, path: &str) -> Option<Vec<u8>> {
    let fd = st.io_open(path, SCE_O_RDONLY);
    if fd < 0 {
        return None;
    }
    let out = st.io_read(fd, 4096);
    st.io_close(fd);
    out
}

#[test]
fn a_save_survives_a_restart_through_the_container() {
    let mut first = state();
    guest_write(&mut first, "savedata0:/DATA.BIN", b"holes in one: 3");
    first.savedata_slot_put("savedata0:", 1, vec![0xAB; 16]);
    first.trophies.unlock("NPWR00001_00", 7, 0x1234);
    let zip = first.game_data().to_zip("PCSA00000");

    // A DIFFERENT state, as a reload is: nothing carried over but the container.
    let mut second = state();
    assert_eq!(guest_read(&mut second, "savedata0:/DATA.BIN"), None);
    let (data, refused) = GameData::from_zip(&zip).expect("read the container");
    assert!(refused.is_empty(), "{refused:?}");
    second.restore_game_data(&data);

    assert_eq!(
        guest_read(&mut second, "savedata0:/DATA.BIN").as_deref(),
        Some(&b"holes in one: 3"[..])
    );
    assert_eq!(
        second.savedata_slot_get("savedata0:", 1),
        Some(vec![0xAB; 16])
    );
    assert!(second.trophies.is_unlocked("NPWR00001_00", 7));
    assert_eq!(second.trophies.unlocked_at("NPWR00001_00", 7), Some(0x1234));
}

#[test]
fn the_separator_after_a_mount_is_optional() {
    // >>> THE BUG THIS PINS COST EVERY RELOAD ITS SAVE.
    // A Vita's colon IS the separator, so `savedata0:x` and `savedata0:/x` are one
    // file. The title that found it SAVES through `sceAppUtilSaveDataDataSave` (which
    // builds `savedata0:/data0_US.bin`) and LOADS with `sceIoOpen`
    // (`savedata0:data0_US.bin`) - two spellings, one file on the device, two keys
    // here. The read got ENOENT on a file the same run had just written.
    assert_eq!(vfs_key("savedata0:data0_US.bin"), "savedata0:/data0_us.bin");
    assert_eq!(
        vfs_key("savedata0:/data0_US.bin"),
        "savedata0:/data0_us.bin"
    );
    assert_eq!(vfs_key("SAVEDATA0:/x"), vfs_key("savedata0:x"));
    assert_eq!(vfs_key("ux0:data/a//b"), "ux0:/data/a/b");
    // The mount alone stays the mount, with no separator invented for it - it is the
    // key `sceIoDopen` on the root of a save uses.
    assert_eq!(vfs_key("savedata0:"), "savedata0:");
    assert_eq!(vfs_key("savedata0:/"), "savedata0:");
    // `..` cannot climb out of a mount, which on a device it cannot either.
    assert_eq!(vfs_key("savedata0:/../../x"), "savedata0:/x");
    // A `:` after a separator is a filename, not a mount.
    assert_eq!(vfs_key("dir/od:d"), "dir/od:d");
    // And both spellings really do reach the same file through the guest API.
    let mut st = state();
    guest_write(&mut st, "savedata0:/DATA.BIN", b"holes in one: 3");
    assert_eq!(
        guest_read(&mut st, "savedata0:DATA.BIN").as_deref(),
        Some(&b"holes in one: 3"[..]),
        "a save written with the separator must be readable without it"
    );
}

#[test]
fn the_installed_title_is_never_in_the_container() {
    // The guest writing over one of its OWN shipped assets makes that file resident,
    // which is the one way game bytes could get into an export. They must not.
    let mut st = state();
    guest_write(&mut st, "app0:/data/level.gxt", b"not yours to carry");
    guest_write(&mut st, "savedata0:/ok", b"mine");
    let data = st.game_data();
    assert_eq!(data.files.len(), 1);
    assert_eq!(data.files[0].0, "savedata0:/ok");
}

#[test]
fn a_run_that_saves_nothing_is_never_dirty() {
    // The flush check runs every tick, so "the guest did not save" has to be free and
    // has to stay false - a persistence layer that rewrites storage on a run that
    // never saved is a battery cost with no purpose.
    let mut st = state();
    assert!(!st.game_data_dirty());
    guest_write(&mut st, "app0:/data/level.gxt", b"an asset, not a save");
    assert!(
        !st.game_data_dirty(),
        "writing a game asset is not saved state"
    );
    assert!(guest_read(&mut st, "app0:/data/level.gxt").is_some());
    assert!(!st.game_data_dirty(), "reading is not saving");
}

#[test]
fn every_way_a_guest_changes_its_save_raises_the_flag() {
    // One missed site here is a save that is silently never written out, which the
    // user discovers as lost progress - so each mutating call is checked by name.
    //
    // >>> THE LIST IS OVER MUTATING PATHS, NOT OVER THE `sceIo*` API, and getting that
    // wrong is how two sites were missed for real. The first version of this test
    // enumerated the file-descriptor calls, which felt exhaustive and was not: a title
    // whose autosave goes through `sceAppUtilSaveDataDataSave` never opens a descriptor
    // at all, and it wrote 806 KB into the file table every boot with the flag reading
    // false and the container never written. Its delete counterpart was missed the same
    // way. So the question to ask of a new entry is not "is there a call for it" but
    // "can this reach `fs.files`, `fs.dirs`, `fs.stats`, the savedata slots or the
    // trophy ledger" - and every path that can, ends at `FileTable::touch`.
    let each: Vec<(&str, Box<dyn Fn(&mut VitaState)>)> = vec![
        (
            "write",
            Box::new(|st: &mut VitaState| guest_write(st, "savedata0:/a", b"x")),
        ),
        (
            "mkdir",
            Box::new(|st: &mut VitaState| assert_eq!(st.io_mkdir("savedata0:/d"), 0)),
        ),
        (
            "pwrite",
            Box::new(|st: &mut VitaState| {
                let fd = st.io_open("savedata0:/p", CREAT_WRITE);
                assert_eq!(st.io_pwrite(fd, 4, b"z"), Some(1));
                st.io_close(fd);
            }),
        ),
        (
            "remove",
            Box::new(|st: &mut VitaState| {
                guest_write(st, "savedata0:/gone", b"x");
                st.clear_game_data_dirty();
                assert_eq!(st.io_remove("savedata0:/gone"), 0);
            }),
        ),
        (
            "rename",
            Box::new(|st: &mut VitaState| {
                guest_write(st, "savedata0:/from", b"x");
                st.clear_game_data_dirty();
                assert_eq!(st.io_rename("savedata0:/from", "savedata0:/to"), 0);
            }),
        ),
        (
            "chstat",
            Box::new(|st: &mut VitaState| {
                guest_write(st, "savedata0:/s", b"x");
                st.clear_game_data_dirty();
                let over = FileStatOverride {
                    mode: Some(0x21b6),
                    ..Default::default()
                };
                assert_eq!(st.io_chstat("savedata0:/s", over), 0);
            }),
        ),
        (
            "slot",
            Box::new(|st: &mut VitaState| st.savedata_slot_put("savedata0:", 0, vec![1])),
        ),
        // The ninth site, and the one that was missed: `sceAppUtilSaveDataDataSave`
        // writes a whole file per entry through `add_file`, NOT through
        // `open`/`write`/`close`. A title whose autosave uses that API - which is the
        // common shape - saved into the file table with the flag still false, so
        // nothing ever wrote the container. The path is the real autosave's.
        (
            "apputil savedata save",
            Box::new(|st: &mut VitaState| {
                st.add_file("savedata0:/-AUTO-/DATA.BIN", b"progress".to_vec())
            }),
        ),
        // And its counterpart, `sceAppUtilSaveDataDataRemove`. A missed DELETE is worse
        // than a missed write: the stale container stays and the deletion is undone at
        // the next load, so the title deletes the same save for ever.
        (
            "apputil savedata remove",
            Box::new(|st: &mut VitaState| {
                st.add_file("savedata0:/gone.bin", b"x".to_vec());
                st.clear_game_data_dirty();
                assert!(st.remove_file("savedata0:/gone.bin"));
            }),
        ),
        (
            "trophy",
            Box::new(|st: &mut VitaState| {
                st.trophies.unlock("NPWR00001_00", 1, 9);
            }),
        ),
    ];
    for (name, act) in each {
        let mut st = state();
        st.clear_game_data_dirty();
        act(&mut st);
        assert!(
            st.game_data_dirty(),
            "{name} left the save looking unchanged"
        );
    }
}

/// The whole-file save path end to end: what `sceAppUtilSaveDataDataSave` leaves
/// behind must both raise the flag AND survive the container - and the loader
/// preloading the title's own shipped assets through the same function must do
/// neither, or every run would flush on its first tick and carry game bytes.
#[test]
fn a_whole_file_save_survives_the_container_and_a_preload_does_not() {
    let mut first = state();
    first.add_file("app0:/usrdir/course.dat", vec![7; 16]);
    assert!(
        !first.game_data_dirty(),
        "preloading a shipped asset is not a save"
    );
    first.add_file("savedata0:/-AUTO-/DATA.BIN", b"progress".to_vec());
    assert!(
        first.game_data_dirty(),
        "a whole-file save left the save looking unchanged"
    );

    let (data, _) = GameData::from_zip(&first.game_data().to_zip("X")).expect("read");
    assert_eq!(
        data.files,
        vec![(
            "savedata0:/-auto-/data.bin".to_string(),
            b"progress".to_vec()
        )],
        "the container carries the save and nothing the title shipped",
    );
    let mut second = state();
    second.restore_game_data(&data);
    assert_eq!(
        second.read_file("savedata0:/-AUTO-/DATA.BIN").as_deref(),
        Some(&b"progress"[..])
    );
}

#[test]
fn a_restore_does_not_ask_to_be_saved_straight_back_out() {
    let mut first = state();
    guest_write(&mut first, "savedata0:/a", b"x");
    let (data, _) = GameData::from_zip(&first.game_data().to_zip("X")).expect("read");
    let mut second = state();
    second.restore_game_data(&data);
    assert!(!second.game_data_dirty());
    // And what it restored is what a second export produces, byte for byte - so a
    // reload cycle is a fixed point rather than something that drifts each run.
    assert_eq!(
        second.game_data().to_zip("X"),
        first.game_data().to_zip("X")
    );
}

#[test]
fn a_restored_empty_directory_is_still_there() {
    // A directory holding nothing is unrepresentable in a flat path map, so it is the
    // piece of state most likely to be lost across a container - and a title that
    // mkdirs its save folder before writing into it notices.
    let mut first = state();
    assert_eq!(first.io_mkdir("savedata0:/EMPTY"), 0);
    let (data, _) = GameData::from_zip(&first.game_data().to_zip("X")).expect("read");
    let mut second = state();
    second.restore_game_data(&data);
    assert!(second.io_is_dir("savedata0:/EMPTY"));
    // ...under the spelling the title used, which is what its own globbing matches.
    assert_eq!(second.io_mkdir("savedata0:/empty"), SCE_ERROR_ERRNO_EEXIST);
}
