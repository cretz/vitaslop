use super::dirty_runs_from_pages;

const PAGE: usize = 1 << 12;

/// Scan `map` (page `p`'s stamp is `map[p]`) over `[off, off + len)` at `stamp`.
fn runs(map: &[u8], off: usize, len: usize, stamp: u8) -> Vec<(usize, usize)> {
    let first = off >> 12;
    let last = (off + len - 1) >> 12;
    let mut out = Vec::new();
    dirty_runs_from_pages(first, last, off, len, stamp, |p| map[p], &mut out);
    out
}

/// A clean map returns nothing at all - the whole point of the instrument.
#[test]
fn a_clean_range_has_no_runs() {
    assert_eq!(runs(&[0; 8], 0, 8 * PAGE, 5), Vec::new());
}

/// >>> THE BUG THIS HELPER EXISTS FOR: a write STRADDLING a boundary in the MIDDLE of the
/// range stamps the lower page only, and the upper page must still be reported.
///
/// Before the two engines shared this, both applied the overhang to the range's FIRST page
/// alone, so page 4 below read clean and the returned run stopped short of bytes the guest
/// had overwritten - a texture re-read that silently misses its changed half.
#[test]
fn a_straddle_inside_the_range_reports_the_page_above_it() {
    let mut map = [0u8; 8];
    map[3] = 9; // a write recorded against page 3 may reach into page 4
    let got = runs(&map, 0, 8 * PAGE, 5);
    assert_eq!(
        got,
        vec![(3 * PAGE, 5 * PAGE)],
        "pages 3 AND 4, not page 3 alone"
    );
}

/// The original overhang still holds: a stamp on the page BELOW the range makes the
/// range's first page dirty.
#[test]
fn a_stamp_below_the_range_reaches_its_first_page() {
    let mut map = [0u8; 8];
    map[2] = 9;
    let got = runs(&map, 3 * PAGE, PAGE, 5);
    assert_eq!(
        got,
        vec![(0, PAGE)],
        "the range's own page is dirty by inheritance"
    );
}

/// Page 0 has no predecessor, and asking for one must not panic or wrap.
#[test]
fn the_first_page_of_memory_has_no_page_below() {
    let map = [0u8; 4];
    assert_eq!(runs(&map, 0, PAGE, 1), Vec::new());
}

/// Runs are CLIPPED to the range, so an offset part way into a page yields an offset of
/// zero, and a length ending part way through the last one stops there.
#[test]
fn runs_are_clipped_to_the_range() {
    let mut map = [0u8; 8];
    map[1] = 9;
    let got = runs(&map, PAGE + 100, 200, 5);
    assert_eq!(
        got,
        vec![(0, 200)],
        "relative to `off`, and no wider than `len`"
    );
}

/// A stamp OLDER than the one asked about is not a change since it.
#[test]
fn an_older_stamp_is_not_dirty() {
    let mut map = [0u8; 8];
    map[3] = 4;
    assert_eq!(runs(&map, 0, 8 * PAGE, 5), Vec::new());
    assert_eq!(
        runs(&map, 0, 8 * PAGE, 4),
        vec![(3 * PAGE, 5 * PAGE)],
        "equal counts as at-or-after"
    );
}

/// Two separated writes are two runs, not one spanning the clean pages between them.
#[test]
fn separated_writes_stay_separate_runs() {
    let mut map = [0u8; 12];
    map[1] = 9;
    map[8] = 9;
    let got = runs(&map, 0, 12 * PAGE, 5);
    assert_eq!(got, vec![(PAGE, 3 * PAGE), (8 * PAGE, 10 * PAGE)]);
}
