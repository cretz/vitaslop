//! `sheet` - montage a run's screenshots into ONE labelled grid image.
//!
//! A gameplay run drops dozens of PNGs. Judging it one file at a time is the
//! slowest thing an agent does: each image is a separate read, and the thing you
//! are actually looking for - where the picture CHANGES, where it stops changing,
//! where it goes wrong - only shows up when consecutive frames sit side by side. A
//! contact sheet turns "read fifty images" into "read one", and puts the frame
//! ordering in front of you while you do it.
//!
//! Usage:
//!   sheet --out <file.png> <png-or-dir> [more...] [options]
//!
//! Options:
//!   --out <file>     the sheet to write (required)
//!   --cols <N>       cells per row (default 4)
//!   --div <N>        shrink each shot by this integer factor (default 3, so a
//!                    960x544 frame becomes a 320x181 thumbnail)
//!   --limit <N>      use at most N images (evenly spaced across the input, so a
//!                    long run still fits on one sheet without losing its shape)
//!
//! Directory arguments contribute their `*.png` files sorted by name, which is why
//! cadence shots are named `<section>-f<frame>`: sorted by name is sorted by frame.

use std::path::PathBuf;
use std::process::ExitCode;

use vitaslop_runtime::render::{png_to_rgba, rgba_to_png};

/// Label bar height under each thumbnail, in pixels (7-pixel glyphs plus padding).
const LABEL_H: u32 = 12;
/// Gap between cells.
const GAP: u32 = 4;
/// Sheet background and label colours.
const BG: [u8; 4] = [24, 24, 28, 255];
const FG: [u8; 4] = [220, 220, 230, 255];

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let mut out: Option<PathBuf> = None;
    let mut cols = 4usize;
    let mut div = 3u32;
    let mut limit: Option<usize> = None;
    let mut inputs: Vec<PathBuf> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        let mut next = || {
            i += 1;
            args.get(i).cloned()
        };
        match a {
            "--out" => out = next().map(PathBuf::from),
            "--cols" => cols = next().and_then(|s| s.parse().ok()).unwrap_or(cols),
            "--div" => div = next().and_then(|s| s.parse().ok()).unwrap_or(div),
            "--limit" => limit = next().and_then(|s| s.parse().ok()),
            "-h" | "--help" => {
                eprintln!("usage: sheet --out <file.png> <png-or-dir>... [--cols N] [--div N] [--limit N]");
                return ExitCode::from(2);
            }
            other => inputs.push(PathBuf::from(other)),
        }
        i += 1;
    }

    let Some(out) = out else {
        eprintln!("error: --out <file.png> is required");
        return ExitCode::from(2);
    };
    if inputs.is_empty() {
        eprintln!("error: no input images or directories");
        return ExitCode::from(2);
    }
    let cols = cols.max(1);
    let div = div.max(1);

    let mut files: Vec<PathBuf> = Vec::new();
    for p in &inputs {
        if p.is_dir() {
            let mut here: Vec<PathBuf> = std::fs::read_dir(p)
                .into_iter()
                .flatten()
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map(|x| x == "png").unwrap_or(false))
                .collect();
            here.sort();
            files.extend(here);
        } else {
            files.push(p.clone());
        }
    }
    // Do not montage a sheet into itself on a re-run.
    files.retain(|f| f != &out);
    if files.is_empty() {
        eprintln!("error: no PNG files found");
        return ExitCode::from(2);
    }
    if let Some(n) = limit {
        files = evenly_spaced(files, n);
    }

    // Decode everything first: a sheet that is missing cells because one file failed
    // silently is worse than no sheet, so a bad image is a hard error naming itself.
    let mut cells: Vec<(String, u32, u32, Vec<u8>)> = Vec::new();
    for f in &files {
        let bytes = match std::fs::read(f) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("error: reading {}: {e}", f.display());
                return ExitCode::FAILURE;
            }
        };
        let (w, h, rgba) = match png_to_rgba(&bytes) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {}: {e}", f.display());
                return ExitCode::FAILURE;
            }
        };
        let (tw, th, thumb) = shrink(w, h, &rgba, div);
        let label = f.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        cells.push((label, tw, th, thumb));
    }

    let cell_w = cells.iter().map(|c| c.1).max().unwrap_or(1);
    let cell_h = cells.iter().map(|c| c.2).max().unwrap_or(1);
    let rows = cells.len().div_ceil(cols);
    let sheet_w = cols as u32 * (cell_w + GAP) + GAP;
    let sheet_h = rows as u32 * (cell_h + LABEL_H + GAP) + GAP;
    let mut sheet = vec![0u8; (sheet_w * sheet_h * 4) as usize];
    for px in sheet.chunks_exact_mut(4) {
        px.copy_from_slice(&BG);
    }

    for (idx, (label, tw, th, thumb)) in cells.iter().enumerate() {
        let cx = GAP + (idx % cols) as u32 * (cell_w + GAP);
        let cy = GAP + (idx / cols) as u32 * (cell_h + LABEL_H + GAP);
        blit(&mut sheet, sheet_w, cx, cy, *tw, *th, thumb);
        draw_text(&mut sheet, sheet_w, sheet_h, cx, cy + cell_h + 2, label, FG);
    }

    if let Some(dir) = out.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Err(e) = std::fs::write(&out, rgba_to_png(sheet_w, sheet_h, &sheet)) {
        eprintln!("error: writing {}: {e}", out.display());
        return ExitCode::FAILURE;
    }
    println!(
        "sheet {} cells ({cols}x{rows}) -> {} ({sheet_w}x{sheet_h})",
        cells.len(),
        out.display()
    );
    ExitCode::SUCCESS
}

/// Keep `n` items spread evenly across `files`, always including the first and last
/// - a run's beginning and end are the two cells you never want dropped.
fn evenly_spaced(files: Vec<PathBuf>, n: usize) -> Vec<PathBuf> {
    if n == 0 || files.len() <= n {
        return files;
    }
    let last = files.len() - 1;
    (0..n).map(|i| files[i * last / (n - 1)].clone()).collect()
}

/// Box-downsample by an integer factor.
fn shrink(w: u32, h: u32, rgba: &[u8], div: u32) -> (u32, u32, Vec<u8>) {
    if div <= 1 {
        return (w, h, rgba.to_vec());
    }
    let (ow, oh) = ((w / div).max(1), (h / div).max(1));
    let mut out = Vec::with_capacity((ow * oh * 4) as usize);
    let inv = (div * div) as u32;
    for oy in 0..oh {
        for ox in 0..ow {
            let mut acc = [0u32; 4];
            for sy in 0..div {
                let row = ((oy * div + sy) * w + ox * div) as usize * 4;
                for sx in 0..div as usize {
                    let p = row + sx * 4;
                    for c in 0..4 {
                        acc[c] += rgba[p + c] as u32;
                    }
                }
            }
            out.extend_from_slice(&[
                (acc[0] / inv) as u8,
                (acc[1] / inv) as u8,
                (acc[2] / inv) as u8,
                (acc[3] / inv) as u8,
            ]);
        }
    }
    (ow, oh, out)
}

/// Copy an image into the sheet at `(x, y)`.
fn blit(sheet: &mut [u8], sheet_w: u32, x: u32, y: u32, w: u32, h: u32, src: &[u8]) {
    for row in 0..h {
        let dst = (((y + row) * sheet_w + x) * 4) as usize;
        let s = (row * w * 4) as usize;
        let n = (w * 4) as usize;
        sheet[dst..dst + n].copy_from_slice(&src[s..s + n]);
    }
}

/// Draw a label in the 5x7 bitmap font, clipped to the sheet.
fn draw_text(sheet: &mut [u8], sw: u32, sh: u32, x: u32, y: u32, text: &str, color: [u8; 4]) {
    let mut cx = x;
    for ch in text.chars() {
        let glyph = glyph(ch);
        for (col, bits) in glyph.iter().enumerate() {
            for row in 0..7u32 {
                if bits >> row & 1 == 0 {
                    continue;
                }
                let px = cx + col as u32;
                let py = y + row;
                if px >= sw || py >= sh {
                    continue;
                }
                let i = ((py * sw + px) * 4) as usize;
                sheet[i..i + 4].copy_from_slice(&color);
            }
        }
        cx += 6;
        if cx + 5 >= sw {
            return;
        }
    }
}

/// The 5x7 glyph for `ch` (five columns, low bit = top row). Lowercase folds to
/// uppercase and anything unknown renders as `?`, so a label is never silently
/// blank.
fn glyph(ch: char) -> [u8; 5] {
    let c = ch.to_ascii_uppercase();
    match c {
        ' ' => [0x00, 0x00, 0x00, 0x00, 0x00],
        '0' => [0x3E, 0x51, 0x49, 0x45, 0x3E],
        '1' => [0x00, 0x42, 0x7F, 0x40, 0x00],
        '2' => [0x42, 0x61, 0x51, 0x49, 0x46],
        '3' => [0x21, 0x41, 0x45, 0x4B, 0x31],
        '4' => [0x18, 0x14, 0x12, 0x7F, 0x10],
        '5' => [0x27, 0x45, 0x45, 0x45, 0x39],
        '6' => [0x3C, 0x4A, 0x49, 0x49, 0x30],
        '7' => [0x01, 0x71, 0x09, 0x05, 0x03],
        '8' => [0x36, 0x49, 0x49, 0x49, 0x36],
        '9' => [0x06, 0x49, 0x49, 0x29, 0x1E],
        'A' => [0x7E, 0x11, 0x11, 0x11, 0x7E],
        'B' => [0x7F, 0x49, 0x49, 0x49, 0x36],
        'C' => [0x3E, 0x41, 0x41, 0x41, 0x22],
        'D' => [0x7F, 0x41, 0x41, 0x22, 0x1C],
        'E' => [0x7F, 0x49, 0x49, 0x49, 0x41],
        'F' => [0x7F, 0x09, 0x09, 0x09, 0x01],
        'G' => [0x3E, 0x41, 0x49, 0x49, 0x7A],
        'H' => [0x7F, 0x08, 0x08, 0x08, 0x7F],
        'I' => [0x00, 0x41, 0x7F, 0x41, 0x00],
        'J' => [0x20, 0x40, 0x41, 0x3F, 0x01],
        'K' => [0x7F, 0x08, 0x14, 0x22, 0x41],
        'L' => [0x7F, 0x40, 0x40, 0x40, 0x40],
        'M' => [0x7F, 0x02, 0x0C, 0x02, 0x7F],
        'N' => [0x7F, 0x04, 0x08, 0x10, 0x7F],
        'O' => [0x3E, 0x41, 0x41, 0x41, 0x3E],
        'P' => [0x7F, 0x09, 0x09, 0x09, 0x06],
        'Q' => [0x3E, 0x41, 0x51, 0x21, 0x5E],
        'R' => [0x7F, 0x09, 0x19, 0x29, 0x46],
        'S' => [0x46, 0x49, 0x49, 0x49, 0x31],
        'T' => [0x01, 0x01, 0x7F, 0x01, 0x01],
        'U' => [0x3F, 0x40, 0x40, 0x40, 0x3F],
        'V' => [0x1F, 0x20, 0x40, 0x20, 0x1F],
        'W' => [0x3F, 0x40, 0x38, 0x40, 0x3F],
        'X' => [0x63, 0x14, 0x08, 0x14, 0x63],
        'Y' => [0x07, 0x08, 0x70, 0x08, 0x07],
        'Z' => [0x61, 0x51, 0x49, 0x45, 0x43],
        '-' => [0x08, 0x08, 0x08, 0x08, 0x08],
        '_' => [0x40, 0x40, 0x40, 0x40, 0x40],
        '.' => [0x00, 0x60, 0x60, 0x00, 0x00],
        ',' => [0x00, 0x50, 0x30, 0x00, 0x00],
        ':' => [0x00, 0x36, 0x36, 0x00, 0x00],
        '/' => [0x20, 0x10, 0x08, 0x04, 0x02],
        '(' => [0x00, 0x1C, 0x22, 0x41, 0x00],
        ')' => [0x00, 0x41, 0x22, 0x1C, 0x00],
        '#' => [0x14, 0x7F, 0x14, 0x7F, 0x14],
        '+' => [0x08, 0x08, 0x3E, 0x08, 0x08],
        '=' => [0x14, 0x14, 0x14, 0x14, 0x14],
        '<' => [0x08, 0x14, 0x22, 0x41, 0x00],
        '>' => [0x00, 0x41, 0x22, 0x14, 0x08],
        '*' => [0x14, 0x08, 0x3E, 0x08, 0x14],
        '!' => [0x00, 0x00, 0x5F, 0x00, 0x00],
        _ => [0x02, 0x01, 0x51, 0x09, 0x06], // '?'
    }
}
