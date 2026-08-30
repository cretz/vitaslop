//! List the guest-visible files an extracted title produces, with sizes.
//!
//! `cargo run -p vitaslop-runtime --example list_game_files -- <extracted-dir> [filter]`
//!
//! Answers "is this asset actually reaching the guest, and how big is it" in seconds,
//! instead of booting the title to find out.
fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args.next().expect("usage: list_game_files <dir> [filter]");
    let filter = args.next().unwrap_or_default().to_lowercase();

    let mut vfs = vitaslop_runtime::ingest::vfs::MemVfs::new();
    let mut stack = Vec::from([std::path::PathBuf::from(&dir)]);
    while let Some(at) = stack.pop() {
        for entry in std::fs::read_dir(&at).expect("read dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if entry.file_type().expect("file type").is_dir() {
                stack.push(path);
            } else {
                let rel = path
                    .strip_prefix(&dir)
                    .expect("under root")
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("/");
                vfs.insert(rel, std::fs::read(&path).expect("read file"));
            }
        }
    }

    let game = vitaslop_runtime::ingest::pipeline::decrypt_container(&mut vfs).expect("decrypt");
    let mut rows: Vec<(String, usize)> =
        game.files.into_files().map(|(p, b)| (p, b.len())).collect();
    rows.sort();
    let mut shown = 0;
    for (path, size) in &rows {
        if filter.is_empty() || path.to_lowercase().contains(&filter) {
            println!("{size:>12}  {path}");
            shown += 1;
        }
    }
    eprintln!("{shown} of {} guest files", rows.len());
}
