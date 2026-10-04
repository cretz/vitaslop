//! The import screen - the browser's `#/import` (`app.js` `renderImport` / `startImport`): four
//! tabs for the four things people have (a `.pkg` with its `work.bin` picked SEPARATELY, a
//! dumped folder, a zip, a homebrew `.vpk`), the drop zone, and one progress card that fills in
//! the title, icon and size as soon as the import has identified them.

use std::collections::VecDeque;
use std::time::Instant;

use egui::{RichText, TextureOptions};

use super::{ACCENT, DIM, EDGE, INK, ImportTab, Screen, Shell, WARN, bold, card, dashed_rect, error_card, fmt_bytes, primary};
use crate::library::{self, DirSource};

/// The import's byte rate over a TRAILING window, as the browser measures it: a container is a
/// few big files and thousands of small ones, and an average from the first byte spends the
/// whole import catching up to the current speed. Nothing is reported until the window is wide
/// enough to mean something.
#[derive(Default)]
pub(super) struct Rate {
    samples: VecDeque<(Instant, u64, u64)>,
    smooth: f64,
}

const WINDOW_SECS: f64 = 8.0;

impl Rate {
    /// Record `(done, total)` at `now`; the smoothed bytes per second, 0 while unknown.
    pub(super) fn sample(&mut self, now: Instant, done: u64, total: u64) -> f64 {
        if total == 0 {
            self.samples.clear();
            self.smooth = 0.0;
            return 0.0;
        }
        if self.samples.front().is_some_and(|s| s.2 != total) {
            self.samples.clear();
        }
        if self.samples.back().is_none_or(|s| s.1 != done) {
            self.samples.push_back((now, done, total));
        }
        while self.samples.len() > 2 && self.samples.front().is_some_and(|s| now.duration_since(s.0).as_secs_f64() > WINDOW_SECS) {
            self.samples.pop_front();
        }
        let Some(&(t0, d0, _)) = self.samples.front() else { return 0.0 };
        let secs = now.duration_since(t0).as_secs_f64();
        let bytes = done.saturating_sub(d0);
        if secs >= 2.0 && bytes > 0 {
            let inst = bytes as f64 / secs;
            self.smooth = if self.smooth > 0.0 { self.smooth * 0.7 + inst * 0.3 } else { inst };
        }
        self.smooth
    }
}

/// A remaining time a person can read at a glance, and one that does not twitch: at this range
/// a second of precision is noise, so it is coarse on purpose (the browser's `fmtLeft`).
pub(super) fn fmt_left(secs: f64) -> String {
    let s = secs.max(0.0).round() as u64;
    if s < 55 {
        return format!("{}s", ((s as f64 / 5.0).round() as u64 * 5).max(5));
    }
    let m = (s as f64 / 60.0).round() as u64;
    if m < 60 { format!("{m} min") } else { format!("{}h {}m", m / 60, m % 60) }
}

/// Whether an import error means "this is not something the emulator knows", which gets the
/// browser's friendlier heading and the list of what it does know.
fn unrecognised(e: &str) -> bool {
    let l = e.to_lowercase();
    ["unknown container", "not a", "no param.sfo", "work.bin"].iter().any(|k| l.contains(k))
}

impl Shell {
    pub(super) fn ui_import(&mut self, ui: &mut egui::Ui) {
        ui.label(bold("Add games").size(22.0).color(INK));
        ui.label(RichText::new("vitaslop only works with games you own, from your own console. It provides no games and downloads nothing.").size(13.0));
        ui.add_space(6.0);
        let progress = self.import.as_ref().map(|p| p.lock().unwrap().clone());
        let running = progress.as_ref().is_some_and(|p| !p.finished);
        // The pickers go away while an import runs: a second pick would start a second import
        // into the same directory.
        if !running {
            self.ui_import_pickers(ui);
        }
        let Some(g) = progress else { return };
        if !g.finished {
            ui.ctx().request_repaint();
            card(ui, |ui| {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = 18.0;
                    let icon = g.probe.as_ref().and_then(|p| p.icon0.as_ref());
                    if self.import_icon.is_none()
                        && let Some(img) = icon.and_then(|b| super::decode_png(b))
                    {
                        self.import_icon = Some(ui.ctx().load_texture("import-icon", img, TextureOptions::LINEAR));
                    }
                    match self.import_icon.as_ref() {
                        Some(t) => {
                            let r = ui.add(egui::Image::new((t.id(), egui::vec2(128.0, 128.0))).corner_radius(64.0));
                            ui.painter().circle_stroke(r.rect.center(), 64.0, egui::Stroke::new(1.0, EDGE));
                        }
                        None => {
                            let (r, _) = ui.allocate_exact_size(egui::vec2(128.0, 128.0), egui::Sense::hover());
                            ui.painter().circle(r.center(), 64.0, super::PANEL_2, egui::Stroke::new(1.0, EDGE));
                        }
                    }
                    ui.vertical(|ui| {
                        let (files, bytes) = g.picked;
                        match g.probe.as_ref() {
                            Some(p) => {
                                ui.label(bold(&p.title).size(16.0).color(INK));
                                let kind = if p.kind == "vpk" { "homebrew".to_string() } else if p.zipped { format!("{} in a zip", p.kind) } else { p.kind.clone() };
                                let version = if p.app_version.is_empty() { String::new() } else { format!(" - v{}", p.app_version) };
                                let replacing = if p.replacing { " - replacing what is in the library (saved data is kept)" } else { "" };
                                ui.label(RichText::new(format!("{}{version} - {kind} - {} files - {}{replacing}", p.title_id, p.files, fmt_bytes(p.bytes))).color(DIM).size(13.0));
                            }
                            None => {
                                ui.label(bold(format!("Reading {files} file{}...", if files == 1 { "" } else { "s" })).size(16.0).color(INK));
                                ui.label(RichText::new(format!("{} picked. Keep this window open; a large title takes a few minutes.", fmt_bytes(bytes))).color(DIM).size(13.0));
                            }
                        }
                        let identifying = g.stage == "reading" || g.probe.is_none();
                        let frac = if g.total > 0 && !identifying { (g.done as f32 / g.total as f32).min(1.0) } else { 0.0 };
                        let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 8.0), egui::Sense::hover());
                        ui.painter().rect(r, 6.0, egui::Color32::from_rgb(0x0f, 0x0f, 0x18), egui::Stroke::new(1.0, EDGE), egui::StrokeKind::Inside);
                        let mut fill = r.shrink(1.0);
                        fill.set_width(fill.width() * frac);
                        ui.painter().rect_filled(fill, 5.0, ACCENT);
                        let text = if identifying {
                            format!("identifying, read {}{}", fmt_bytes(g.done), if g.file.is_empty() { String::new() } else { format!(" - {}", g.file) })
                        } else if g.total == 0 {
                            "loading".to_string()
                        } else {
                            let rate = self.import_rate.sample(Instant::now(), g.done, g.total);
                            let at = if rate > 0.0 { format!(" at {:.0} MB/s", rate / 1e6) } else { String::new() };
                            let left = if rate > 0.0 { format!(", about {} left", fmt_left((g.total - g.done.min(g.total)) as f64 / rate)) } else { String::new() };
                            format!("loading {} / {}{at}{left} - {}", fmt_bytes(g.done), fmt_bytes(g.total), g.file)
                        };
                        ui.label(RichText::new(text).color(DIM));
                    });
                });
            });
        } else if let Some(e) = g.error {
            error_card(ui, |ui| {
                let unknown = unrecognised(&e);
                ui.label(bold(if unknown { "Not a title this emulator recognises" } else { "Import failed" }).size(16.0).color(INK));
                ui.label(RichText::new(&e).monospace().color(egui::Color32::from_rgb(0xff, 0xd9, 0xd9)));
                if unknown {
                    ui.label(RichText::new("It looks for a .pkg (with its work.bin picked alongside), a folder with sce_pfs/files.db, or a homebrew .vpk.").color(DIM));
                }
                ui.label(RichText::new("Pick again above to retry.").color(DIM));
            });
        } else if let Some(id) = g.title_id {
            self.import = None;
            self.pick_pkg = None;
            self.pick_work = None;
            self.titles = library::list_titles();
            self.icons.remove(&id);
            self.pictures.remove(&id);
            self.screen = Screen::Title(id);
        }
    }

    fn ui_import_pickers(&mut self, ui: &mut egui::Ui) {
        // `.tabs`: the chosen one filled with the accent, the card below joined to it.
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            for (tab, label) in [
                (ImportTab::Pkg, "Package (.pkg + work.bin)"),
                (ImportTab::Folder, "Dumped folder"),
                (ImportTab::Zip, "Zip"),
                (ImportTab::Vpk, "Homebrew (.vpk)"),
            ] {
                let on = self.import_tab == tab;
                let b = egui::Button::new(bold(label).color(if on { super::ACCENT_INK } else { DIM }))
                    .fill(if on { ACCENT } else { egui::Color32::TRANSPARENT })
                    .stroke(if on { egui::Stroke::NONE } else { egui::Stroke::new(1.0, EDGE) })
                    .corner_radius(egui::CornerRadius { nw: 8, ne: 8, sw: 0, se: 0 });
                if ui.add(b).clicked() {
                    self.import_tab = tab;
                }
            }
        });
        ui.add_space(-ui.spacing().item_spacing.y);
        let mut go: Option<DirSource> = None;
        // `.card.mode`: joined to the tab above it, so its top-left corner is square.
        egui::Frame::new()
            .fill(super::PANEL)
            .stroke(egui::Stroke::new(1.0, EDGE))
            .corner_radius(egui::CornerRadius { nw: 0, ne: 12, sw: 12, se: 12 })
            .inner_margin(egui::Margin::symmetric(16, 14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                match self.import_tab {
                    ImportTab::Pkg => {
                        code_text(
                            ui,
                            &[("A ", false), (".pkg", true), (" from a console and the ", false), ("work.bin", true), (" licence that was made for it (from NoNpDrm or a dump). Both are needed: the pkg is encrypted and the licence holds its key.", false)],
                        );
                        let pkg_name = self.pick_pkg.as_ref().map(|p| {
                            let size = std::fs::metadata(p).map(|m| fmt_bytes(m.len())).unwrap_or_default();
                            format!("{} ({size})", file_name(p))
                        });
                        if pick_row(ui, "1. The package", "Choose .pkg", pkg_name.as_deref())
                            && let Some(p) = rfd::FileDialog::new().add_filter("Vita package", &["pkg", "PKG"]).pick_file()
                        {
                            // A licence beside the package is picked for the person, as the folder pick would find it.
                            if self.pick_work.is_none()
                                && let Some(w) = p.parent().map(|d| d.join("work.bin")).filter(|w| w.is_file())
                            {
                                self.pick_work = Some(w);
                            }
                            self.pick_pkg = Some(p);
                        }
                        ui.separator();
                        let work_name = self.pick_work.as_ref().map(|p| file_name(p));
                        if pick_row(ui, "2. Its licence", "Choose work.bin", work_name.as_deref()) {
                            let mut d = rfd::FileDialog::new().add_filter("Licence", &["bin", "BIN"]);
                            if let Some(dir) = self.pick_pkg.as_ref().and_then(|p| p.parent()) {
                                d = d.set_directory(dir);
                            }
                            if let Some(p) = d.pick_file() {
                                self.pick_work = Some(p);
                            }
                        }
                        ui.add_space(4.0);
                        let ready = self.pick_pkg.is_some() && self.pick_work.is_some();
                        if ui.add_enabled(ready, primary("Import")).clicked()
                            && let (Some(pkg), Some(work)) = (self.pick_pkg.as_ref(), self.pick_work.as_ref())
                        {
                            go = Some(DirSource::pkg_and_licence(pkg, work));
                        }
                    }
                    ImportTab::Folder => {
                        code_text(
                            ui,
                            &[("A folder dumped from a console (with ", false), ("sce_pfs", true), (" and ", false), ("sce_sys", true), (" inside, and the ", false), ("work.bin", true), (" under ", false), ("sce_sys/package", true), (").", false)],
                        );
                        if ui.add(primary("Choose a folder")).clicked()
                            && let Some(p) = rfd::FileDialog::new().pick_folder()
                        {
                            go = DirSource::open(&p).ok();
                        }
                    }
                    ImportTab::Zip => {
                        ui.label(RichText::new("A zip of either of the above.").color(DIM));
                        if ui.add(primary("Choose a .zip")).clicked()
                            && let Some(p) = rfd::FileDialog::new().add_filter("Zip", &["zip", "ZIP"]).pick_file()
                        {
                            go = DirSource::open(&p).ok();
                        }
                    }
                    ImportTab::Vpk => {
                        code_text(ui, &[("A homebrew app as its ", false), (".vpk", true), (", as distributed. Nothing is encrypted, so no licence is needed.", false)]);
                        if ui.add(primary("Choose a .vpk")).clicked()
                            && let Some(p) = rfd::FileDialog::new().add_filter("Homebrew", &["vpk", "VPK"]).pick_file()
                        {
                            go = DirSource::open(&p).ok();
                        }
                    }
                }
            });
        ui.add_space(4.0);
        // The browser's drop zone - the window takes a dropped file or folder on any screen.
        let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 60.0), egui::Sense::hover());
        if self.drop_hover {
            ui.painter().rect_filled(r.shrink(1.0), 12.0, egui::Color32::from_rgb(0x10, 0x16, 0x1a));
        }
        dashed_rect(ui.painter(), r.shrink(1.0), egui::Stroke::new(2.0, if self.drop_hover { ACCENT } else { EDGE }));
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "Or drop files or a folder anywhere here.", egui::FontId::proportional(14.0), DIM);
        if let Some(src) = go {
            self.start_import(src);
        }
    }
}

/// A line of the browser's import help, with its `<code>` spans in the code colour.
fn code_text(ui: &mut egui::Ui, parts: &[(&str, bool)]) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        for (text, code) in parts {
            if *code {
                ui.label(RichText::new(*text).monospace().color(WARN));
            } else {
                ui.label(RichText::new(*text).color(DIM));
            }
        }
    });
}

/// `.pick`: a numbered step, its button and what was chosen. True when the button was clicked.
fn pick_row(ui: &mut egui::Ui, step: &str, button: &str, chosen: Option<&str>) -> bool {
    let mut clicked = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(step).color(INK));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(chosen.unwrap_or("none chosen")).color(DIM));
            clicked = ui.button(bold(button)).clicked();
        });
    });
    clicked
}

fn file_name(p: &std::path::Path) -> String {
    p.file_name().map_or_else(|| p.display().to_string(), |n| n.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn the_rate_waits_for_a_window_then_follows_the_current_speed() {
        let mut r = Rate::default();
        let t0 = Instant::now();
        assert_eq!(r.sample(t0, 0, 1000), 0.0);
        assert_eq!(r.sample(t0 + Duration::from_secs(1), 100, 1000), 0.0, "under two seconds is not a rate yet");
        let at2 = r.sample(t0 + Duration::from_secs(2), 200, 1000);
        assert!((at2 - 100.0).abs() < 1e-6, "{at2}");
        assert_eq!(r.sample(t0 + Duration::from_secs(3), 0, 0), 0.0, "a stage with no total resets it");
    }

    #[test]
    fn time_left_is_coarse() {
        assert_eq!(fmt_left(3.0), "5s");
        assert_eq!(fmt_left(42.0), "40s");
        assert_eq!(fmt_left(130.0), "2 min");
        assert_eq!(fmt_left(3.0 * 3600.0 + 20.0 * 60.0), "3h 20m");
    }

    #[test]
    fn unrecognised_inputs_get_the_friendly_heading() {
        assert!(unrecognised("this pkg has no work.bin and none was found beside it"));
        assert!(unrecognised("no param.sfo, so no title id to file this under"));
        assert!(!unrecognised("disk full"));
    }
}
