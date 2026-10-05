//! The native shell: library, title page, settings, import, diagnostics, about, and the
//! in-game menu, drawn with egui over the same wgpu surface the game presents to.
//!
//! One window, one surface. When no title runs, egui owns the whole frame; while
//! one runs, the game is presented first and egui draws the frame-rate badge, the menu
//! and mute buttons and (on Esc or the menu button) the menu over it in the same command
//! encoder. The rules - what a setting is, which knob it means, what a title record holds -
//! are the shared ones in `vitaslop-frontend`; this file is only the drawing and the wiring.
//! The screens follow the browser front end's (`web/index.html`, `app.js`, `player.js`)
//! control for control; where the desktop has no use for one (the touch pad) it says so here.

mod import;
mod play;
mod touchpad;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{Color32, RichText, TextureHandle, TextureOptions};
use vitaslop_frontend::input::Button;
use vitaslop_frontend::meta::TitleMeta;
use vitaslop_frontend::settings::{self, FullscreenStart, PadMode, Scaling, Settings};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::bindings::{Capture, Kind};
use crate::input::Input;
use crate::library::{self, ImportProgress};
use crate::navpad::{Dir, NavPad};
use crate::retail::{DesktopInput, GameDraw, RetailGfx, RetailGuest, SharedInput};
use crate::session::{Session, Stats};
use crate::vitapic;

// The browser front end's palette (`web/app.css` `:root`), so the two read as one product.
pub(crate) const BG: Color32 = Color32::from_rgb(0x0b, 0x0b, 0x12);
pub(crate) const PANEL: Color32 = Color32::from_rgb(0x14, 0x14, 0x1f);
pub(crate) const PANEL_2: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x29);
pub(crate) const EDGE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x3a);
pub(crate) const INK: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xec);
pub(crate) const ACCENT: Color32 = Color32::from_rgb(0x8f, 0xe0, 0xa0);
pub(crate) const ACCENT_INK: Color32 = Color32::from_rgb(0x08, 0x12, 0x0c);
pub(crate) const DIM: Color32 = Color32::from_rgb(0x8f, 0x8f, 0xa3);
pub(crate) const DANGER: Color32 = Color32::from_rgb(0xe8, 0xa0, 0xa8);
pub(crate) const DANGER_EDGE: Color32 = Color32::from_rgb(0x4a, 0x2a, 0x34);
pub(crate) const WARN: Color32 = Color32::from_rgb(0xe0, 0xc0, 0x7f);
/// `.card.error`.
const ERROR_FILL: Color32 = Color32::from_rgb(0x1d, 0x10, 0x15);
const ERROR_EDGE: Color32 = Color32::from_rgb(0x5a, 0x2a, 0x34);

/// The library's order - the browser's sort menu, same four.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum SortMode {
    #[default]
    Recent,
    Played,
    Name,
    Id,
}

impl SortMode {
    const ALL: [SortMode; 4] = [SortMode::Recent, SortMode::Played, SortMode::Name, SortMode::Id];

    fn label(self) -> &'static str {
        match self {
            SortMode::Recent => "Recently added",
            SortMode::Played => "Recently played",
            SortMode::Name => "Name",
            SortMode::Id => "Title id",
        }
    }

    fn apply(self, list: &mut [TitleMeta]) {
        match self {
            SortMode::Recent => list.sort_by_key(|t| std::cmp::Reverse(t.imported_at)),
            SortMode::Played => list.sort_by_key(|t| (std::cmp::Reverse(t.last_played_at), std::cmp::Reverse(t.imported_at))),
            SortMode::Name => list.sort_by_cached_key(|t| t.title.to_lowercase()),
            SortMode::Id => list.sort_by(|a, b| a.title_id.cmp(&b.title_id)),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
enum Screen {
    Library,
    Title(String),
    Settings(Option<String>),
    Import,
    /// What this run reported. The warnings are captured whether or not anyone is watching
    /// stderr (see `crate::log`), so this screen is the only place a person who launched the
    /// app by double-clicking it can read them - beside the run log's path on disk.
    Diagnostics,
    About,
}

/// The import screen's tabs - the browser's four, same order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum ImportTab {
    #[default]
    Pkg,
    Folder,
    Zip,
    Vpk,
}

/// A settings form being edited: the record, the text of the knobs box, and which leaves the
/// title overrides (marked "changed", as the browser marks them).
struct Draft {
    title_id: Option<String>,
    s: Settings,
    knobs_text: String,
    overridden: HashSet<String>,
    saved_at: Option<Instant>,
}

impl Draft {
    /// Whether the title's own record sets `key` (a leaf path such as `keyboard.cross`) or
    /// anything under it.
    fn changed(&self, key: &str) -> bool {
        self.overridden.iter().any(|o| o == key || o.starts_with(&format!("{key}.")))
    }
}

/// The guest being built on a thread (decrypt + link + transpile).
struct Loading {
    title_id: String,
    title: String,
    rx: Receiver<Result<RetailGuest, String>>,
    input: SharedInput,
    settings: Settings,
    started: Instant,
}

/// A question the pages ask before doing something that cannot be undone - the browser's
/// `confirm()` dialogs, same wording.
enum Ask {
    RemoveTitle(String),
    ClearSave { profile: String, id: String },
    RestoreSave { profile: String, id: String, zip: Vec<u8>, summary: String },
    ClearAll { profile: String },
    RestoreBundle { profile: String, entries: Vec<(String, Vec<u8>)> },
    ResetAll,
}

/// The in-game menu's Yes/No panel - the browser's `askConfirm`, which stands in for the
/// menu body until it is answered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MenuAsk {
    Restart,
    Quit,
}

pub(crate) struct Shell {
    window: Option<Arc<Window>>,
    gfx: Option<RetailGfx>,
    egui: egui::Context,
    egui_state: Option<egui_winit::State>,
    renderer: Option<egui_wgpu::Renderer>,
    screen: Screen,
    titles: Vec<TitleMeta>,
    icons: HashMap<String, Option<TextureHandle>>,
    /// The title pages' backdrops (`pic0.png`), loaded on first view like the icons.
    pictures: HashMap<String, Option<TextureHandle>>,
    search: String,
    sort: SortMode,
    draft: Option<Draft>,
    loading: Option<Loading>,
    session: Option<Session>,
    /// The title the session is playing - what an exec (`Session::take_exec`) reboots.
    playing: Option<String>,
    /// The guest frame last presented, and whether the next frame must present anyway - see
    /// `frame`. A guest frame is ENCODED once; later presents re-scale it.
    presented: Option<u64>,
    force_present: bool,
    session_settings: Settings,
    menu_open: bool,
    menu_ask: Option<MenuAsk>,
    /// The outcome of the last thing the menu did (a copy, a save, a remap), shown in it.
    menu_note: Option<String>,
    /// Which binding the menu's controller picture shows and captures.
    menu_ctl: vitapic::Mode,
    stats: Option<Stats>,
    /// A key or pad button being captured for a binding - see `crate::bindings`.
    capture: Option<Capture>,
    /// The on-screen controls' mouse grab - see `touchpad`.
    touchpad: touchpad::PadState,
    /// Which binding the settings page's controller picture shows and captures.
    settings_ctl: vitapic::Mode,
    import: Option<Arc<Mutex<ImportProgress>>>,
    import_tab: ImportTab,
    pick_pkg: Option<PathBuf>,
    pick_work: Option<PathBuf>,
    /// The import's byte rate, over a trailing window - see `import::Rate`.
    import_rate: import::Rate,
    import_icon: Option<TextureHandle>,
    /// Paths dropped on the window since the last frame: a drop of several files arrives as
    /// one event per file, and they are one import (a `.pkg` with its `work.bin`).
    dropped: Vec<PathBuf>,
    /// A file is being dragged over the window - the browser's `.drop.over`.
    drop_hover: bool,
    ask: Option<Ask>,
    /// A page action's outcome (a saved-data download, an upload), shown where it was done.
    page_note: Option<String>,
    /// The settings page's "New profile" name being typed.
    new_profile: Option<String>,
    error: Option<String>,
    /// The browser's FATAL panel: the run is over (an error, a panic, a lost device, or the
    /// title simply exiting) and this says why, with the diagnostics a click away.
    fatal: Option<String>,
    /// A game-only screenshot to write at the next present - the menu's Screenshot.
    shot_to: Option<PathBuf>,
    /// The window went fullscreen because the settings said so when the title started, and
    /// goes back when it stops.
    fullscreen_by_play: bool,
    /// The window title last set, so it is set only when it changes.
    window_title: String,
    /// The gamepad driving the shell's own screens - see `crate::navpad`.
    navpad: NavPad,
    nav: Nav,
    /// `VITASLOP_SHELL_SHOT` - see [`ShellShot`].
    shot: Option<ShellShot>,
    /// The window's normal-state rect, followed as it moves and resizes, written to
    /// `window.json` on close so the next launch opens where this one was. `None` while a rig
    /// owns the geometry (a shell shot, `VITASLOP_WINDOW_SIZE`), which neither reads nor writes it.
    place: Option<library::WindowPlace>,
    /// The window moved or resized since `place` was taken. Read once the events settle (in
    /// `about_to_wait`), not in the event: on Windows a maximize's `Resized` arrives BEFORE
    /// the window reports itself maximized, and taking it there saved the maximized rect as the
    /// normal one - restoring then left a screen-sized window.
    place_moved: bool,
}

/// `VITASLOP_SHELL_SHOT=<png>` (+ `VITASLOP_SHELL_SCREEN=library|settings|settings:<ID>|import|
/// import:<tab>|import-pick:<pkg>|<work.bin>|import-go:<pkg>|<licence> (the package tab's Import)
/// or import-go:<path>[|<path>...] (a drop)|diagnostics|about|capture|
/// title:<ID>|play:<ID>|menu:<ID>`): open that screen - or PLAY that title through the library's
/// own Play path, loading screen and LoadExec included (`menu:` also opens the in-game menu once
/// the title has run `VITASLOP_SHELL_MENU_AT` frames, default 120) - write the window as it SHOWS
/// it after `VITASLOP_SHELL_SHOT_AT` redraws (default 40: egui lays out over a couple of passes, an
/// icon loads in one; for `play:` the count starts when the title runs, for `menu:` when the menu
/// opens), and exit. The shell's
/// pages side by side with the browser's, without anyone clicking through them.
struct ShellShot {
    path: PathBuf,
    screen: String,
    opened: bool,
    redraws: u32,
    at: u32,
    menu_at: u64,
}

impl ShellShot {
    fn from_knobs() -> Option<ShellShot> {
        let path = std::env::var("VITASLOP_SHELL_SHOT").ok()?;
        let screen = std::env::var("VITASLOP_SHELL_SCREEN").unwrap_or_else(|_| "library".into());
        let at = std::env::var("VITASLOP_SHELL_SHOT_AT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(40);
        let menu_at = std::env::var("VITASLOP_SHELL_MENU_AT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(120);
        Some(ShellShot { path: PathBuf::from(path), screen, opened: false, redraws: 0, at, menu_at })
    }
}

/// What the pad asked of the UI this frame, and what the UI told the pad last frame.
#[derive(Default)]
struct Nav {
    /// East: back out of whatever is open. Handled at the top of `ui`.
    back: bool,
    /// Start: the settings from a page, or close the in-game menu.
    start: bool,
    /// A value step for the focused combo box - left/right, or south, change its value as they
    /// do a `<select>` in the browser, instead of moving focus or opening its list.
    adjust: i32,
    /// Whether a combo box had focus at the end of the last pass - set by the combos themselves.
    combo_focused: bool,
    /// Passes until the focused widget is scrolled into view: focus moves at the END of the pass
    /// that read the arrow, so the scroll is asked for at the start of the next one.
    scroll_in: u8,
}

/// Put a settings record's run knobs in the environment, where the engine reads them.
/// Browser-only knobs are left out (the shell fast-forwards itself).
fn apply_run_knobs(s: &Settings) {
    // >>> A TITLE'S KNOBS END WITH ITS RUN. Every knob an earlier apply set is put back to what
    // the process started with (or removed) first, so one title's advanced-box knob does not
    // follow the person into the next title they play.
    static SET: Mutex<Vec<(String, Option<std::ffi::OsString>)>> = Mutex::new(Vec::new());
    let knobs = s.run_knobs();
    let mut set = SET.lock().unwrap_or_else(|e| e.into_inner());
    for (k, original) in set.iter() {
        if !knobs.contains_key(k) {
            // SAFETY: as below.
            match original {
                Some(v) => unsafe { std::env::set_var(k, v) },
                None => unsafe { std::env::remove_var(k) },
            }
        }
    }
    for (k, v) in knobs {
        if k == "VITASLOP_BROWSER_FASTFORWARD" {
            continue;
        }
        if !set.iter().any(|(s, _)| *s == k) {
            set.push((k.clone(), std::env::var_os(&k)));
        }
        // SAFETY: set on the UI thread, before the window's renderer is made or a title's
        // loader thread is spawned - nothing reads the environment concurrently.
        unsafe { std::env::set_var(&k, &v) };
    }
    crate::log::follow_env();
}

pub fn run() -> Result<(), String> {
    // The global settings' knobs before the window and its renderer exist, so a knob read at
    // construction (the device's features, say) sees the person's settings and not none.
    apply_run_knobs(&library::effective(None));
    let event_loop = EventLoop::new().map_err(|e| format!("create event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = Shell::new(NavPad::new(), ShellShot::from_knobs());
    app.titles = library::list_titles();
    load_ui_fonts(&app.egui);
    apply_theme(&app.egui);
    event_loop.run_app(&mut app).map_err(|e| format!("run event loop: {e}"))?;
    if let Some(s) = app.session.as_mut() {
        s.guest.flush_save(true);
    }
    Ok(())
}

impl ApplicationHandler for Shell {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let knob = crate::retail::window_size_knob();
        let (w, h) = knob.unwrap_or((1100, 700));
        let mut attrs = crate::icon::with_icon(Window::default_attributes()).with_title("vitaslop").with_inner_size(LogicalSize::new(w, h));
        let remember = knob.is_none() && self.shot.is_none();
        let saved = if remember { library::window_place() } else { None };
        if let Some(p) = saved {
            attrs = attrs.with_inner_size(PhysicalSize::new(p.width, p.height));
            // Only where a monitor still is: a window saved on a display since unplugged would
            // open out of reach, so then the OS places it (at the saved size).
            if on_a_monitor(event_loop, &p) {
                attrs = attrs.with_position(PhysicalPosition::new(p.x, p.y));
            }
            attrs = attrs.with_maximized(p.maximized);
            if p.fullscreen {
                attrs = attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
            }
        }
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        if remember {
            self.place = Some(saved.unwrap_or_else(|| {
                let pos = window.outer_position().unwrap_or_default();
                let size = window.inner_size();
                library::WindowPlace { x: pos.x, y: pos.y, width: size.width, height: size.height, maximized: false, fullscreen: false }
            }));
        }
        match RetailGfx::new(window.clone()) {
            Ok(g) => {
                self.renderer = Some(egui_wgpu::Renderer::new(g.device(), g.render_format(), egui_wgpu::RendererOptions::default()));
                self.gfx = Some(g);
            }
            Err(e) => {
                crate::log::file_line(&format!("failed to init GPU surface: {e}"));
                eprintln!("failed to init GPU surface: {e}");
                let _ = rfd::MessageDialog::new()
                    .set_level(rfd::MessageLevel::Error)
                    .set_title("vitaslop")
                    .set_description(format!("vitaslop could not open its window on the GPU: {e}"))
                    .show();
                event_loop.exit();
                return;
            }
        }
        self.egui_state = Some(egui_winit::State::new(
            self.egui.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        ));
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else { return };
        let size = {
            let s = window.inner_size();
            (s.width.max(1) as f64, s.height.max(1) as f64)
        };
        // A capture takes the next key before anyone else sees it - egui would move focus on an
        // arrow, the game would get a press - and Esc cancels it.
        if let (Some(c), WindowEvent::KeyboardInput { event: k, .. }) = (self.capture, &event) {
            if let PhysicalKey::Code(code) = k.physical_key
                && k.state == ElementState::Pressed
                && !k.repeat
            {
                if code == KeyCode::Escape {
                    self.cancel_capture();
                } else if c.kind == Kind::Keyboard {
                    self.finish_capture(crate::bindings::key_name(code));
                }
            }
            return;
        }
        // egui sees every event when it owns the screen. While a game runs it sees the pointer
        // (the menu and mute buttons sit over the game, as the browser's do) and, with the menu
        // open, everything; the game gets what egui does not take.
        let pointer = matches!(
            event,
            WindowEvent::CursorMoved { .. } | WindowEvent::CursorLeft { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. }
        );
        let ui_wants = self.session.is_none() || self.menu_open || self.fatal.is_some() || pointer;
        let mut consumed = false;
        if ui_wants && let Some(st) = self.egui_state.as_mut() {
            consumed = st.on_window_event(&window, &event).consumed;
        }
        match &event {
            WindowEvent::CloseRequested => {
                if let Some(s) = self.session.as_mut() {
                    s.guest.flush_save(true);
                }
                self.save_place(&window);
                event_loop.exit();
            }
            WindowEvent::Resized(sz) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(sz.width, sz.height);
                }
                self.force_present = true;
                self.place_moved = true;
            }
            WindowEvent::Moved(_) => self.place_moved = true,
            // The browser's "drop files or a folder anywhere here": what is dropped is imported,
            // from any screen but a running game. Gathered, then imported together next frame.
            WindowEvent::DroppedFile(path) if self.session.is_none() => {
                self.dropped.push(path.clone());
                self.drop_hover = false;
            }
            WindowEvent::HoveredFile(_) => self.drop_hover = self.session.is_none(),
            WindowEvent::HoveredFileCancelled => self.drop_hover = false,
            WindowEvent::KeyboardInput { event: k, .. } => {
                if let PhysicalKey::Code(code) = k.physical_key {
                    let pressed = k.state == ElementState::Pressed;
                    // Esc opens and closes the menu - or backs out of its Yes/No panel first, as
                    // the browser's `menuBack` does.
                    if pressed && code == KeyCode::Escape && !k.repeat && self.session.is_some() && self.fatal.is_none() {
                        if self.menu_ask.take().is_none() {
                            self.toggle_menu();
                        }
                        return;
                    }
                    if pressed && code == KeyCode::F11 && !k.repeat {
                        self.toggle_fullscreen();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                self.frame(size, &window);
                return;
            }
            _ => {}
        }
        if let Some(s) = self.session.as_mut() {
            if !self.menu_open && !consumed && self.fatal.is_none() {
                s.event(&event, Some(size));
            } else if let WindowEvent::Focused(_) = &event {
                s.event(&event, Some(size));
            }
        }
    }

    /// While a game runs, SLEEP until its next frame is due (a redraw with nothing new is
    /// skipped - see `frame`); the library, the menu and a pause redraw as before.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if std::mem::take(&mut self.place_moved)
            && let Some(w) = self.window.clone()
        {
            self.follow_place(&w);
        }
        if self.shot.as_ref().is_some_and(|s| s.redraws > s.at + 1) {
            event_loop.exit();
            return;
        }
        let due = match self.session.as_ref() {
            Some(s) if !self.menu_open && s.live() && !self.force_present => s.due_in(),
            _ => Duration::ZERO,
        };
        if due > Duration::from_millis(1) {
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + due));
        } else {
            event_loop.set_control_flow(ControlFlow::Poll);
            if let Some(w) = self.window.as_ref() {
                w.request_redraw();
            }
        }
    }
}

impl Shell {
    fn new(navpad: NavPad, shot: Option<ShellShot>) -> Shell {
        Shell {
            window: None,
            gfx: None,
            egui: egui::Context::default(),
            egui_state: None,
            renderer: None,
            screen: Screen::Library,
            titles: Vec::new(),
            icons: HashMap::new(),
            pictures: HashMap::new(),
            search: String::new(),
            sort: SortMode::default(),
            draft: None,
            loading: None,
            session: None,
            playing: None,
            presented: None,
            force_present: true,
            session_settings: Settings::default(),
            menu_open: false,
            menu_ask: None,
            menu_note: None,
            menu_ctl: vitapic::Mode::Keyboard,
            stats: None,
            capture: None,
            touchpad: touchpad::PadState::default(),
            settings_ctl: vitapic::Mode::Keyboard,
            import: None,
            import_tab: ImportTab::default(),
            pick_pkg: None,
            pick_work: None,
            import_rate: import::Rate::default(),
            import_icon: None,
            dropped: Vec::new(),
            drop_hover: false,
            ask: None,
            page_note: None,
            new_profile: None,
            error: None,
            fatal: None,
            shot_to: None,
            fullscreen_by_play: false,
            window_title: String::new(),
            navpad,
            nav: Nav::default(),
            shot,
            place: None,
            place_moved: false,
        }
    }

    /// A stand-in so `ui` can borrow the whole shell while the context runs.
    fn placeholder() -> Shell {
        Shell::new(NavPad::inert(), None)
    }

    fn toggle_menu(&mut self) {
        self.menu_open = !self.menu_open;
        self.menu_ask = None;
        self.cancel_capture();
        if let Some(s) = self.session.as_mut() {
            s.paused = self.menu_open;
            if self.menu_open {
                s.input.release_all();
            } else {
                // The press that closed the menu must not reach the game.
                s.input.hand_back();
            }
        }
    }

    fn toggle_fullscreen(&mut self) {
        if let Some(w) = self.window.as_ref() {
            let full = w.fullscreen().is_some();
            w.set_fullscreen(if full { None } else { Some(winit::window::Fullscreen::Borderless(None)) });
            // Leaving fullscreen by hand is a choice: stopping the title does not undo it again.
            self.fullscreen_by_play = false;
        }
    }

    /// Takes the window's rect into `place` while it is in its normal state - a maximized,
    /// fullscreen or minimized window's rect is not where it returns to.
    fn follow_place(&mut self, window: &Window) {
        let Some(p) = self.place.as_mut() else { return };
        if window.is_maximized() || window.fullscreen().is_some() || window.is_minimized() == Some(true) {
            return;
        }
        if let Ok(pos) = window.outer_position() {
            (p.x, p.y) = (pos.x, pos.y);
        }
        let size = window.inner_size();
        if size.width > 0 && size.height > 0 {
            (p.width, p.height) = (size.width, size.height);
        }
    }

    fn save_place(&mut self, window: &Window) {
        if std::mem::take(&mut self.place_moved) {
            self.follow_place(window);
        }
        let Some(mut p) = self.place else { return };
        p.maximized = window.is_maximized();
        // Fullscreen the settings entered for a title is theirs to enter again, not the window's.
        p.fullscreen = window.fullscreen().is_some() && !self.fullscreen_by_play;
        if let Err(e) = library::save_window_place(&p) {
            crate::log::file_line(&format!("could not save the window position: {e}"));
        }
    }

    fn is_fullscreen(&self) -> bool {
        self.window.as_ref().is_some_and(|w| w.fullscreen().is_some())
    }

    fn set_window_title(&mut self, title: &str) {
        if self.window_title != title {
            self.window_title = title.to_string();
            if let Some(w) = self.window.as_ref() {
                w.set_title(title);
            }
        }
    }

    // ------------------------------- bindings -------------------------------

    /// Start capturing `button`'s binding: for the running game (`live`) or the settings draft.
    fn begin_capture(&mut self, button: Button, kind: Kind, live: bool) {
        self.capture = Some(Capture { button, kind, live });
        if kind == Kind::Gamepad {
            self.navpad.begin_capture();
        }
    }

    fn cancel_capture(&mut self) {
        self.capture = None;
        self.navpad.cancel_capture();
    }

    /// The captured key or pad control becomes the binding: in the draft (kept on Save), or -
    /// from the in-game menu - in the running input at once and on disk (see `crate::bindings`).
    fn finish_capture(&mut self, value: String) {
        let Some(c) = self.capture.take() else { return };
        self.navpad.cancel_capture();
        let set = |s: &mut Settings| {
            let map = match c.kind {
                Kind::Keyboard => &mut s.keyboard,
                Kind::Gamepad => &mut s.gamepad,
            };
            map.insert(c.button.name().to_string(), value.clone());
        };
        if c.live {
            set(&mut self.session_settings);
            if let Some(s) = self.session.as_mut() {
                s.input.apply(&self.session_settings);
            }
            if let Some(id) = self.playing.clone() {
                self.menu_note = Some(match crate::bindings::save_live(&id, c.kind, c.button, &value) {
                    Ok(crate::bindings::Target::Title) => format!("{} is now {value}, saved for this title (its settings already change it).", c.button.label()),
                    Ok(crate::bindings::Target::Global) => format!("{} is now {value}, saved for every title.", c.button.label()),
                    Err(e) => format!("{} is now {value} for this run, but it could not be saved: {e}", c.button.label()),
                });
            }
        } else if let Some(d) = self.draft.as_mut() {
            set(&mut d.s);
        }
    }

    // ------------------------------- the pad -------------------------------

    /// Read the shell's pad and turn this frame's asks into egui input: a move is an arrow key
    /// (egui's own spatial focus), or Tab when nothing has focus yet; south is Enter, which
    /// clicks a focused button or toggles a checkbox. The pad is the shell's only while the shell
    /// owns the screen - no game, or the in-game menu open; otherwise the game's input has it.
    fn pad_to_egui(&mut self, raw: &mut egui::RawInput) {
        self.navpad.attach(self.session.is_none() || self.menu_open || self.fatal.is_some());
        let pad = self.navpad.poll();
        if let Some(control) = pad.captured {
            self.finish_capture(control.to_string());
            return;
        }
        if pad.is_empty() {
            return;
        }
        let key = push_key;
        let focused = self.egui.memory(|m| m.focused()).is_some();
        for d in pad.moves {
            match d {
                Dir::Left | Dir::Right if self.nav.combo_focused => self.nav.adjust += if d == Dir::Right { 1 } else { -1 },
                _ if !focused => key(raw, egui::Key::Tab),
                Dir::Up => key(raw, egui::Key::ArrowUp),
                Dir::Down => key(raw, egui::Key::ArrowDown),
                Dir::Left => key(raw, egui::Key::ArrowLeft),
                Dir::Right => key(raw, egui::Key::ArrowRight),
            }
            self.nav.scroll_in = 2;
        }
        if pad.south {
            if self.nav.combo_focused {
                self.nav.adjust += 1;
            } else if focused {
                key(raw, egui::Key::Enter);
            } else {
                key(raw, egui::Key::Tab);
            }
        }
        self.nav.back |= pad.east;
        self.nav.start |= pad.start;
    }

    /// The pad's east and start, and the scroll that follows a pad move - at the top of each pass.
    fn nav_pass(&mut self, ctx: &egui::Context) {
        self.nav.combo_focused = false;
        if self.nav.scroll_in > 0 {
            self.nav.scroll_in -= 1;
            if self.nav.scroll_in == 0
                && let Some(r) = ctx.memory(|m| m.focused()).and_then(|id| ctx.read_response(id))
            {
                r.scroll_to_me(None);
            }
        }
        if std::mem::take(&mut self.nav.start) {
            if self.session.is_some() {
                if self.menu_open {
                    self.toggle_menu();
                }
            } else if !matches!(self.screen, Screen::Settings(_)) {
                self.open_draft(None);
            }
        }
        if !std::mem::take(&mut self.nav.back) {
            return;
        }
        // The innermost open thing first, as a browser's back would.
        if self.capture.is_some() {
            self.cancel_capture();
            return;
        }
        if egui::Popup::is_any_open(ctx) {
            egui::Popup::close_all(ctx);
            return;
        }
        if self.ask.take().is_some() || self.menu_ask.take().is_some() {
            return;
        }
        if self.session.is_some() {
            if self.menu_open {
                self.toggle_menu();
            }
            return;
        }
        self.screen = match &self.screen {
            Screen::Settings(Some(id)) => Screen::Title(id.clone()),
            _ => Screen::Library,
        };
    }

    // ------------------------------- the frame -------------------------------

    /// The first redraw of a `VITASLOP_SHELL_SHOT` run: open the screen it names.
    fn open_shot_screen(&mut self) {
        let Some(sh) = self.shot.as_mut().filter(|s| !s.opened) else { return };
        sh.opened = true;
        let screen = sh.screen.clone();
        let (name, arg) = screen.split_once(':').unwrap_or((screen.as_str(), ""));
        match (name, arg) {
            ("settings", "") => self.open_draft(None),
            ("settings", id) => self.open_draft(Some(id.to_string())),
            ("import", tab) => {
                self.screen = Screen::Import;
                self.import_tab = match tab {
                    "folder" => ImportTab::Folder,
                    "zip" => ImportTab::Zip,
                    "vpk" => ImportTab::Vpk,
                    _ => ImportTab::Pkg,
                };
            }
            ("import-pick", files) => {
                self.screen = Screen::Import;
                let mut it = files.split('|');
                self.pick_pkg = it.next().filter(|p| !p.is_empty()).map(PathBuf::from);
                self.pick_work = it.next().filter(|p| !p.is_empty()).map(PathBuf::from);
            }
            ("import-go", files) => {
                self.screen = Screen::Import;
                let paths: Vec<PathBuf> = files.split('|').filter(|p| !p.is_empty()).map(PathBuf::from).collect();
                match paths.as_slice() {
                    // A package and its licence: the package tab's Import button.
                    [pkg, work] if pkg.extension().is_some_and(|e| e.eq_ignore_ascii_case("pkg")) => {
                        self.pick_pkg = Some(pkg.clone());
                        self.pick_work = Some(work.clone());
                        self.start_import(library::DirSource::pkg_and_licence(pkg, work));
                    }
                    // Anything else: a drop.
                    _ => self.dropped = paths,
                }
            }
            ("diagnostics", _) => self.screen = Screen::Diagnostics,
            ("about", _) => self.screen = Screen::About,
            ("capture", _) => {
                self.open_draft(None);
                self.begin_capture(Button::Cross, Kind::Keyboard, false);
            }
            ("title", id) => self.screen = Screen::Title(id.to_string()),
            ("play", id) | ("menu", id) => self.start(id),
            _ => {}
        }
    }

    fn frame(&mut self, size: (f64, f64), window: &Arc<Window>) {
        self.open_shot_screen();
        // A panic on another thread (the loader, the guest's own) ends the run, not the window.
        if let Some(p) = crate::log::take_panic() {
            if self.session.is_some() || self.loading.is_some() {
                self.loading = None;
                self.fatal = Some(format!("RUST PANIC\n{p}"));
            } else {
                self.error = Some(p);
            }
        }
        self.take_drops();
        self.poll_loading();
        // The game, if one runs.
        let mut exec = None;
        let mut menu_request = false;
        if let Some(s) = self.session.as_mut() {
            s.tick(Some(size));
            // The advanced Fast-forward: unpaced and unpresented to the frame asked for, in slices
            // short enough that the window stays responsive.
            let target = u64::from(self.session_settings.fast_forward);
            if target > 0 && s.live() && s.guest.frames() < target {
                let t0 = Instant::now();
                while s.guest.frames() < target && !s.guest.finished() && t0.elapsed() < Duration::from_millis(50) {
                    s.guest.advance();
                }
                // The burst is not wall time the pacer owes the guest.
                s.last_tick = Instant::now();
            }
            exec = s.take_exec();
            // The pad's way into the menu (home, or start+select held) - see `Input::pump_gamepad`.
            if s.input.take_menu_request() && !self.menu_open {
                menu_request = true;
            }
            if let Some(st) = s.stats(Instant::now()) {
                self.stats = Some(st);
            }
            if exec.is_none() && s.guest.finished() && self.fatal.is_none() {
                self.fatal = Some(match s.guest.error() {
                    Some(e) => format!("ERROR\n{e}"),
                    None => format!("The game exited after {} frames.", s.guest.frames()),
                });
                crate::log::file_line(&format!("run ended: {}", self.fatal.as_deref().unwrap_or_default()));
            }
        }
        if menu_request && self.fatal.is_none() {
            self.toggle_menu();
        }
        if let Some(sh) = self.shot.as_ref()
            && sh.screen.starts_with("menu:")
            && !self.menu_open
            && self.session.as_ref().is_some_and(|s| s.guest.frames() >= sh.menu_at)
        {
            self.toggle_menu();
        }
        // A title that replaced its own process (`sceAppMgrLoadExec`): boot that executable in its
        // place through the ordinary loading screen, its saves flushed first.
        if let (Some(path), Some(id)) = (exec, self.playing.clone()) {
            if let Some(mut s) = self.session.take() {
                s.guest.flush_save(true);
            }
            self.start_exec(&id, Some(path));
        }
        if self.gfx.is_none() || self.renderer.is_none() {
            return;
        }
        let Some(st) = self.egui_state.as_mut() else { return };

        // egui runs on every frame (it needs to, to repaint the badge), but its input
        // is taken only when it owns the screen; otherwise it gets an empty frame.
        let mut raw = st.take_egui_input(window);
        self.pad_to_egui(&mut raw);
        let ctx = self.egui.clone();
        let mut ui_shell = std::mem::replace(self, Shell::placeholder());
        let full = ctx.run_ui(raw, |ui| ui_shell.ui(ui));
        *self = ui_shell;
        let Some(st) = self.egui_state.as_mut() else { return };
        let Some(gfx) = self.gfx.as_mut() else { return };
        let Some(renderer) = self.renderer.as_mut() else { return };
        st.handle_platform_output(window, full.platform_output);
        let prims = ctx.tessellate(full.shapes, full.pixels_per_point);
        let (w, h) = gfx.size();
        let desc = egui_wgpu::ScreenDescriptor { size_in_pixels: [w, h], pixels_per_point: full.pixels_per_point };
        for (id, deltas) in &full.textures_delta.set {
            for delta in deltas {
                renderer.update_texture(gfx.device(), gfx.queue(), *id, delta);
            }
        }
        // >>> WHILE THE GAME RUNS, A FRAME IS PRESENTED ONCE - the menu, the library, a pause and
        // egui asking to repaint still present every redraw. Re-rendering an unchanged game frame
        // each display period held the guest back by a whole chain encode (see `RetailApp`).
        let game_live = !self.menu_open && self.session.as_ref().is_some_and(Session::live);
        let frame_now = self.session.as_ref().map(|s| s.guest.frames());
        let egui_wants = full.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|v| v.repaint_delay.is_zero());
        let shooting = self.shot_to.is_some();
        if game_live && !egui_wants && !self.force_present && !shooting && self.presented == frame_now {
            for id in &full.textures_delta.free {
                renderer.free_texture(id);
            }
            return;
        }
        // >>> AND A GUEST FRAME IS ENCODED ONCE: a present that is not a new guest frame (the
        // menu, the badge, a resize) re-scales the stage - see `RetailGfx::frame`.
        let fresh = self.presented != frame_now;
        gfx.set_scaling(if self.session.is_some() { self.session_settings.scaling } else { Scaling::Fit });
        let rect = gfx.game_rect();
        let mut drew_new = false;
        let no_game = self.session.is_none();
        // Marks the present so the parallel guest runs on through a slow one (a first-draw
        // pipeline build) instead of freezing with its sound - see `smp::late_present`.
        let late = self.session.as_ref().and_then(|s| s.guest.late_handle());
        let game = match self.session.as_mut() {
            None => GameDraw::Clear,
            Some(s) => {
                // The mouse is the touch screen inside the picture, wherever the scaling put it.
                s.game_rect = Some((f64::from(rect.0), f64::from(rect.1), f64::from(rect.2), f64::from(rect.3)));
                let (scenes, display, presents) = s.scenes();
                if scenes.is_empty() {
                    if self.presented.is_some() { GameDraw::Again } else { GameDraw::Clear }
                } else if fresh {
                    drew_new = true;
                    GameDraw::Fresh(scenes, display, presents)
                } else {
                    GameDraw::Again
                }
            }
        };
        if drew_new || no_game {
            self.presented = frame_now;
        }
        self.force_present = false;
        if let Some(path) = self.shot_to.take()
            && !gfx.request_game_capture(path.clone())
        {
            self.menu_note = Some(format!("this window cannot be read back, so {} was not written", path.display()));
        }
        if let Some(shot) = self.shot.as_mut() {
            // A `menu:` shot counts from the menu opening, a `play:` one from the title running
            // (the build before it takes as long as it takes); anything else from the first redraw.
            let counting = if shot.screen.starts_with("menu:") {
                self.menu_open
            } else if shot.screen.starts_with("play:") {
                !no_game
            } else {
                true
            };
            if counting {
                shot.redraws += 1;
            }
            if shot.redraws == shot.at && !gfx.request_capture(shot.path.clone()) {
                eprintln!("VITASLOP_SHELL_SHOT: this surface cannot be read back");
            }
        }
        let late = late.filter(|_| drew_new);
        if let Some(l) = &late {
            l.begin();
        }
        let result = gfx.frame(game, |device, queue, encoder, view, _| {
            let cmds = renderer.update_buffers(device, queue, encoder, &prims, &desc);
            debug_assert!(cmds.is_empty());
            let mut rpass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            renderer.render(&mut rpass, &prims, &desc);
        });
        if let Some(l) = &late {
            l.end();
        }
        for id in &full.textures_delta.free {
            renderer.free_texture(id);
        }
        if let Err(e) = result {
            crate::log::file_line(&format!("present failed: {e}"));
            self.fatal = Some(format!("ERROR\n{e}"));
            if let Some(s) = self.session.as_mut() {
                s.paused = true;
            }
        } else if drew_new && let Some(s) = self.session.as_mut() {
            // Only a frame just ENCODED has render targets to read back for the guest.
            let wb = gfx.rtt_writebacks();
            s.apply_writebacks(&wb);
        }
    }

    // ------------------------------- state -------------------------------

    /// Import what was dropped on the window since the last frame, as one set.
    fn take_drops(&mut self) {
        if self.dropped.is_empty() || self.session.is_some() || self.loading.is_some() {
            self.dropped.clear();
            return;
        }
        let busy = self.import.as_ref().is_some_and(|p| !p.lock().unwrap().finished);
        let paths = std::mem::take(&mut self.dropped);
        if busy {
            return;
        }
        self.screen = Screen::Import;
        match library::DirSource::open_many(&paths) {
            Ok(src) => self.start_import(src),
            Err(e) => self.error = Some(format!("could not read what was dropped: {e}")),
        }
    }

    fn poll_loading(&mut self) {
        let Some(l) = self.loading.as_ref() else { return };
        match l.rx.try_recv() {
            Ok(Ok(mut guest)) => {
                let l = self.loading.take().unwrap();
                if let Err(e) = guest.persist_to(&library::saves_dir(&l.settings.profile), &library::title_dir(&l.title_id)) {
                    self.fatal = Some(format!("COULD NOT START\n{e}"));
                    return;
                }
                let input = Input::new(&l.settings);
                if let Some(g) = self.gfx.as_ref() {
                    guest.install_complete_scene_hook(g.completion_hook());
                }
                let session = Session::new(guest, l.input, input, l.settings.pause_on_blur);
                if let Some(a) = session.audio.as_ref()
                    && library::muted()
                {
                    a.set_muted(true);
                }
                crate::log::file_line(&format!("playing {} ({}), loaded in {:.1} s", l.title, l.title_id, l.started.elapsed().as_secs_f32()));
                self.session = Some(session);
                self.playing = Some(l.title_id.clone());
                // Fullscreen on start - the browser's setting. On a desktop `Automatic` plays in the
                // window (it takes over the screen only on a touch device).
                if l.settings.fullscreen_on_start == FullscreenStart::Always && !self.is_fullscreen() {
                    if let Some(w) = self.window.as_ref() {
                        w.set_fullscreen(Some(winit::window::Fullscreen::Borderless(None)));
                    }
                    self.fullscreen_by_play = true;
                }
                self.session_settings = l.settings;
                self.menu_open = false;
                self.menu_note = None;
                self.stats = None;
                self.presented = None;
                if let Some(mut m) = self.titles.iter().find(|t| t.title_id == l.title_id).cloned() {
                    m.last_played_at = library::now_ms();
                    let _ = library::write_meta(&m);
                    self.titles = library::list_titles();
                }
            }
            Ok(Err(e)) => {
                self.loading = None;
                crate::log::file_line(&format!("could not start: {e}"));
                self.fatal = Some(format!("COULD NOT START\n{e}"));
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.loading = None;
                // The loader thread panicked; the hook has the message and the backtrace.
                let why = crate::log::take_panic().unwrap_or_else(|| "the loader thread died".into());
                self.fatal = Some(format!("COULD NOT START\n{why}"));
            }
        }
    }

    fn start(&mut self, id: &str) {
        self.start_exec(id, None);
    }

    /// [`Self::start`] booting `main_exec` in the title's eboot's place - what a guest's
    /// `sceAppMgrLoadExec` asks for (see `Session::take_exec`).
    fn start_exec(&mut self, id: &str, main_exec: Option<String>) {
        let s = library::effective(Some(id));
        apply_run_knobs(&s);
        // The renderer reads knobs when it is made, and it was made with the window - before
        // this title's knobs were in the environment. See `RetailGfx::rebuild_renderer`.
        if let Some(g) = self.gfx.as_mut() {
            g.rebuild_renderer();
        }
        let dir = library::title_dir(id);
        let input: SharedInput = Arc::new(Mutex::new(DesktopInput::default()));
        let (tx, rx) = channel();
        let recipe = (!s.recipe.trim().is_empty()).then(|| s.recipe.clone());
        let input2 = input.clone();
        std::thread::Builder::new()
            .name("loader".into())
            .spawn(move || {
                let r = RetailGuest::new_with_exec(&dir, input2, recipe.as_deref(), main_exec.as_deref());
                let _ = tx.send(r);
            })
            .expect("spawn the loader thread");
        let title = self.titles.iter().find(|t| t.title_id == id).map_or_else(|| id.to_string(), |t| t.title.clone());
        self.fatal = None;
        self.loading = Some(Loading { title_id: id.to_string(), title, rx, input, settings: s, started: Instant::now() });
    }

    /// End the run. `to_library` also leaves fullscreen if the run entered it.
    fn stop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.guest.flush_save(true);
        }
        if let Some(id) = self.playing.as_ref() {
            crate::log::file_line(&format!("stopped {id}"));
        }
        self.loading = None;
        self.menu_open = false;
        self.menu_ask = None;
        self.cancel_capture();
        self.stats = None;
        self.fatal = None;
        self.presented = None;
        if self.fullscreen_by_play {
            if let Some(w) = self.window.as_ref() {
                w.set_fullscreen(None);
            }
            self.fullscreen_by_play = false;
        }
    }

    /// Back to the title page the run started from, or the library - the browser's `onExit`.
    fn quit_to_library(&mut self) {
        let id = self.playing.take();
        self.stop();
        self.titles = library::list_titles();
        self.screen = match id {
            Some(id) if self.titles.iter().any(|t| t.title_id == id) => Screen::Title(id),
            _ => Screen::Library,
        };
    }

    /// Tear the run down and start the same title again; the window stays as it is.
    fn restart(&mut self) {
        let Some(id) = self.playing.clone() else { return };
        let full = self.fullscreen_by_play;
        self.fullscreen_by_play = false;
        self.stop();
        self.fullscreen_by_play = full;
        self.start(&id);
    }

    /// The title's backdrop, `pic0.png`, as the browser's title page shows it.
    fn picture(&mut self, ctx: &egui::Context, id: &str) -> Option<TextureHandle> {
        if let Some(t) = self.pictures.get(id) {
            return t.clone();
        }
        let tex = std::fs::read(library::title_dir(id).join("pic0.png"))
            .ok()
            .and_then(|b| decode_png(&b))
            .map(|img| ctx.load_texture(format!("pic-{id}"), img, TextureOptions::LINEAR));
        self.pictures.insert(id.to_string(), tex.clone());
        tex
    }

    fn icon(&mut self, ctx: &egui::Context, id: &str) -> Option<TextureHandle> {
        if let Some(t) = self.icons.get(id) {
            return t.clone();
        }
        let tex = std::fs::read(library::title_dir(id).join("icon0.png"))
            .ok()
            .and_then(|b| decode_png(&b))
            .map(|img| ctx.load_texture(format!("icon-{id}"), img, TextureOptions::LINEAR));
        self.icons.insert(id.to_string(), tex.clone());
        tex
    }

    fn open_draft(&mut self, title_id: Option<String>) {
        let s = library::effective(title_id.as_deref());
        self.open_draft_with(title_id, s);
    }

    fn open_draft_with(&mut self, title_id: Option<String>, s: Settings) {
        let knobs_text = s.knobs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("\n");
        let overridden = title_id.as_deref().and_then(library::title_patch).map(|p| leaf_paths(&p, "")).unwrap_or_default().into_iter().collect();
        self.draft = Some(Draft { title_id: title_id.clone(), s, knobs_text, overridden, saved_at: None });
        self.new_profile = None;
        self.page_note = None;
        self.cancel_capture();
        self.screen = Screen::Settings(title_id);
    }

    fn save_draft(&mut self) {
        let Some(d) = self.draft.as_mut() else { return };
        d.s.knobs = settings::parse_knobs(&d.knobs_text);
        let r = match &d.title_id {
            Some(id) => {
                // Only what differs from the global settings is the title's own.
                let global = library::effective(None).to_value();
                let mine = d.s.to_value();
                let patch = deep_diff(&global, &mine);
                d.overridden = leaf_paths(&patch, "").into_iter().collect();
                library::save_title_patch(id, Some(&patch))
            }
            None => library::save_global_settings(&d.s),
        };
        match r {
            Ok(()) => d.saved_at = Some(Instant::now()),
            Err(e) => self.error = Some(format!("could not save settings: {e}")),
        }
    }

    fn start_import(&mut self, src: library::DirSource) {
        let progress = Arc::new(Mutex::new(ImportProgress::default()));
        let p2 = progress.clone();
        std::thread::Builder::new()
            .name("import".into())
            .spawn(move || {
                let r = library::import_from(src, &p2);
                let mut g = p2.lock().unwrap();
                g.finished = true;
                match r {
                    Ok(m) => g.title_id = Some(m.title_id),
                    Err(e) => g.error = Some(e),
                }
            })
            .expect("spawn the import thread");
        self.import = Some(progress);
        self.import_rate = import::Rate::default();
        self.import_icon = None;
    }

    /// The diagnostics file: the browser's `diagText` - what was running, how, on what - then
    /// everything the run reported.
    fn diag_text(&self) -> String {
        let title = self.playing.as_ref().map(|id| {
            let name = self.titles.iter().find(|t| &t.title_id == id).map_or(id.as_str(), |t| t.title.as_str());
            format!("{name} ({id})")
        });
        let knobs: Vec<String> = self.session_settings.run_knobs().into_iter().map(|(k, v)| format!("{k}={v}")).collect();
        let window = self.window.as_ref().map_or_else(String::new, |w| {
            let s = w.inner_size();
            format!("{}x{} scale {:.2}{}", s.width, s.height, w.scale_factor(), if w.fullscreen().is_some() { " fullscreen" } else { "" })
        });
        let audio = match self.session.as_ref().and_then(|s| s.audio.as_ref()) {
            Some(a) => {
                let s = a.stats();
                format!(
                    "audio: {} peak={:.4} underrun {:.2}s overrun {:.2}s latency skip {:.2}s rejoins {}{}",
                    a.device,
                    s.peak,
                    s.underrun_s,
                    s.overrun_s,
                    s.latency_skip_s,
                    s.rejoins,
                    if a.muted() { " (muted)" } else { "" }
                )
            }
            None if self.session.is_some() => "audio: no output device - this run is SILENT".into(),
            None => "audio: (no title running)".into(),
        };
        let mut out = [
            "vitaslop diagnostics".to_string(),
            format!("title: {}", title.as_deref().unwrap_or("(none running)")),
            format!("settings: {}", self.session_settings.to_value()),
            format!("knobs: {}", knobs.join(" ")),
            format!("build: vitaslop {} desktop ({}, {} {})", env!("CARGO_PKG_VERSION"), if cfg!(debug_assertions) { "debug" } else { "release" }, std::env::consts::OS, std::env::consts::ARCH),
            format!("window: {window}"),
            format!("adapter: {}", self.gfx.as_ref().map_or("(none)", RetailGfx::adapter_name)),
            self.stats.as_ref().map_or_else(|| "fps: (no title running)".into(), |s| format!("fps: {}", s.title_line())),
            audio,
            format!("home: {}", library::home().display()),
        ]
        .join("\n");
        if let Some(f) = self.fatal.as_ref() {
            out.push_str(&format!("\n\nFATAL\n{f}"));
        }
        out.push_str("\n\n");
        out.push_str(&crate::log::snapshot());
        out
    }

    /// Ask where to save the diagnostics file, named as the browser's download is.
    fn save_diag(&mut self, name: &str) -> Option<String> {
        let p = rfd::FileDialog::new().set_file_name(name).add_filter("Text", &["txt"]).save_file()?;
        Some(match std::fs::write(&p, self.diag_text()) {
            Ok(()) => format!("Saved {}", p.display()),
            Err(e) => format!("Could not save {}: {e}", p.display()),
        })
    }

    // ------------------------------- drawing -------------------------------

    fn ui(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        self.nav_pass(&ctx);
        if self.session.is_some() || self.loading.is_some() || self.fatal.is_some() {
            self.ui_player(root);
            return;
        }
        let page = match &self.screen {
            Screen::Library => "Library - vitaslop".to_string(),
            Screen::Settings(None) => "Settings - vitaslop".to_string(),
            Screen::Import => "Add games - vitaslop".to_string(),
            Screen::About => "About - vitaslop".to_string(),
            Screen::Diagnostics => "Diagnostics - vitaslop".to_string(),
            Screen::Title(id) => format!("{} - vitaslop", self.titles.iter().find(|t| &t.title_id == id).map_or(id.as_str(), |t| t.title.as_str())),
            Screen::Settings(Some(id)) => format!("{} settings - vitaslop", self.titles.iter().find(|t| &t.title_id == id).map_or(id.as_str(), |t| t.title.as_str())),
        };
        self.set_window_title(&page);
        // The browser's header: the name at the left, the pages as plain links at the right.
        let header = egui::Panel::top("top")
            .frame(egui::Frame::new().fill(PANEL).inner_margin(egui::Margin::symmetric(16, 10)))
            .show(root, |ui| {
                ui.horizontal(|ui| {
                    if ui.add(egui::Label::new(bold("vitaslop").size(17.0).color(ACCENT)).sense(egui::Sense::click())).clicked() {
                        self.titles = library::list_titles();
                        self.screen = Screen::Library;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.spacing_mut().item_spacing.x = 14.0;
                        if nav_link(ui, "About", self.screen == Screen::About).clicked() {
                            self.screen = Screen::About;
                        }
                        let (held, _, dropped) = vitaslop_platform::diag::counts(vitaslop_platform::diag::Channel::Warning);
                        let diag = if held + dropped == 0 { "Diagnostics".to_string() } else { format!("Diagnostics ({})", held + dropped) };
                        if nav_link(ui, &diag, self.screen == Screen::Diagnostics).clicked() {
                            self.screen = Screen::Diagnostics;
                        }
                        if nav_link(ui, "Settings", self.screen == Screen::Settings(None)).clicked() {
                            self.open_draft(None);
                        }
                        if nav_link(ui, "Add games", self.screen == Screen::Import).clicked() {
                            self.screen = Screen::Import;
                        }
                        if nav_link(ui, "Library", matches!(self.screen, Screen::Library | Screen::Title(_))).clicked() {
                            self.titles = library::list_titles();
                            self.screen = Screen::Library;
                        }
                    });
                });
            });
        let foot = header.response.rect;
        ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("header-edge")))
            .hline(foot.x_range(), foot.bottom() - 0.5, egui::Stroke::new(1.0, EDGE));
        if let Some(e) = self.error.clone() {
            egui::Panel::top("error")
                .frame(egui::Frame::new().fill(ERROR_FILL).stroke(egui::Stroke::new(1.0, ERROR_EDGE)).inner_margin(egui::Margin::symmetric(16, 8)))
                .show(root, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new(&e).color(DANGER));
                        if ui.small_button("Dismiss").clicked() {
                            self.error = None;
                        }
                    });
                });
        }
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::symmetric(16, 12)))
            .show(root, |ui| {
                // The browser's `#view`: at most 1100 px wide, centred.
                let w = ui.available_width().min(1100.0);
                let pad = ((ui.available_width() - w) / 2.0).max(0.0);
                ui.horizontal_top(|ui| {
                    ui.add_space(pad);
                    ui.vertical(|ui| {
                        ui.set_width(w);
                        match self.screen.clone() {
                            Screen::Library => self.ui_library(ui),
                            Screen::Title(id) => self.ui_title(ui, &id),
                            Screen::Settings(_) => self.ui_settings(ui),
                            Screen::Import => self.ui_import(ui),
                            Screen::Diagnostics => self.ui_diagnostics(ui),
                            Screen::About => self.ui_about(ui),
                        }
                    });
                });
            });
        self.ui_ask(&ctx);
        self.ui_capture(&ctx);
    }

    fn ui_library(&mut self, ui: &mut egui::Ui) {
        // The browser's toolbar: search filling the row, the sort, and the primary action.
        ui.horizontal(|ui| {
            let n = self.titles.len();
            let search_w = (ui.available_width() - 170.0 - 120.0 - 2.0 * ui.spacing().item_spacing.x).max(120.0);
            ui.add_sized(
                [search_w, 36.0],
                egui::TextEdit::singleline(&mut self.search).hint_text(format!("Search {n} title{}", if n == 1 { "" } else { "s" })).margin(egui::Margin::symmetric(10, 9)),
            );
            let r = egui::ComboBox::from_id_salt("sort").width(150.0).selected_text(self.sort.label()).show_ui(ui, |ui| {
                for s in SortMode::ALL {
                    ui.selectable_value(&mut self.sort, s, s.label());
                }
            });
            combo_nav(&r.response, &mut self.nav, &mut self.sort, &SortMode::ALL);
            if ui.add(primary("Add games").min_size(egui::vec2(110.0, 36.0))).clicked() {
                self.screen = Screen::Import;
            }
        });
        ui.add_space(8.0);
        if self.titles.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(60.0);
                ui.label(RichText::new("No titles yet.").color(DIM).size(15.0));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let text = "from a .pkg and work.bin, a dumped folder, or a zip of either.";
                    let w = 110.0 + ui.spacing().item_spacing.x + ui.fonts_mut(|f| f.layout_no_wrap(text.into(), egui::FontId::proportional(14.0), DIM).size().x);
                    ui.add_space(((ui.available_width() - w) / 2.0).max(0.0));
                    if ui.add(primary("Add a game").min_size(egui::vec2(110.0, 36.0))).clicked() {
                        self.screen = Screen::Import;
                    }
                    ui.label(RichText::new(text).color(DIM));
                });
            });
            return;
        }
        let q = self.search.trim().to_lowercase();
        let mut list: Vec<TitleMeta> = self.titles.iter().filter(|t| q.is_empty() || t.search_key().contains(&q)).cloned().collect();
        self.sort.apply(&mut list);
        // The browser's grid: as many columns of at least 112 px as fit, the slack shared out.
        let gap = egui::vec2(10.0, 14.0);
        let avail = ui.available_width();
        let cols = (((avail + gap.x) / (112.0 + gap.x)).floor() as usize).max(1);
        let tile = (avail - gap.x * (cols as f32 - 1.0)) / cols as f32;
        let icon = (tile - 8.0).min(150.0);
        let ctx = ui.ctx().clone();
        let mut open: Option<String> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for row in list.chunks(cols) {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = gap.x;
                    for t in row {
                        let tex = self.icon(&ctx, &t.title_id);
                        ui.allocate_ui(egui::vec2(tile, icon + 46.0), |ui| {
                            ui.vertical_centered(|ui| {
                                ui.spacing_mut().item_spacing.y = 4.0;
                                let r = match tex {
                                    Some(tex) => ui.add(egui::Button::image(egui::Image::new((tex.id(), egui::vec2(icon, icon))).corner_radius(icon / 2.0)).frame(false)),
                                    None => ui.add_sized([icon, icon], egui::Button::new(RichText::new(&t.title_id).small()).corner_radius(icon / 2.0)),
                                };
                                // The icon's rim, lit for the pad's cursor or the mouse - the frameless tile has no other.
                                let lit = r.has_focus() || r.hovered();
                                let rim = if r.has_focus() { egui::Stroke::new(3.0, ACCENT) } else if lit { egui::Stroke::new(1.0, ACCENT) } else { egui::Stroke::new(1.0, EDGE) };
                                ui.painter().circle_stroke(r.rect.center(), icon / 2.0 + if r.has_focus() { 2.0 } else { 0.0 }, rim);
                                if r.clicked() {
                                    open = Some(t.title_id.clone());
                                }
                                ui.add(egui::Label::new(RichText::new(&t.title).size(12.0)).truncate());
                                ui.label(RichText::new(&t.title_id).color(DIM).size(10.0));
                            });
                        });
                    }
                });
                ui.add_space(gap.y - ui.spacing().item_spacing.y);
            }
        });
        if let Some(id) = open {
            self.page_note = None;
            self.screen = Screen::Title(id);
        }
    }

    fn ui_title(&mut self, ui: &mut egui::Ui, id: &str) {
        let Some(meta) = self.titles.iter().find(|t| t.title_id == id).cloned() else {
            card(ui, |ui| {
                ui.label(bold(id).size(16.0));
                ui.label("This title is not in the library.");
                if ui.button("Add it").clicked() {
                    self.screen = Screen::Import;
                }
            });
            return;
        };
        if let Some(to) = crumbs(ui, &[("Library", Some(Screen::Library)), (&meta.title, None)]) {
            self.screen = to;
            return;
        }
        let ctx = ui.ctx().clone();
        let icon = self.icon(&ctx, id);
        let pic = self.picture(&ctx, id);
        // The browser's hero: the title's own backdrop (pic0) under a left-to-right shade, the big
        // round icon, the name, one line of facts, and the actions.
        let bg = ui.painter().add(egui::Shape::Noop);
        let hero = egui::Frame::new().inner_margin(egui::Margin::symmetric(16, 28)).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 20.0;
                match icon {
                    Some(tex) => {
                        let r = ui.add(egui::Image::new((tex.id(), egui::vec2(128.0, 128.0))).corner_radius(64.0));
                        ui.painter().circle_stroke(r.rect.center(), 64.0, egui::Stroke::new(1.0, EDGE));
                    }
                    None => {
                        let (r, _) = ui.allocate_exact_size(egui::vec2(128.0, 128.0), egui::Sense::hover());
                        ui.painter().circle(r.center(), 64.0, PANEL_2, egui::Stroke::new(1.0, EDGE));
                    }
                }
                ui.vertical(|ui| {
                    ui.label(bold(&meta.title).size(26.0).color(INK));
                    let version = if meta.app_version.is_empty() { String::new() } else { format!("  \u{b7}  v{}", meta.app_version) };
                    ui.label(
                        RichText::new(format!(
                            "{}{version}  \u{b7}  {}  \u{b7}  added {}  \u{b7}  played {}",
                            meta.title_id,
                            fmt_bytes(meta.bytes),
                            fmt_date(meta.imported_at),
                            fmt_date(meta.last_played_at)
                        ))
                        .color(DIM)
                        .size(13.0),
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.add(primary("Play").min_size(egui::vec2(96.0, 44.0))).clicked() {
                            self.start(id);
                        }
                        if ui.add(egui::Button::new(bold("Settings for this title")).min_size(egui::vec2(0.0, 36.0))).clicked() {
                            self.open_draft(Some(id.to_string()));
                        }
                        if ui.add(danger("Remove").min_size(egui::vec2(0.0, 36.0))).clicked() {
                            self.ask = Some(Ask::RemoveTitle(id.to_string()));
                        }
                    });
                });
            });
        });
        ui.painter().set(bg, hero_backdrop(hero.response.rect, pic.as_ref()));
        ui.add_space(4.0);
        let eff = library::effective(Some(id));
        let profile = eff.profile.clone();
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(bold("Saved data").size(16.0));
                ui.label(RichText::new(format!("profile: {profile}")).color(DIM).size(16.0));
            });
            let info = library::gamedata_info(&profile, id);
            ui.label(RichText::new(match info {
                Some((bytes, modified)) => format!("{}, last written {}", fmt_bytes(bytes), fmt_datetime(modified)),
                None => "nothing saved yet - it appears here the first time the game saves.".into(),
            }).color(DIM));
            ui.horizontal(|ui| {
                if ui.add_enabled(info.is_some(), egui::Button::new(bold("Download"))).clicked() {
                    self.page_note = Some(download_gamedata(&profile, id));
                }
                if ui.button(bold("Upload")).clicked()
                    && let Some(p) = rfd::FileDialog::new().add_filter("Saved data", &["zip"]).pick_file()
                {
                    match std::fs::read(&p).map_err(|e| e.to_string()).and_then(|zip| library::describe_gamedata(&zip).map(|s| (zip, s))) {
                        Ok((zip, summary)) => self.ask = Some(Ask::RestoreSave { profile: profile.clone(), id: id.to_string(), zip, summary }),
                        Err(e) => self.page_note = Some(format!("Not restored: {e}")),
                    }
                }
                if ui.add_enabled(info.is_some(), danger("Clear")).clicked() {
                    self.ask = Some(Ask::ClearSave { profile: profile.clone(), id: id.to_string() });
                }
                if ui.add(egui::Button::new(RichText::new("Open the saves folder").color(DIM)).frame(false)).clicked() {
                    let dir = library::saves_dir(&profile).join(id);
                    let _ = std::fs::create_dir_all(&dir);
                    open_path(&dir);
                }
            });
            if let Some(n) = self.page_note.as_ref() {
                ui.label(RichText::new(n).color(DIM).small());
            }
            ui.label(RichText::new("What the game saved - its save files and trophies - and nothing of the game itself. A download is a file you own; upload it on another device to continue there.").color(DIM));
        });
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui) {
        let Some(mut d) = self.draft.take() else { return };
        let title_name = d.title_id.as_ref().map(|id| self.titles.iter().find(|t| &t.title_id == id).map_or(id.clone(), |t| t.title.clone()));
        let trail: Vec<(&str, Option<Screen>)> = match (&d.title_id, &title_name) {
            (Some(id), Some(name)) => vec![("Library", Some(Screen::Library)), (name.as_str(), Some(Screen::Title(id.clone()))), ("Settings", None)],
            _ => vec![("Library", Some(Screen::Library)), ("Settings", None)],
        };
        if let Some(to) = crumbs(ui, &trail) {
            self.draft = Some(d);
            self.screen = to;
            return;
        }
        let heading = match &title_name {
            Some(n) => format!("Settings for {n}"),
            None => "Settings".into(),
        };
        ui.label(bold(heading).size(22.0).color(INK));
        let mut go_global = false;
        if d.title_id.is_some() {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.label(RichText::new("Only what you change here is kept for this title (marked ").color(DIM));
                ui.label(bold("changed").color(WARN));
                ui.label(RichText::new("); everything else follows the ").color(DIM));
                if ui.add(egui::Button::new(RichText::new("global settings").color(ACCENT)).frame(false)).clicked() {
                    go_global = true;
                }
                ui.label(RichText::new(", including later changes to them.").color(DIM));
            });
        }
        let mut save = false;
        let mut use_global = false;
        let mut reset_controls = false;
        let mut capture: Option<Button> = None;
        let global_page = d.title_id.is_none();
        // The browser's sticky Save bar: the cards scroll above it, the bar stays.
        let bar = 52.0;
        egui::ScrollArea::vertical().max_height((ui.available_height() - bar).max(80.0)).show(ui, |ui| {
            card(ui, |ui| {
                ui.label(bold("General").size(16.0));
                let c = d.changed("pauseOnBlur");
                check_row(ui, &mut d.s.pause_on_blur, "Pause when the window is hidden or loses focus", Some("Not the game's pause menu: the emulator stops, as a console in a pocket would."), c);
                ui.separator();
                let c = d.changed("showFps");
                check_row(ui, &mut d.s.show_fps, "Show the frame rate over the game", None, c);
                ui.separator();
                let c = d.changed("fpsInTitle");
                check_row(ui, &mut d.s.fps_in_title, "Show the frame rate in the window title", None, c);
                ui.separator();
                setting_row(ui, "Scaling", None, d.changed("scaling"), |ui| {
                    let r = egui::ComboBox::from_id_salt("scaling").width(170.0).selected_text(scaling_label(d.s.scaling)).show_ui(ui, |ui| {
                        for s in [Scaling::Fit, Scaling::Integer, Scaling::Stretch] {
                            ui.selectable_value(&mut d.s.scaling, s, scaling_label(s));
                        }
                    });
                    combo_nav(&r.response, &mut self.nav, &mut d.s.scaling, &[Scaling::Fit, Scaling::Integer, Scaling::Stretch]);
                });
                ui.separator();
                setting_row(
                    ui,
                    "Start in fullscreen",
                    Some("Automatic plays in the window on a desktop and takes over the screen on a touch device. F11 or the in-game menu toggles it either way."),
                    d.changed("fullscreenOnStart"),
                    |ui| {
                        let all = [FullscreenStart::Auto, FullscreenStart::Always, FullscreenStart::Never];
                        let r = egui::ComboBox::from_id_salt("fullscreen").width(170.0).selected_text(fullscreen_label(d.s.fullscreen_on_start)).show_ui(ui, |ui| {
                            for f in all {
                                ui.selectable_value(&mut d.s.fullscreen_on_start, f, fullscreen_label(f));
                            }
                        });
                        combo_nav(&r.response, &mut self.nav, &mut d.s.fullscreen_on_start, &all);
                    },
                );
                ui.separator();
                setting_row(ui, "Save profile", Some("Each profile keeps its own saved data for every game."), d.changed("profile"), |ui| {
                    let mut profiles = library::profiles();
                    if !profiles.contains(&d.s.profile) {
                        profiles.push(d.s.profile.clone());
                    }
                    match self.new_profile.as_mut() {
                        Some(name) => {
                            let ok = library::valid_profile(name.trim());
                            if ui.add_enabled(ok, egui::Button::new(bold("Add")).small()).clicked() {
                                d.s.profile = name.trim().to_string();
                                self.new_profile = None;
                            } else {
                                ui.add(egui::TextEdit::singleline(name).hint_text("letters, digits, - and _").desired_width(170.0));
                            }
                        }
                        None => {
                            if ui.add(egui::Button::new(bold("New")).small()).clicked() {
                                self.new_profile = Some(String::new());
                            }
                            let r = egui::ComboBox::from_id_salt("profile").width(150.0).selected_text(&d.s.profile).show_ui(ui, |ui| {
                                for p in &profiles {
                                    ui.selectable_value(&mut d.s.profile, p.clone(), p);
                                }
                            });
                            combo_nav(&r.response, &mut self.nav, &mut d.s.profile, &profiles);
                        }
                    }
                });
            });
            card(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(bold("Controls").size(16.0));
                    if d.changed("keyboard") || d.changed("gamepad") {
                        ui.label(bold("changed").size(11.0).color(WARN));
                    }
                });
                seg(ui, &mut self.settings_ctl);
                ui.label(RichText::new(if self.settings_ctl == vitapic::Mode::Gamepad {
                    "Click a control on the picture, then press the controller button for it (or pick it from the list). Standard layout positions: south is the bottom face button (A or Cross), east the right one (B or Circle)."
                } else {
                    "Click a control on the picture, then press the key for it."
                }).color(DIM));
                let listening = self.capture.filter(|c| !c.live).map(|c| c.button);
                capture = vitapic::show(ui, &d.s.keyboard, &d.s.gamepad, self.settings_ctl, listening, 760.0);
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(bold("Reset to defaults")).small()).clicked() {
                        reset_controls = true;
                    }
                });
                ui.separator();
                setting_row(ui, "Stick dead zone", Some("How far a stick must move before the game sees it."), d.changed("stickDeadzone"), |ui| {
                    ui.add(egui::Slider::new(&mut d.s.stick_deadzone, 0.0..=0.5).step_by(0.01));
                });
                // The browser's "On-screen controls" card, reduced to what a window can do: over
                // the game or not at all (a computer has no touch, so Auto keeps them hidden).
                let mut on_screen = touchpad::shown(&d.s.pad);
                if check_row(ui, &mut on_screen, "Show the on-screen controls", Some("The browser's touch controls over the game: they light up as you press keys or pad buttons, the sticks follow your pad, and the mouse can press them."), d.changed("pad")) {
                    d.s.pad.mode = if on_screen { PadMode::Overlay } else { PadMode::Auto };
                }
                if on_screen {
                    setting_row(ui, "Opacity", None, d.changed("pad"), |ui| {
                        ui.add(egui::Slider::new(&mut d.s.pad.opacity, 0.1..=1.0).step_by(0.05));
                    });
                    setting_row(ui, "Size", None, d.changed("pad"), |ui| {
                        ui.add(egui::Slider::new(&mut d.s.pad.scale, 0.5..=2.0).step_by(0.05));
                    });
                }
                ui.label(RichText::new("In a game, a controller's home button opens the menu; so does holding Start and Select together for a second, on any pad. Outside a game the controller moves around this app: the d-pad or left stick moves, the south button (A or Cross) picks, the east button (B or Circle) goes back, Start (Options) opens the settings, and left/right change a choice.").color(DIM));
            });
            card(ui, |ui| {
                egui::CollapsingHeader::new(bold("Advanced").size(16.0)).id_salt("advanced").show(ui, |ui| {
                    field_label(ui, "Knobs", "One VITASLOP_NAME=value per line, set in the environment when the title starts.", d.changed("knobs"));
                    ui.add(egui::TextEdit::multiline(&mut d.knobs_text).code_editor().desired_rows(4).desired_width(f32::INFINITY));
                    field_label(ui, "Recipe", "A scripted-input recipe, replayed from the first frame; live input still works.", d.changed("recipe"));
                    ui.add(egui::TextEdit::multiline(&mut d.s.recipe).code_editor().desired_rows(3).desired_width(f32::INFINITY));
                    ui.separator();
                    setting_row(ui, "Fast-forward to frame", Some("Runs unpaced to here first."), d.changed("fastForward"), |ui| {
                        ui.add(egui::DragValue::new(&mut d.s.fast_forward).range(0..=10_000_000).speed(100));
                    });
                    ui.separator();
                    let c = d.changed("debugCapture");
                    check_row(ui, &mut d.s.debug_capture, "Capture debug timings", Some("Times every host call for the diagnostics; roughly doubles the frame cost."), c);
                    ui.separator();
                    let c = d.changed("consoleNotes");
                    check_row(ui, &mut d.s.console_notes, "Mirror run notes to the console", None, c);
                });
            });
            if global_page {
                self.ui_settings_global_cards(ui, &d.s.profile);
            }
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.add(primary("Save").min_size(egui::vec2(64.0, 36.0))).clicked() {
                save = true;
            }
            if d.title_id.is_some() {
                if ui.add(egui::Button::new(bold("Use global settings")).min_size(egui::vec2(0.0, 36.0))).clicked() {
                    use_global = true;
                }
            } else if ui.add(egui::Button::new(bold("Reset to defaults")).min_size(egui::vec2(0.0, 36.0))).clicked() {
                self.ask = Some(Ask::ResetAll);
            }
            if let Some(t) = d.saved_at
                && t.elapsed().as_secs_f32() < 2.0
            {
                ui.label(RichText::new("saved").color(DIM));
            }
        });
        if reset_controls {
            d.s.keyboard = vitaslop_frontend::input::default_keyboard();
            d.s.gamepad = vitaslop_frontend::input::default_gamepad();
        }
        let id = d.title_id.clone();
        self.draft = Some(d);
        if let Some(b) = capture {
            let kind = if self.settings_ctl == vitapic::Mode::Gamepad { Kind::Gamepad } else { Kind::Keyboard };
            self.begin_capture(b, kind, false);
        }
        if save {
            self.save_draft();
        }
        if use_global && let Some(id) = id {
            let _ = library::save_title_patch(&id, None);
            self.open_draft(Some(id));
        }
        if go_global {
            self.open_draft(None);
        }
    }

    /// The global settings page's last three cards: every game's saved data, the storage, and
    /// this computer - the browser's Saved data, Storage and This browser.
    fn ui_settings_global_cards(&mut self, ui: &mut egui::Ui, profile: &str) {
        card(ui, |ui| {
            ui.label(bold("Saved data").size(16.0));
            ui.label(RichText::new("Every game's saved data in the current profile, as one file.").color(DIM));
            let ids = library::titles_with_gamedata(profile);
            ui.horizontal(|ui| {
                if ui.add_enabled(!ids.is_empty(), egui::Button::new(bold("Download all"))).clicked()
                    && let Some(p) = rfd::FileDialog::new().set_file_name(format!("vitaslop-{profile}-all-gamedata.zip")).add_filter("Zip", &["zip"]).save_file()
                {
                    self.page_note = Some(match std::fs::write(&p, library::gamedata_bundle(profile)) {
                        Ok(()) => format!("Saved {}", p.display()),
                        Err(e) => format!("Could not save {}: {e}", p.display()),
                    });
                }
                if ui.button(bold("Upload a bundle")).clicked()
                    && let Some(p) = rfd::FileDialog::new().add_filter("Zip", &["zip"]).pick_file()
                {
                    match std::fs::read(&p).map_err(|e| e.to_string()).and_then(|b| library::read_gamedata_bundle(&b)) {
                        Ok(entries) => self.ask = Some(Ask::RestoreBundle { profile: profile.to_string(), entries }),
                        Err(e) => self.page_note = Some(format!("Not restored: {e}")),
                    }
                }
                if ui.add_enabled(!ids.is_empty(), danger("Clear all")).clicked() {
                    self.ask = Some(Ask::ClearAll { profile: profile.to_string() });
                }
            });
            ui.label(
                RichText::new(if ids.is_empty() {
                    format!("no saved data in profile \"{profile}\".")
                } else {
                    format!("{} game{} have saved data in profile \"{profile}\".", ids.len(), if ids.len() == 1 { "" } else { "s" })
                })
                .color(DIM),
            );
            if let Some(n) = self.page_note.as_ref() {
                ui.label(RichText::new(n).color(DIM).small());
            }
        });
        card(ui, |ui| {
            ui.label(bold("Storage").size(16.0));
            let home = library::home();
            let lib = library::dir_bytes(&library::library_dir());
            let saves = library::dir_bytes(&home.join("saves"));
            ui.label(RichText::new(format!("{} of games and {} of saved data in {}.", fmt_bytes(lib), fmt_bytes(saves), home.display())).color(DIM));
            if ui.add(egui::Button::new(bold("Open the folder")).small()).clicked() {
                let _ = std::fs::create_dir_all(&home);
                open_path(&home);
            }
        });
        card(ui, |ui| {
            ui.label(bold("This computer").size(16.0));
            self.computer_checks(ui);
        });
    }

    /// The browser's feature checks, for the desktop: what this run found to draw, sound and
    /// play with.
    fn computer_checks(&self, ui: &mut egui::Ui) {
        let gpu = self.gfx.as_ref().map(RetailGfx::adapter_name).unwrap_or("none");
        let pads: Vec<String> = gilrs::Gilrs::new().map(|g| g.gamepads().filter(|(_, p)| p.is_connected()).map(|(_, p)| p.name().to_string()).collect()).unwrap_or_default();
        let audio = match cpal_device_name() {
            Some(n) => (true, format!("Sound: {n}")),
            None => (false, "Sound: no output device - titles run silent".to_string()),
        };
        let rows: [(bool, String); 4] = [
            (self.gfx.is_some(), format!("Graphics: {gpu}")),
            audio,
            (!pads.is_empty(), if pads.is_empty() { "Controllers: none connected - the keyboard plays".into() } else { format!("Controllers: {}", pads.join(", ")) }),
            (true, format!("System: {} {}", std::env::consts::OS, std::env::consts::ARCH)),
        ];
        for (i, (ok, text)) in rows.iter().enumerate() {
            if i > 0 {
                ui.separator();
            }
            ui.label(bold(text).color(if *ok { ACCENT } else { WARN }));
        }
    }

    /// Everything this run reported, in the two channels it reported them on, and where the
    /// run's log is on disk.
    ///
    /// # Why a warning being HERE is not a warning being hidden
    /// A run that approximates something is required to say so, and it still does - every one
    /// of these lines was emitted exactly as before. What changed is that a player is not
    /// shouted at in a terminal they did not open. The count sits in the top bar on every
    /// screen, so a run with findings never looks like a run without them, and `Save report`
    /// writes the whole thing out for a bug report.
    fn ui_diagnostics(&mut self, ui: &mut egui::Ui) {
        use vitaslop_platform::diag::{Channel, counts, report};
        let (held, total, dropped) = counts(Channel::Warning);
        ui.horizontal(|ui| {
            ui.label(bold("Diagnostics").size(22.0).color(INK));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(bold("Save report")).clicked()
                    && let Some(m) = self.save_diag("vitaslop-diag.txt")
                {
                    self.page_note = Some(m);
                }
                if ui.button(bold("Copy")).clicked() {
                    ui.ctx().copy_text(self.diag_text());
                    self.page_note = Some("Copied".into());
                }
            });
        });
        ui.label(RichText::new("Warnings are the emulator saying it did something the console would not have done. A title can look and play perfectly with warnings here.").color(DIM));
        card(ui, |ui| {
            ui.label(bold("Run log").size(16.0));
            match crate::log::log_path() {
                Some(p) => ui.label(RichText::new(format!("Everything below, and the details of any crash, is also written to {}", p.display())).color(DIM)),
                None => ui.label(RichText::new("This run has no log file.").color(DIM)),
            };
            ui.label(RichText::new(format!("The last {} runs' logs are kept in {}.", crate::log::KEEP_LOGS, library::logs_dir().display())).color(DIM));
            if ui.add(egui::Button::new(bold("Open the logs folder")).small()).clicked() {
                let _ = std::fs::create_dir_all(library::logs_dir());
                open_path(&library::logs_dir());
            }
            if let Some(n) = self.page_note.as_ref() {
                ui.label(RichText::new(n).color(DIM).small());
            }
        });
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(bold(format!(
                "{held} distinct warning(s), {total} in total{}",
                if dropped > 0 { format!(", {dropped} older one(s) dropped") } else { String::new() }
            )));
            match report(Channel::Warning) {
                Some(text) => ui.label(RichText::new(text).monospace().small()),
                None => ui.label(RichText::new("Nothing was reported.").color(DIM)),
            };
            ui.add_space(12.0);
            ui.label(bold("Status"));
            ui.label(RichText::new("What the run did rather than what went wrong: the adapter, the archive, the shapes it saw.").color(DIM).small());
            match report(Channel::Status) {
                Some(text) => ui.label(RichText::new(text).monospace().small()),
                None => ui.label(RichText::new("Nothing yet.").color(DIM)),
            };
        });
    }

    fn ui_about(&mut self, ui: &mut egui::Ui) {
        ui.label(bold("About").size(22.0).color(INK));
        card(ui, |ui| {
            ui.label(format!(
                "vitaslop is a clean-room PlayStation Vita emulator. This is its desktop app: a title's ARM code is translated to WebAssembly and run natively, its GPU stream is drawn with your graphics card, and everything you import stays in this computer's vitaslop folder ({}). No server, no console firmware, no downloads from anywhere.",
                library::home().display()
            ));
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                if ui.add(egui::Button::new(RichText::new("Source and documentation on GitHub").color(ACCENT)).frame(false)).clicked() {
                    open_url("https://github.com/cretz/vitaslop");
                }
                ui.label(". Titles are never provided; import your own from a console you own.");
            });
            ui.label(RichText::new(format!("Version {}.", env!("CARGO_PKG_VERSION"))).color(DIM));
        });
        card(ui, |ui| {
            ui.label(bold("This computer").size(16.0));
            self.computer_checks(ui);
        });
    }

    /// The pages' confirmation dialog - see [`Ask`].
    fn ui_ask(&mut self, ctx: &egui::Context) {
        let Some(ask) = self.ask.as_ref() else { return };
        let name = |id: &str| self.titles.iter().find(|t| t.title_id == id).map_or(id.to_string(), |t| t.title.clone());
        let (text, yes) = match ask {
            Ask::RemoveTitle(id) => {
                let size = self.titles.iter().find(|t| &t.title_id == id).map_or_else(String::new, |t| fmt_bytes(t.bytes));
                (format!("Remove {} ({id}) from this computer?\n\nThis deletes the imported game ({size}) and its prepared code. Saved data is kept; clear it separately if you want it gone.", name(id)), "Remove")
            }
            Ask::ClearSave { id, .. } => (format!("Delete the saved data for {}? This cannot be undone.", name(id)), "Delete"),
            Ask::RestoreSave { profile, id, summary, .. } => {
                let existing = library::gamedata_info(profile, id).map_or_else(String::new, |(b, _)| format!("\n\nThis REPLACES the {} already saved.", fmt_bytes(b)));
                (format!("Restore this save into {}?\n\n{summary}{existing}", name(id)), "Restore")
            }
            Ask::ClearAll { profile } => {
                let n = library::titles_with_gamedata(profile).len();
                (format!("Delete the saved data of {n} game{} in profile \"{profile}\"? This cannot be undone.", if n == 1 { "" } else { "s" }), "Delete")
            }
            Ask::RestoreBundle { profile, entries } => (
                format!("Restore saved data for {} game{} into profile \"{profile}\"? Existing saves for the same games are replaced.", entries.len(), if entries.len() == 1 { "" } else { "s" }),
                "Restore",
            ),
            Ask::ResetAll => ("Reset every global setting to its default?".into(), "Reset"),
        };
        let mut answer = None;
        egui::Modal::new(egui::Id::new("ask"))
            .backdrop_color(Color32::from_rgba_unmultiplied(5, 5, 10, 200))
            .frame(panel_frame())
            .show(ctx, |ui| {
                ui.set_max_width(460.0);
                ui.label(RichText::new(text).size(15.0).color(INK));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.add(danger(yes)).clicked() {
                        answer = Some(true);
                    }
                    let no = ui.add(primary("Cancel"));
                    if no.clicked() {
                        answer = Some(false);
                    }
                    // The safe answer has the focus, as the browser's menu Yes/No panel does.
                    if ui.memory(|m| m.focused()).is_none() {
                        no.request_focus();
                    }
                });
            });
        let Some(yes) = answer else { return };
        let ask = self.ask.take().unwrap();
        if !yes {
            return;
        }
        let r: Result<(), String> = match ask {
            Ask::RemoveTitle(id) => {
                let r = library::remove_title(&id).map_err(|e| e.to_string());
                self.icons.remove(&id);
                self.pictures.remove(&id);
                self.titles = library::list_titles();
                let _ = library::save_title_patch(&id, None);
                self.screen = Screen::Library;
                r
            }
            Ask::ClearSave { profile, id } => library::clear_gamedata(&profile, &id).map_err(|e| e.to_string()),
            Ask::RestoreSave { profile, id, zip, .. } => library::write_gamedata(&profile, &id, &zip),
            Ask::ClearAll { profile } => library::titles_with_gamedata(&profile).iter().try_for_each(|id| library::clear_gamedata(&profile, id)).map_err(|e| e.to_string()),
            Ask::RestoreBundle { profile, entries } => entries.iter().try_for_each(|(id, zip)| library::write_gamedata(&profile, id, zip)),
            Ask::ResetAll => {
                let r = library::save_global_settings(&Settings::default()).map_err(|e| e.to_string());
                self.open_draft(None);
                r
            }
        };
        if let Err(e) = r {
            self.page_note = Some(format!("Failed: {e}"));
        }
    }

    /// The key-capture dialog - the browser's `.capture` modal: nothing else is reachable
    /// until a key (or pad button) is pressed or the capture is cancelled.
    fn ui_capture(&mut self, ctx: &egui::Context) {
        let Some(c) = self.capture else { return };
        let current = match (c.live, c.kind) {
            (true, Kind::Keyboard) => self.session_settings.keyboard.get(c.button.name()).cloned(),
            (true, Kind::Gamepad) => self.session_settings.gamepad.get(c.button.name()).cloned(),
            (false, Kind::Keyboard) => self.draft.as_ref().and_then(|d| d.s.keyboard.get(c.button.name()).cloned()),
            (false, Kind::Gamepad) => self.draft.as_ref().and_then(|d| d.s.gamepad.get(c.button.name()).cloned()),
        }
        .unwrap_or_else(|| "-".into());
        let mut cancel = false;
        let mut picked: Option<&'static str> = None;
        egui::Modal::new(egui::Id::new("capture"))
            .backdrop_color(Color32::from_rgba_unmultiplied(5, 5, 10, 217))
            .frame(panel_frame().stroke(egui::Stroke::new(1.0, ACCENT)).inner_margin(egui::Margin::symmetric(36, 24)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new(if c.kind == Kind::Keyboard { "Press the key for" } else { "Press the controller button for" }).color(DIM));
                    ui.label(bold(c.button.label()).size(26.0).color(ACCENT));
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.label(RichText::new("Esc cancels. Currently ").color(DIM));
                        ui.label(RichText::new(&current).monospace().color(WARN));
                        ui.label(RichText::new(".").color(DIM));
                    });
                    if c.kind == Kind::Gamepad {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Or pick one:").color(DIM));
                            egui::ComboBox::from_id_salt("capture-pick").width(130.0).selected_text(&current).show_ui(ui, |ui| {
                                for control in vitaslop_frontend::input::GAMEPAD_CONTROLS {
                                    if ui.selectable_label(current == control, control).clicked() {
                                        picked = Some(control);
                                    }
                                }
                            });
                        });
                    }
                    if ui.add(egui::Button::new(bold("Cancel")).small()).clicked() {
                        cancel = true;
                    }
                });
            });
        if let Some(p) = picked {
            self.finish_capture(p.to_string());
        } else if cancel {
            self.cancel_capture();
        }
    }
}

/// One tap of `key` - a press and its release - as egui input.
fn push_key(raw: &mut egui::RawInput, key: egui::Key) {
    for pressed in [true, false] {
        raw.events.push(egui::Event::Key { key, physical_key: None, pressed, repeat: false, modifiers: egui::Modifiers::NONE });
    }
}

/// The browser's look on egui: its palette, its 8 px controls and 12 px cards, its type sizes.
fn apply_theme(ctx: &egui::Context) {
    ctx.all_styles_mut(|style| {
        use egui::{FontId, TextStyle};
        style.text_styles = [
            (TextStyle::Small, FontId::proportional(12.0)),
            (TextStyle::Body, FontId::proportional(14.0)),
            (TextStyle::Button, FontId::proportional(14.0)),
            (TextStyle::Heading, FontId::proportional(22.0)),
            (TextStyle::Monospace, FontId::monospace(13.0)),
        ]
        .into();
        style.spacing.item_spacing = egui::vec2(8.0, 8.0);
        style.spacing.button_padding = egui::vec2(14.0, 7.0);
        style.spacing.interact_size.y = 30.0;
        let v = &mut style.visuals;
        *v = egui::Visuals::dark();
        v.panel_fill = BG;
        v.window_fill = PANEL;
        v.extreme_bg_color = Color32::from_rgb(0x0f, 0x0f, 0x18);
        v.faint_bg_color = PANEL;
        v.hyperlink_color = ACCENT;
        v.selection.bg_fill = ACCENT.linear_multiply(0.35);
        v.selection.stroke = egui::Stroke::new(1.0, ACCENT);
        v.window_stroke = egui::Stroke::new(1.0, EDGE);
        v.window_corner_radius = 12.into();
        for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active, &mut v.widgets.open] {
            w.corner_radius = 8.into();
            w.weak_bg_fill = PANEL_2;
            w.bg_fill = PANEL_2;
            w.bg_stroke = egui::Stroke::new(1.0, EDGE);
            w.fg_stroke = egui::Stroke::new(1.0, INK);
            w.expansion = 0.0;
        }
        v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, DIM);
        // Focus - the pad's cursor - wears the accent, as the browser's `.navfocus` does.
        v.widgets.active.bg_stroke = egui::Stroke::new(2.0, ACCENT);
        v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, INK);
        v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, EDGE);
    });
}

/// Text in the UI font's BOLD face (see [`load_ui_fonts`]) - egui's `strong` only recolours.
pub(crate) fn bold(text: impl Into<String>) -> RichText {
    RichText::new(text).family(egui::FontFamily::Name("bold".into()))
}

/// The browser renders in the platform's own UI face (`system-ui`): Segoe UI on Windows,
/// the system sans elsewhere. Load it, regular and bold, from where each platform keeps it,
/// ahead of egui's built-in face - which stays as the fallback, for a glyph the system font
/// lacks or a system with neither installed.
fn load_ui_fonts(ctx: &egui::Context) {
    const REGULAR: &[&str] = &[
        "C:/Windows/Fonts/segoeui.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ];
    const BOLD: &[&str] = &[
        "C:/Windows/Fonts/seguisb.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    ];
    let first = |paths: &[&str]| paths.iter().find_map(|p| std::fs::read(p).ok());
    let mut fonts = egui::FontDefinitions::default();
    let fallback = fonts.families.get(&egui::FontFamily::Proportional).cloned().unwrap_or_default();
    let mut bold_family = fallback.clone();
    if let Some(bytes) = first(REGULAR) {
        fonts.font_data.insert("ui".into(), egui::FontData::from_owned(bytes).into());
        fonts.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "ui".into());
        bold_family.insert(0, "ui".into());
    }
    if let Some(bytes) = first(BOLD) {
        fonts.font_data.insert("ui-bold".into(), egui::FontData::from_owned(bytes).into());
        bold_family.insert(0, "ui-bold".into());
    }
    fonts.families.insert(egui::FontFamily::Name("bold".into()), bold_family);
    ctx.set_fonts(fonts);
}

/// The browser's `.btn.primary`: accent fill, dark ink.
pub(crate) fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(bold(text).color(ACCENT_INK)).fill(ACCENT).stroke(egui::Stroke::NONE)
}

/// The browser's `.btn.danger`.
pub(crate) fn danger(text: &str) -> egui::Button<'static> {
    egui::Button::new(bold(text).color(DANGER)).stroke(egui::Stroke::new(1.0, DANGER_EDGE))
}

/// The browser's `.menu-in` / `.capture-in` panel: the card colours, 12 px corners, 16 px in.
pub(crate) fn panel_frame() -> egui::Frame {
    egui::Frame::new().fill(PANEL).stroke(egui::Stroke::new(1.0, EDGE)).corner_radius(12).inner_margin(egui::Margin::same(16))
}

/// A header link: plain text, ink when it is the page shown, an underline under the pad's cursor.
fn nav_link(ui: &mut egui::Ui, text: &str, current: bool) -> egui::Response {
    let r = ui.add(egui::Button::new(RichText::new(text).color(if current { INK } else { DIM })).frame(false));
    if r.has_focus() {
        let y = r.rect.bottom() - 2.0;
        ui.painter().hline(r.rect.x_range(), y, egui::Stroke::new(2.0, ACCENT));
    }
    r
}

/// The browser's `.seg`: the Keyboard / Gamepad switch over the controller picture.
pub(crate) fn seg(ui: &mut egui::Ui, mode: &mut vitapic::Mode) {
    egui::Frame::new().stroke(egui::Stroke::new(1.0, EDGE)).corner_radius(8).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            for (m, label) in [(vitapic::Mode::Keyboard, "Keyboard"), (vitapic::Mode::Gamepad, "Gamepad")] {
                let on = *mode == m;
                let b = egui::Button::new(bold(label).size(13.0).color(if on { ACCENT_INK } else { DIM }))
                    .fill(if on { ACCENT } else { Color32::TRANSPARENT })
                    .stroke(egui::Stroke::NONE)
                    .corner_radius(if on { 7 } else { 0 });
                if ui.add(b).clicked() {
                    *mode = m;
                }
            }
        });
    });
}

/// The browser's `.card`: a panel-filled, edged, 12 px-rounded box the width of the page.
pub(crate) fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let r = egui::Frame::new()
        .fill(PANEL)
        .stroke(egui::Stroke::new(1.0, EDGE))
        .corner_radius(12)
        .inner_margin(egui::Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner;
    ui.add_space(4.0);
    r
}

/// The browser's `.card.error`.
pub(crate) fn error_card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    let r = egui::Frame::new()
        .fill(ERROR_FILL)
        .stroke(egui::Stroke::new(1.0, ERROR_EDGE))
        .corner_radius(12)
        .inner_margin(egui::Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner;
    ui.add_space(4.0);
    r
}

/// The browser's breadcrumbs: `Library / <title> / Settings`, every step but the last a link.
/// Returns the screen a clicked step leads to.
fn crumbs(ui: &mut egui::Ui, trail: &[(&str, Option<Screen>)]) -> Option<Screen> {
    let mut to = None;
    ui.horizontal(|ui| {
        for (i, (text, screen)) in trail.iter().enumerate() {
            if i > 0 {
                ui.label(RichText::new("/").color(DIM.linear_multiply(0.5)).size(13.0));
            }
            match screen {
                Some(sc) => {
                    let r = ui.add(egui::Button::new(RichText::new(*text).color(ACCENT).size(13.0)).frame(false));
                    if r.clicked() {
                        to = Some(sc.clone());
                    }
                }
                None => {
                    ui.label(RichText::new(*text).color(INK).size(13.0));
                }
            }
        }
    });
    to
}

/// A label, the browser's " changed" mark after it when the title overrides it.
fn label_with_mark(ui: &mut egui::Ui, label: &str, changed: bool, size: f32) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;
        ui.label(RichText::new(label).color(INK).size(size));
        if changed {
            ui.label(bold("changed").size(11.0).color(WARN));
        }
    });
}

/// One settings row: the label (and its hint under it) at the left, the control at the right.
fn setting_row(ui: &mut egui::Ui, label: &str, hint: Option<&str>, changed: bool, control: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.set_max_width((ui.available_width() - 260.0).max(200.0));
            ui.spacing_mut().item_spacing.y = 2.0;
            label_with_mark(ui, label, changed, 14.0);
            if let Some(h) = hint {
                ui.label(RichText::new(h).color(DIM).size(12.0));
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), control);
    });
}

/// A text field's label above it (the browser's `label.col`), hint underneath.
fn field_label(ui: &mut egui::Ui, label: &str, hint: &str, changed: bool) {
    ui.add_space(4.0);
    label_with_mark(ui, label, changed, 14.0);
    ui.label(RichText::new(hint).color(DIM).size(12.0));
}

/// A checkbox row: the box, its label, and the hint under the label.
pub(crate) fn check_row(ui: &mut egui::Ui, value: &mut bool, label: &str, hint: Option<&str>, changed: bool) -> bool {
    const BOX: f32 = 20.0;
    let mut flipped = false;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 16.0;
        let (rect, mut r) = ui.allocate_exact_size(egui::vec2(BOX, BOX), egui::Sense::click());
        if r.clicked() {
            *value = !*value;
            flipped = true;
            r.mark_changed();
        }
        let p = ui.painter();
        if *value {
            p.rect_filled(rect, 4.0, ACCENT);
            let tick = [rect.left_center() + egui::vec2(4.5, 0.5), rect.center_bottom() + egui::vec2(-2.0, -5.0), rect.right_top() + egui::vec2(-4.5, 5.5)];
            p.line(tick.to_vec(), egui::Stroke::new(2.5, ACCENT_INK));
        } else {
            p.rect_filled(rect, 4.0, Color32::WHITE);
        }
        if r.has_focus() {
            p.rect_stroke(rect.expand(3.0), 6.0, egui::Stroke::new(2.0, ACCENT), egui::StrokeKind::Outside);
        }
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                if ui.add(egui::Label::new(RichText::new(label).color(INK).size(15.0)).sense(egui::Sense::click())).clicked() {
                    *value = !*value;
                    flipped = true;
                }
                if changed {
                    ui.label(bold("changed").size(11.0).color(WARN));
                }
            });
            if let Some(h) = hint {
                ui.label(RichText::new(h).color(DIM).size(12.0));
            }
        });
    });
    flipped
}

fn scaling_label(s: Scaling) -> &'static str {
    match s {
        Scaling::Fit => "Fit (smooth)",
        Scaling::Integer => "Integer (crisp)",
        Scaling::Stretch => "Stretch",
    }
}

fn fullscreen_label(f: FullscreenStart) -> &'static str {
    match f {
        FullscreenStart::Auto => "Automatic (touch screens)",
        FullscreenStart::Always => "Always",
        FullscreenStart::Never => "Never",
    }
}

/// The title page's backdrop: `pic0` covering the hero (cropped, never stretched), under the
/// browser's left-to-right shade (`.hero-in`), with the edge line along its foot.
fn hero_backdrop(rect: egui::Rect, pic: Option<&TextureHandle>) -> egui::Shape {
    let mut shapes = vec![egui::Shape::rect_filled(rect, 0.0, Color32::BLACK)];
    if let Some(tex) = pic {
        let [w, h] = tex.size().map(|v| v as f32);
        // `background-size: cover`, centred: the UV window of the picture the rect shows.
        let (ra, ta) = (rect.width() / rect.height(), w / h);
        let uv = if ta > ra {
            let f = ra / ta;
            egui::Rect::from_min_max(egui::pos2((1.0 - f) / 2.0, 0.0), egui::pos2((1.0 + f) / 2.0, 1.0))
        } else {
            let f = ta / ra;
            egui::Rect::from_min_max(egui::pos2(0.0, (1.0 - f) / 2.0), egui::pos2(1.0, (1.0 + f) / 2.0))
        };
        let mut mesh = egui::Mesh::with_texture(tex.id());
        mesh.add_rect_with_uv(rect, uv, Color32::WHITE);
        shapes.push(egui::Shape::mesh(mesh));
    }
    // rgba(11,11,18,.96) -> .75 at 60% -> .4 at the right.
    let shade = |a: f32| Color32::from_rgba_unmultiplied(0x0b, 0x0b, 0x12, (a * 255.0) as u8);
    let mut mesh = egui::Mesh::default();
    let mid = rect.left() + rect.width() * 0.6;
    for (x0, x1, a0, a1) in [(rect.left(), mid, 0.96, 0.75), (mid, rect.right(), 0.75, 0.4)] {
        let base = mesh.vertices.len() as u32;
        for (x, a) in [(x0, a0), (x1, a1)] {
            mesh.colored_vertex(egui::pos2(x, rect.top()), shade(a));
            mesh.colored_vertex(egui::pos2(x, rect.bottom()), shade(a));
        }
        mesh.add_triangle(base, base + 1, base + 2);
        mesh.add_triangle(base + 1, base + 3, base + 2);
    }
    shapes.push(egui::Shape::mesh(mesh));
    shapes.push(egui::Shape::hline(rect.x_range(), rect.bottom(), egui::Stroke::new(1.0, EDGE)));
    egui::Shape::Vec(shapes)
}

/// The browser's `.drop` box: a dashed outline, accent while something is dragged over.
fn dashed_rect(painter: &egui::Painter, r: egui::Rect, stroke: egui::Stroke) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    painter.extend(egui::Shape::dashed_line(&pts, stroke, 6.0, 4.0));
}

/// Days since 1970-01-01 to a civil `(year, month, day)` (Howard Hinnant's algorithm).
fn civil(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// A Unix-milliseconds date as the title page shows it, `2026-10-03`, or `never`.
fn fmt_date(ms: u64) -> String {
    if ms == 0 {
        return "never".into();
    }
    let (y, m, d) = civil((ms / 86_400_000) as i64);
    format!("{y:04}-{m:02}-{d:02}")
}

/// A Unix-milliseconds moment, `2026-10-03 13:51 UTC`.
fn fmt_datetime(ms: u64) -> String {
    if ms == 0 {
        return "never".into();
    }
    let mins = ms / 60_000;
    format!("{} {:02}:{:02} UTC", fmt_date(ms), (mins / 60) % 24, mins % 60)
}

/// The pad on a focused combo box: `nav.adjust` steps `value` through `options` (wrapping), as
/// left/right and south do on a `<select>` in the browser. Records that a combo had focus, so the
/// next frame's left/right come here instead of moving focus. True when the value changed.
fn combo_nav<T: PartialEq + Clone>(resp: &egui::Response, nav: &mut Nav, value: &mut T, options: &[T]) -> bool {
    if !resp.has_focus() {
        return false;
    }
    nav.combo_focused = true;
    let step = std::mem::take(&mut nav.adjust);
    if step == 0 || options.is_empty() {
        return false;
    }
    let n = options.len() as i32;
    let at = options.iter().position(|o| o == value).map_or(0, |i| i as i32);
    *value = options[(at + step).rem_euclid(n) as usize].clone();
    true
}

/// The LEAVES of `mine` that differ from `base` - nested objects recurse, so a title
/// that remaps one key stores one key and every other global change still reaches it.
fn deep_diff(base: &serde_json::Value, mine: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    let (Some(b), Some(m)) = (base.as_object(), mine.as_object()) else { return mine.clone() };
    let mut out = serde_json::Map::new();
    for (k, mv) in m {
        match b.get(k) {
            Some(bv) if bv == mv => {}
            Some(bv) if bv.is_object() && mv.is_object() => {
                let d = deep_diff(bv, mv);
                if d.as_object().map(|o| !o.is_empty()).unwrap_or(false) {
                    out.insert(k.clone(), d);
                }
            }
            _ => {
                out.insert(k.clone(), mv.clone());
            }
        }
    }
    for k in b.keys() {
        if !m.contains_key(k) {
            out.insert(k.clone(), Value::Null);
        }
    }
    Value::Object(out)
}

/// Every leaf of a settings patch as a dotted path (`keyboard.cross`) - the browser's
/// `leafPaths`, which is how the settings page knows what to mark "changed".
fn leaf_paths(v: &serde_json::Value, prefix: &str) -> Vec<String> {
    match v.as_object() {
        Some(o) => o.iter().flat_map(|(k, v)| if v.is_object() { leaf_paths(v, &format!("{prefix}{k}.")) } else { vec![format!("{prefix}{k}")] }).collect(),
        None => Vec::new(),
    }
}

pub(crate) fn fmt_bytes(n: u64) -> String {
    if n < 1_000_000 {
        format!("{} KB", n / 1000)
    } else if n < 1_000_000_000 {
        format!("{} MB", n / 1_000_000)
    } else {
        format!("{:.2} GB", n as f64 / 1e9)
    }
}

/// Save a title's saved data where the person picks, named as the browser's download is.
fn download_gamedata(profile: &str, id: &str) -> String {
    let Some(p) = rfd::FileDialog::new().set_file_name(format!("vitaslop-{id}-{profile}-gamedata.zip")).add_filter("Zip", &["zip"]).save_file() else {
        return String::new();
    };
    match std::fs::copy(library::gamedata_path(profile, id), &p) {
        Ok(_) => format!("Saved {}", p.display()),
        Err(e) => format!("Could not save {}: {e}", p.display()),
    }
}

/// The default sound output's name, if there is one.
fn cpal_device_name() -> Option<String> {
    use cpal::traits::{DeviceTrait, HostTrait};
    let d = cpal::default_host().default_output_device()?;
    d.description().ok().map(|d| d.to_string())
}

pub(crate) fn open_path(p: &std::path::Path) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(p).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(p).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(p).spawn();
}

fn open_url(url: &str) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(url).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(url).spawn();
}

pub(crate) fn decode_png(bytes: &[u8]) -> Option<egui::ColorImage> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width as usize, info.height as usize);
    let rgba: Vec<u8> = match info.color_type {
        png::ColorType::Rgba => buf[..w * h * 4].to_vec(),
        png::ColorType::Rgb => buf[..w * h * 3].chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        png::ColorType::GrayscaleAlpha => buf[..w * h * 2].chunks(2).flat_map(|p| [p[0], p[0], p[0], p[1]]).collect(),
        png::ColorType::Grayscale => buf[..w * h].iter().flat_map(|&g| [g, g, g, 255]).collect(),
        _ => return None,
    };
    Some(egui::ColorImage::from_rgba_unmultiplied([w, h], &rgba))
}

/// Whether the saved window's title bar would land on a connected monitor - enough of it to grab.
fn on_a_monitor(event_loop: &ActiveEventLoop, p: &library::WindowPlace) -> bool {
    let (bar_w, bar_h) = (i64::from(p.width.min(200)), 32);
    let (x, y) = (i64::from(p.x), i64::from(p.y));
    event_loop.available_monitors().any(|m| {
        let (mp, ms) = (m.position(), m.size());
        let (mx, my) = (i64::from(mp.x), i64::from(mp.y));
        x < mx + i64::from(ms.width) && x + bar_w > mx && y < my + i64::from(ms.height) && y + bar_h > my
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One pass of a column of three buttons over `keys`; returns the clicked button, if any.
    fn pass(ctx: &egui::Context, keys: &[egui::Key]) -> Option<usize> {
        let mut raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0))),
            ..Default::default()
        };
        for k in keys {
            push_key(&mut raw, *k);
        }
        let mut clicked = None;
        let _ = ctx.run_ui(raw, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                for i in 0..3 {
                    if ui.button(format!("button {i}")).clicked() {
                        clicked = Some(i);
                    }
                }
            });
        });
        clicked
    }

    /// What the pad's translation rests on: with nothing focused a Tab focuses the first widget,
    /// an arrow moves focus spatially, and Enter clicks the focused button - egui's contract, which
    /// a version bump could change under the shell without a compile error.
    #[test]
    fn tab_arrows_and_enter_drive_egui_focus_and_clicks() {
        let ctx = egui::Context::default();
        pass(&ctx, &[]);
        assert_eq!(ctx.memory(|m| m.focused()), None);
        pass(&ctx, &[egui::Key::Tab]);
        pass(&ctx, &[]);
        assert!(ctx.memory(|m| m.focused()).is_some(), "Tab focuses the first widget");
        pass(&ctx, &[egui::Key::ArrowDown]);
        pass(&ctx, &[]);
        assert_eq!(pass(&ctx, &[egui::Key::Enter]), Some(1), "down then Enter clicks the second button");
        pass(&ctx, &[egui::Key::ArrowUp]);
        pass(&ctx, &[]);
        assert_eq!(pass(&ctx, &[egui::Key::Enter]), Some(0));
    }

    #[test]
    fn dates_read_as_the_calendar_does() {
        assert_eq!(fmt_date(0), "never");
        assert_eq!(fmt_date(86_400_000), "1970-01-02");
        assert_eq!(fmt_date(1_791_035_486_827), "2026-10-03");
        assert_eq!(fmt_date(951_782_400_000), "2000-02-29");
        assert_eq!(fmt_datetime(1_791_035_486_827), "2026-10-03 13:51 UTC");
    }

    #[test]
    fn a_focused_combo_steps_its_value_and_wraps() {
        let ctx = egui::Context::default();
        let mut nav = Nav { adjust: -1, ..Default::default() };
        let mut value = "south";
        let options = ["south", "east", "west"];
        let mut changed = false;
        let mut run = |nav: &mut Nav, value: &mut &str, focus: bool| {
            let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
                let r = ui.button("combo");
                if focus {
                    r.request_focus();
                }
                changed |= combo_nav(&r, nav, value, &options);
            });
        };
        for _ in 0..4 {
            run(&mut nav, &mut value, true);
        }
        assert!(changed && nav.combo_focused);
        assert_eq!(value, "west", "left from the first option wraps to the last");
        assert_eq!(nav.adjust, 0, "the step is consumed once");
    }

    #[test]
    fn a_capture_lands_in_the_draft_or_the_running_settings() {
        let mut sh = Shell::placeholder();
        sh.open_draft_with(None, Settings::default());
        sh.begin_capture(Button::Cross, Kind::Keyboard, false);
        sh.finish_capture("KeyJ".into());
        assert_eq!(sh.draft.as_ref().unwrap().s.keyboard["cross"], "KeyJ");
        assert!(sh.capture.is_none(), "one capture, one key");
        sh.begin_capture(Button::Start, Kind::Gamepad, true);
        sh.finish_capture("home".into());
        assert_eq!(sh.session_settings.gamepad["start"], "home", "a live capture changes the running settings");
        assert_eq!(sh.draft.as_ref().unwrap().s.gamepad["start"], "start", "and not the draft");
        sh.begin_capture(Button::Circle, Kind::Keyboard, false);
        sh.cancel_capture();
        sh.finish_capture("KeyK".into());
        assert_eq!(sh.draft.as_ref().unwrap().s.keyboard["circle"], "KeyX", "a cancelled capture assigns nothing");
    }

    #[test]
    fn a_title_patch_marks_exactly_its_own_leaves() {
        let patch = serde_json::json!({ "keyboard": { "cross": "KeyJ" }, "showFps": true, "pad": { "opacity": 0.3 } });
        let mut paths = leaf_paths(&patch, "");
        paths.sort();
        assert_eq!(paths, ["keyboard.cross", "pad.opacity", "showFps"]);
        let d = Draft { title_id: Some("PCSA00000".into()), s: Settings::default(), knobs_text: String::new(), overridden: paths.into_iter().collect(), saved_at: None };
        assert!(d.changed("keyboard") && d.changed("keyboard.cross") && d.changed("showFps"));
        assert!(!d.changed("keyboard.circle") && !d.changed("scaling") && !d.changed("show"));
    }

    #[test]
    fn the_picture_is_placed_as_the_browser_places_its_canvas() {
        use crate::retail::game_rect;
        assert_eq!(game_rect(Scaling::Fit, (960, 544)), (0, 0, 960, 544), "1:1 fills the window");
        assert_eq!(game_rect(Scaling::Fit, (1100, 700)), (0, 38, 1100, 623), "wider than tall: bars above and below");
        assert_eq!(game_rect(Scaling::Fit, (1920, 600)), (430, 0, 1059, 600), "bars at the sides");
        assert_eq!(game_rect(Scaling::Integer, (2000, 1200)), (40, 56, 1920, 1088), "the largest whole multiple");
        assert_eq!(game_rect(Scaling::Integer, (1100, 700)), (70, 78, 960, 544), "one whole multiple");
        assert_eq!(game_rect(Scaling::Integer, (800, 600)), game_rect(Scaling::Fit, (800, 600)), "too small for one: fit");
        assert_eq!(game_rect(Scaling::Stretch, (1100, 700)), (0, 0, 1100, 700));
    }
}

