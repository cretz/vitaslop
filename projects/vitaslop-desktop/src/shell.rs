//! The native shell: library, title page, settings, import, and the in-game menu,
//! drawn with egui over the same wgpu surface the game presents to.
//!
//! One window, one surface. When no title runs, egui owns the whole frame; while
//! one runs, the game is presented first and egui draws the frame-rate badge and
//! (on Esc) the menu over it in the same command encoder. The rules - what a
//! setting is, which knob it means, what a title record holds - are the shared
//! ones in `vitaslop-frontend`; this file is only the drawing and the wiring.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{Color32, RichText, TextureHandle, TextureOptions};
use vitaslop_frontend::input::{Button, GAMEPAD_CONTROLS};
use vitaslop_frontend::meta::TitleMeta;
use vitaslop_frontend::settings::{self, PadMode, Scaling, Settings};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::input::Input;
use crate::library::{self, ImportProgress};
use crate::navpad::{Dir, NavPad};
use crate::retail::{DesktopInput, RetailGfx, RetailGuest, SharedInput};
use crate::session::{Session, Stats};

// The browser front end's palette (`web/app.css` `:root`), so the two read as one product.
const BG: Color32 = Color32::from_rgb(0x0b, 0x0b, 0x12);
const PANEL: Color32 = Color32::from_rgb(0x14, 0x14, 0x1f);
const PANEL_2: Color32 = Color32::from_rgb(0x1b, 0x1b, 0x29);
const EDGE: Color32 = Color32::from_rgb(0x2a, 0x2a, 0x3a);
const INK: Color32 = Color32::from_rgb(0xe4, 0xe4, 0xec);
const ACCENT: Color32 = Color32::from_rgb(0x8f, 0xe0, 0xa0);
const ACCENT_INK: Color32 = Color32::from_rgb(0x08, 0x12, 0x0c);
const DIM: Color32 = Color32::from_rgb(0x8f, 0x8f, 0xa3);
const DANGER: Color32 = Color32::from_rgb(0xe8, 0xa0, 0xa8);
const DANGER_EDGE: Color32 = Color32::from_rgb(0x4a, 0x2a, 0x34);

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

enum Screen {
    Library,
    Title(String),
    Settings(Option<String>),
    Import,
    /// What this run reported. The warnings are captured whether or not anyone is watching
    /// stderr (see `crate::log`), so this screen is the only place a person who launched the
    /// app by double-clicking it can read them.
    Diagnostics,
}

/// A settings form being edited: the record plus the text of the knobs box.
struct Draft {
    title_id: Option<String>,
    s: Settings,
    knobs_text: String,
    /// The button whose key is being captured.
    capturing: Option<Button>,
    saved_at: Option<Instant>,
}

/// The guest being built on a thread (decrypt + link + transpile).
struct Loading {
    title_id: String,
    rx: Receiver<Result<RetailGuest, String>>,
    input: SharedInput,
    settings: Settings,
    started: Instant,
}

struct Shell {
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
    /// `frame`. A frame is presented ONCE while the game runs.
    presented: Option<u64>,
    force_present: bool,
    session_settings: Settings,
    menu_open: bool,
    stats: Option<Stats>,
    import: Option<Arc<Mutex<ImportProgress>>>,
    import_msg: Option<Result<String, String>>,
    confirm_remove: Option<String>,
    error: Option<String>,
    last_key: Option<String>,
    /// The outcome of the last diagnostics save, shown in the in-game menu.
    diag_saved: Option<String>,
    /// The gamepad driving the shell's own screens - see `crate::navpad`.
    navpad: NavPad,
    nav: Nav,
    /// `VITASLOP_SHELL_SHOT` - see [`ShellShot`].
    shot: Option<ShellShot>,
}

/// `VITASLOP_SHELL_SHOT=<png>` (+ `VITASLOP_SHELL_SCREEN=library|settings|import|diagnostics|
/// title:<ID>|play:<ID>`): open that screen - or PLAY that title through the library's own Play
/// path, loading screen and LoadExec included - write the window as it SHOWS it after
/// `VITASLOP_SHELL_SHOT_AT` redraws (default 40: egui lays out over a couple of passes, an icon
/// loads in one), and exit. The shell's pages side by side with the browser's, without anyone
/// clicking through them.
struct ShellShot {
    path: PathBuf,
    screen: String,
    redraws: u32,
    at: u32,
}

impl ShellShot {
    fn from_knobs() -> Option<ShellShot> {
        let path = std::env::var("VITASLOP_SHELL_SHOT").ok()?;
        let screen = std::env::var("VITASLOP_SHELL_SCREEN").unwrap_or_else(|_| "library".into());
        let at = std::env::var("VITASLOP_SHELL_SHOT_AT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(40);
        Some(ShellShot { path: PathBuf::from(path), screen, redraws: 0, at })
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

pub fn run() -> Result<(), String> {
    let event_loop = EventLoop::new().map_err(|e| format!("create event loop: {e}"))?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = Shell {
        window: None,
        gfx: None,
        egui: egui::Context::default(),
        egui_state: None,
        renderer: None,
        screen: Screen::Library,
        titles: library::list_titles(),
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
        stats: None,
        import: None,
        import_msg: None,
        confirm_remove: None,
        error: None,
        last_key: None,
        diag_saved: None,
        navpad: NavPad::new(),
        nav: Nav::default(),
        shot: ShellShot::from_knobs(),
    };
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
        let attrs = Window::default_attributes().with_title("vitaslop").with_inner_size(LogicalSize::new(1100, 700));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        match RetailGfx::new(window.clone()) {
            Ok(g) => {
                self.renderer = Some(egui_wgpu::Renderer::new(g.device(), g.render_format(), egui_wgpu::RendererOptions::default()));
                self.gfx = Some(g);
            }
            Err(e) => {
                eprintln!("failed to init GPU surface: {e}");
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
        // egui sees every event first when it owns the screen; while a game runs it
        // sees them only when the menu is open, so the game gets the keys.
        let ui_wants = self.session.is_none() || self.menu_open;
        let mut consumed = false;
        if ui_wants
            && let Some(st) = self.egui_state.as_mut() {
                consumed = st.on_window_event(&window, &event).consumed;
            }
        match &event {
            WindowEvent::CloseRequested => {
                if let Some(s) = self.session.as_mut() {
                    s.guest.flush_save(true);
                }
                event_loop.exit();
            }
            WindowEvent::Resized(sz) => {
                if let Some(g) = self.gfx.as_mut() {
                    g.resize(sz.width, sz.height);
                }
                self.force_present = true;
            }
            // The browser's "drop files or a folder anywhere here": a dropped package, zip,
            // homebrew or dump folder is imported, from any screen but a running game.
            WindowEvent::DroppedFile(path) if self.session.is_none() => {
                let busy = self.import.as_ref().is_some_and(|p| !p.lock().unwrap().finished);
                if !busy {
                    self.screen = Screen::Import;
                    self.start_import(path.clone());
                }
            }
            WindowEvent::KeyboardInput { event: k, .. } => {
                if let PhysicalKey::Code(code) = k.physical_key {
                    let pressed = k.state == ElementState::Pressed;
                    if pressed && !k.repeat {
                        self.last_key = Some(format!("{code:?}"));
                    }
                    if pressed && code == KeyCode::Escape && !k.repeat && self.session.is_some() {
                        self.toggle_menu();
                        return;
                    }
                    if pressed && code == KeyCode::F11 && !k.repeat {
                        let full = window.fullscreen().is_some();
                        window.set_fullscreen(if full { None } else { Some(winit::window::Fullscreen::Borderless(None)) });
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
            if !self.menu_open && !consumed {
                s.event(&event, Some(size));
            } else if let WindowEvent::Focused(_) = &event {
                s.event(&event, Some(size));
            }
        }
    }

    /// While a game runs, SLEEP until its next frame is due (a redraw with nothing new is
    /// skipped - see `frame`); the library, the menu and a pause redraw as before.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
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
    fn toggle_menu(&mut self) {
        self.menu_open = !self.menu_open;
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

    /// Read the shell's pad and turn this frame's asks into egui input: a move is an arrow key
    /// (egui's own spatial focus), or Tab when nothing has focus yet; south is Enter, which
    /// clicks a focused button or toggles a checkbox. The pad is the shell's only while the shell
    /// owns the screen - no game, or the in-game menu open; otherwise the game's input has it.
    fn pad_to_egui(&mut self, raw: &mut egui::RawInput) {
        self.navpad.attach(self.session.is_none() || self.menu_open);
        let pad = self.navpad.poll();
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
                && let Some(r) = ctx.memory(|m| m.focused()).and_then(|id| ctx.read_response(id)) {
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
        if egui::Popup::is_any_open(ctx) {
            egui::Popup::close_all(ctx);
            return;
        }
        if let Some(d) = self.draft.as_mut()
            && d.capturing.is_some() {
                d.capturing = None;
                return;
            }
        if self.confirm_remove.take().is_some() {
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

    fn frame(&mut self, size: (f64, f64), window: &Arc<Window>) {
        if let Some(screen) = self.shot.as_ref().filter(|s| s.redraws == 0).map(|s| s.screen.clone()) {
            match screen.as_str() {
                "settings" => self.open_draft(None),
                "import" => self.screen = Screen::Import,
                "diagnostics" => self.screen = Screen::Diagnostics,
                s => {
                    if let Some(id) = s.strip_prefix("title:") {
                        self.screen = Screen::Title(id.to_string());
                    } else if let Some(id) = s.strip_prefix("play:") {
                        self.start(id);
                    }
                }
            }
        }
        self.poll_loading();
        // The game, if one runs.
        let mut exec = None;
        let mut menu_request = false;
        if let Some(s) = self.session.as_mut() {
            s.tick(Some(size));
            exec = s.take_exec();
            // The pad's way into the menu (home, or start+select held) - see `Input::pump_gamepad`.
            if s.input.take_menu_request() && !self.menu_open {
                menu_request = true;
            }
            if let Some(st) = s.stats(Instant::now()) {
                if self.session_settings.fps_in_title {
                    window.set_title(&format!("vitaslop  |  {}", st.title_line()));
                }
                self.stats = Some(st);
            }
        }
        if menu_request {
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
        if game_live && !egui_wants && !self.force_present && self.presented == frame_now {
            for id in &full.textures_delta.free {
                renderer.free_texture(id);
            }
            return;
        }
        self.presented = frame_now;
        self.force_present = false;
        let scenes = self.session.as_mut().map(|s| s.scenes());
        let scenes = scenes.filter(|(sc, _, _)| !sc.is_empty());
        let drew_game = scenes.is_some();
        if let Some(shot) = self.shot.as_mut() {
            shot.redraws += 1;
            if shot.redraws == shot.at && !gfx.request_capture(shot.path.clone()) {
                eprintln!("VITASLOP_SHELL_SHOT: this surface cannot be read back");
            }
        }
        let result = gfx.frame(scenes, |device, queue, encoder, view, _| {
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
        for id in &full.textures_delta.free {
            renderer.free_texture(id);
        }
        if let Err(e) = result {
            self.error = Some(e);
            self.session = None;
        } else if drew_game
            && let Some(s) = self.session.as_mut() {
                let wb = gfx.rtt_writebacks();
                s.apply_writebacks(&wb);
            }
    }

    /// A stand-in so `ui` can borrow the whole shell while the context runs.
    fn placeholder() -> Shell {
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
            stats: None,
            import: None,
            import_msg: None,
            confirm_remove: None,
            error: None,
            last_key: None,
            diag_saved: None,
            navpad: NavPad::inert(),
            nav: Nav::default(),
            shot: None,
        }
    }

    // ------------------------------- state -------------------------------

    fn poll_loading(&mut self) {
        let Some(l) = self.loading.as_ref() else { return };
        match l.rx.try_recv() {
            Ok(Ok(mut guest)) => {
                let l = self.loading.take().unwrap();
                if let Err(e) = guest.persist_to(&library::saves_dir(&l.settings.profile), &library::title_dir(&l.title_id)) {
                    self.error = Some(e);
                    return;
                }
                let input = Input::new(&l.settings);
                if let Some(g) = self.gfx.as_ref() {
                    guest.install_complete_scene_hook(g.completion_hook());
                }
                let session = Session::new(guest, l.input, input, l.settings.pause_on_blur);
                if let Some(a) = session.audio.as_ref()
                    && library::muted() {
                        a.set_muted(true);
                    }
                self.session = Some(session);
                self.playing = Some(l.title_id.clone());
                self.session_settings = l.settings;
                self.menu_open = false;
                self.stats = None;
                if let Some(mut m) = self.titles.iter().find(|t| t.title_id == l.title_id).cloned() {
                    m.last_played_at = library::now_ms();
                    let _ = library::write_meta(&m);
                    self.titles = library::list_titles();
                }
                if let Some(w) = self.window.as_ref() {
                    w.set_title("vitaslop");
                }
            }
            Ok(Err(e)) => {
                self.loading = None;
                self.error = Some(e);
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.loading = None;
                self.error = Some("the title failed to load (the loader thread died)".into());
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
        // The engine reads its knobs from the environment; the settings are that
        // environment here. Browser-only knobs are left out.
        for (k, v) in s.run_knobs() {
            if k == "VITASLOP_BROWSER_FASTFORWARD" {
                continue;
            }
            unsafe { std::env::set_var(&k, &v) };
        }
        let dir = library::title_dir(id);
        let input: SharedInput = Arc::new(Mutex::new(DesktopInput::default()));
        let (tx, rx) = channel();
        let recipe = (!s.recipe.trim().is_empty()).then(|| s.recipe.clone());
        let input2 = input.clone();
        std::thread::spawn(move || {
            let r = RetailGuest::new_with_exec(&dir, input2, recipe.as_deref(), main_exec.as_deref());
            let _ = tx.send(r);
        });
        self.loading = Some(Loading { title_id: id.to_string(), rx, input, settings: s, started: Instant::now() });
    }

    fn stop(&mut self) {
        if let Some(mut s) = self.session.take() {
            s.guest.flush_save(true);
        }
        self.menu_open = false;
        self.stats = None;
        if let Some(w) = self.window.as_ref() {
            w.set_title("vitaslop");
        }
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
        let knobs_text = s.knobs.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join("\n");
        self.draft = Some(Draft { title_id: title_id.clone(), s, knobs_text, capturing: None, saved_at: None });
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
                library::save_title_patch(id, Some(&deep_diff(&global, &mine)))
            }
            None => library::save_global_settings(&d.s),
        };
        match r {
            Ok(()) => d.saved_at = Some(Instant::now()),
            Err(e) => self.error = Some(format!("could not save settings: {e}")),
        }
    }

    fn start_import(&mut self, path: PathBuf) {
        let progress = Arc::new(Mutex::new(ImportProgress::default()));
        let p2 = progress.clone();
        std::thread::spawn(move || {
            let r = library::import(&path, &p2);
            let mut g = p2.lock().unwrap();
            g.finished = true;
            match r {
                Ok(m) => g.title_id = Some(m.title_id),
                Err(e) => g.error = Some(e),
            }
        });
        self.import = Some(progress);
        self.import_msg = None;
    }

    // ------------------------------- drawing -------------------------------

    fn ui(&mut self, root: &mut egui::Ui) {
        let ctx = root.ctx().clone();
        self.nav_pass(&ctx);
        if self.session.is_some() {
            self.ui_playing(&ctx);
            return;
        }
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
                        let (held, _, dropped) = vitaslop_platform::diag::counts(vitaslop_platform::diag::Channel::Warning);
                        let diag = if held + dropped == 0 { "Diagnostics".to_string() } else { format!("Diagnostics ({})", held + dropped) };
                        if nav_link(ui, &diag, matches!(self.screen, Screen::Diagnostics)).clicked() {
                            self.screen = Screen::Diagnostics;
                        }
                        if nav_link(ui, "Settings", matches!(self.screen, Screen::Settings(None))).clicked() {
                            self.open_draft(None);
                        }
                        if nav_link(ui, "Add games", matches!(self.screen, Screen::Import)).clicked() {
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
                .frame(egui::Frame::new().fill(Color32::from_rgb(0x1d, 0x10, 0x15)).stroke(egui::Stroke::new(1.0, DANGER_EDGE)).inner_margin(egui::Margin::symmetric(16, 8)))
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
                if let Some(l) = self.loading.as_ref() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(120.0);
                        ui.heading(&l.title_id);
                        ui.label(RichText::new(format!("preparing the title ({:.0} s)...", l.started.elapsed().as_secs_f32())).color(DIM));
                        ui.spinner();
                    });
                    return;
                }
                match std::mem::replace(&mut self.screen, Screen::Library) {
                    Screen::Library => {
                        self.screen = Screen::Library;
                        self.ui_library(ui);
                    }
                    Screen::Title(id) => {
                        self.screen = Screen::Title(id.clone());
                        self.ui_title(ui, &id);
                    }
                    Screen::Settings(id) => {
                        self.screen = Screen::Settings(id);
                        self.ui_settings(ui);
                    }
                    Screen::Import => {
                        self.screen = Screen::Import;
                        self.ui_import(ui);
                    }
                    Screen::Diagnostics => {
                        self.screen = Screen::Diagnostics;
                        self.ui_diagnostics(ui);
                    }
                }
            });
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
                            self.confirm_remove = Some(id.to_string());
                        }
                    });
                });
            });
        });
        ui.painter().set(bg, hero_backdrop(hero.response.rect, pic.as_ref()));
        ui.add_space(4.0);
        let eff = library::effective(Some(id));
        let saves = library::saves_dir(&eff.profile);
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(bold("Saved data").size(16.0));
                ui.label(RichText::new(format!("profile: {}", eff.profile)).color(DIM));
            });
            ui.label(RichText::new(format!("Kept under {}", saves.display())).color(DIM));
            ui.horizontal(|ui| {
                if ui.button("Open the saves folder").clicked() {
                    let _ = std::fs::create_dir_all(&saves);
                    open_path(&saves);
                }
                if ui.button("Open the game folder").clicked() {
                    open_path(&library::title_dir(id));
                }
            });
            ui.label(RichText::new("What the game saved - its save files and trophies - and nothing of the game itself.").color(DIM));
        });
        if let Some(rid) = self.confirm_remove.clone() {
            egui::Window::new("Remove this title?").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.label(format!("{} ({}) will be deleted from the library. Saved data is kept.", meta.title, rid));
                ui.horizontal(|ui| {
                    if ui.add(danger("Remove")).clicked() {
                        if let Err(e) = library::remove_title(&rid) {
                            self.error = Some(e.to_string());
                        }
                        self.icons.remove(&rid);
                        self.pictures.remove(&rid);
                        self.titles = library::list_titles();
                        self.confirm_remove = None;
                        self.screen = Screen::Library;
                    }
                    if ui.button("Keep").clicked() {
                        self.confirm_remove = None;
                    }
                });
            });
        }
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui) {
        // A pressed key lands in the capturing button.
        let key = self.last_key.take();
        let Some(mut d) = self.draft.take() else { return };
        if let (Some(b), Some(k)) = (d.capturing, key.as_ref()) {
            if k != "Escape" {
                d.s.keyboard.insert(b.name().to_string(), k.clone());
            }
            d.capturing = None;
        }
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
        ui.label(bold("Settings").size(22.0).color(INK));
        if d.title_id.is_some() {
            ui.label(RichText::new("Only what you change here is kept for this title; everything else follows the global settings, including later changes to them.").color(DIM));
        }
        let mut save = false;
        let mut use_global = false;
        // The browser's sticky Save bar: the cards scroll above it, the bar stays.
        let bar = 52.0;
        egui::ScrollArea::vertical().max_height((ui.available_height() - bar).max(80.0)).show(ui, |ui| {
            card(ui, |ui| {
                ui.label(bold("General").size(16.0));
                check_row(ui, &mut d.s.pause_on_blur, "Pause when the window loses focus", Some("Not the game's pause menu: the emulator stops, as a console in a pocket would."));
                ui.separator();
                check_row(ui, &mut d.s.show_fps, "Show the frame rate over the game", None);
                ui.separator();
                check_row(ui, &mut d.s.fps_in_title, "Show the frame rate in the window title", None);
                ui.separator();
                setting_row(ui, "Scaling", None, |ui| {
                    let r = egui::ComboBox::from_id_salt("scaling").width(150.0).selected_text(scaling_label(d.s.scaling)).show_ui(ui, |ui| {
                        for s in [Scaling::Fit, Scaling::Integer, Scaling::Stretch] {
                            ui.selectable_value(&mut d.s.scaling, s, scaling_label(s));
                        }
                    });
                    combo_nav(&r.response, &mut self.nav, &mut d.s.scaling, &[Scaling::Fit, Scaling::Integer, Scaling::Stretch]);
                });
                ui.separator();
                setting_row(ui, "Save profile", Some("Each profile keeps its own saved data for every game."), |ui| {
                    ui.add(egui::TextEdit::singleline(&mut d.s.profile).desired_width(150.0));
                });
            });
            card(ui, |ui| {
                ui.label(bold("Keyboard").size(16.0));
                ui.label(RichText::new("Pick a control, then press the key for it.").color(DIM));
                egui::Grid::new("kb").num_columns(4).spacing([16.0, 6.0]).show(ui, |ui| {
                    for (i, b) in Button::ALL.iter().enumerate() {
                        ui.label(b.label());
                        let listening = d.capturing == Some(*b);
                        let text = if listening { "press a key...".to_string() } else { d.s.keyboard.get(b.name()).cloned().unwrap_or_else(|| "-".into()) };
                        let mut key_button = egui::Button::new(RichText::new(text).monospace().color(if listening { ACCENT } else { INK })).min_size(egui::vec2(120.0, 0.0));
                        if listening {
                            key_button = key_button.stroke(egui::Stroke::new(1.0, ACCENT));
                        }
                        if ui.add(key_button).clicked() {
                            d.capturing = Some(*b);
                        }
                        if i % 2 == 1 {
                            ui.end_row();
                        }
                    }
                });
                if ui.small_button("Reset keyboard").clicked() {
                    d.s.keyboard = vitaslop_frontend::input::default_keyboard();
                }
            });
            card(ui, |ui| {
                ui.label(bold("Gamepad").size(16.0));
                ui.label(RichText::new("Which control of a standard pad presses each Vita button. In the menus the pad moves with the d-pad or stick, picks with south and backs out with east; home, or start and select held together, opens the in-game menu.").color(DIM));
                egui::Grid::new("gp").num_columns(4).spacing([16.0, 6.0]).show(ui, |ui| {
                    for (i, b) in Button::ALL.iter().enumerate() {
                        ui.label(b.label());
                        let cur = d.s.gamepad.get(b.name()).cloned().unwrap_or_default();
                        let mut pick: Option<&str> = None;
                        let r = egui::ComboBox::from_id_salt(format!("gp-{}", b.name())).width(120.0).selected_text(&cur).show_ui(ui, |ui| {
                            for c in GAMEPAD_CONTROLS {
                                if ui.selectable_label(cur == c, c).clicked() {
                                    pick = Some(c);
                                }
                            }
                        });
                        let mut step = GAMEPAD_CONTROLS.iter().copied().find(|c| *c == cur).unwrap_or("");
                        if combo_nav(&r.response, &mut self.nav, &mut step, &GAMEPAD_CONTROLS) {
                            pick = Some(step);
                        }
                        if let Some(c) = pick {
                            d.s.gamepad.insert(b.name().to_string(), c.to_string());
                        }
                        if i % 2 == 1 {
                            ui.end_row();
                        }
                    }
                });
                ui.separator();
                setting_row(ui, "Stick dead zone", Some("How far a stick must move before the game sees it."), |ui| {
                    ui.add(egui::Slider::new(&mut d.s.stick_deadzone, 0.0..=0.5));
                });
            });
            card(ui, |ui| {
                ui.collapsing(bold("Advanced").size(16.0), |ui| {
                    ui.label("Knobs, one VITASLOP_NAME=value per line:");
                    ui.add(egui::TextEdit::multiline(&mut d.knobs_text).code_editor().desired_rows(4).desired_width(f32::INFINITY));
                    ui.label("Recipe (scripted input, replayed from the first frame):");
                    ui.add(egui::TextEdit::multiline(&mut d.s.recipe).code_editor().desired_rows(3).desired_width(f32::INFINITY));
                    ui.checkbox(&mut d.s.debug_capture, "Capture debug timings (roughly doubles the frame cost)");
                    let _ = PadMode::Auto;
                });
            });
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
                d.s = Settings::default();
                d.knobs_text.clear();
            }
            if let Some(t) = d.saved_at
                && t.elapsed().as_secs_f32() < 2.0 {
                    ui.label(RichText::new("Saved").color(ACCENT));
                }
        });
        let id = d.title_id.clone();
        self.draft = Some(d);
        if save {
            self.save_draft();
        }
        if use_global
            && let Some(id) = id {
                let _ = library::save_title_patch(&id, None);
                self.open_draft(Some(id));
            }
    }

    fn ui_import(&mut self, ui: &mut egui::Ui) {
        ui.label(bold("Add games").size(22.0).color(INK));
        ui.label("vitaslop only works with games you own, from your own console. It provides no games and downloads nothing.");
        ui.add_space(4.0);
        let busy = self.import.as_ref().map(|p| !p.lock().unwrap().finished).unwrap_or(false);
        card(ui, |ui| {
            ui.label(RichText::new("A .pkg with its work.bin licence beside it, a folder dumped from a console (with sce_pfs and sce_sys inside), a zip of either, or a homebrew .vpk.").color(DIM));
            ui.separator();
            setting_row(ui, "A dumped folder, or one holding a .pkg and its work.bin", None, |ui| {
                if ui.add_enabled(!busy, egui::Button::new(bold("Choose a folder"))).clicked()
                    && let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.start_import(p);
                    }
            });
            ui.separator();
            setting_row(ui, "A .pkg (its work.bin beside it), a .zip, or a .vpk", None, |ui| {
                if ui.add_enabled(!busy, egui::Button::new(bold("Choose a file"))).clicked()
                    && let Some(p) = rfd::FileDialog::new().add_filter("Vita package, zip or homebrew vpk", &["pkg", "zip", "vpk", "PKG", "ZIP", "VPK"]).pick_file() {
                        self.start_import(p);
                    }
            });
        });
        // The browser's drop zone - the window takes a dropped file or folder on any screen.
        let (r, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 72.0), egui::Sense::hover());
        dashed_rect(ui.painter(), r.shrink(1.0), egui::Stroke::new(2.0, EDGE));
        ui.painter().text(r.center(), egui::Align2::CENTER_CENTER, "Or drop files or a folder anywhere here.", egui::FontId::proportional(14.0), DIM);
        if let Some(p) = self.import.clone() {
            let g = p.lock().unwrap().clone();
            ui.add_space(12.0);
            if !g.finished {
                let frac = if g.total > 0 { g.done as f32 / g.total as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(frac).show_percentage());
                // The ingest's stage names (copy, decrypt, unwrap) are for whoever reads
                // the code; what a person watching an import needs is that it is working.
                ui.label(RichText::new(format!("loading {} / {} - {}", fmt_bytes(g.done), fmt_bytes(g.total), g.file)).color(DIM).small());
            } else if let Some(e) = g.error {
                ui.label(RichText::new(format!("Import failed: {e}")).color(DANGER));
            } else if let Some(id) = g.title_id {
                self.import = None;
                self.titles = library::list_titles();
                self.icons.remove(&id);
                self.pictures.remove(&id);
                self.screen = Screen::Title(id);
            }
        }
    }

    /// Everything this run reported, in the two channels it reported them on.
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
            ui.heading("Diagnostics");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Save report").clicked() {
                    let name = format!("vitaslop-diagnostics-{}.txt", std::process::id());
                    if let Some(p) = rfd::FileDialog::new().set_file_name(&name).save_file() {
                        match std::fs::write(&p, crate::log::snapshot()) {
                            Ok(()) => self.import_msg = Some(Ok(format!("saved {}", p.display()))),
                            Err(e) => self.error = Some(format!("could not save the report: {e}")),
                        }
                    }
                }
                if ui.button("Copy").clicked() {
                    ui.ctx().copy_text(crate::log::snapshot());
                }
            });
        });
        ui.label(
            RichText::new(
                "Warnings are the emulator saying it did something the console would not have done. A title can look and play perfectly with warnings here.",
            )
            .color(DIM)
            .small(),
        );
        ui.separator();
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.label(RichText::new(format!(
                "{held} distinct warning(s), {total} in total{}",
                if dropped > 0 { format!(", {dropped} older one(s) dropped") } else { String::new() }
            )).family(egui::FontFamily::Name("bold".into())));
            match report(Channel::Warning) {
                Some(text) => ui.label(RichText::new(text).monospace().small()),
                None => ui.label(RichText::new("Nothing was reported.").color(DIM)),
            };
            ui.add_space(12.0);
            ui.label(bold("Status"));
            ui.label(
                RichText::new("What the run did rather than what went wrong: the adapter, the archive, the shapes it saw.")
                    .color(DIM)
                    .small(),
            );
            match report(Channel::Status) {
                Some(text) => ui.label(RichText::new(text).monospace().small()),
                None => ui.label(RichText::new("Nothing yet.").color(DIM)),
            };
        });
    }

    fn ui_playing(&mut self, ctx: &egui::Context) {
        if self.session_settings.show_fps
            && let Some(st) = self.stats.as_ref() {
                egui::Area::new(egui::Id::new("fps")).fixed_pos([8.0, 8.0]).show(ctx, |ui| {
                    egui::Frame::new().fill(Color32::from_black_alpha(140)).inner_margin(4.0).show(ui, |ui| {
                        ui.label(RichText::new(format!("{:.0} fps  {:.0}%", st.fps, st.speed_pct)).monospace().color(ACCENT));
                    });
                });
            }
        if !self.menu_open {
            return;
        }
        let mut close = false;
        let mut quit = false;
        // Held across the window closure so the outcome can be shown in the same menu.
        let mut diag_saved = self.diag_saved.clone();
        egui::Window::new("vitaslop").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            if let Some(st) = self.stats.as_ref() {
                ui.label(RichText::new(st.title_line()).color(DIM).small());
            }
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(bold(" Resume ")).fill(ACCENT.linear_multiply(0.25))).clicked() {
                    close = true;
                }
                if ui.add(egui::Button::new(RichText::new("Quit to library").color(DANGER))).clicked() {
                    quit = true;
                }
            });
            ui.separator();
            let mut changed = false;
            changed |= ui.checkbox(&mut self.session_settings.show_fps, "Show frame rate").changed();
            changed |= ui.checkbox(&mut self.session_settings.fps_in_title, "Frame rate in the window title").changed();
            changed |= ui.checkbox(&mut self.session_settings.pause_on_blur, "Pause when the window loses focus").changed();
            match self.session.as_ref().and_then(|s| s.audio.as_ref()) {
                Some(a) => {
                    let mut muted = a.muted();
                    if ui.checkbox(&mut muted, "Mute").changed() {
                        a.set_muted(muted);
                        let _ = library::save_muted(muted);
                    }
                }
                None => {
                    ui.label(RichText::new("No audio output device - the title runs silent").color(DIM).small());
                }
            }
            if changed {
                if let Some(s) = self.session.as_mut() {
                    s.pause_on_blur = self.session_settings.pause_on_blur;
                }
                // A change made while playing is a global preference.
                let mut g = library::effective(None);
                g.show_fps = self.session_settings.show_fps;
                g.fps_in_title = self.session_settings.fps_in_title;
                g.pause_on_blur = self.session_settings.pause_on_blur;
                let _ = library::save_global_settings(&g);
                if !self.session_settings.fps_in_title
                    && let Some(w) = self.window.as_ref() {
                        w.set_title("vitaslop");
                    }
            }
            ui.separator();
            let (held, _, dropped) = vitaslop_platform::diag::counts(vitaslop_platform::diag::Channel::Warning);
            if ui.button(format!("Save a diagnostics report ({} warning(s))", held + dropped)).clicked() {
                let name = format!("vitaslop-diagnostics-{}.txt", std::process::id());
                if let Some(p) = rfd::FileDialog::new().set_file_name(&name).save_file() {
                    match std::fs::write(&p, crate::log::snapshot()) {
                        Ok(()) => diag_saved = Some(format!("saved {}", p.display())),
                        Err(e) => diag_saved = Some(format!("could not save the report: {e}")),
                    }
                }
            }
            if let Some(m) = diag_saved.as_ref() {
                ui.label(RichText::new(m).color(DIM).small());
            }
            ui.label(RichText::new("Esc (or the pad's east button) closes this menu; home, or start+select held, opens it. F11 toggles fullscreen.").color(DIM).small());
        });
        self.diag_saved = diag_saved;
        if close {
            self.toggle_menu();
        }
        if quit {
            self.stop();
            self.screen = Screen::Library;
            self.titles = library::list_titles();
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
fn bold(text: impl Into<String>) -> RichText {
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
fn primary(text: &str) -> egui::Button<'static> {
    egui::Button::new(bold(text).color(ACCENT_INK)).fill(ACCENT).stroke(egui::Stroke::NONE)
}

/// The browser's `.btn.danger`.
fn danger(text: &str) -> egui::Button<'static> {
    egui::Button::new(bold(text).color(DANGER)).stroke(egui::Stroke::new(1.0, DANGER_EDGE))
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

/// The browser's `.card`: a panel-filled, edged, 12 px-rounded box the width of the page.
fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
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
                        to = Some(match sc {
                            Screen::Library => Screen::Library,
                            Screen::Title(id) => Screen::Title(id.clone()),
                            Screen::Settings(id) => Screen::Settings(id.clone()),
                            Screen::Import => Screen::Import,
                            Screen::Diagnostics => Screen::Diagnostics,
                        });
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

/// One settings row: the label (and its hint under it) at the left, the control at the right.
fn setting_row(ui: &mut egui::Ui, label: &str, hint: Option<&str>, control: impl FnOnce(&mut egui::Ui)) {
    ui.horizontal(|ui| {
        ui.vertical(|ui| {
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.label(RichText::new(label).color(INK));
            if let Some(h) = hint {
                ui.label(RichText::new(h).color(DIM).size(12.0));
            }
        });
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), control);
    });
}

/// A checkbox row: the box, its label, and the hint under the label.
fn check_row(ui: &mut egui::Ui, value: &mut bool, label: &str, hint: Option<&str>) {
    const BOX: f32 = 20.0;
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 16.0;
        let (rect, mut r) = ui.allocate_exact_size(egui::vec2(BOX, BOX), egui::Sense::click());
        if r.clicked() {
            *value = !*value;
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
            if ui.add(egui::Label::new(RichText::new(label).color(INK).size(15.0)).sense(egui::Sense::click())).clicked() {
                *value = !*value;
            }
            if let Some(h) = hint {
                ui.label(RichText::new(h).color(DIM).size(12.0));
            }
        });
    });
}

fn scaling_label(s: Scaling) -> &'static str {
    match s {
        Scaling::Fit => "Fit",
        Scaling::Integer => "Integer",
        Scaling::Stretch => "Stretch",
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

/// The browser's `.drop` box: a dashed, rounded-looking outline.
fn dashed_rect(painter: &egui::Painter, r: egui::Rect, stroke: egui::Stroke) {
    let pts = [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom(), r.left_top()];
    painter.extend(egui::Shape::dashed_line(&pts, stroke, 6.0, 4.0));
}

/// A Unix-milliseconds date as the title page shows it, `2026-10-03`, or `never`.
fn fmt_date(ms: u64) -> String {
    if ms == 0 {
        return "never".into();
    }
    // Days since 1970-01-01 to a civil date (Howard Hinnant's algorithm).
    let z = (ms / 86_400_000) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
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

fn fmt_bytes(n: u64) -> String {
    if n < 1_000_000 {
        format!("{} KB", n / 1000)
    } else if n < 1_000_000_000 {
        format!("{} MB", n / 1_000_000)
    } else {
        format!("{:.2} GB", n as f64 / 1e9)
    }
}

fn open_path(p: &std::path::Path) {
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(p).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(p).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(p).spawn();
}

fn decode_png(bytes: &[u8]) -> Option<egui::ColorImage> {
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
}
