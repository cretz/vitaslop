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
use std::time::Instant;

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
use crate::retail::{DesktopInput, RetailGfx, RetailGuest, SharedInput};
use crate::session::{Session, Stats};

const ACCENT: Color32 = Color32::from_rgb(0x8f, 0xe0, 0xa0);
const DIM: Color32 = Color32::from_rgb(0x8f, 0x8f, 0xa3);
const DANGER: Color32 = Color32::from_rgb(0xe8, 0xa0, 0xa8);

enum Screen {
    Library,
    Title(String),
    Settings(Option<String>),
    Import,
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
    search: String,
    draft: Option<Draft>,
    loading: Option<Loading>,
    session: Option<Session>,
    session_settings: Settings,
    menu_open: bool,
    stats: Option<Stats>,
    import: Option<Arc<Mutex<ImportProgress>>>,
    import_msg: Option<Result<String, String>>,
    confirm_remove: Option<String>,
    error: Option<String>,
    last_key: Option<String>,
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
        search: String::new(),
        draft: None,
        loading: None,
        session: None,
        session_settings: Settings::default(),
        menu_open: false,
        stats: None,
        import: None,
        import_msg: None,
        confirm_remove: None,
        error: None,
        last_key: None,
    };
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = Color32::from_rgb(0x0b, 0x0b, 0x12);
    visuals.window_fill = Color32::from_rgb(0x14, 0x14, 0x1f);
    visuals.selection.bg_fill = ACCENT.linear_multiply(0.35);
    app.egui.set_visuals(visuals);
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
        if ui_wants {
            if let Some(st) = self.egui_state.as_mut() {
                consumed = st.on_window_event(&window, &event).consumed;
            }
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

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = self.window.as_ref() {
            w.request_redraw();
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
            }
        }
    }

    fn frame(&mut self, size: (f64, f64), window: &Arc<Window>) {
        self.poll_loading();
        // The game, if one runs.
        if let Some(s) = self.session.as_mut() {
            s.tick(Some(size));
            if let Some(st) = s.stats(Instant::now()) {
                if self.session_settings.fps_in_title {
                    window.set_title(&format!("vitaslop  |  {}", st.title_line()));
                }
                self.stats = Some(st);
            }
        }
        if self.gfx.is_none() || self.renderer.is_none() {
            return;
        }
        let Some(st) = self.egui_state.as_mut() else { return };

        // egui runs on every frame (it needs to, to repaint the badge), but its input
        // is taken only when it owns the screen; otherwise it gets an empty frame.
        let raw = st.take_egui_input(window);
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
        let scenes = self.session.as_mut().map(|s| s.scenes());
        let scenes = scenes.filter(|(sc, _, _)| !sc.is_empty());
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
            search: String::new(),
            draft: None,
            loading: None,
            session: None,
            session_settings: Settings::default(),
            menu_open: false,
            stats: None,
            import: None,
            import_msg: None,
            confirm_remove: None,
            error: None,
            last_key: None,
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
                self.session = Some(Session::new(guest, l.input, input, l.settings.pause_on_blur));
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
            let r = RetailGuest::new(&dir, input2, recipe.as_deref());
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
        if self.session.is_some() {
            self.ui_playing(&ctx);
            return;
        }
        egui::Panel::top("top").show(root, |ui| {
            ui.horizontal(|ui| {
                ui.add_space(6.0);
                if ui.add(egui::Label::new(RichText::new("vitaslop").strong().color(ACCENT)).sense(egui::Sense::click())).clicked() {
                    self.screen = Screen::Library;
                }
                ui.add_space(12.0);
                if ui.button("Library").clicked() {
                    self.titles = library::list_titles();
                    self.screen = Screen::Library;
                }
                if ui.button("Add games").clicked() {
                    self.screen = Screen::Import;
                }
                if ui.button("Settings").clicked() {
                    self.open_draft(None);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(library::home().display().to_string()).color(DIM).small());
                });
            });
        });
        if let Some(e) = self.error.clone() {
            egui::Panel::top("error").show(root, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&e).color(DANGER));
                    if ui.small_button("dismiss").clicked() {
                        self.error = None;
                    }
                });
            });
        }
        egui::CentralPanel::default().show(root, |ui| {
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
            }
        });
    }

    fn ui_library(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.search).hint_text(format!("Search {} titles", self.titles.len())).desired_width(300.0));
            if ui.button("Add games").clicked() {
                self.screen = Screen::Import;
            }
        });
        ui.add_space(8.0);
        if self.titles.is_empty() {
            ui.vertical_centered(|ui| {
                ui.add_space(80.0);
                ui.label(RichText::new("No titles yet.").color(DIM));
                if ui.button("Add a game").clicked() {
                    self.screen = Screen::Import;
                }
            });
            return;
        }
        let q = self.search.trim().to_lowercase();
        let list: Vec<TitleMeta> = self.titles.iter().filter(|t| q.is_empty() || t.search_key().contains(&q)).cloned().collect();
        let tile = 120.0;
        let cols = ((ui.available_width() / (tile + 12.0)).floor() as usize).max(1);
        let ctx = ui.ctx().clone();
        let mut open: Option<String> = None;
        egui::ScrollArea::vertical().show(ui, |ui| {
            for row in list.chunks(cols) {
                ui.horizontal_top(|ui| {
                    for t in row {
                        let icon = self.icon(&ctx, &t.title_id);
                        ui.allocate_ui(egui::vec2(tile, tile + 44.0), |ui| {
                            ui.vertical_centered(|ui| {
                                let r = match icon {
                                    Some(tex) => ui.add(egui::Button::image(egui::Image::new((tex.id(), egui::vec2(tile - 16.0, tile - 16.0))).corner_radius((tile - 16.0) / 2.0)).frame(false)),
                                    None => ui.add_sized([tile - 16.0, tile - 16.0], egui::Button::new(RichText::new(&t.title_id).small())),
                                };
                                if r.clicked() {
                                    open = Some(t.title_id.clone());
                                }
                                let name = if t.title.chars().count() > 30 { format!("{}...", t.title.chars().take(28).collect::<String>()) } else { t.title.clone() };
                                ui.label(RichText::new(name).small());
                                ui.label(RichText::new(&t.title_id).color(DIM).small());
                            });
                        });
                    }
                });
            }
        });
        if let Some(id) = open {
            self.screen = Screen::Title(id);
        }
    }

    fn ui_title(&mut self, ui: &mut egui::Ui, id: &str) {
        let Some(meta) = self.titles.iter().find(|t| t.title_id == id).cloned() else {
            ui.label("this title is not in the library");
            return;
        };
        let ctx = ui.ctx().clone();
        let icon = self.icon(&ctx, id);
        ui.horizontal(|ui| {
            if let Some(tex) = icon {
                ui.add(egui::Image::new((tex.id(), egui::vec2(128.0, 128.0))).corner_radius(64.0));
            }
            ui.vertical(|ui| {
                ui.heading(&meta.title);
                let played = if meta.last_played_at > 0 { "played before" } else { "never played" };
                ui.label(RichText::new(format!("{}  |  v{}  |  {}  |  {}", meta.title_id, meta.app_version, fmt_bytes(meta.bytes), played)).color(DIM));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(RichText::new("  Play  ").strong().size(18.0)).fill(ACCENT.linear_multiply(0.25))).clicked() {
                        self.start(id);
                    }
                    if ui.button("Settings for this title").clicked() {
                        self.open_draft(Some(id.to_string()));
                    }
                    if ui.add(egui::Button::new(RichText::new("Remove").color(DANGER))).clicked() {
                        self.confirm_remove = Some(id.to_string());
                    }
                });
            });
        });
        ui.add_space(16.0);
        ui.separator();
        let eff = library::effective(Some(id));
        let saves = library::saves_dir(&eff.profile);
        ui.label(RichText::new(format!("Saved data (profile: {})", eff.profile)).strong());
        ui.label(RichText::new(format!("kept under {}", saves.display())).color(DIM).small());
        ui.horizontal(|ui| {
            if ui.button("Open the saves folder").clicked() {
                let _ = std::fs::create_dir_all(&saves);
                open_path(&saves);
            }
            if ui.button("Open the game folder").clicked() {
                open_path(&library::title_dir(id));
            }
        });
        if let Some(rid) = self.confirm_remove.clone() {
            egui::Window::new("Remove this title?").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(&ctx, |ui| {
                ui.label(format!("{} ({}) will be deleted from the library. Saved data is kept.", meta.title, rid));
                ui.horizontal(|ui| {
                    if ui.add(egui::Button::new(RichText::new("Remove").color(DANGER))).clicked() {
                        if let Err(e) = library::remove_title(&rid) {
                            self.error = Some(e.to_string());
                        }
                        self.icons.remove(&rid);
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
        let title = match &d.title_id {
            Some(id) => format!("Settings for {id}"),
            None => "Settings".to_string(),
        };
        ui.heading(title);
        if d.title_id.is_some() {
            ui.label(RichText::new("Only what you change here differs from the global settings.").color(DIM));
        }
        let mut save = false;
        let mut use_global = false;
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.add_space(6.0);
            ui.label(RichText::new("General").strong());
            ui.checkbox(&mut d.s.pause_on_blur, "Pause when the window loses focus");
            ui.checkbox(&mut d.s.show_fps, "Show the frame rate over the game");
            ui.checkbox(&mut d.s.fps_in_title, "Show the frame rate in the window title");
            ui.horizontal(|ui| {
                ui.label("Scaling");
                egui::ComboBox::from_id_salt("scaling").selected_text(format!("{:?}", d.s.scaling)).show_ui(ui, |ui| {
                    ui.selectable_value(&mut d.s.scaling, Scaling::Fit, "Fit");
                    ui.selectable_value(&mut d.s.scaling, Scaling::Integer, "Integer");
                    ui.selectable_value(&mut d.s.scaling, Scaling::Stretch, "Stretch");
                });
            });
            ui.horizontal(|ui| {
                ui.label("Save profile");
                ui.text_edit_singleline(&mut d.s.profile);
            });
            ui.add_space(10.0);
            ui.label(RichText::new("Keyboard").strong());
            ui.label(RichText::new("Click a control, then press the key for it.").color(DIM).small());
            egui::Grid::new("kb").num_columns(4).spacing([16.0, 4.0]).show(ui, |ui| {
                for (i, b) in Button::ALL.iter().enumerate() {
                    ui.label(b.label());
                    let text = if d.capturing == Some(*b) { "press a key...".to_string() } else { d.s.keyboard.get(b.name()).cloned().unwrap_or_else(|| "-".into()) };
                    if ui.add(egui::Button::new(RichText::new(text).monospace()).min_size(egui::vec2(120.0, 0.0))).clicked() {
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
            ui.add_space(10.0);
            ui.label(RichText::new("Gamepad").strong());
            egui::Grid::new("gp").num_columns(4).spacing([16.0, 4.0]).show(ui, |ui| {
                for (i, b) in Button::ALL.iter().enumerate() {
                    ui.label(b.label());
                    let cur = d.s.gamepad.get(b.name()).cloned().unwrap_or_default();
                    let mut pick: Option<&str> = None;
                    egui::ComboBox::from_id_salt(format!("gp-{}", b.name())).selected_text(&cur).show_ui(ui, |ui| {
                        for c in GAMEPAD_CONTROLS {
                            if ui.selectable_label(cur == c, c).clicked() {
                                pick = Some(c);
                            }
                        }
                    });
                    if let Some(c) = pick {
                        d.s.gamepad.insert(b.name().to_string(), c.to_string());
                    }
                    if i % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label("Stick dead zone");
                ui.add(egui::Slider::new(&mut d.s.stick_deadzone, 0.0..=0.5));
            });
            ui.add_space(10.0);
            ui.collapsing(RichText::new("Advanced").strong(), |ui| {
                ui.label("Knobs, one VITASLOP_NAME=value per line:");
                ui.add(egui::TextEdit::multiline(&mut d.knobs_text).code_editor().desired_rows(4).desired_width(f32::INFINITY));
                ui.label("Recipe (scripted input, replayed from the first frame):");
                ui.add(egui::TextEdit::multiline(&mut d.s.recipe).code_editor().desired_rows(3).desired_width(f32::INFINITY));
                ui.checkbox(&mut d.s.debug_capture, "Capture debug timings (roughly doubles the frame cost)");
                let _ = PadMode::Auto;
            });
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(RichText::new(" Save ").strong()).fill(ACCENT.linear_multiply(0.25))).clicked() {
                    save = true;
                }
                if d.title_id.is_some() {
                    if ui.button("Use global settings").clicked() {
                        use_global = true;
                    }
                } else if ui.button("Reset to defaults").clicked() {
                    d.s = Settings::default();
                    d.knobs_text.clear();
                }
                if let Some(t) = d.saved_at {
                    if t.elapsed().as_secs_f32() < 2.0 {
                        ui.label(RichText::new("saved").color(ACCENT));
                    }
                }
            });
        });
        let id = d.title_id.clone();
        self.draft = Some(d);
        if save {
            self.save_draft();
        }
        if use_global {
            if let Some(id) = id {
                let _ = library::save_title_patch(&id, None);
                self.open_draft(Some(id));
            }
        }
    }

    fn ui_import(&mut self, ui: &mut egui::Ui) {
        ui.heading("Add games");
        ui.label(RichText::new("A .pkg with its work.bin beside it, a folder dumped from a console (with sce_pfs and sce_sys inside), a zip of either, or a homebrew .vpk.").color(DIM));
        ui.add_space(8.0);
        let busy = self.import.as_ref().map(|p| !p.lock().unwrap().finished).unwrap_or(false);
        ui.horizontal(|ui| {
            if ui.add_enabled(!busy, egui::Button::new("Pick a folder")).clicked() {
                if let Some(p) = rfd::FileDialog::new().pick_folder() {
                    self.start_import(p);
                }
            }
            if ui.add_enabled(!busy, egui::Button::new("Pick a .pkg, .zip or .vpk")).clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Vita package, zip or homebrew vpk", &["pkg", "zip", "vpk", "PKG", "ZIP", "VPK"]).pick_file() {
                    self.start_import(p);
                }
            }
        });
        if let Some(p) = self.import.clone() {
            let g = p.lock().unwrap().clone();
            ui.add_space(12.0);
            if !g.finished {
                let frac = if g.total > 0 { g.done as f32 / g.total as f32 } else { 0.0 };
                ui.add(egui::ProgressBar::new(frac).show_percentage());
                ui.label(RichText::new(format!("{} {} / {} - {}", g.stage, fmt_bytes(g.done), fmt_bytes(g.total), g.file)).color(DIM).small());
            } else if let Some(e) = g.error {
                ui.label(RichText::new(format!("Import failed: {e}")).color(DANGER));
            } else if let Some(id) = g.title_id {
                self.import = None;
                self.titles = library::list_titles();
                self.icons.remove(&id);
                self.screen = Screen::Title(id);
            }
        }
    }

    fn ui_playing(&mut self, ctx: &egui::Context) {
        if self.session_settings.show_fps {
            if let Some(st) = self.stats.as_ref() {
                egui::Area::new(egui::Id::new("fps")).fixed_pos([8.0, 8.0]).show(ctx, |ui| {
                    egui::Frame::new().fill(Color32::from_black_alpha(140)).inner_margin(4.0).show(ui, |ui| {
                        ui.label(RichText::new(format!("{:.0} fps  {:.0}%", st.fps, st.speed_pct)).monospace().color(ACCENT));
                    });
                });
            }
        }
        if !self.menu_open {
            return;
        }
        let mut close = false;
        let mut quit = false;
        egui::Window::new("vitaslop").collapsible(false).resizable(false).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).show(ctx, |ui| {
            if let Some(st) = self.stats.as_ref() {
                ui.label(RichText::new(st.title_line()).color(DIM).small());
            }
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new(RichText::new(" Resume ").strong()).fill(ACCENT.linear_multiply(0.25))).clicked() {
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
                if !self.session_settings.fps_in_title {
                    if let Some(w) = self.window.as_ref() {
                        w.set_title("vitaslop");
                    }
                }
            }
            ui.label(RichText::new("Esc closes this menu. F11 toggles fullscreen.").color(DIM).small());
        });
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
