//! The play screen's own UI - the browser's `#player` (`index.html`, `player.js`): the loading
//! panel, the frame-rate badge, the mute and menu buttons over the game, the "paused" shade,
//! the in-game menu with every option the browser's has, and the FATAL panel.
//!
//! The browser's on-screen touch controls (placement, opacity, size, vibration) have no desktop
//! counterpart - nothing here is a touch screen - so the menu leaves them out; everything else
//! is there, in the browser's order, plus the window title's frame rate, which only a window has.

use egui::{Color32, Pos2, Rect, RichText, Stroke, vec2};

use super::{
    ACCENT, DANGER, DIM, EDGE, INK, MenuAsk, Shell, bold, check_row, danger, panel_frame, primary, seg, touchpad,
};
use crate::bindings::Kind;
use crate::library;
use crate::vitapic;

/// `#menu .menu-in`: the panel is see-through, so the paused frame stays visible behind it.
const MENU_FILL: Color32 = Color32::from_rgba_premultiplied(16, 16, 25, 209);

impl Shell {
    pub(super) fn ui_player(&mut self, root: &mut egui::Ui) {
        let ctx = &root.ctx().clone();
        let name = self
            .loading
            .as_ref()
            .map(|l| l.title.clone())
            .or_else(|| self.playing.as_ref().map(|id| self.titles.iter().find(|t| &t.title_id == id).map_or(id.clone(), |t| t.title.clone())))
            .unwrap_or_default();
        let title = match self.stats.as_ref().filter(|_| self.session_settings.fps_in_title && self.session.is_some()) {
            Some(st) => format!("{name} - vitaslop  |  {}", st.title_line()),
            None => format!("{name} - vitaslop"),
        };
        self.set_window_title(&title);
        if self.fatal.is_some() {
            self.ui_fatal(root);
            return;
        }
        if let Some(l) = self.loading.as_ref() {
            let (title, secs) = (l.title.clone(), l.started.elapsed().as_secs_f32());
            egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::from_rgb(5, 5, 10))).show(root, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space((ui.available_height() / 2.0 - 70.0).max(0.0));
                    ui.label(bold(&title).size(20.0).color(INK));
                    ui.add_space(4.0);
                    ui.label(RichText::new(format!("preparing the title ({secs:.0} s - a few seconds, longer the first time)...")).color(DIM));
                    ui.add_space(14.0);
                    ui.add(egui::Spinner::new().size(28.0).color(ACCENT));
                });
            });
            ctx.request_repaint();
            return;
        }
        self.ui_hud(ctx);
        if self.menu_open {
            self.ui_menu(ctx);
        }
        self.ui_capture(ctx);
    }

    /// The badge, the paused shade, and the two buttons that sit over the game.
    fn ui_hud(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        self.ui_touchpad(ctx);
        if self.session_settings.show_fps
            && let Some(st) = self.stats.as_ref()
        {
            // `#fpsbadge`: top left, out of the pointer's way.
            egui::Area::new(egui::Id::new("fps")).fixed_pos([8.0, 6.0]).interactable(false).show(ctx, |ui| {
                // One line: an Area remembers its first width, and the badge's first text is
                // shorter than the steady one - it wrapped at every space without this.
                ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                egui::Frame::new().fill(Color32::from_black_alpha(140)).corner_radius(4).inner_margin(egui::Margin::symmetric(6, 4)).show(ui, |ui| {
                    ui.add(egui::Label::new(RichText::new(format!("{:.0} ({:.0}% speed)", st.guest_fps, st.speed_pct)).font(egui::FontId::monospace(12.0)).strong().color(ACCENT)).extend());
                });
            });
        }
        // `#player.paused`: a hard pause the person did not ask for by opening the menu.
        if !self.menu_open && self.session.as_ref().is_some_and(|s| s.paused_by_blur || s.paused) {
            let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("paused")));
            p.rect_filled(screen, 0.0, Color32::from_black_alpha(128));
            p.text(screen.center(), egui::Align2::CENTER_CENTER, "P A U S E D", egui::FontId::proportional(14.0), DIM);
        }
        // `#mutebtn` and `#menubtn`: 34 px, top right, half-transparent until the pointer is on them.
        let muted = self.session.as_ref().and_then(|s| s.audio.as_ref()).map_or_else(library::muted, |a| a.muted());
        let mut toggle_mute = false;
        let mut toggle_menu = false;
        egui::Area::new(egui::Id::new("hud-buttons")).fixed_pos([screen.right() - 8.0 - 34.0 - 8.0 - 34.0, 6.0]).order(egui::Order::Middle).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 8.0;
                if hud_button(ui, if muted { "Unmute" } else { "Mute" }, |p, r, c| draw_speaker(p, r, c, muted)).clicked() {
                    toggle_mute = true;
                }
                if hud_button(ui, "Menu", draw_hamburger).clicked() {
                    toggle_menu = true;
                }
            });
        });
        if toggle_mute {
            self.set_muted(!muted);
        }
        if toggle_menu {
            self.toggle_menu();
        }
    }

    /// The on-screen controls, when the settings put them on screen - see `touchpad`. Off (and
    /// holding nothing) while the menu is open, so a click in the menu presses no button.
    fn ui_touchpad(&mut self, ctx: &egui::Context) {
        let show = touchpad::shown(&self.session_settings.pad) && !self.menu_open;
        let ppp = ctx.pixels_per_point();
        let Some(s) = self.session.as_mut() else { return };
        let out = match (show, s.game_rect) {
            (true, Some((x, y, w, h))) => {
                let game = egui::Rect::from_min_size(
                    egui::pos2(x as f32 / ppp, y as f32 / ppp),
                    egui::vec2(w as f32 / ppp, h as f32 / ppp),
                );
                Some(touchpad::show(ctx, game, &self.session_settings.pad, s.last_ctrl, &mut self.touchpad))
            }
            _ => None,
        };
        s.input.overlay_buttons = out.as_ref().map_or(0, |o| o.buttons);
        s.input.overlay_sticks = out.as_ref().map_or([None, None], |o| o.sticks);
        s.overlay_pointer = out.is_some_and(|o| o.pointer);
    }

    fn set_muted(&mut self, muted: bool) {
        if let Some(a) = self.session.as_ref().and_then(|s| s.audio.as_ref()) {
            a.set_muted(muted);
        }
        let _ = library::save_muted(muted);
    }

    /// `#menu`: every in-game option the browser's menu has.
    fn ui_menu(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        let width = (screen.width() - 32.0).min(960.0);
        let mut close = false;
        let mut ask: Option<MenuAsk> = self.menu_ask;
        let mut answer: Option<bool> = None;
        let mut fullscreen = false;
        let mut capture = None;
        let mut to_settings = false;
        let mut copy = false;
        let mut download = false;
        let mut shot = false;
        let frame = panel_frame().fill(MENU_FILL);
        egui::Modal::new(egui::Id::new("menu")).backdrop_color(Color32::from_rgba_unmultiplied(5, 5, 10, 89)).frame(frame).show(ctx, |ui| {
            if let Some(a) = ask {
                // `.menu-confirm`: the Yes/No panel in place of the menu body.
                ui.set_width(width.min(420.0));
                ui.vertical_centered(|ui| {
                    ui.label(RichText::new(match a {
                        MenuAsk::Restart => "Restart this game from the beginning? Anything not saved is lost.",
                        MenuAsk::Quit => "Quit to the library? Anything not saved is lost.",
                    }).size(16.0).color(INK));
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.add_space(((ui.available_width() - 130.0) / 2.0).max(0.0));
                        if ui.add(danger("Yes")).clicked() {
                            answer = Some(true);
                        }
                        let no = ui.add(primary("No"));
                        if no.clicked() {
                            answer = Some(false);
                        }
                        if ui.memory(|m| m.focused()).is_none() {
                            no.request_focus();
                        }
                    });
                });
                return;
            }
            ui.set_width(width);
            // As tall as the window allows: a scroll area otherwise settles at its own minimum.
            let tall = (screen.height() - 64.0).max(200.0);
            egui::ScrollArea::vertical().max_height(tall).min_scrolled_height(tall).show(ui, |ui| {
                let name = self.playing.as_ref().map(|id| self.titles.iter().find(|t| &t.title_id == id).map_or(id.clone(), |t| t.title.clone())).unwrap_or_default();
                ui.label(bold(name).size(18.0).color(ACCENT));
                ui.add_space(2.0);
                ui.horizontal_wrapped(|ui| {
                    let resume = ui.add(primary("Resume"));
                    if resume.clicked() {
                        close = true;
                    }
                    if ui.memory(|m| m.focused()).is_none() && self.capture.is_none() {
                        resume.request_focus();
                    }
                    if ui.button(bold(if self.is_fullscreen() { "Exit fullscreen" } else { "Fullscreen" })).clicked() {
                        fullscreen = true;
                    }
                    if ui.button(bold("Restart game")).clicked() {
                        ask = Some(MenuAsk::Restart);
                    }
                    if ui.add(danger("Quit to library")).clicked() {
                        ask = Some(MenuAsk::Quit);
                    }
                });
                if let Some(st) = self.stats.as_ref() {
                    ui.label(RichText::new(st.title_line()).color(DIM).small());
                }
                menu_rule(ui);
                let mut muted = self.session.as_ref().and_then(|s| s.audio.as_ref()).map_or_else(library::muted, |a| a.muted());
                if check_row(ui, &mut muted, "Mute", None, false) {
                    self.set_muted(muted);
                }
                if self.session.as_ref().is_some_and(|s| s.audio.is_none()) {
                    ui.label(RichText::new("No sound output device - this title runs silent.").color(DIM).small());
                }
                menu_rule(ui);
                let mut changed = check_row(ui, &mut self.session_settings.show_fps, "Show frame rate", None, false);
                menu_rule(ui);
                changed |= check_row(ui, &mut self.session_settings.fps_in_title, "Show the frame rate in the window title", None, false);
                menu_rule(ui);
                changed |= check_row(ui, &mut self.session_settings.pause_on_blur, "Pause when hidden or unfocused", None, false);
                menu_rule(ui);
                let mut on_screen = touchpad::shown(&self.session_settings.pad);
                if check_row(ui, &mut on_screen, "Show the on-screen controls", None, false) {
                    self.session_settings.pad.mode =
                        if on_screen { vitaslop_frontend::settings::PadMode::Overlay } else { vitaslop_frontend::settings::PadMode::Auto };
                    changed = true;
                }
                menu_rule(ui);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Scaling").color(INK).size(15.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let all = [vitaslop_frontend::settings::Scaling::Fit, vitaslop_frontend::settings::Scaling::Integer, vitaslop_frontend::settings::Scaling::Stretch];
                        let before = self.session_settings.scaling;
                        let r = egui::ComboBox::from_id_salt("menu-scaling").width(170.0).selected_text(super::scaling_label(before)).show_ui(ui, |ui| {
                            for s in all {
                                ui.selectable_value(&mut self.session_settings.scaling, s, super::scaling_label(s));
                            }
                        });
                        super::combo_nav(&r.response, &mut self.nav, &mut self.session_settings.scaling, &all);
                        changed |= self.session_settings.scaling != before;
                    });
                });
                if changed {
                    self.menu_setting_changed();
                }
                ui.add_space(6.0);
                egui::CollapsingHeader::new(RichText::new("Controls").color(DIM)).id_salt("menu-controls").default_open(true).show(ui, |ui| {
                    seg(ui, &mut self.menu_ctl);
                    let listening = self.capture.filter(|c| c.live).map(|c| c.button);
                    capture = vitapic::show(ui, &self.session_settings.keyboard, &self.session_settings.gamepad, self.menu_ctl, listening, 760.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.label(RichText::new("Pick a control, then press the key or button for it; it takes effect at once and is kept for every title (or this one, where its settings already change that control). Everything else: ").color(DIM));
                        if ui.add(egui::Button::new(RichText::new("this title's settings").color(ACCENT)).frame(false)).clicked() {
                            to_settings = true;
                        }
                        ui.label(RichText::new(" (leaves the game).").color(DIM));
                    });
                });
                egui::CollapsingHeader::new(RichText::new("Diagnostics").color(DIM)).id_salt("menu-diag").show(ui, |ui| {
                    ui.horizontal(|ui| {
                        if ui.add(egui::Button::new(bold("Copy")).small()).clicked() {
                            copy = true;
                        }
                        if ui.add(egui::Button::new(bold("Download")).small()).clicked() {
                            download = true;
                        }
                        if ui.add(egui::Button::new(bold("Screenshot")).small()).clicked() {
                            shot = true;
                        }
                    });
                    if let Some(log) = crate::log::log_path() {
                        ui.label(RichText::new(format!("The run log is {}", log.display())).color(DIM).small());
                    }
                    egui::Frame::new().fill(Color32::from_rgb(0x0d, 0x0d, 0x16)).corner_radius(6).inner_margin(8).show(ui, |ui| {
                        egui::ScrollArea::vertical().id_salt("menu-diag-text").max_height(screen.height() * 0.3).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new(self.diag_text()).font(egui::FontId::monospace(11.0)).color(Color32::from_rgb(0xa8, 0xb8, 0xc8)));
                        });
                    });
                });
                if let Some(n) = self.menu_note.as_ref() {
                    ui.add_space(4.0);
                    ui.label(RichText::new(n).color(ACCENT).small());
                }
                ui.add_space(6.0);
                ui.label(RichText::new("Esc or the menu button opens and closes this; on a controller, the home button or Start and Select held together for a second. The game is paused while it is open. F11 toggles fullscreen.").color(DIM));
            });
        });
        let id = self.playing.clone().unwrap_or_default();
        if copy {
            ctx.copy_text(self.diag_text());
            self.menu_note = Some("Copied".into());
        }
        if download && let Some(m) = self.save_diag(&format!("vitaslop-{id}-diag.txt")) {
            self.menu_note = Some(m);
        }
        if shot
            && let Some(p) = rfd::FileDialog::new().set_file_name(format!("vitaslop-{id}-{}.png", library::now_ms())).add_filter("PNG", &["png"]).save_file()
        {
            self.menu_note = Some(format!("Saved {}", p.display()));
            self.shot_to = Some(p);
        }
        if let Some(b) = capture {
            let kind = if self.menu_ctl == vitapic::Mode::Gamepad { Kind::Gamepad } else { Kind::Keyboard };
            self.begin_capture(b, kind, true);
        }
        if fullscreen {
            self.toggle_fullscreen();
        }
        match answer {
            Some(true) => {
                self.menu_ask = None;
                match ask {
                    Some(MenuAsk::Restart) => self.restart(),
                    Some(MenuAsk::Quit) => self.quit_to_library(),
                    None => {}
                }
                return;
            }
            Some(false) => ask = None,
            None => {}
        }
        self.menu_ask = ask;
        if to_settings {
            let id = self.playing.clone();
            self.quit_to_library();
            self.open_draft(id);
            return;
        }
        if close {
            self.toggle_menu();
        }
    }

    /// A change made in the menu applies to the run at once and is kept as a GLOBAL setting:
    /// the person changed how they want to play, not this title - the browser's `onRuntimeSetting`.
    fn menu_setting_changed(&mut self) {
        if let Some(s) = self.session.as_mut() {
            s.pause_on_blur = self.session_settings.pause_on_blur;
            if !s.pause_on_blur {
                s.paused_by_blur = false;
            }
        }
        let mut g = library::effective(None);
        g.show_fps = self.session_settings.show_fps;
        g.fps_in_title = self.session_settings.fps_in_title;
        g.pause_on_blur = self.session_settings.pause_on_blur;
        g.scaling = self.session_settings.scaling;
        g.pad.mode = self.session_settings.pad.mode;
        if let Err(e) = library::save_global_settings(&g) {
            self.menu_note = Some(format!("could not save the setting: {e}"));
        }
        self.force_present = true;
    }

    /// `#fatal`: the run is over and this is why, with the diagnostics a click away.
    fn ui_fatal(&mut self, root: &mut egui::Ui) {
        let ctx = &root.ctx().clone();
        let text = self.fatal.clone().unwrap_or_default();
        let clean = text.starts_with("The game exited");
        let mut copy = false;
        let mut download = false;
        let mut back = false;
        egui::CentralPanel::default().frame(egui::Frame::new().fill(Color32::from_rgba_unmultiplied(5, 5, 10, 230))).show(root, |ui| {
            let w = (ui.available_width() - 32.0).min(520.0);
            ui.vertical_centered(|ui| {
                ui.add_space((ui.available_height() / 2.0 - 160.0).max(16.0));
                panel_frame().show(ui, |ui| {
                    ui.set_width(w);
                    ui.label(bold(if clean { "The game ended" } else { "The emulator stopped" }).size(16.0).color(if clean { INK } else { DANGER }));
                    egui::Frame::new().fill(Color32::from_rgb(0x2a, 0x0d, 0x12)).corner_radius(6).inner_margin(8).show(ui, |ui| {
                        egui::ScrollArea::vertical().max_height(ui.ctx().content_rect().height() * 0.5).show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(RichText::new(&text).font(egui::FontId::monospace(11.0)).color(Color32::from_rgb(0xff, 0xd9, 0xd9)));
                        });
                    });
                    if let Some(log) = crate::log::log_path() {
                        ui.label(RichText::new(format!("The run log is {}", log.display())).color(DIM).small());
                    }
                    ui.horizontal(|ui| {
                        let first = ui.add(primary("Copy diagnostics"));
                        if first.clicked() {
                            copy = true;
                        }
                        if ui.memory(|m| m.focused()).is_none() {
                            first.request_focus();
                        }
                        if ui.button(bold("Download diagnostics")).clicked() {
                            download = true;
                        }
                        if ui.button(bold("Back to library")).clicked() {
                            back = true;
                        }
                    });
                    if let Some(n) = self.menu_note.as_ref() {
                        ui.label(RichText::new(n).color(DIM).small());
                    }
                });
            });
        });
        let id = self.playing.clone().or_else(|| self.loading.as_ref().map(|l| l.title_id.clone())).unwrap_or_else(|| "unknown".into());
        if copy {
            ctx.copy_text(self.diag_text());
            self.menu_note = Some("Copied".into());
        }
        if download && let Some(m) = self.save_diag(&format!("vitaslop-{id}-fatal.txt")) {
            self.menu_note = Some(m);
        }
        if back || self.nav.back {
            self.nav.back = false;
            self.quit_to_library();
        }
    }
}

/// The thin line between the menu's rows (`.menu-in label.row` border-top).
fn menu_rule(ui: &mut egui::Ui) {
    let r = ui.available_rect_before_wrap();
    ui.painter().hline(r.x_range(), r.top(), Stroke::new(1.0, EDGE));
    ui.add_space(1.0);
}

/// A 34 px square button over the game, drawn by `icon`, faint until hovered or focused.
fn hud_button(ui: &mut egui::Ui, label: &str, icon: impl FnOnce(&egui::Painter, Rect, Color32)) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(vec2(34.0, 34.0), egui::Sense::click());
    let lit = resp.hovered() || resp.has_focus();
    let alpha = if lit { 1.0 } else { 0.55 };
    let p = ui.painter();
    p.rect(rect, 8.0, Color32::from_black_alpha((115.0 * alpha) as u8), Stroke::new(1.0, Color32::from_white_alpha((38.0 * alpha) as u8)), egui::StrokeKind::Inside);
    icon(p, rect, Color32::from_white_alpha((255.0 * alpha) as u8));
    resp.on_hover_text(label)
}

/// The menu button's three bars.
fn draw_hamburger(p: &egui::Painter, r: Rect, c: Color32) {
    for dy in [-5.0, 0.0, 5.0] {
        let y = r.center().y + dy;
        p.line_segment([Pos2::new(r.left() + 10.0, y), Pos2::new(r.right() - 10.0, y)], Stroke::new(2.0, c));
    }
}

/// The mute button's speaker, with sound waves - or a cross when muted.
fn draw_speaker(p: &egui::Painter, r: Rect, c: Color32, muted: bool) {
    let o = r.center() + vec2(-5.0, 0.0);
    let body = vec![o + vec2(-6.0, -3.5), o + vec2(-2.0, -3.5), o + vec2(3.0, -8.0), o + vec2(3.0, 8.0), o + vec2(-2.0, 3.5), o + vec2(-6.0, 3.5)];
    p.add(egui::Shape::convex_polygon(body, c, Stroke::NONE));
    let s = Stroke::new(1.6, c);
    if muted {
        p.line_segment([o + vec2(7.0, -4.0), o + vec2(14.0, 4.0)], s);
        p.line_segment([o + vec2(7.0, 4.0), o + vec2(14.0, -4.0)], s);
    } else {
        for (rad, n) in [(7.0_f32, 6), (11.0, 8)] {
            let pts: Vec<Pos2> = (0..=n).map(|i| {
                let a = -0.8 + 1.6 * i as f32 / n as f32;
                o + vec2(a.cos() * rad, a.sin() * rad)
            }).collect();
            p.add(egui::Shape::line(pts, s));
        }
    }
}

