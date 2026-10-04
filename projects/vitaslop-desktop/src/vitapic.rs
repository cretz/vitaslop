//! The controller as a picture - the browser's `vita.js`: a Vita outline with a callout on
//! every control showing what drives it. On the settings page and in the in-game menu the
//! callouts are buttons (pick one, press the key or pad button for it); read-only they show
//! both bindings, so a person who forgot which key is Circle can look instead of guess.
//!
//! The geometry is the browser's, unit for unit: the same 400 x 190 drawing and the same
//! callout positions as percentages of it, so the two front ends draw one picture.

use std::collections::BTreeMap;

use egui::{Pos2, Rect, Stroke, vec2};
use vitaslop_frontend::input::Button;

use crate::shell::{ACCENT, BG, DIM, EDGE, INK, PANEL, PANEL_2};

/// What the callouts show and do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The keyboard binding, as a button.
    Keyboard,
    /// The gamepad binding, as a button.
    Gamepad,
    /// Both bindings, not clickable.
    ReadOnly,
}

/// `vita.js`'s `LAYOUT`: each control's callout centre, in percent of the drawing.
fn at(b: Button) -> (f32, f32) {
    match b {
        Button::L => (16.0, 6.0),
        Button::R => (84.0, 6.0),
        Button::Up => (17.0, 30.0),
        Button::Left => (8.0, 44.0),
        Button::Right => (26.0, 44.0),
        Button::Down => (17.0, 58.0),
        Button::Triangle => (83.0, 30.0),
        Button::Square => (74.0, 44.0),
        Button::Circle => (92.0, 44.0),
        Button::Cross => (83.0, 58.0),
        Button::Select => (68.0, 84.0),
        Button::Start => (90.0, 84.0),
    }
}

/// The browser's short callout name (`vita.js` prints the vocabulary label, upper-cased).
fn short(b: Button) -> &'static str {
    match b {
        Button::Up => "UP",
        Button::Down => "DOWN",
        Button::Left => "LEFT",
        Button::Right => "RIGHT",
        Button::Cross => "CROSS",
        Button::Circle => "CIRCLE",
        Button::Square => "SQUARE",
        Button::Triangle => "TRIANGLE",
        Button::L => "L",
        Button::R => "R",
        Button::Start => "START",
        Button::Select => "SELECT",
    }
}

/// Draw the picture at most `max_w` wide, centred. Returns the control whose callout was
/// clicked (never, read-only). `listening` lights the callout being captured.
pub fn show(
    ui: &mut egui::Ui,
    keyboard: &BTreeMap<String, String>,
    gamepad: &BTreeMap<String, String>,
    mode: Mode,
    listening: Option<Button>,
    max_w: f32,
) -> Option<Button> {
    let w = ui.available_width().min(max_w);
    let s = w / 400.0;
    // Room above the drawing for the shoulder callouts, which sit across its top edge.
    let pad = 16.0;
    let (outer, _) = ui.allocate_exact_size(vec2(ui.available_width(), 190.0 * s + 2.0 * pad), egui::Sense::hover());
    let origin = Pos2::new(outer.center().x - w / 2.0, outer.top() + pad);
    let p = |x: f32, y: f32| origin + vec2(x * s, y * s);
    let r = |x: f32, y: f32, rw: f32, rh: f32| Rect::from_min_max(p(x, y), p(x + rw, y + rh));
    let painter = ui.painter().clone();
    let edge = Stroke::new(1.0, EDGE);
    painter.rect(r(4.0, 12.0, 392.0, 166.0), 40.0 * s, PANEL, Stroke::new(2.0, EDGE), egui::StrokeKind::Inside);
    painter.rect(r(98.0, 34.0, 204.0, 116.0), 4.0 * s, BG, edge, egui::StrokeKind::Inside);
    for x in [40.0, 300.0] {
        painter.rect(r(x, 6.0, 60.0, 6.0), 0.0, PANEL_2, edge, egui::StrokeKind::Inside);
    }
    for x in [52.0, 348.0] {
        painter.circle(p(x, 140.0), 14.0 * s, PANEL_2, edge);
    }
    painter.rect(r(46.0, 52.0, 12.0, 36.0), 2.0 * s, PANEL_2, edge, egui::StrokeKind::Inside);
    painter.rect(r(34.0, 64.0, 36.0, 12.0), 2.0 * s, PANEL_2, edge, egui::StrokeKind::Inside);
    for (x, y) in [(348.0, 56.0), (332.0, 72.0), (364.0, 72.0), (348.0, 88.0)] {
        painter.circle(p(x, y), 6.0 * s, PANEL_2, edge);
    }
    for x in [286.0, 340.0] {
        painter.rect(r(x, 160.0, 14.0, 6.0), 3.0 * s, PANEL_2, edge, egui::StrokeKind::Inside);
    }

    let mono = egui::FontId::monospace(11.0);
    let mut clicked = None;
    for b in Button::ALL {
        let (px, py) = at(b);
        let centre = origin + vec2(px / 100.0 * w, py / 100.0 * 190.0 * s);
        let kb = keyboard.get(b.name()).map_or("-", String::as_str);
        let gp = gamepad.get(b.name()).map_or("-", String::as_str);
        let shown = if mode == Mode::Gamepad { gp } else { kb };
        let text_w = |t: &str| ui.fonts_mut(|f| f.layout_no_wrap(t.to_string(), mono.clone(), INK).size().x);
        let box_w = (text_w(shown).max(if mode == Mode::ReadOnly { text_w(gp) } else { 0.0 }) + 14.0).max(64.0);
        let rows = if mode == Mode::ReadOnly { 44.0 } else { 36.0 };
        let top = centre.y - rows / 2.0;
        painter.text(Pos2::new(centre.x, top), egui::Align2::CENTER_TOP, short(b), egui::FontId::proportional(10.0), DIM);
        let key_rect = Rect::from_center_size(Pos2::new(centre.x, top + 14.0 + 11.0), vec2(box_w, 22.0));
        if mode == Mode::ReadOnly {
            let kr = Rect::from_center_size(Pos2::new(centre.x, top + 14.0 + 9.0), vec2(box_w, 18.0));
            painter.rect(kr, 6.0, PANEL_2, edge, egui::StrokeKind::Inside);
            painter.text(kr.center(), egui::Align2::CENTER_CENTER, kb, mono.clone(), INK);
            let gr = Rect::from_center_size(Pos2::new(centre.x, kr.bottom() + 2.0 + 8.0), vec2(box_w, 16.0));
            painter.rect(gr, 6.0, PANEL_2, edge, egui::StrokeKind::Inside);
            painter.text(gr.center(), egui::Align2::CENTER_CENTER, gp, egui::FontId::monospace(10.0), DIM);
            continue;
        }
        let lit = listening == Some(b);
        // Painted and sensed in place rather than laid out: a positioned widget would push the
        // picture's layout cursor around, and these sit at fixed points of the drawing.
        let resp = ui.interact(key_rect, ui.id().with(("vita-callout", b.name())), egui::Sense::click());
        let edge = if lit { ACCENT } else if resp.hovered() { DIM } else { EDGE };
        painter.rect(key_rect, 6.0, PANEL_2, Stroke::new(1.0, edge), egui::StrokeKind::Inside);
        painter.text(key_rect.center(), egui::Align2::CENTER_CENTER, shown, mono.clone(), if lit { ACCENT } else { INK });
        if resp.has_focus() {
            painter.rect_stroke(key_rect.expand(3.0), 8.0, Stroke::new(2.0, ACCENT), egui::StrokeKind::Outside);
        }
        if resp.clicked() {
            clicked = Some(b);
        }
    }
    clicked
}
