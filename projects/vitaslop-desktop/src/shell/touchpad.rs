//! The on-screen controls over the game - the browser's `web/touchpad.js` in its landscape
//! "overlay" placement: L and R at the top corners, the d-pad and left stick at the left, the
//! faces and right stick at the right, Select and Start at the bottom middle.
//!
//! On a desktop there is no touch screen, so the mouse plays the finger: a press on a button
//! holds it until release (the browser's pointer capture), a drag on a stick moves it. Every
//! control also SHOWS the controller state the guest was handed - a button lit while a key or
//! pad button holds it, a stick's knob where the pad's stick is - so the picture agrees with the
//! game whichever way a control was pushed. Shown when the settings' pad placement is Overlay
//! (or Beside, which a window has no room for); Auto is hidden, a desktop having no touch.

use egui::{pos2, vec2, Align2, Color32, FontId, Pos2, Rect, Stroke};
use vitaslop_frontend::input::Button;
use vitaslop_frontend::settings::{PadMode, PadSettings};
use vitaslop_runtime::CtrlFrame;

use super::ACCENT;

/// The browser's stick dead zone for a finger (`STICK_DEADZONE` in `touchpad.js`).
const STICK_DEADZONE: f32 = 0.14;

/// What the mouse is holding, from its press to its release.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Grab {
    Button(Button),
    Stick(usize),
}

#[derive(Default)]
pub(crate) struct PadState {
    grab: Option<Grab>,
}

/// What the controls hand the input this frame.
pub(crate) struct PadOutput {
    pub buttons: u32,
    pub sticks: [Option<(u8, u8)>; 2],
    /// The mouse is on a control (or holding one), so it is not touching the game screen.
    pub pointer: bool,
}

/// Whether these settings put the controls on screen in a window.
pub(crate) fn shown(pad: &PadSettings) -> bool {
    matches!(pad.mode, PadMode::Overlay | PadMode::Beside)
}

/// One control's place, in window points.
struct Layout {
    buttons: Vec<(Button, Rect, &'static str)>,
    sticks: [(Pos2, f32); 2],
}

fn layout(game: Rect, scale: f32) -> Layout {
    // One unit is a face button at the console's 960x544, scaled with the picture.
    let u = (game.width() / 960.0).min(game.height() / 544.0) * 56.0 * scale.clamp(0.5, 2.0);
    let (l, r, t, b) = (game.left(), game.right(), game.top(), game.bottom());
    let sq = |c: Pos2| Rect::from_center_size(c, vec2(u * 0.92, u * 0.92));
    let dpad = pos2(l + u * 1.7, b - u * 2.9);
    let faces = pos2(r - u * 1.7, b - u * 2.9);
    let buttons = vec![
        (Button::L, Rect::from_min_size(pos2(l + u * 0.3, t + u * 0.3), vec2(u * 1.7, u * 0.7)), "L"),
        (Button::R, Rect::from_min_size(pos2(r - u * 2.0, t + u * 0.3), vec2(u * 1.7, u * 0.7)), "R"),
        (Button::Up, sq(dpad + vec2(0.0, -u)), ""),
        (Button::Down, sq(dpad + vec2(0.0, u)), ""),
        (Button::Left, sq(dpad + vec2(-u, 0.0)), ""),
        (Button::Right, sq(dpad + vec2(u, 0.0)), ""),
        (Button::Triangle, sq(faces + vec2(0.0, -u)), ""),
        (Button::Cross, sq(faces + vec2(0.0, u)), ""),
        (Button::Square, sq(faces + vec2(-u, 0.0)), ""),
        (Button::Circle, sq(faces + vec2(u, 0.0)), ""),
        (Button::Select, Rect::from_center_size(pos2(game.center().x - u * 0.95, b - u * 0.5), vec2(u * 1.6, u * 0.5)), "SELECT"),
        (Button::Start, Rect::from_center_size(pos2(game.center().x + u * 0.95, b - u * 0.5), vec2(u * 1.6, u * 0.5)), "START"),
    ];
    let sticks = [(pos2(l + u * 3.9, b - u * 1.2), u * 0.85), (pos2(r - u * 3.9, b - u * 1.2), u * 0.85)];
    Layout { buttons, sticks }
}

/// The guest's 0..255 stick byte as -1..1.
fn axis(v: u8) -> f32 {
    ((f32::from(v) - 128.0) / 127.0).clamp(-1.0, 1.0)
}

/// -1..1 as the guest's 0..255 byte - `encodeAxis` in `touchpad.js`.
fn byte(v: f32) -> u8 {
    (128.0 + v * 127.0).round().clamp(0.0, 255.0) as u8
}

/// A drag from a stick's centre as a deflection: clamped to the rim, zero inside the dead zone.
fn deflection(centre: Pos2, radius: f32, at: Pos2) -> (f32, f32) {
    let d = (at - centre) / radius.max(1.0);
    let mag = d.length();
    let d = if mag > 1.0 { d / mag } else { d };
    if mag < STICK_DEADZONE { (0.0, 0.0) } else { (d.x, d.y) }
}

/// Draw the controls over `game` and read the mouse on them.
pub(crate) fn show(ctx: &egui::Context, game: Rect, pad: &PadSettings, ctrl: CtrlFrame, state: &mut PadState) -> PadOutput {
    let lay = layout(game, pad.scale);
    let (pos, down, pressed) = ctx.input(|i| (i.pointer.interact_pos(), i.pointer.primary_down(), i.pointer.primary_pressed()));
    // Press: grab whatever is under the pointer. Release: let go.
    if pressed && let Some(p) = pos {
        state.grab = lay
            .buttons
            .iter()
            .find(|(_, r, _)| r.contains(p))
            .map(|(b, _, _)| Grab::Button(*b))
            .or_else(|| lay.sticks.iter().position(|(c, rad)| (p - *c).length() <= *rad * 1.3).map(Grab::Stick));
    }
    if !down {
        state.grab = None;
    }
    let mut out = PadOutput { buttons: 0, sticks: [None, None], pointer: state.grab.is_some() };
    if let Some(p) = pos {
        out.pointer |= lay.buttons.iter().any(|(_, r, _)| r.contains(p)) || lay.sticks.iter().any(|(c, rad)| (p - *c).length() <= *rad * 1.3);
    }
    match state.grab {
        Some(Grab::Button(b)) => out.buttons = b.bit(),
        Some(Grab::Stick(i)) => {
            if let Some(p) = pos {
                let (dx, dy) = deflection(lay.sticks[i].0, lay.sticks[i].1, p);
                out.sticks[i] = Some((byte(dx), byte(dy)));
            }
        }
        None => {}
    }
    let alpha = pad.opacity.clamp(0.1, 1.0);
    let fill = Color32::from_white_alpha((70.0 * alpha) as u8);
    let edge = Stroke::new(1.5, Color32::from_white_alpha((150.0 * alpha) as u8));
    let ink = Color32::from_white_alpha((230.0 * alpha) as u8);
    let lit = ACCENT.gamma_multiply(0.35 + 0.65 * alpha);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Middle, egui::Id::new("touchpad")));
    let held = ctrl.buttons | out.buttons;
    for (b, r, glyph) in &lay.buttons {
        let on = held & b.bit() != 0;
        let small = matches!(b, Button::Select | Button::Start);
        let radius = if small || matches!(b, Button::L | Button::R) { r.height() * 0.5 } else { r.width() * 0.5 };
        painter.rect_filled(*r, radius, if on { lit } else { fill });
        painter.rect_stroke(*r, radius, edge, egui::StrokeKind::Inside);
        let col = if on { Color32::BLACK } else { ink };
        // The face symbols and arrows are DRAWN: the UI font has no PlayStation shapes, and a
        // missing glyph renders as an empty box.
        let c = r.center();
        let h = r.height() * 0.22;
        let stroke = Stroke::new((r.height() * 0.07).max(1.5), col);
        match b {
            Button::Triangle => {
                let pts = vec![c + vec2(0.0, -h), c + vec2(h * 0.95, h * 0.7), c + vec2(-h * 0.95, h * 0.7)];
                painter.add(egui::Shape::closed_line(pts, stroke));
            }
            Button::Circle => {
                painter.circle_stroke(c, h * 0.9, stroke);
            }
            Button::Cross => {
                painter.line_segment([c + vec2(-h, -h) * 0.8, c + vec2(h, h) * 0.8], stroke);
                painter.line_segment([c + vec2(-h, h) * 0.8, c + vec2(h, -h) * 0.8], stroke);
            }
            Button::Square => {
                painter.rect_stroke(Rect::from_center_size(c, vec2(h * 1.5, h * 1.5)), 0.0, stroke, egui::StrokeKind::Middle);
            }
            Button::Up | Button::Down | Button::Left | Button::Right => {
                let (d, s) = match b {
                    Button::Up => (vec2(0.0, -1.0), vec2(1.0, 0.0)),
                    Button::Down => (vec2(0.0, 1.0), vec2(1.0, 0.0)),
                    Button::Left => (vec2(-1.0, 0.0), vec2(0.0, 1.0)),
                    _ => (vec2(1.0, 0.0), vec2(0.0, 1.0)),
                };
                let pts = vec![c + d * h, c - d * h * 0.6 + s * h * 0.9, c - d * h * 0.6 - s * h * 0.9];
                painter.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
            }
            _ => {
                painter.text(c, Align2::CENTER_CENTER, *glyph, FontId::proportional(r.height() * 0.55), col);
            }
        }
    }
    // Each knob sits where the stick the guest was handed is - a drag here, or a pad's stick.
    let pads = [(ctrl.lx, ctrl.ly), (ctrl.rx, ctrl.ry)];
    for (i, (c, rad)) in lay.sticks.iter().enumerate() {
        let (x, y) = out.sticks[i].unwrap_or(pads[i]);
        let knob = *c + vec2(axis(x), axis(y)) * *rad * 0.6;
        let moved = out.sticks[i].is_some() || (axis(x).abs() + axis(y).abs()) > 0.05;
        painter.circle_filled(*c, *rad, fill);
        painter.circle_stroke(*c, *rad, edge);
        painter.circle_filled(knob, *rad * 0.5, if moved { lit } else { Color32::from_white_alpha((120.0 * alpha) as u8) });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drag_is_clamped_to_the_rim_and_dead_at_the_centre() {
        let c = pos2(100.0, 100.0);
        assert_eq!(deflection(c, 50.0, pos2(103.0, 100.0)), (0.0, 0.0));
        let (x, y) = deflection(c, 50.0, pos2(300.0, 100.0));
        assert!((x - 1.0).abs() < 1e-6 && y.abs() < 1e-6);
        assert_eq!(byte(1.0), 255);
        assert_eq!(byte(-1.0), 1);
        assert_eq!(byte(0.0), 128);
        assert!((axis(byte(0.5)) - 0.5).abs() < 0.01);
    }

    #[test]
    fn only_the_overlay_placements_show_in_a_window() {
        let mut p = PadSettings::default();
        assert!(!shown(&p), "Auto is hidden on a desktop");
        p.mode = PadMode::Overlay;
        assert!(shown(&p));
        p.mode = PadMode::Hidden;
        assert!(!shown(&p));
    }

    #[test]
    fn every_vita_button_has_a_control() {
        let lay = layout(Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 544.0)), 1.0);
        for b in Button::ALL {
            assert!(lay.buttons.iter().any(|(x, _, _)| *x == b), "{b:?} has no on-screen control");
        }
    }
}
