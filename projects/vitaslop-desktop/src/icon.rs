//! The app's icon on every window the desktop opens.
//!
//! The executable carries the icon as a resource (`build.rs`), which is what Explorer shows
//! for the file - but a window shows the icon of its window CLASS, and winit registers its
//! class with none. So the shell and the `--game` window both sat in the taskbar under the
//! system's blank program icon. The same picture is decoded here and handed to winit for the
//! title bar and, on Windows, the taskbar.

use winit::window::{Icon, WindowAttributes};

/// `assets/icon.png`, rasterised from the web front end's `icon.svg` like `icon.ico`.
const PNG: &[u8] = include_bytes!("../assets/icon.png");

/// The icon as winit takes it, `None` if the embedded PNG cannot be decoded (a window
/// without its icon is still a window).
pub fn window_icon() -> Option<Icon> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(PNG));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).ok()?;
    let (w, h) = (info.width, info.height);
    let rgba = match info.color_type {
        png::ColorType::Rgba => buf[..(w * h * 4) as usize].to_vec(),
        png::ColorType::Rgb => buf[..(w * h * 3) as usize].chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect(),
        _ => return None,
    };
    Icon::from_rgba(rgba, w, h).ok()
}

/// `attrs` with the app's icon on the window and (Windows) in the taskbar.
pub fn with_icon(attrs: WindowAttributes) -> WindowAttributes {
    let icon = window_icon();
    #[cfg(windows)]
    let attrs = {
        use winit::platform::windows::WindowAttributesExtWindows;
        attrs.with_taskbar_icon(icon.clone())
    };
    attrs.with_window_icon(icon)
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_embedded_icon_decodes() {
        assert!(super::window_icon().is_some());
    }
}
