//! Probe: is any GPU adapter reachable in this environment? Informational.
#[test]
fn list_adapters() {
    let instance = wgpu::Instance::default();
    // enumerate_adapters is async in wgpu 30.
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    eprintln!("found {} adapter(s)", adapters.len());
    for a in &adapters {
        let info = a.get_info();
        eprintln!(
            "  {:?} {} ({:?}) vendor {:#06x} device {:#06x} driver {:?} {:?}",
            info.backend, info.name, info.device_type, info.vendor, info.device, info.driver, info.driver_info
        );
    }
    // ...and the one the renderer actually takes, with how it classifies it - what a test that
    // gates on `software` (see `vita_gxmconf_real.rs`) will see on this machine.
    match vitaslop_native::wgpu_render::GeneralRenderer::new() {
        Some(g) => eprintln!("GeneralRenderer picks: {} (software: {})", g.adapter_name, g.software),
        None => eprintln!("GeneralRenderer: no adapter"),
    }
}
