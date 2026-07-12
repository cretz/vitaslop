//! Probe: is any GPU adapter reachable in this environment? Informational.
#[test]
fn list_adapters() {
    let instance = wgpu::Instance::default();
    // enumerate_adapters is async in wgpu 30.
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    eprintln!("found {} adapter(s)", adapters.len());
    for a in &adapters {
        let info = a.get_info();
        eprintln!("  {:?} {} ({:?})", info.backend, info.name, info.device_type);
    }
}
