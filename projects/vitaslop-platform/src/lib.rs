//! Web/native abstraction seam. Defines the trait contracts that the frontends
//! implement: random-access asset storage (OPFS on web, mmap on desktop),
//! input, and audio. Renderer and window are thin wrappers over wgpu and winit,
//! which already span web and native from one API.
//!
//! This crate stays dependency-light. It pulls in no wasm-bindgen, js-sys, or
//! OS-specific crates. Concrete impls live in vitaslop-web and vitaslop-desktop
//! and are injected into the runtime at startup.
//!
//! Trait signatures (starting with the async storage trait) are pinned during
//! the conformance design pass.
