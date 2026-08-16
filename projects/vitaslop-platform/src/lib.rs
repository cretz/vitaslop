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
//!
//! The GPU seam is live: [`gpu`] holds the shared cube render pipeline and the
//! neutral [`gpu::DrawBatch`] type. The pipeline itself is behind the `gpu`
//! feature so the engine-agnostic runtime can depend on this crate for the
//! neutral types without pulling in wgpu.

pub mod diag;
/// A fast, non-cryptographic hasher for the per-draw maps on both sides of the seam. Here rather
/// than in the runtime because the renderer's own caches are hit just as often as the capture's,
/// and two copies of a hash function is exactly the kind of duplicate that drifts.
pub mod fasthash;
pub mod gpu;
pub mod knobs;
/// The GPU texture transcoder: guest blocks -> compressed blocks, in compute shaders, with no
/// CPU decode and no readback. Behind the `gpu` feature for the same reason the pipelines are -
/// the engine-agnostic runtime depends on this crate for the neutral types and must not pull in
/// wgpu.
#[cfg(feature = "gpu")]
pub mod texenc;
