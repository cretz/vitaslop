//! Browser app entry. Rust cdylib compiled via wasm-bindgen, paired with a
//! vite/TypeScript static-page layer (added when we build the web frontend).
//! Implements the vitaslop-platform traits against browser APIs (OPFS, Gamepad,
//! Web Audio) and drives vitaslop-runtime.
