# Builds the browser bundle: compile the crate to wasm, then run wasm-bindgen to
# generate the JS glue + processed wasm into web/pkg/. Serve web/ over HTTP
# afterwards (a module + WebGPU need an http origin, not file://), e.g.:
#   python -m http.server -d projects/vitaslop-web/web 8080
# then open http://localhost:8080/ in a WebGPU-capable browser.
#
# The wasm-bindgen CLI version must equal the crate's wasm-bindgen version
# (pinned in Cargo.toml). If it does not, run:
#   cargo install -f wasm-bindgen-cli --version <the pinned version>
param([switch]$Debug)
# Not "Stop": cargo and wasm-bindgen write normal progress to stderr, which under
# Stop would abort the script even on success. We gate on $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"

$crateDir = $PSScriptRoot
$projects = Split-Path $crateDir -Parent
# Release for a real perf read; pass -Debug to iterate faster.
$buildProfile = if ($Debug) { "dev" } else { "wasm-release" }
$profileDir = if ($Debug) { "debug" } else { "wasm-release" }

Write-Host "Building vitaslop-web ($buildProfile) for wasm32..."
cargo build --manifest-path "$projects\Cargo.toml" -p vitaslop-web `
    --target wasm32-unknown-unknown --profile $buildProfile
if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }

$wasm = "$projects\target\wasm32-unknown-unknown\$profileDir\vitaslop_web.wasm"
$out = "$crateDir\web\pkg"

Write-Host "Running wasm-bindgen -> $out"
wasm-bindgen --target web --out-dir $out --out-name vitaslop_web $wasm
if ($LASTEXITCODE -ne 0) { throw "wasm-bindgen failed (CLI/crate version mismatch?)" }

Write-Host "Done. Serve projects/vitaslop-web/web over HTTP and open index.html."
