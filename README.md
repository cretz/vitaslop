# vitaslop

A PlayStation Vita emulator that runs in a browser tab, and as a desktop app.

Everything happens on your own machine. There is no server, no Sony firmware, and
nothing is uploaded anywhere.

**Early days.** Some commercial games boot and play. Many do not. Homebrew works.
See the FAQ before expecting much.

## Play in a browser

<https://cretz.github.io/vitaslop/>

Your browser needs WebGPU, JSPI, shared memory, and OPFS. In practice that means a
recent Chrome or Edge on desktop or Android. The page checks all of it on load and
tells you what is missing. Firefox and Safari are not there yet.

Games you add are stored in the browser and stay there.

## Desktop

Download a build from [Releases](https://github.com/cretz/vitaslop/releases) for
Windows, macOS, or Linux. Nothing to install. The desktop app also has a `serve`
command that hosts the same web version from your own machine.

## FAQ

#### How do I load my games?

vitaslop only works with games you own, from your own console, or with homebrew.
It provides no games and downloads nothing. Pirated games are not supported. The
"Add games" screen lists what it accepts.

#### This code was written by AI, right?

Yes, 100%. vitaslop is an AI built and maintained project, and the code quality
reflects that. It is also why the usual expectations of a maintained repo do not
apply here.

#### Will you accept a pull request?

No. This is an AI project, and the agent maintaining it writes all of the code.

#### Can I file an issue?

Yes, and they get read. There is no promise that any of them get fixed. Reports
about one specific game are the least likely to, because most of them need that
game to investigate.

#### Which games work?

There is no compatibility list and there will not be one. Keeping it accurate is
more work than the project itself, and it turns into a promise.

#### Why is it slow on my phone?

Often it is not. Flagship Android phones on Chrome run plenty of this at full
speed. Demanding scenes are where it gives out, because a phone does the same work
as a desktop with roughly a third of the CPU, and full speed in those is not
always reachable.

#### Does anything get uploaded?

No. There is no server side to this project at all. The emulator runs entirely on
your device, and the hosted page is a static site with nothing behind it. Games,
saves, and settings never leave your browser or your disk, and once the page has
loaded it makes no network requests.

## How it works

Most of this had to be built from scratch, and some of it was more interesting
than expected.

- **ARM to WebAssembly.** The game's ARMv7, Thumb-2, NEON, and VFP code is lifted
  and translated to WebAssembly ahead of running it, then executed by the
  browser's own engine. Around 25,000 functions and 8.7 million instructions on a
  large title.
- **JSPI.** The Vita's blocking calls are synchronous, and the browser's are not.
  JavaScript Promise Integration bridges the two so the guest can wait on things
  without the page freezing.
- **A GXP shader recompiler.** The Vita's shaders are compiled SGX543 USSE
  microcode. There is no public compiler for it, so vitaslop decodes the
  instruction set and recompiles each shader pair to WGSL at runtime.
- **GXM to WebGPU.** The whole graphics command stream is captured and replayed as
  WebGPU render passes, including render targets, blending baked into shader
  epilogues, and swizzled and compressed textures.
- **Audio.** ATRAC9 decoding and an NGS voice mixer, feeding a shared ring buffer
  the audio thread reads directly.
- **Video.** H.264 and AAC behind one interface, using the platform decoder on
  each host, which is WebCodecs in a browser and Media Foundation, Video Toolbox,
  or VA-API natively.
- **Streaming import.** A retail package can be 3.3 GB, which does not fit in a
  32-bit wasm heap, so import streams from the file straight into browser storage
  and never holds it in memory.
- **A cooperative scheduler** with an emulated vblank clock, because guest threads
  expect to be preempted and the browser has one main thread.
- **Cross-origin isolation on a static host.** Shared memory needs COOP and COEP
  headers, and GitHub Pages cannot send them, so a service worker adds them to
  every response.
- **Archive mounting.** A title's data is one large compressed archive, mounted
  read-only and read on demand rather than unpacked to disk first.
- **Homebrew.** vita2d and SceSharedFb apps run too, which took eight fixes to the
  ARM lifter that no commercial game had ever reached.
- **A pixel oracle.** Headless runs render deterministic frames that tests compare
  bit for bit, so a rendering change either matches the previous build exactly or
  says which draw moved.

## Build

Rust and Node are the only requirements.

```
# desktop
cargo build --release -p vitaslop-desktop --manifest-path projects/Cargo.toml

# web (needs the wasm32 target and a matching wasm-bindgen CLI)
cd projects/vitaslop-web && node build.mjs && node serve.mjs
```

## License

MIT, see [LICENSE](LICENSE).

Developed clean-room, with no copyleft code. Built from these permissive or
neutral references:

- [vita-headers](https://github.com/vitasdk/vita-headers) (MIT) for the `sce*` API
  surface and NID database.
- [vita-toolchain](https://github.com/vitasdk/vita-toolchain) (MIT) for the
  SELF/ELF and VELF executable formats.
- [dynarmic](https://github.com/yuzu-mirror/dynarmic) (0BSD) as an ARMv7,
  Thumb-2, NEON, and VFP decode and semantics reference.
- [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) (MIT) as an ATRAC9
  reference.
- [psdevwiki](https://www.psdevwiki.com/vita/) and
  [henkaku wiki](https://wiki.henkaku.xyz/) for hardware documentation.
- The ARM Architecture Reference Manual.

Independent reverse engineering fills the rest.

Not affiliated with or endorsed by Sony Interactive Entertainment. PlayStation and
PS Vita are their trademarks.
