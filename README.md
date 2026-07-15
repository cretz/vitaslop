# vitaslop

A client-side, in-browser PlayStation Vita emulator.

> **Status: under active early development.** Nothing here is usable yet. This
> repository is greenfield and the public API, structure, and scope are all in
> flux.

## License

MIT (see [LICENSE](LICENSE)). The project is developed clean-room: no copyleft
(GPL/LGPL) code. It is built only from these permissive or neutral references:

- [vita-headers](https://github.com/vitasdk/vita-headers) (MIT) - the `sce*` API
  surface and NID database.
- [vita-toolchain](https://github.com/vitasdk/vita-toolchain) (MIT) - the
  SELF/ELF and VELF executable-format tools.
- [dynarmic](https://github.com/yuzu-mirror/dynarmic) (0BSD) - a permissive
  ARMv7 + Thumb-2 + NEON + VFP recompiler, used as decode and semantics reference.
- [LibAtrac9](https://github.com/Thealexbarney/LibAtrac9) (MIT) - an ATRAC9
  (the Vita's standard audio codec) decoder, referenced for the audio path.
- [psdevwiki](https://www.psdevwiki.com/vita/) and
  [henkaku wiki](https://wiki.henkaku.xyz/) - hardware and reverse-engineering
  documentation.
- The ARM Architecture Reference Manual.

Independent reverse engineering fills the rest.
