//! Vita ROM front half: parses SELF/ELF, applies relocations, reads NID
//! import/export tables, and performs provenance-based entry-point
//! enumeration. Produces the input that vitaslop-transpiler consumes.
