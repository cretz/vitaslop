//! A C `printf`-family formatter for the Vita clib host calls.
//!
//! Why the host formats: `sceClibPrintf` is a single NID the guest calls with a
//! format string plus a variadic tail. The Vita kernel formats internally, so the
//! host owns the formatting - we implement the C conversion semantics here.
//!
//! The variadic tail follows AAPCS: variadic arguments are always marshaled as
//! for the base standard (no VFP), so every argument sits in the core registers
//! (r0..r3) then on the stack, and a `double`/`long long` occupies an 8-byte
//! aligned slot (an even-numbered word pair). This module models that as a word
//! cursor over [`GuestCtx::arg`], which already reads word `n` from r0..r3 then
//! `sp + (n-4)*4`. Verified against arm-vita-eabi-gcc output for the hello corpus.
//!
//! Supported conversions: `d i u o x X c s p f F e E g G %`, with the flags
//! `- + space 0 #`, a numeric or `*` width, a numeric or `*` precision, and the
//! length modifiers `hh h l ll z j t` (they only affect how wide an integer
//! argument is read). This is a pragmatic subset covering what homebrew logging
//! uses; it is not the full C99 surface (no wide chars, no `%a`, no `%n`).

use crate::host::GuestCtx;

/// Upper bound on a format string or a `%s` argument we will read from guest
/// memory, so a missing NUL cannot make us scan the whole address space.
const MAX_STR: usize = 4096;

/// A cursor over the variadic argument area, in 4-byte words. Word `n` is r`n`
/// for `n < 4`, else the stack slot at `sp + (n-4)*4` (see [`GuestCtx::arg`]).
struct ArgCursor<'a, 'b> {
    ctx: &'a GuestCtx<'b>,
    word: usize,
}

impl<'a, 'b> ArgCursor<'a, 'b> {
    fn new(ctx: &'a GuestCtx<'b>, first_word: usize) -> Self {
        ArgCursor { ctx, word: first_word }
    }

    /// Read a 4-byte argument (int, unsigned, pointer, promoted char).
    fn next_word(&mut self) -> u32 {
        let v = self.ctx.arg(self.word);
        self.word += 1;
        v
    }

    /// Read an 8-byte argument (double, `long long`). AAPCS 8-byte-aligns it: the
    /// word cursor rounds up to an even index, which - because sp is 8-byte
    /// aligned at a public call - is always an 8-byte boundary across both the
    /// core registers and the stack.
    fn next_dword(&mut self) -> u64 {
        self.word = (self.word + 1) & !1;
        let lo = self.ctx.arg(self.word) as u64;
        let hi = self.ctx.arg(self.word + 1) as u64;
        self.word += 2;
        lo | (hi << 32)
    }
}

/// Parsed `printf` flags.
#[derive(Default, Clone, Copy)]
struct Flags {
    left: bool,  // '-': left-justify within the field
    plus: bool,  // '+': always show a sign on signed conversions
    space: bool, // ' ': leading space on non-negative signed conversions
    zero: bool,  // '0': pad with zeros
    alt: bool,   // '#': alternate form (0x/0 prefix, forced '.')
}

/// How wide an integer argument is, from the length modifier.
#[derive(Clone, Copy, PartialEq)]
enum IntWidth {
    W32,
    W64,
}

/// One parsed conversion specifier (everything between `%` and the conversion
/// letter): flags, field width, precision, and integer-argument width.
#[derive(Clone, Copy)]
struct Spec {
    flags: Flags,
    width: usize,
    precision: Option<usize>,
    iw: IntWidth,
}

/// Format the C string at `fmt_addr`, pulling variadic arguments starting at word
/// `first_word` (1 for `printf`: word 0 is the format-string pointer itself).
/// Appends the formatted bytes to `out`.
pub fn format_into(out: &mut Vec<u8>, ctx: &GuestCtx, fmt_addr: u32, first_word: usize) {
    let fmt = read_cstr_bytes(ctx, fmt_addr);
    let mut args = ArgCursor::new(ctx, first_word);
    let mut i = 0;
    while i < fmt.len() {
        let c = fmt[i];
        if c != b'%' {
            out.push(c);
            i += 1;
            continue;
        }
        // Parse one conversion beginning at the '%'.
        i += 1;
        if i >= fmt.len() {
            out.push(b'%');
            break;
        }

        // Flags.
        let mut flags = Flags::default();
        loop {
            match fmt.get(i) {
                Some(b'-') => flags.left = true,
                Some(b'+') => flags.plus = true,
                Some(b' ') => flags.space = true,
                Some(b'0') => flags.zero = true,
                Some(b'#') => flags.alt = true,
                _ => break,
            }
            i += 1;
        }

        // Width (numeric or '*').
        let width = if fmt.get(i) == Some(&b'*') {
            i += 1;
            let w = args.next_word() as i32;
            // A negative width means left-justify with the magnitude as width.
            if w < 0 {
                flags.left = true;
                w.unsigned_abs() as usize
            } else {
                w as usize
            }
        } else {
            let mut w = 0usize;
            while let Some(d) = fmt.get(i).and_then(|b| (*b as char).to_digit(10)) {
                w = w * 10 + d as usize;
                i += 1;
            }
            w
        };

        // Precision ('.' then numeric or '*'); present-but-empty means 0.
        let mut precision: Option<usize> = None;
        if fmt.get(i) == Some(&b'.') {
            i += 1;
            if fmt.get(i) == Some(&b'*') {
                i += 1;
                let p = args.next_word() as i32;
                // A negative precision is taken as if omitted.
                precision = if p < 0 { None } else { Some(p as usize) };
            } else {
                let mut p = 0usize;
                while let Some(d) = fmt.get(i).and_then(|b| (*b as char).to_digit(10)) {
                    p = p * 10 + d as usize;
                    i += 1;
                }
                precision = Some(p);
            }
        }

        // Length modifier (affects only how wide an integer argument is read).
        let mut iw = IntWidth::W32;
        loop {
            match fmt.get(i) {
                Some(b'l') => {
                    // 'l' then another 'l' is long long (64-bit). On the Vita,
                    // long is 32-bit, so a single 'l' stays 32.
                    if fmt.get(i + 1) == Some(&b'l') {
                        iw = IntWidth::W64;
                        i += 1;
                    }
                }
                // long long, intmax_t: 64-bit on this ABI.
                Some(b'j') => iw = IntWidth::W64,
                // short/char widths and size_t/ptrdiff_t: all read as a 32-bit
                // word here (they are promoted to int in the variadic tail).
                Some(b'h') | Some(b'z') | Some(b't') | Some(b'L') => {}
                _ => break,
            }
            i += 1;
        }

        // Conversion.
        let Some(&conv) = fmt.get(i) else {
            out.push(b'%');
            break;
        };
        i += 1;
        format_conv(out, conv, &Spec { flags, width, precision, iw }, &mut args, ctx);
    }
}

/// Emit one conversion's output.
fn format_conv(out: &mut Vec<u8>, conv: u8, spec: &Spec, args: &mut ArgCursor, ctx: &GuestCtx) {
    let Spec { flags, width, precision, iw } = *spec;
    match conv {
        b'%' => pad_and_emit(out, b"", b"%", flags, width, false),
        b'd' | b'i' => {
            let v = read_signed(args, iw);
            let neg = v < 0;
            let digits = to_decimal(v.unsigned_abs());
            emit_signed(out, &digits, neg, flags, width, precision);
        }
        b'u' => {
            let v = read_unsigned(args, iw);
            let digits = to_decimal(v);
            emit_unsigned(out, &digits, b"", flags, width, precision);
        }
        b'o' => {
            let v = read_unsigned(args, iw);
            let digits = to_radix(v, 8, false);
            let prefix: &[u8] = if flags.alt && !digits.starts_with(b"0") { b"0" } else { b"" };
            emit_unsigned(out, &digits, prefix, flags, width, precision);
        }
        b'x' | b'X' => {
            let upper = conv == b'X';
            let v = read_unsigned(args, iw);
            let digits = to_radix(v, 16, upper);
            let prefix: &[u8] = if flags.alt && v != 0 {
                if upper { b"0X" } else { b"0x" }
            } else {
                b""
            };
            emit_unsigned(out, &digits, prefix, flags, width, precision);
        }
        b'p' => {
            // Vita/glibc style: "0x" + lowercase hex, or "(nil)" for null.
            let v = args.next_word();
            if v == 0 {
                pad_and_emit(out, b"", b"(nil)", flags, width, false);
            } else {
                let digits = to_radix(v as u64, 16, false);
                emit_unsigned(out, &digits, b"0x", Flags { zero: false, ..flags }, width, None);
            }
        }
        b'c' => {
            let ch = [args.next_word() as u8];
            pad_and_emit(out, b"", &ch, flags, width, false);
        }
        b's' => {
            let ptr = args.next_word();
            let mut s = read_cstr_bytes(ctx, ptr);
            if let Some(p) = precision {
                s.truncate(p);
            }
            pad_and_emit(out, b"", &s, flags, width, false);
        }
        b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
            let bits = args.next_dword();
            let val = f64::from_bits(bits);
            emit_float(out, conv, val, flags, width, precision);
        }
        // Unknown conversion: echo it literally, like a lenient libc.
        other => {
            out.push(b'%');
            out.push(other);
        }
    }
}

/// Read a signed integer argument of the given width, sign-extended to i64.
fn read_signed(args: &mut ArgCursor, iw: IntWidth) -> i64 {
    match iw {
        IntWidth::W32 => args.next_word() as i32 as i64,
        IntWidth::W64 => args.next_dword() as i64,
    }
}

/// Read an unsigned integer argument of the given width, zero-extended to u64.
fn read_unsigned(args: &mut ArgCursor, iw: IntWidth) -> u64 {
    match iw {
        IntWidth::W32 => args.next_word() as u64,
        IntWidth::W64 => args.next_dword(),
    }
}

/// Decimal digits of a non-negative value (no sign), "0" for zero.
fn to_decimal(v: u64) -> Vec<u8> {
    to_radix(v, 10, false)
}

/// Digits of `v` in `radix`, most-significant first; "0" for zero.
fn to_radix(mut v: u64, radix: u64, upper: bool) -> Vec<u8> {
    if v == 0 {
        return vec![b'0'];
    }
    let lut: &[u8] = if upper {
        b"0123456789ABCDEF"
    } else {
        b"0123456789abcdef"
    };
    let mut digits = Vec::new();
    while v > 0 {
        digits.push(lut[(v % radix) as usize]);
        v /= radix;
    }
    digits.reverse();
    digits
}

/// Emit a signed integer conversion (d/i): choose the sign prefix, apply the
/// precision (minimum digit count), then pad to the field width.
fn emit_signed(
    out: &mut Vec<u8>,
    digits: &[u8],
    neg: bool,
    flags: Flags,
    width: usize,
    precision: Option<usize>,
) {
    let sign: &[u8] = if neg {
        b"-"
    } else if flags.plus {
        b"+"
    } else if flags.space {
        b" "
    } else {
        b""
    };
    emit_number(out, digits, sign, flags, width, precision);
}

/// Emit an unsigned integer conversion (u/o/x/X) with an optional base prefix.
fn emit_unsigned(
    out: &mut Vec<u8>,
    digits: &[u8],
    prefix: &[u8],
    flags: Flags,
    width: usize,
    precision: Option<usize>,
) {
    emit_number(out, digits, prefix, flags, width, precision);
}

/// Shared integer emit: `prefix` (sign or base marker) then zero-extended digits
/// to `precision`, all padded to `width`. A precision of 0 with a zero value
/// yields no digits (C rule). Zero-padding is suppressed when a precision is
/// given or when left-justifying.
fn emit_number(
    out: &mut Vec<u8>,
    digits: &[u8],
    prefix: &[u8],
    flags: Flags,
    width: usize,
    precision: Option<usize>,
) {
    // Apply precision: minimum number of digits (0 precision + value 0 => none).
    let digits: Vec<u8> = match precision {
        Some(0) if digits == b"0" => Vec::new(),
        Some(p) if p > digits.len() => {
            let mut d = vec![b'0'; p - digits.len()];
            d.extend_from_slice(digits);
            d
        }
        _ => digits.to_vec(),
    };

    let body_len = prefix.len() + digits.len();
    let pad = width.saturating_sub(body_len);

    if flags.left {
        out.extend_from_slice(prefix);
        out.extend_from_slice(&digits);
        out.extend(std::iter::repeat_n(b' ', pad));
    } else if flags.zero && precision.is_none() {
        // Zero-pad goes after the prefix (e.g. "-0007", "0x00ff").
        out.extend_from_slice(prefix);
        out.extend(std::iter::repeat_n(b'0', pad));
        out.extend_from_slice(&digits);
    } else {
        out.extend(std::iter::repeat_n(b' ', pad));
        out.extend_from_slice(prefix);
        out.extend_from_slice(&digits);
    }
}

/// Emit a floating-point conversion. The magnitude is rendered by Rust's
/// formatter (correct fixed/scientific rounding); we own the sign, the field
/// width, and zero-padding to match C.
fn emit_float(out: &mut Vec<u8>, conv: u8, val: f64, flags: Flags, width: usize, precision: Option<usize>) {
    let prec = precision.unwrap_or(6);
    let neg = val.is_sign_negative();
    let mag = val.abs();

    let mut body: Vec<u8> = if mag.is_nan() {
        b"nan".to_vec()
    } else if mag.is_infinite() {
        b"inf".to_vec()
    } else {
        match conv {
            b'f' | b'F' => format!("{mag:.prec$}").into_bytes(),
            b'e' => format!("{mag:.prec$e}").into_bytes(),
            b'E' => format!("{mag:.prec$e}").to_uppercase().into_bytes(),
            // %g/%G: pick the shorter of %e/%f; approximate with Rust's default
            // shortest representation at the given precision.
            _ => {
                let p = prec.max(1);
                let s = format!("{mag:.p$}");
                // Trim trailing zeros (C %g default), keep at least one digit.
                let trimmed = trim_g(&s);
                if conv == b'G' {
                    trimmed.to_uppercase().into_bytes()
                } else {
                    trimmed.into_bytes()
                }
            }
        }
    };

    let sign: &[u8] = if neg {
        b"-"
    } else if flags.plus {
        b"+"
    } else if flags.space {
        b" "
    } else {
        b""
    };

    let total = sign.len() + body.len();
    let pad = width.saturating_sub(total);
    if flags.left {
        out.extend_from_slice(sign);
        out.append(&mut body);
        out.extend(std::iter::repeat_n(b' ', pad));
    } else if flags.zero {
        out.extend_from_slice(sign);
        out.extend(std::iter::repeat_n(b'0', pad));
        out.append(&mut body);
    } else {
        out.extend(std::iter::repeat_n(b' ', pad));
        out.extend_from_slice(sign);
        out.append(&mut body);
    }
}

/// Trim trailing zeros (and a trailing dot) from a fixed-notation number, the
/// %g default behavior.
fn trim_g(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.trim_end_matches('.').to_string()
}

/// Emit `prefix` then `body`, space-padded to `width` (no numeric semantics).
/// Used for `%c`, `%s`, `%%`, and null `%p`.
fn pad_and_emit(out: &mut Vec<u8>, prefix: &[u8], body: &[u8], flags: Flags, width: usize, _zero: bool) {
    let total = prefix.len() + body.len();
    let pad = width.saturating_sub(total);
    if flags.left {
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
        out.extend(std::iter::repeat_n(b' ', pad));
    } else {
        out.extend(std::iter::repeat_n(b' ', pad));
        out.extend_from_slice(prefix);
        out.extend_from_slice(body);
    }
}

/// Read a NUL-terminated byte string from guest memory (bounded by [`MAX_STR`]),
/// preserving raw bytes (no UTF-8 lossy substitution, unlike `read_cstr`).
fn read_cstr_bytes(ctx: &GuestCtx, addr: u32) -> Vec<u8> {
    if addr == 0 {
        return b"(null)".to_vec();
    }
    let bytes = ctx.read_bytes(addr, MAX_STR);
    let n = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    bytes[..n].to_vec()
}

#[cfg(test)]
mod tests {
    //! Verify the conversion semantics against known C `printf` output. The
    //! variadic tail is laid out here with the SAME AAPCS rule the walker reads
    //! back (core words then stack, doubles 8-byte aligned), so these cover the
    //! formatting logic; the alignment itself is proven end to end against real
    //! arm-vita-eabi-gcc output in the hello conformance test.
    use super::*;
    use crate::host::{GuestCtx, SliceMemory};
    use crate::VFP_ARG_COUNT;
    use vitaslop_transpiler::abi::{REG_COUNT, SP};

    /// A typed variadic argument, laid out per AAPCS by [`render`].
    enum Arg<'a> {
        /// A 4-byte word (int, unsigned, char, pointer value).
        W(u32),
        /// An 8-byte double (variadic float promoted).
        D(f64),
        /// A string placed in guest memory; its pointer is passed.
        S(&'a str),
    }

    /// Format `fmt` with `args`, mirroring how a compiler marshals the variadic
    /// tail, and return the produced text.
    fn render(fmt: &str, args: &[Arg]) -> String {
        let mut mem = vec![0u8; 16 * 1024];
        let fmt_addr = 0x100u32;
        mem[fmt_addr as usize..fmt_addr as usize + fmt.len()].copy_from_slice(fmt.as_bytes());

        let mut regs = [0u32; REG_COUNT];
        let mut vfp = [0u32; VFP_ARG_COUNT];
        regs[0] = fmt_addr;
        let sp = 0x2000u32; // 8-byte aligned, as at a public call
        regs[SP] = sp;

        // Place a word into the argument sequence (word i is r`i`, or the stack
        // slot at sp + (i-4)*4).
        let mut put = |mem: &mut [u8], regs: &mut [u32; REG_COUNT], word: usize, v: u32| {
            if word < 4 {
                regs[word] = v;
            } else {
                let a = (sp as usize) + (word - 4) * 4;
                mem[a..a + 4].copy_from_slice(&v.to_le_bytes());
            }
        };

        let mut w = 1usize; // word 0 is the format-string pointer
        let mut str_cursor = 0x400u32;
        for arg in args {
            match arg {
                Arg::W(v) => {
                    put(&mut mem, &mut regs, w, *v);
                    w += 1;
                }
                Arg::S(s) => {
                    let addr = str_cursor;
                    mem[addr as usize..addr as usize + s.len()].copy_from_slice(s.as_bytes());
                    str_cursor += s.len() as u32 + 1;
                    put(&mut mem, &mut regs, w, addr);
                    w += 1;
                }
                Arg::D(d) => {
                    w = (w + 1) & !1; // 8-byte align
                    let bits = d.to_bits();
                    put(&mut mem, &mut regs, w, bits as u32);
                    put(&mut mem, &mut regs, w + 1, (bits >> 32) as u32);
                    w += 2;
                }
            }
        }

        let mut sm = SliceMemory(&mut mem);
        let ctx = GuestCtx::new(&mut regs, &mut vfp, &mut sm, 0);
        let mut out = Vec::new();
        format_into(&mut out, &ctx, fmt_addr, 1);
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn plain_and_percent() {
        assert_eq!(render("Hello, world\n", &[]), "Hello, world\n");
        assert_eq!(render("100%% done", &[]), "100% done");
    }

    #[test]
    fn signed_and_unsigned() {
        assert_eq!(render("%d", &[Arg::W((-42i32) as u32)]), "-42");
        assert_eq!(render("%i", &[Arg::W(7)]), "7");
        assert_eq!(render("%u", &[Arg::W((-1i32) as u32)]), "4294967295");
    }

    #[test]
    fn radix_conversions() {
        assert_eq!(render("%x", &[Arg::W(0xBEEF)]), "beef");
        assert_eq!(render("%X", &[Arg::W(0xBEEF)]), "BEEF");
        assert_eq!(render("%o", &[Arg::W(64)]), "100");
        assert_eq!(render("%#x", &[Arg::W(0xFF)]), "0xff");
        assert_eq!(render("%#o", &[Arg::W(64)]), "0100");
        assert_eq!(render("%#x", &[Arg::W(0)]), "0"); // no prefix for zero
    }

    #[test]
    fn char_string_pointer() {
        assert_eq!(render("%c", &[Arg::W(b'!' as u32)]), "!");
        assert_eq!(render("%s", &[Arg::S("vitaslop")]), "vitaslop");
        assert_eq!(render("%p", &[Arg::W(0x8100_0000)]), "0x81000000");
        assert_eq!(render("%p", &[Arg::W(0)]), "(nil)");
    }

    #[test]
    fn width_and_flags() {
        assert_eq!(render("[%5d]", &[Arg::W(42)]), "[   42]");
        assert_eq!(render("[%-5d]", &[Arg::W(42)]), "[42   ]");
        assert_eq!(render("[%05d]", &[Arg::W(42)]), "[00042]");
        assert_eq!(render("[%+d]", &[Arg::W(42)]), "[+42]");
        assert_eq!(render("[% d]", &[Arg::W(42)]), "[ 42]");
        assert_eq!(render("[%05d]", &[Arg::W((-42i32) as u32)]), "[-0042]");
    }

    #[test]
    fn precision() {
        assert_eq!(render("%.3d", &[Arg::W(42)]), "042");
        assert_eq!(render("%.0d", &[Arg::W(0)]), ""); // precision 0 + value 0 => empty
        assert_eq!(render("%8.3d", &[Arg::W(42)]), "     042");
        assert_eq!(render("%.3s", &[Arg::S("vitaslop")]), "vit");
    }

    #[test]
    fn star_width_and_precision() {
        assert_eq!(render("%*d", &[Arg::W(5), Arg::W(42)]), "   42");
        assert_eq!(render("%.*d", &[Arg::W(3), Arg::W(42)]), "042");
        // A negative star width means left-justify.
        assert_eq!(render("[%*d]", &[Arg::W((-5i32) as u32), Arg::W(42)]), "[42   ]");
    }

    #[test]
    fn stack_spilled_ints() {
        // Six ints after the format string: the 4th..6th spill past r3 onto the
        // stack. Exercises the word cursor crossing the register/stack boundary.
        assert_eq!(
            render(
                "%d,%d,%d,%d,%d,%d",
                &[Arg::W(1), Arg::W(2), Arg::W(3), Arg::W(4), Arg::W(5), Arg::W(6)]
            ),
            "1,2,3,4,5,6"
        );
    }

    #[test]
    fn doubles_default_precision() {
        assert_eq!(render("%f", &[Arg::D(1.5)]), "1.500000");
        assert_eq!(render("%f", &[Arg::D(-3.5)]), "-3.500000");
        assert_eq!(render("%.2f", &[Arg::D(0.25)]), "0.25");
        assert_eq!(render("%8.2f", &[Arg::D(1.5)]), "    1.50");
        // Two doubles: the second spills to the stack, 8-byte aligned.
        assert_eq!(render("%f %f", &[Arg::D(1.5), Arg::D(0.25)]), "1.500000 0.250000");
    }

    #[test]
    fn long_long() {
        // %lld reads an 8-byte argument.
        assert_eq!(render("%lld", &[Arg::D(f64::from_bits(0x0000_0001_0000_0000))]), "4294967296");
    }
}
