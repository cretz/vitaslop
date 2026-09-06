// A stored (uncompressed) zip writer and reader, for bundling every title's save
// into one file and back. Entries are already zips, so compressing them again buys
// nothing, and "stored" keeps both directions to a few dozen lines with no library.

const CRC = new Int32Array(256);
for (let n = 0; n < 256; n++) {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  CRC[n] = c;
}
function crc32(bytes) {
  let c = -1;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

const enc = new TextEncoder();
const dec = new TextDecoder();

/// `entries`: [{ name, bytes: Uint8Array }] -> Uint8Array of a stored zip.
export function writeZip(entries) {
  const parts = [];
  const central = [];
  let off = 0;
  const u16 = (v) => [v & 0xff, (v >> 8) & 0xff];
  const u32 = (v) => [v & 0xff, (v >> 8) & 0xff, (v >> 16) & 0xff, (v >>> 24) & 0xff];
  for (const { name, bytes } of entries) {
    const n = enc.encode(name);
    const crc = crc32(bytes);
    const local = new Uint8Array([
      0x50, 0x4b, 0x03, 0x04, ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
      ...u32(crc), ...u32(bytes.length), ...u32(bytes.length), ...u16(n.length), ...u16(0), ...n,
    ]);
    central.push(
      new Uint8Array([
        0x50, 0x4b, 0x01, 0x02, ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
        ...u32(crc), ...u32(bytes.length), ...u32(bytes.length), ...u16(n.length), ...u16(0), ...u16(0),
        ...u16(0), ...u16(0), ...u32(0), ...u32(off), ...n,
      ])
    );
    parts.push(local, bytes);
    off += local.length + bytes.length;
  }
  const cdSize = central.reduce((a, c) => a + c.length, 0);
  const eocd = new Uint8Array([
    0x50, 0x4b, 0x05, 0x06, ...u16(0), ...u16(0), ...u16(entries.length), ...u16(entries.length),
    ...u32(cdSize), ...u32(off), ...u16(0),
  ]);
  const total = off + cdSize + eocd.length;
  const out = new Uint8Array(total);
  let p = 0;
  for (const part of [...parts, ...central, eocd]) {
    out.set(part, p);
    p += part.length;
  }
  return out;
}

/// A stored zip -> [{ name, bytes }]. Refuses compressed entries by name.
export function readZip(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 22 - 0xffff); i--) {
    if (dv.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) throw new Error("not a zip file");
  const count = dv.getUint16(eocd + 10, true);
  let cd = dv.getUint32(eocd + 16, true);
  const out = [];
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(cd, true) !== 0x02014b50) throw new Error("bad zip directory");
    const method = dv.getUint16(cd + 10, true);
    const size = dv.getUint32(cd + 24, true);
    const nameLen = dv.getUint16(cd + 28, true);
    const extraLen = dv.getUint16(cd + 30, true);
    const commentLen = dv.getUint16(cd + 32, true);
    const local = dv.getUint32(cd + 42, true);
    const name = dec.decode(bytes.subarray(cd + 46, cd + 46 + nameLen));
    if (method !== 0) throw new Error(`${name} is compressed; this bundle format is stored-only`);
    const ln = dv.getUint16(local + 26, true);
    const le = dv.getUint16(local + 28, true);
    const start = local + 30 + ln + le;
    out.push({ name, bytes: bytes.slice(start, start + size) });
    cd += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}
