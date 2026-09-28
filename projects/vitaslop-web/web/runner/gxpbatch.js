// gxpbatch.js - ONE BATCH of graphics conformance / corpus cases on THIS browser's GPU.
//
// Moved out of `e2e/gxpexec.mjs` (2026-09-26) so the SAME code runs in two places: the desktop
// harness imports it into its page with `page.evaluate`, and the device runner
// (`web/runner/`) imports it on a phone. One copy, for the reason `caseinputs.mjs` gives for
// the seeded inputs: a second copy that drifted would report its drift as a GPU defect.
//
// Needs, set up by the caller before the first batch: `globalThis.__dev` (a GPUDevice, with
// `shader-f16` when the adapter has it), `globalThis.__laneValue` (from caseinputs.mjs), and
// for render-rig cases `globalThis.__texb64` / `__texSize` / `__texUnits`.
export async function runBatch({ batch }) {
  const device = globalThis.__dev;

  // The seeded inputs, from `caseinputs.mjs` - ONE copy of the generator across the
  // runners, for the same reason the Rust side keeps one copy of `lane_value`: a second one
  // that drifted would hand the GPU different numbers and the difference would be reported
  // as a translation defect.
  const laneValue = globalThis.__laneValue;
  // One scratch pair for reading a seeded lane back as BITS - the `pa` baseline below is a
  // bit pattern, and the input buffer stores the same value through a `Float32Array`.
  const fv = new Float32Array(1);
  const uv = new Uint32Array(fv.buffer);

  const out = [];
  const pending = [];
  // Window buffers are per-case (their size varies), so unlike the pooled register buffers
  // they are destroyed once the batch has run - which is what keeps the resident set flat.
  const retired = [];
  const encoder = device.createCommandEncoder();

  // >>> THE BUFFERS ARE A POOL, REUSED FOR THE LIFE OF THE RUN. Allocating three per case and
  // leaving them to the collector cost 0.6 s a case over 670 - the run grew a buffer set it
  // never released, and the driver slowed down under it. Every case has the same fixed layout
  // (one bank-pair in, three banks out), so one slot per batch position serves all of them.
  // >>> THE RENDER RIG'S TEXTURE SET, BUILT ONCE FOR THE WHOLE RUN. A texture per case would
  // be thousands of allocations for bytes that never change, and the same resident-set growth
  // the buffer pool below exists to avoid.
  const texSet = () => {
    if (globalThis.__texset) return globalThis.__texset;
    // >>> THE 1x1 COLOUR TARGET IS WHAT MAKES A RENDER CASE A RENDER CASE; the stand-in
    // TEXTURES are what makes one that SAMPLES. A directory whose only render cases are the
    // authored discard and written-depth ones carries no texel file at all, and demanding one
    // refused the whole directory over a binding no case in it declares.
    const target = device.createTexture({
      size: [1, 1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    if (!globalThis.__texb64) {
      globalThis.__texset = { views: [], cubeViews: [], sampler: null, target: target.createView() };
      return globalThis.__texset;
    }
    const bin = atob(globalThis.__texb64);
    const bytes = new Uint8Array(bin.length);
    for (let j = 0; j < bin.length; j++) bytes[j] = bin.charCodeAt(j);
    const size = globalThis.__texSize;
    const units = globalThis.__texUnits;
    const per = size * size * 4;
    const views = [];
    for (let u = 0; u < units; u++) {
      const t = device.createTexture({
        size: [size, size, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: t },
        bytes.subarray(u * per, (u + 1) * per),
        { bytesPerRow: size * 4, rowsPerImage: size },
        [size, size, 1],
      );
      views.push(t.createView());
    }
    // The CUBE set: six faces per unit, contiguous in the file, uploaded as one depth-6 write
    // and viewed as a cube. A unit is a flat texture in one program and a cube in another, so
    // both exist for every unit and the CASE says which its module declared.
    const cubeViews = [];
    const cubeBase = units * per;
    for (let u = 0; u < units; u++) {
      const t = device.createTexture({
        size: [size, size, 6],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      });
      device.queue.writeTexture(
        { texture: t },
        bytes.subarray(cubeBase + u * per * 6, cubeBase + (u + 1) * per * 6),
        { bytesPerRow: size * 4, rowsPerImage: size },
        [size, size, 6],
      );
      cubeViews.push(t.createView({ dimension: "cube" }));
    }
    // >>> NEAREST, CLAMP, NO MIPS - the one sampler configuration whose result is an EXACT
    // function of the coordinate. Anything filtered or mipped is a number the specification
    // lets the device approximate, and holding it to a CPU model would report that permitted
    // freedom as a translation defect. See `wrap_render_case_module_for`.
    const sampler = device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
      mipmapFilter: "nearest",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    });
    globalThis.__texset = { views, cubeViews, sampler, target: target.createView() };
    return globalThis.__texset;
  };

  const n = batch[0].lanes;
  // FOUR banks: r, o, i and pa. `pa` is an output too - a fragment program's colour can land
  // there (`ColorOutput::NonNativePa`), and while it went uncaptured such a program's entire
  // effect was invisible to this comparison.
  //
  // A RENDER case adds two words - the kill flag and the fragment depth - and the pool is
  // sized for the larger so one slot serves both rigs. A compute case's copy stops at its own
  // four banks, so the two stale words can never reach a comparison.
  const outBytes = n * 4 * 4 + 8;
  if (!globalThis.__pool || globalThis.__poolLanes !== n) {
    globalThis.__pool = [];
    globalThis.__poolLanes = n;
  }
  const slot = (k) => {
    if (!globalThis.__pool[k]) {
      globalThis.__pool[k] = {
        inBuf: device.createBuffer({ size: n * 2 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }),
        outBuf: device.createBuffer({ size: outBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC }),
        readBuf: device.createBuffer({ size: outBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
      };
    }
    return globalThis.__pool[k];
  };

  // ONE error scope for the whole batch rather than one per case: popping is a round trip to
  // the GPU process, and awaiting it per case was the second half of the cost. A batch that
  // reports an error re-runs its cases one at a time to attribute it.
  device.pushErrorScope("validation");
  for (let k = 0; k < batch.length; k++) {
    const c = batch[k];
    const mod = device.createShaderModule({ code: c.src });
    const isRender = c.stage === "fragment";
    let pipeline = null;
    try {
      pipeline = isRender
        ? device.createRenderPipeline({
            layout: "auto",
            vertex: { module: mod, entryPoint: "vs_main" },
            fragment: { module: mod, entryPoint: "fs_main", targets: [{ format: "rgba8unorm" }] },
            primitive: { topology: "triangle-list", cullMode: "none" },
          })
        : device.createComputePipeline({
            layout: "auto",
            compute: { module: mod, entryPoint: "cs_main" },
          });
    } catch (e) {
      const info = await mod.getCompilationInfo();
      out.push({
        name: c.name,
        status: "compile",
        messages: info.messages
          .filter((m) => m.type === "error")
          .map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
          .slice(0, 3),
      });
      continue;
    }

    const { inBuf, outBuf, readBuf } = slot(k);
    const input = new Float32Array(n * 2);
    for (let j = 0; j < n; j++) {
      input[j] = laneValue(c.seed, j);
      input[n + j] = laneValue(c.seed, n + j);
    }
    // The lanes this program reads as a count, an index or a pointer, chosen by the case
    // writer and carried as data - see `caseinputs.mjs`. AFTER the seeded fill, because they
    // replace it; the module's own prologue then puts the window bases on top, which is the
    // same order the reference used.
    globalThis.__applyCaseInputs(c, input);
    device.queue.writeBuffer(inBuf, 0, input);
    // NO CLEAR of the pooled output buffer, deliberately: the wrapper's epilogue writes
    // EVERY lane of all three banks unconditionally from `var<private>` arrays that WGSL
    // zero-initialises per dispatch, so nothing of the previous case survives. (A
    // `clearBuffer` here is not merely redundant - the pooled buffer carries no COPY_DST
    // usage, so it fails `finish()` and silently voids the whole batch's submit, which reads
    // back as every case computing zero. That is what the batch guard below now catches.)
    // A program with 0xE8 loads reads its bound guest-memory WINDOW at binding 2. The words
    // come from the case file, which is also what the reference interpreter was handed, so
    // there is one copy of the bytes rather than two generators that have to agree.
    const entries = [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: outBuf } },
    ];
    if (c.mem && c.mem.length) {
      const mem = new Uint32Array(c.mem);
      const memBuf = device.createBuffer({
        size: mem.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(memBuf, 0, mem);
      entries.push({ binding: 2, resource: { buffer: memBuf } });
      retired.push(memBuf);
    }
    // A render case's textures go at binding 4 + 2k / 5 + 2k, in the case's own ASCENDING
    // unit order - which is the order the module declared them in, and the only thing that
    // says which stand-in texture is unit 7's.
    let tex = null;
    if (isRender) {
      tex = texSet();
      (c.units ?? []).forEach((u, k) => {
        const isCube = (c.cubes ?? [])[k] === 1;
        entries.push({ binding: 4 + 2 * k, resource: isCube ? tex.cubeViews[u] : tex.views[u] });
        entries.push({ binding: 5 + 2 * k, resource: tex.sampler });
      });
    }
    const bind = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    if (isRender) {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: tex.target,
            loadOp: "clear",
            storeOp: "store",
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      // Three vertices, one triangle, the whole clip square over a 1x1 target - so exactly
      // one invocation is real and the rest of its quad are helpers, whose stores do not land.
      pass.draw(3);
      pass.end();
    } else {
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(1);
      pass.end();
    }
    encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, outBytes);
    pending.push({ c, readBuf });
  }

  // ONE submit for the whole batch, then every readback mapped in parallel.
  device.queue.submit([encoder.finish()]);
  const scoped = await device.popErrorScope();
  // >>> A BATCH WHOSE SUBMIT DID NOT HAPPEN MUST NOT REPORT RESULTS. An encoder error voids
  // the WHOLE command buffer, so every case in the batch reads its output buffer back
  // untouched - which compares as "the GPU computed zero" and would be reported as hundreds
  // of divergences that are entirely this script's fault. (It already happened once: an
  // invalid `clearBuffer` voided all seven batches and turned a 9-second run into 220
  // fabricated findings.) The batch is returned as INVALID and the run refuses to pass.
  if (scoped) {
    return { invalid: batch.map((c) => c.name), error: scoped.message.split("\n").slice(0, 4).join(" | ") };
  }
  await Promise.all(pending.map((p) => p.readBuf.mapAsync(GPUMapMode.READ)));

  for (const { c, readBuf } of pending) {
    const got = new Uint32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    // A RENDER case's answer is two words longer: the kill flag and the fragment depth. A
    // compute case stops at its four banks, so the pooled buffer's two stale words - which
    // nothing wrote this dispatch - are never read.
    const words = c.lanes * 4 + (c.stage === "fragment" ? 2 : 0);
    const want = new Uint32Array(words);
    // 0 = one f32, 1 = two packed f16s, 2 = four unorm bytes, 3 = a raw word compared
    // EXACTLY (the kill flag: "within one ULP of killed" is not a thing).
    const prec = new Uint8Array(words);
    // >>> `pa`'s BASELINE IS THE SEEDED INPUT, NOT ZERO. Every other bank starts zeroed on
    // both sides, so a lane nobody wrote needs no transcription. `pa` starts holding the
    // case's seeded values, so the case carries only the lanes the program CHANGED and the
    // rest are rebuilt here from the same seed - which keeps "every lane is compared" true
    // without a 512-lane expectation per case.
    // The baseline is the seeded fill WITH the overrides, because that is what the case
    // actually started `pa` at. A baseline that used the seed alone would read every
    // overridden lane as a lane the program changed.
    const paBase = new Float32Array(c.lanes);
    for (let j = 0; j < c.lanes; j++) paBase[j] = laneValue(c.seed, j);
    globalThis.__applyCaseInputs(c, paBase);
    const paBaseBits = new Uint32Array(paBase.buffer);
    for (let j = 0; j < c.lanes; j++) {
      want[c.lanes * 3 + j] = paBaseBits[j];
    }
    for (const [idx, bits, p] of c.expect) {
      want[idx] = bits >>> 0;
      prec[idx] = p;
    }

    const f = new Float32Array(1);
    const u = new Uint32Array(f.buffer);
    const asF32 = (bits) => {
      u[0] = bits >>> 0;
      return f[0];
    };
    // Decode one IEEE binary16 out of a packed word (0 = low half, 1 = high half).
    const halfBits = (bits, half) => (half ? (bits >>> 16) & 0xffff : bits & 0xffff);
    const asF16 = (bits, half) => {
      const x = halfBits(bits, half);
      const sign = x & 0x8000 ? -1 : 1;
      const exp = (x >>> 10) & 0x1f;
      const frac = x & 0x3ff;
      if (exp === 0) return sign * frac * Math.pow(2, -24);
      if (exp === 31) return frac ? NaN : sign * Infinity;
      return sign * (1 + frac / 1024) * Math.pow(2, exp - 15);
    };

    // >>> DISTANCE IN ULPs, NOT RELATIVE ERROR, IS WHAT SEPARATES A ROUNDING FROM A DEFECT.
    // A relative-error bound has to be guessed per magnitude and says nothing about how many
    // representable values lie between the two answers. Mapping a float's bits to a
    // sign-magnitude ordinal makes "how many representable values apart" a subtraction - so
    // "the last bit differs" and "a lane was dropped" stop being the same measurement. One
    // ULP of binary16 near 50 is 0.03; near 10000 it is 8. Both are one rounding.
    const ordinal = (bits, signBit) => {
      const neg = (bits & signBit) !== 0;
      const mag = bits & (signBit - 1);
      return neg ? -mag : mag;
    };
    const ulpsApart = (a, b, signBit) => Math.abs(ordinal(a, signBit) - ordinal(b, signBit));

    let worst = 0;
    const bad = [];
    let inexact = 0;
    // Lanes graded leniently because the case writer could not say which view they hold.
    let ambiguous = 0;
    // Lanes where one side held a subnormal and the other an exact zero - a flush the
    // specification permits. Counted per case, reported in total; see `judgeAt`.
    let flushedHere = 0;
    // >>> COMPARE A LANE IN THE VIEW IT ACTUALLY HOLDS. A word carrying two F16s read as one
    // f32 turns a one-ULP difference in the high half into a difference of thousands in a
    // number neither side ever computed, and hides a difference in the low half completely.
    // The tolerances differ per view for the same reason: binary16 carries about three
    // decimal digits, so a rounding difference there is ~1e-3, not ~1e-7.
    for (let k = 0; k < want.length; k++) {
      if (want[k] === got[k]) continue;
      // Temps 248 and up (the `r` bank is lanes 0..c.lanes) cannot be NAMED by any encoding -
      // the doubled field's top codes select the internal registers instead - so no program
      // observes them. The decoder parks a dual-issue word's first result there when each of
      // its operations reads the other's destination; the Rust suite skips them the same way.
      if (k >= 248 && k < c.lanes) continue;
      const p = prec[k];
      if (p === 3) {
        // A FLAG. There is no tolerance to apply and no float view to read it in.
        if (bad.length < 6) {
          bad.push({ lane: k, want: want[k], got: got[k], ulps: Infinity, view: "flag", rawWant: want[k], rawGot: got[k] });
        }
        worst = Infinity;
        continue;
      }
      if (p === 2) {
        // Four unorm bytes: agreement is within one quantisation step per channel.
        let off = 0;
        for (let byte = 0; byte < 4; byte++) {
          const a = (want[k] >>> (8 * byte)) & 0xff;
          const b = (got[k] >>> (8 * byte)) & 0xff;
          off = Math.max(off, Math.abs(a - b));
        }
        if (off <= 1) inexact++;
        else {
          if (off > worst) worst = off;
          if (bad.length < 6) bad.push({ lane: k, want: want[k], got: got[k], ulps: off, view: "u8x4", rawWant: want[k], rawGot: got[k] });
        }
        continue;
      }
      // >>> A LANE WHOSE VIEW IS NOT STATICALLY DETERMINED IS GRADED IN BOTH, AND AGREES IF
      // >>> EITHER DOES.
      //
      // Code 4 means the writer that decided the lane's view sits behind a predicate or a
      // branch, so the case writer cannot say whether the word holds one f32 or two halves -
      // see `written_lane_precision`. Grading it in the wrong one manufactures a divergence:
      // two NaN halves that agree, read as an f32, came out as 4,227,955,712 ULP.
      //
      // This is WEAKER than a determined lane and is counted as such below.
      const views = p === 4 ? [1, 0] : [p];
      if (p === 4) ambiguous++;
      let laneBad = null;
      for (const view of views) {
        const attempt = judgeAt(k, view);
        if (!attempt) {
          laneBad = null;
          break;
        }
        // Keep the FIRST view's report, which is the packed one - the view an ambiguous lane
        // is in whenever this rule actually mattered.
        laneBad = laneBad ?? attempt;
      }
      if (!laneBad) {
        inexact++;
      } else {
        if (laneBad.ulps > worst) worst = laneBad.ulps;
        if (bad.length < 6) bad.push(laneBad);
      }
    }

    // The per-view comparison of ONE lane: `null` when the two sides agree in that view,
    // otherwise the report. A function so an ambiguous lane can ask it twice.
    function judgeAt(k, p) {
      const halves = p === 1 ? [0, 1] : [null];
      let laneBad = null;
      for (const half of halves) {
        const a = half === null ? asF32(want[k]) : asF16(want[k], half);
        const b = half === null ? asF32(got[k]) : asF16(got[k], half);
        if (a === b) continue;
        const view = p === 1 ? `f16[${half}]` : "f32";
        // >>> A SUBNORMAL AGAINST A ZERO IS A PERMITTED FLUSH, NOT A TRANSLATION DEFECT.
        //
        // WGSL lets an implementation flush subnormal values to zero, and every backend this
        // runner reaches does. So a reference of `0x0000000b` against a GPU zero is the
        // hardware exercising a freedom the specification grants it - holding the emitter to
        // the other choice reports conformance as a bug.
        //
        // MEASURED, and it cost a whole corpus run: a change that fed programs small integer
        // bit patterns turned 5 divergences into 23, and **18 of the new ones were this and
        // nothing else** - each reported as 5 to 11 ULP of translation error against an
        // emitter that was right every time.
        //
        // It is COUNTED and printed, because it is genuinely weaker coverage: inside this
        // window a dropped write and a flush look identical, and so does every value a
        // subnormal expectation could have been. That is a hole the hardware creates and this
        // cannot close - but an uncounted one would read as the differential getting cleaner.
        const flushed = (x, y) =>
          y === 0 && x !== 0 && Math.abs(x) < (half === null ? 1.1754944e-38 : 6.103515625e-5);
        if (flushed(a, b) || flushed(b, a)) {
          flushedHere++;
          continue;
        }
        // A non-finite on one side and not the other is a real divergence; both non-finite is
        // agreement (which NaN payload a unit produces is not our translation's business).
        if (!Number.isFinite(a) || !Number.isFinite(b)) {
          if (!Number.isFinite(a) && !Number.isFinite(b)) continue;
          laneBad = { lane: k, want: a, got: b, ulps: Infinity, view, rawWant: want[k], rawGot: got[k] };
          break;
        }
        const ulps =
          half === null
            ? ulpsApart(want[k], got[k], 0x80000000)
            : ulpsApart(halfBits(want[k], half), halfBits(got[k], half), 0x8000);
        // >>> A LANE'S BOUND IS SET BY THE NARROWEST PRECISION THE PROGRAM USES, NOT BY THE
        // WIDTH OF ITS OWN STORE. An f32 lane in a program that did its arithmetic in
        // half-precision holds an f16-rounded value widened, so one upstream f16 rounding
        // shows up as thousands of f32 ULPs - 19 cases sat at exactly 8192 of them, which is
        // one f16 ULP wearing an f32 measurement. Comparing such a lane at f32 tightness asks
        // for precision the value never carried.
        //
        // The bound is in REPRESENTABLE VALUES, so it means the same thing at every
        // magnitude. Four of them absorbs a differently-associated multiply-add, a reciprocal
        // computed to a different last bit, and a driver that evaluates a packed expression at
        // half precision. It does not absorb a dropped lane, a swapped operand or an index off
        // by one - which is what this harness exists to find.
        const f16Scale = c.half > 0 && half === null;
        if (f16Scale) {
          // f32 lane, f16-precision value: bound it at four binary16 steps, relatively.
          const rel = Math.abs(a - b) / Math.max(1e-6, Math.abs(a), Math.abs(b));
          if (rel <= 4 * Math.pow(2, -11)) continue;
        } else if (ulps <= 4) continue;
        if (!laneBad || ulps > laneBad.ulps) {
          // >>> THE RAW WORDS TRAVEL WITH THE LANE. A decoded float cannot be read back as
          // bits, and the bits are what say which KIND of disagreement this is: two NaNs
          // differing only in their sign, one f16 ULP in a high half, or an actual wrong
          // number. Two "catastrophic" divergences of 4.2e9 ULP were `0xfe00fe00` against
          // `0x7e007e00` - both halves a NaN on both sides, differing in the sign bit alone,
          // read as an f32 because the lane's last writer was an f32 move.
          laneBad = { lane: k, want: a, got: b, ulps, view, rawWant: want[k], rawGot: got[k] };
        }
      }
      return laneBad;
    }
    // Name the bank a diverging lane belongs to here, where the layout is known, so the
    // report does not have to re-derive it.
    for (const b of bad) {
      // The two fragment-stage results sit past the four banks and are named, not indexed:
      // "the pixel was killed and the reference says it was not" has to read as that.
      if (b.lane === c.lanes * 4) {
        b.bank = "kill";
        b.reg = "";
      } else if (b.lane === c.lanes * 4 + 1) {
        b.bank = "depth";
        b.reg = "";
      } else {
        b.bank = ["r", "o", "i", "pa"][Math.floor(b.lane / c.lanes)] ?? "?";
        b.reg = b.lane % c.lanes;
      }
    }
    if (bad.length) out.push({ name: c.name, status: "diverged", worst, bad, instrs: c.instrs, kind: c.kind, half: c.half, ambiguous, flushed: flushedHere });
    else out.push({ name: c.name, status: inexact ? "close" : "exact", inexact, half: c.half, ambiguous, flushed: flushedHere });
  }
  for (const b of retired) b.destroy();
  return out;
}
