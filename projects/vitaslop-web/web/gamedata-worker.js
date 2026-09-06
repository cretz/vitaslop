// A throwaway worker that says what is inside a game-data container, using the SAME parser
// the run uses.
//
// # Why a worker, and why not just check it in JS here
// The rule about what a container may contain - only the guest's own writable mounts,
// everything else refused - lives in Rust, in `vitaslop_runtime::gamedata`, because that is
// the code that will actually perform the restore. A second implementation of the rule in
// JS, written to give the user a nicer upload dialog, would be a copy that can drift; the
// day it drifted, the page would promise something the run does not do.
//
// So the page asks the real parser. It costs one wasm instantiation, which is why it
// happens in a worker that is terminated straight afterwards rather than on the page that
// is about to start a game.
//
// # This does not store anything
// It answers a question. The page decides what to do with the answer, and `gamedata.js`
// owns storage.
import init, { game_data_describe } from "./pkg/vitaslop_web.js";

const ready = init();

self.onmessage = async (e) => {
  try {
    await ready;
    // `game_data_describe` throws on anything that is not a readable container - a wrong
    // file picked by mistake, a truncated download - and the message it throws names the
    // reason. That is the whole value of asking: the user finds out now, not at the boot
    // after next.
    const summary = game_data_describe(new Uint8Array(e.data.zip));
    self.postMessage({ ok: true, summary });
  } catch (err) {
    self.postMessage({ ok: false, error: String((err && err.message) || err) });
  }
};
