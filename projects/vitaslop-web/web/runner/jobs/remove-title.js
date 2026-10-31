// remove-title - delete one title from the device's OPFS, as the main page's Remove does, so a
// device with a small storage quota can make room for the next title a live job needs.
//
// params: { titleId }. Removes the title's bytes (with its add-ons and rePatch overlay) and its
// library record. The saves are kept, exactly as the main page keeps them.

import { removeTitle, isComplete, storageRoom } from "../../opfs.js";
import { removeTitleRecord } from "../../store.js";

export async function run(params) {
  const id = params.titleId;
  if (!id) throw new Error("params.titleId is required");
  const had = await isComplete(id);
  await removeTitle(id);
  await removeTitleRecord(id);
  const room = await storageRoom();
  return { summary: `${id} ${had ? "removed" : "was not imported"}`, room };
}
