// The game's on-screen keyboard (SceIme), typed on with the device's own keyboard.
//
// On a Vita the system draws a keyboard over the game while the game has its IME open. Here the
// worker says when that is ({type: "ime", open}), and this shows a text field over the top of
// the game. A phone raises its keyboard only for a focused text field, and only when a tap does
// the focusing, so the field says "Tap to type"; a desktop keyboard types into it straight away.
//
// What is typed is read as EDITS of the field's value (a phone keyboard composes words and
// reports its keys as "Unidentified"), turned into the keys the guest's IME understands - a
// character, Backspace, Enter, Escape, ArrowLeft, ArrowRight - and posted as {type: "text", key}.
// While the field has focus, its keys do not also reach the game's pad mapping.

export function mountImeInput(worker, host) {
  const bar = document.createElement("div");
  bar.className = "ime-bar";
  bar.hidden = true;
  const field = document.createElement("input");
  field.type = "text";
  field.autocomplete = "off";
  field.autocapitalize = "off";
  field.spellcheck = false;
  field.enterKeyHint = "done";
  field.placeholder = "Tap to type";
  field.setAttribute("aria-label", "Type for the game");
  bar.append(field);
  host.append(bar);

  const send = (key) => worker.postMessage({ type: "text", key });
  // What the field held at the last edit: an edit is diffed against it.
  let last = "";

  field.addEventListener("input", () => {
    const now = field.value;
    let same = 0;
    while (same < last.length && same < now.length && last[same] === now[same]) same++;
    for (let i = same; i < last.length; i++) send("Backspace");
    for (const c of now.slice(same)) send(c);
    last = now;
    // A long run is cleared so the field never grows without bound; nothing is sent for it.
    if (last.length > 64 && !composing) {
      field.value = "";
      last = "";
    }
  });
  let composing = false;
  field.addEventListener("compositionstart", () => (composing = true));
  field.addEventListener("compositionend", () => (composing = false));

  const editing = { Enter: 1, Escape: 1, ArrowLeft: 1, ArrowRight: 1 };
  field.addEventListener("keydown", (e) => {
    e.stopPropagation();
    // A key inside a COMPOSITION belongs to the device's own IME: Enter there picks a candidate,
    // and forwarding it would close the game's keyboard before the word lands. The committed
    // text still arrives, as an edit of the field.
    if (composing || e.isComposing || e.keyCode === 229) return;
    if (editing[e.key]) {
      e.preventDefault();
      send(e.key);
    } else if (e.key === "Backspace" && field.value === "") {
      // Nothing left in the field to delete, but the game's text may still have some.
      send("Backspace");
    }
  });
  field.addEventListener("keyup", (e) => e.stopPropagation());

  return {
    setOpen(open) {
      bar.hidden = !open;
      field.value = "";
      last = "";
      if (open) field.focus({ preventScroll: true });
      else field.blur();
    },
    destroy() {
      bar.remove();
    },
  };
}
