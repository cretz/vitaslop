-- FIFA Soccer 13 (PCSE00093): boot, the first-run profile and tips, the practice arena, then
-- KICK-OFF > EXHIBITION MATCH with every default (Chicago Fire v Columbus Crew), the broadcast
-- intro, and a live match played with the face buttons and the left stick.
--
-- Converted from the old frame-keyed script `exhibition-match`, which was already timed by host calls
-- (`@wait call ... pulse`): each hold below is one of its waits. Frames are from its native run:
--
--   f3030  title "Press START button"; START -> the autosave notice (game-drawn)
--   f3520  CROSS answers it: the save slots are probed (sceAppUtilSaveDataSlotGetParam)
--   ...    profile setup (skill / controls / favourite team), five tip pages, a tips dialog
--          - all CROSS; sceNetCtlInetGetInfo is called once the first-run flow is over
--   f6637  the practice ARENA builds its models (a burst of sceGxmPrecomputedDrawInit)
--   f6782  START in the arena opens the main menu, which builds its background the same way
--   f11379 the stadium draws (sceGxmDrawInstanced, first) - "PRESS START BUTTON TO PLAY"
--   f13555 START: sceGxmDrawInstanced again, the broadcast intro begins
--   f17600 KICK-OFF
--
-- Each hold PULSES the button its screens advance with, so a screen that is already up, or
-- one that opens late, is answered either way. START is only ever pulsed into a hold that a
-- START ends: in a match it pauses.
--
-- CONTROLS (Classic, the first-run default): left stick move, R sprint, CROSS short pass,
-- CIRCLE shoot, SQUARE lob, TRIANGLE through ball.

-- Tap `button` every `gap` input frames until `name` is called. The cue is declared HERE, so
-- it counts only calls from this hold on (a call cue counts from its own declaration).
local function pulse_until(name, button, gap)
  local called = cue.call(name):happened()
  repeat tap(button, 2, gap - 2) until try_wait(called, 0.01)
end

phase("boot", { timeout = 180 }, function()
  -- The title takes START and ignores CROSS; the autosave notice behind it takes CROSS and
  -- ignores START; and the calls around the intro movie differ between engines (its file
  -- closes ~f2660 natively, before f1500 in the browser). So one hold pulses BOTH from the
  -- legal screens on: START also skips the movie, and whichever screen is up takes its own.
  pulse_until("sceAppUtilSaveDataSlotGetParam", "start+cross", 60)
  split("autosave answered")
  -- Three profile pages, five tip pages and a dialog, all CROSS.
  pulse_until("sceNetCtlInetGetInfo", "cross", 60)
  split("first-run done")
end)

phase("menus", { timeout = 240 }, function()
  -- START in the arena opens the main menu; a START during the arena's load is ignored, so
  -- the first hold ends on the arena's own model build and the second on the menu's.
  pulse_until("sceGxmPrecomputedDrawInit", "start", 120)
  split("practice arena")
  -- The arena's build is a burst; let it end, or its tail would release the next hold.
  idle(30)
  shot("practice-arena")
  pulse_until("sceGxmPrecomputedDrawInit", "start", 120)
  split("main menu")
  idle(30)
  shot("main-menu")
  -- KICK-OFF > EXHIBITION MATCH, controller select, both teams, both kits and PLAY MATCH are
  -- all the default item, so CROSS walks the chain until the stadium starts drawing.
  pulse_until("sceGxmDrawInstanced", "cross", 90)
  split("stadium")
  idle(60)
  shot("press-start")
  -- "PRESS START BUTTON TO PLAY": the next instanced draw comes once START was taken.
  idle(30)
  pulse_until("sceGxmDrawInstanced", "start", 120)
  split("broadcast intro")
end)

-- One cycle of the old script's match loop (400 display frames): sprint right and pass, run
-- up-right and shoot, cut left and play a through ball, sprint down-right and lob. MEASURED:
-- this title reads the pad on every other display flip, so the old script's gaps are halved.
local function cycle()
  stick("left", 255, 128); press("r"); idle(45); release("r")
  tap("cross", 6, 0)
  stick("left", 200, 40); idle(44)
  tap("circle", 6, 0)
  stick("left", 60, 128); idle(39)
  tap("triangle", 6, 0)
  stick("left", 255, 200); press("r"); idle(39); release("r")
  tap("square", 6, 0)
  stick("left", 128, 128); idle(9)
end

phase("match", { timeout = 400 }, function()
  -- The broadcast intro runs ~4000 frames (walk-out, line-ups, both formations); CROSS skips a
  -- shot of it where it can, and in play it is a short pass - so the loop starts at once.
  local _, _, start = now()
  local n = 0
  repeat
    cycle()
    n = n + 1
    if n % 5 == 0 then shot(string.format("match-%02d", n // 5)) end
    local _, _, t = now()
  until t - start > 200
end)
