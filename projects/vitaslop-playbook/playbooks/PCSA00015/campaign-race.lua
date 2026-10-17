-- WipEout 2048 (PCSA00015): boot through the attract demo, PRESS ANY BUTTON, GAME MODE ->
-- SINGLE PLAYER CAMPAIGN, the campaign intro, the campaign MAP, the first event (EMPIRE CLIMB),
-- its briefing and START, the load, the pre-race RACE card, and a LIVE RACE (LAP 1/3, POS 8/8).
--
-- Converted from the recipes `boot` (the boot ladder), `front-end` (to the campaign map) and
-- `campaign-race` (to the live race). Each recipe's end point is a phase here.
--
-- HOW THE CUES WERE FOUND: `--discover` on `campaign-race`'s run (deterministic, 10000 frames)
-- next to a shot every 200 frames, and a value search over dumps taken twice on each of six
-- screens. The front end is preloaded, so only the event makes calls (frames from that run):
--
--   f5200  (screen: a heap object)              title, PRESS ANY BUTTON TO START
--   f5750  (screen: the next slot)              GAME MODE, four tiles
--   f6000  (screen: the next slot)              the campaign intro ("It's the year 2048...")
--   f6700  (screen: the next slot)              the campaign MAP, first node TOUCH TO START
--   f7000  load trackscreens/Tower.gxt          the EMPIRE CLIMB briefing
--   f8403  load environments/tower/track        START: the race loads (CONTROLS/RACER legend)
--   f9409  load audio/music/01                  the race music, part way through the load; the
--                                               pre-race RACE card follows, and its checkmark
--                                               starts the race (judge a live race by the HUD
--                                               - the flyby behind the card is a world scene)
--
-- "screen" is the u32 at 0x8152c468: the screen manager's top screen, a pointer into a pool of
-- screen slots 0x228 bytes apart in the image's data - the next slot for each screen pushed,
-- so its value says how many screens came before, not which screen is up. Only the title is a
-- heap object. So the title is recognised by its address, and after it each front-end screen
-- is left by ONE input and waited out as a CHANGE of the pointer. (First drafts waited for
-- fixed slot addresses taken from one run, and stalled in another whose boot pushed one screen
-- fewer.)
--
-- A CROSS during the logos skips the attract movie entirely (the title is up by ~f400).
--
-- THE TITLE SCREEN WANTS A BUTTON; everything after it is TOUCH (panel coordinates are screen
-- x2). The map's node is tapped ONCE: the gesture recognizer defaults to type 1 (a tap), and
-- reporting all types made the map only pan.

local RESOLVE = "_sceFiosKernelOverlayResolveWithRangeSync"
local function loads(fragment) return cue.call(RESOLVE, { arg = 2, has = fragment }):happened() end

local screen     = cue.u32(0x8152c468):trace("screen")
local on_title   = screen:atleast(0x82000000)
local briefing = loads("trackscreens/Tower.gxt")
local race_ld  = loads("environments/tower/track.rcsmodel")
local live     = loads("audio/music/")

local function touch_tap(x, y)
  touch(x, y); idle(20); untouch()
end

-- Repeat `act()` until the top screen is no longer the one that was up when this began.
local function leave_screen(act)
  local from = screen:value()
  repeat act(); idle(60) until screen:value() ~= from
end

phase("boot", { timeout = 240 }, function()
  -- Loading, the logos and the attract movie; a CROSS skips them to the title screen.
  repeat tap("cross", 20, 0); idle(40) until try_wait(on_title, 0.01)
  split("title")
  idle(30)
  shot("title-screen")
  -- "PRESS ANY BUTTON TO START" - a button; a touch does not take it.
  leave_screen(function() tap("cross", 20, 0) end)
  split("game mode")
  idle(60)
  shot("game-mode")
end)

phase("front-end", { timeout = 120 }, function()
  -- The SINGLE PLAYER CAMPAIGN tile (screen 236,262). ONLINE CAMPAIGN and CROSS-PLAY are
  -- greyed out (signed out); ADHOC needs a second unit.
  leave_screen(function() touch_tap(472, 524) end)
  idle(60)
  shot("campaign-intro")
  -- The intro's confirm checkmark, bottom right (screen 882,480).
  leave_screen(function() touch_tap(1764, 960) end)
  split("campaign map")
  idle(120)
  shot("campaign-map")
end)

phase("event", { timeout = 180 }, function()
  -- The first event node (screen 508,285), the only unlocked one.
  repeat touch_tap(1016, 570); idle(100) until try_wait(briefing, 0.01)
  split("briefing")
  idle(120)
  shot("event-briefing")
  -- EMPIRE CLIMB / NO WEAPONS: a ship button, a cancel X and START along the bottom
  -- (screen 622/752/883, 480). START.
  repeat touch_tap(1766, 960); idle(100) until try_wait(race_ld, 0.01)
  split("race loading")
  idle(60)
  -- The CONTROLS/RACER legend: STEER/PITCH left stick, R THRUST, X FIRE/USE PICKUP.
  shot("event-loading")
  -- MEASURED: the race music loads at ~26% of this load, so it only says the load is well on.
  -- The load screen then gives way to the RACE card (over a flyby of the circuit) on its own;
  -- the card's checkmark starts the race, and is tapped until the card's screen goes.
  wait(live)
  local loading = screen:value()
  wait(screen:isnt(loading))
  idle(60)
  shot("pre-race")
  leave_screen(function() touch_tap(1762, 960) end)
  split("race")
end)

phase("race", { timeout = 60 }, function()
  idle(120)
  shot("race-start")
  -- Hold THRUST off the line: the frame is a race frame (a 1024x1024 shadow pass, reflection
  -- and post targets, a world pass of several hundred draws into an offscreen target).
  press("r"); idle(240)
  shot("racing")
  idle(240)
  shot("racing-2")
  release("r")
end)
