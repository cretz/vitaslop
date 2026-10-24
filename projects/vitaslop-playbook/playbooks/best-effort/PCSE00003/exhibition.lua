-- VIRTUA TENNIS 4 (PCSE00003): close the first-run save dialogs, PRESS START, EXHIBITION with
-- the default setup (singles, Normal, random court), take the default position and players
-- (Federer v the COM's Nadal), through the tips, matchup card and fly-over onto the court, and
-- SERVE until a serve lands (the serve-speed badge shows).
-- last verified: 2026-10-24, uncommitted next-work tree (desktop headless, one-baton)
--
-- Every screen is read off the frame (960x544, measured on shots of every screen of the route):
--
--   okbtn       the purple OK button of each first-run dialog     ~5-9k px; TOUCH only - CROSS
--                                                                 does not take them
--   menubar     the main menu's dark purple bar                   6-9k px, 0 once it fades out
--   rowlit      any lit row of the EXHIBITION setup               the screen FADES IN, and a
--                                                                 DOWN pressed before a row is
--                                                                 lit is dropped (it then
--                                                                 stopped on COURT, whose CROSS
--                                                                 opens the court picker)
--   ready_lit   the READY row lit - ITS OWN COLOUR, a red-blue    ~10k px; the other rows lit
--               gradient with no green                           are (133,38,220)
--   green       the POSITION SELECT court                         ~38k px
--   badge       the serve-speed "km/h" badge                      ~158 px, only after a serve
--                                                                 lands (0 on every other shot)
--
-- CONTROLS: CROSS selects and serves (toss, then hit ~25 frames later). The player does NOT
-- run to a return on its own: a rally needs the left stick toward the ball, not done here.

local okbtn = cue.pixels{ x = 200, y = 300, w = 560, h = 120, rgb = { 162, 0, 214 }, tol = 20 }
local okat = cue.locate{ x = 200, y = 300, w = 560, h = 120, rgb = { 162, 0, 214 }, tol = 20 }
local menubar = cue.pixels{ x = 100, y = 15, w = 350, h = 380, rgb = { 127, 0, 159 }, tol = 25 }
local LIT = { 133, 38, 220 }
local rowlit = cue.pixels{ x = 30, y = 40, w = 390, h = 450, rgb = LIT, tol = 25 }
local ready_lit = cue.pixels{ x = 30, y = 448, w = 390, h = 34, rgb = { 125, 0, 192 }, tol = 20 }
local green = cue.pixels{ x = 500, y = 90, w = 400, h = 400, rgb = { 67, 178, 9 }, tol = 30 }
local badge = cue.pixels{ x = 720, y = 185, w = 140, h = 50, rgb = { 200, 53, 214 }, tol = 30 }

-- A tap in PANEL coordinates (the screen doubled): touch, hold, lift, give the screen time.
local function tap_at(x, y, after)
  touch(x, y)
  idle(6)
  untouch()
  idle(after or 30)
end

local function btn(b, after)
  tap(b, 6, after or 20)
end

phase("boot", { timeout = 180 }, function()
  -- One or two OK dialogs (Save data created, the autosave warning): tap each OK where it is.
  wait(okbtn:above(5000))
  shot("dialog")
  for _ = 1, 20 do
    if okbtn:value() < 100 then break end
    local x, y = okat:xy()
    if x then tap_at((x + 20) * 2, (y + 15) * 2, 60) end
  end
  split("dialogs closed")
  repeat btn("start", 60) until try_wait(menubar:above(3000), 1)
  shot("menu")
end)

phase("exhibition", { timeout = 180 }, function()
  -- EXHIBITION is the third entry.
  btn("down") btn("down")
  repeat btn("cross", 60) until try_wait(menubar:below(100), 1)
  wait(rowlit:above(3000))
  for _ = 1, 10 do
    if ready_lit:value() > 1500 then break end
    btn("down", 30)
  end
  check(ready_lit:above(1500), "READY is lit")
  shot("ready")
  repeat btn("cross", 60) until try_wait(green:above(20000), 1)
  shot("position")
  split("position select")
end)

phase("to the court", { timeout = 180 }, function()
  -- Position, our player, the COM's player, the tips, the matchup card, the fly-over: each a
  -- CROSS, and CROSS is ignored while a screen is still coming in, so the gaps are generous.
  btn("cross", 120)
  btn("cross", 120)
  btn("cross", 300)
  btn("cross", 150)
  btn("cross", 150)
  btn("cross", 300)
  shot("court")
end)

phase("serve", { timeout = 240 }, function()
  for _ = 1, 8 do
    btn("cross", 25)
    btn("cross", 30)
    if try_wait(badge:above(100), 3) then
      shot("served")
      split("serve landed")
      return
    end
    idle(300)
  end
  fail("no serve landed in 8 attempts")
end)
