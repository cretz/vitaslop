-- RESISTANCE: BURNING SKIES (PCSA00008): boot, touch the title's "Touch here to Start", accept the
-- "There is no saved data" notice with CROSS, open Single Player, start a New Game on Normal, sit
-- through the opening movie and the fire-truck ride, and at the first objective ("Break Door - press
-- R to swing your axe") break the power station's door and walk into the burning building.
-- last verified: 2026-10-28, uncommitted next-work tree (desktop headless SMP, DONE - control at
-- f4637 on the HUD cue; shots viewed: the first objective, the door broken, the interior)
--
-- HOW THE CUES WERE FOUND: shots every 100-200 frames of exploratory runs. The front end is touch
-- driven (every menu says "Select by using the screen"); positions are panel coordinates, twice the
-- screen's:
--
--   title         "Touch here to Start" box                 (960, 756)
--   main menu     Single Player, top of the left column      (540, 360)
--   single player New Game, top of the left column           (534, 448)
--   difficulty    Normal, the middle of three                (960, 602)
--
-- The title is cued on its orange logo (the first frames are the seizure warning and a publisher
-- card, whose white text an earlier cue matched). The menus after the title are timed: each screen
-- takes its touch as soon as it is drawn. The opening movie and ride are waited out on the in-game
-- HUD's axe badge (below), ~3000 frames after New Game.

local function tap_at(x, y, after)
  touch(x, y)
  idle(6)
  untouch()
  idle(after or 30)
end

local logo = cue.pixels{ x = 100, y = 90, w = 760, h = 60, rgb = { 110, 35, 0 }, tol = 40 }

phase("title", { timeout = 120 }, function()
  wait(logo:above(3000))
  idle(120)
  shot("title")
  tap_at(960, 756, 240)
  shot("no-save-data")
  tap("cross", 6, 200)
  split("main menu")
  shot("main-menu")
end)

phase("new game", { timeout = 120 }, function()
  tap_at(540, 360, 150)
  shot("single-player")
  tap_at(534, 448, 200)
  shot("difficulty")
  tap_at(960, 602, 200)
  split("new game")
end)

-- The in-game HUD's axe badge (right edge): ~1,524 white pixels inside ~6,790 dark ones. The
-- PAIR, each in a narrow band, is the cue - one movie frame has the white count (1,526) but only
-- 5,426 dark. Measured on every shot of two runs (2026-10-28): the HUD reads 6,789-6,795.
local axe_white = cue.pixels{ x = 850, y = 310, w = 100, h = 95, rgb = { 220, 220, 220 }, tol = 70 }
local axe_dark = cue.pixels{ x = 850, y = 310, w = 100, h = 95, rgb = { 30, 30, 30 }, tol = 30 }
local white_lo, white_hi = axe_white:above(1400), axe_white:below(1650)
local dark_lo, dark_hi = axe_dark:above(6500), axe_dark:below(7100)

local function hud_up()
  return try_wait(white_lo, 0.3) and try_wait(white_hi, 0.01) and try_wait(dark_lo, 0.01)
    and try_wait(dark_hi, 0.01)
end

phase("opening", { timeout = 300 }, function()
  -- The opening movie and the fire-truck ride do not skip: start (tap or hold), a front tap,
  -- cross, circle, select and triangle were each tried mid-movie on 2026-10-28. So it is waited
  -- out, and play starts the moment the HUD is drawn.
  local n = 0
  repeat
    n = n + 1
    if n % 200 == 0 then shot(string.format("opening-%02d", n // 200)) end
  until hud_up()
  split("first objective")
  shot("break-door")
end)

phase("the door", { timeout = 120 }, function()
  for i = 1, 4 do tap("r", 8, 60) end
  shot("door-broken")
  stick("left", 128, 0)
  idle(360)
  stick("left", 128, 128)
  shot("inside")
end)
