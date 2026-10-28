-- RESISTANCE: BURNING SKIES (PCSA00008): boot, touch the title's "Touch here to Start", accept the
-- "There is no saved data" notice with CROSS, open Single Player, start a New Game on Normal, sit
-- through the opening movie and the fire-truck ride, and at the first objective ("Break Door - press
-- R to swing your axe") break the power station's door and walk into the burning building.
-- last verified: 2026-10-25, uncommitted next-work tree (desktop headless SMP, every frame rendered;
-- shots viewed: menus with text, the street and fire truck, the door, the axe, the interior)
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
-- card, whose white text an earlier cue matched). Everything after the title is timed: each screen
-- takes its touch as soon as it is drawn, and the opening movie and ride run to the first objective
-- by about 3000 frames after New Game.

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

phase("opening", { timeout = 300 }, function()
  for i = 1, 15 do
    idle(200)
    if i % 3 == 0 then shot("opening-" .. i) end
  end
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
