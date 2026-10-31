-- KILLZONE: MERCENARY (PCSA00107): boot, confirm the CONNECTION PREFERENCES dialog, dismiss the
-- valor card, open the campaign, pick TROOPER, sit through the opening cutscene in the lobby, and
-- once the HUD is up PLAY: walk up the stairwell, look right and left, fire a burst, walk on.
-- last verified: 2026-10-27, uncommitted next-work tree (desktop headless SMP and desktop browser
-- SMP, both DONE - browser stairwell 96% speed; shots viewed: valor card, menu, difficulty, the
-- cutscene, control with the HUD up, the lit stairwell, the burst (ammo 32 -> 17), the walk on)
--
-- HOW THE CUES WERE FOUND: shots every 60-200 frames of exploratory runs. The front end is touch
-- driven; positions are panel coordinates, twice the screen's:
--
--   connection preferences  the confirm button            (690, 916)
--   valor card              the continue arrow            (1562, 826)
--   main menu               CAMPAIGN, top left            (506, 426)
--   difficulty              TROOPER, the middle of three  (960, 640)
--
-- The dialog and the valor card are cued on their teal panels. Control is cued on the in-game
-- HUD's weapon-switch ring (right edge, cyan), which is drawn only once the cutscene hands over.
-- The ring ALONE is not enough: its box takes 120-1350 px of that colour in play depending on what
-- is behind it, and single cutscene frames carry it too (a cue on it handed over control mid
-- cutscene). So control waits for the ring AND the white VALOR / BALANCE labels top left -
-- MEASURED on shots every 60 frames: labels 63-86 / 86-137 px with the HUD up, 0 in every
-- cutscene frame.
--
-- CONTROLS (the title's own): left stick moves, right stick looks, R fires.

local function tap_at(x, y, after)
  touch(x, y)
  idle(6)
  untouch()
  idle(after or 30)
end

local prefs = cue.pixels{ x = 470, y = 430, w = 180, h = 60, rgb = { 0, 110, 90 }, tol = 40 }
local valor = cue.pixels{ x = 650, y = 390, w = 160, h = 50, rgb = { 10, 80, 60 }, tol = 45 }
local hud = cue.pixels{ x = 855, y = 280, w = 70, h = 65, rgb = { 140, 225, 230 }, tol = 45 }
local valor_label = cue.pixels{ x = 36, y = 16, w = 40, h = 12, rgb = { 235, 235, 235 }, tol = 30 }
local balance_label = cue.pixels{ x = 105, y = 16, w = 55, h = 12, rgb = { 235, 235, 235 }, tol = 30 }

-- The HUD is up when the ring AND both top-left labels are drawn at once.
local function hud_up()
  return try_wait(hud:above(200), 0.01)
    and try_wait(valor_label:above(40), 0.01)
    and try_wait(balance_label:above(60), 0.01)
end

phase("preferences", { timeout = 120 }, function()
  wait(prefs:above(2000))
  idle(30)
  shot("preferences")
  tap_at(690, 916, 120)
end)

phase("valor card", { timeout = 200 }, function()
  wait(valor:above(1500))
  idle(240)
  shot("valor")
  tap_at(1562, 826, 240)
  split("main menu")
  shot("menu")
end)

phase("campaign", { timeout = 200 }, function()
  tap_at(506, 426, 200)
  shot("difficulty")
  tap_at(960, 640, 60)
  split("new game")
  local i = 0
  repeat
    idle(30)
    i = i + 1
    if i % 6 == 0 then shot("cutscene-" .. i) end
  until hud_up() or i > 200
  split("control")
  shot("control")
end)

phase("the stairwell", { timeout = 600 }, function()
  stick("left", 128, 0)
  for i = 1, 5 do idle(60) shot("walk-" .. i) end
  stick("left", 128, 128)
  stick("right", 255, 128) idle(30) stick("right", 128, 128) idle(20)
  shot("look-right")
  stick("left", 128, 0)
  for i = 6, 10 do idle(60) shot("walk-" .. i) end
  stick("left", 128, 128)
  stick("right", 0, 128) idle(60) stick("right", 128, 128) idle(20)
  shot("look-left")
  tap("r", 30, 30)
  shot("burst")
  stick("left", 128, 0)
  for i = 11, 16 do idle(60) shot("walk-" .. i) end
  stick("left", 128, 128)
  idle(60)
  shot("end")
end)
