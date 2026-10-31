-- HELLDIVERS (PCSA00134): boot through the save notice and the intro movie, cross through the
-- front end into the TRAINING FACILITY, page through the tutorial robot's dialogs, mash cross
-- to wake the recruit, then PLAY: walk in four directions and fire after each.
-- last verified: 2026-10-28, uncommitted next-work tree (desktop headless SMP, DONE - the hub at
-- f6827, against f7543 for the timed version; shots viewed: the hub with the tutorial robot, the
-- wake prompt, the recruit up, the walk, Lesson 1)
--
-- The interpreter the title runs its game logic in is reached only through code found at run
-- time (a dispatch table built in memory), so this is also the run that exercises runtime
-- translation end to end: ~75 entries, all in the first second of the front end.
--
-- HOW THE TIMINGS WERE FOUND: shots every 90-150 input frames of exploratory runs. Cross skips
-- the epilepsy warning and the logos (~600 input frames sooner than waiting); the Bink intro
-- movie that follows does NOT skip - cross, start (tap or hold), circle, triangle, square,
-- select, R, a front tap and a rear tap were each tried mid-movie on 2026-10-28 and none moved
-- it. So the intro mashes cross until the title's PLAY bar shows, and every later step presses
-- as soon as its screen is up. START opens the controls page (circle closes it), so it is never
-- pressed here.
--
-- CONTROLS (the title's own controls page): left stick moves, R fires, cross interacts.

-- The title screen's gold PLAY bar - a FLAT bar, so the bands just above and below it are not
-- gold. The movie has gold frames too (a helmet, a desert); the bands are what tells them apart,
-- checked against every boot and movie shot of the exploratory runs. It blinks, so it is
-- waited for rather than polled once.
local play_bar = cue.pixels{ x = 360, y = 400, w = 240, h = 30, rgb = { 150, 135, 50 }, tol = 30 }
local above_bar = cue.pixels{ x = 360, y = 370, w = 240, h = 20, rgb = { 150, 135, 50 }, tol = 30 }
local below_bar = cue.pixels{ x = 360, y = 440, w = 240, h = 20, rgb = { 150, 135, 50 }, tol = 30 }
-- The hub HUD, bottom left: its red rifle icon on the dark player panel. Either alone also
-- matches a movie frame; together they hold only once the facility is up and in control.
local hud_rifle = cue.pixels{ x = 20, y = 505, w = 60, h = 25, rgb = { 200, 40, 40 }, tol = 55 }
local hud_panel = cue.pixels{ x = 0, y = 495, w = 140, h = 49, rgb = { 40, 40, 40 }, tol = 30 }

-- The conditions, declared once (each `:above` declares a cue, and these are polled in loops).
local bar_lit, above_clear, below_clear = play_bar:above(2000), above_bar:below(200), below_bar:below(200)
local rifle_lit, panel_up = hud_rifle:above(40), hud_panel:above(1500)

local function title_up()
  return try_wait(bar_lit, 0.2) and try_wait(above_clear, 0.01) and try_wait(below_clear, 0.01)
end

local function hub_up()
  return try_wait(rifle_lit, 0.3) and try_wait(panel_up, 0.01)
end

phase("boot", { timeout = 300 }, function()
  -- The warning and logos skip on cross, the movie plays out under the mashing, and the same
  -- cross then takes PLAY, LOADING and the facility banner - straight on into the hub. The
  -- title is only noted on the way (its bar blinks, so a mash can pass it unseen).
  local titled = false
  repeat
    tap("cross")
    if not titled and title_up() then
      titled = true
      split("title")
    end
  until hub_up()
  shot("hub")
end)

phase("tutorial", { timeout = 120 }, function()
  -- The tutorial robot's dialogs, each closed with cross.
  for i = 1, 12 do
    tap("cross")
    idle(90)
  end
  shot("wake-prompt")
  -- "Repeatedly tap X to wake up": a bar under the recruit fills with each tap.
  for _ = 1, 80 do
    tap("cross", 3, 3)
  end
  idle(120)
  shot("awake")
end)

-- A LESSON dialog's white "CONTINUE" label, bottom right of its panel: 246 matching pixels on
-- every dialog frame of two desktop runs, at most 94 on any other shot (2026-10-28). While one
-- is up the game takes no other input - the recruit stands still until cross closes it.
local continue_lit = cue.pixels{ x = 390, y = 410, w = 80, h = 22, rgb = { 230, 230, 230 }, tol = 30 }:above(200)
local function close_lessons()
  local n = 0
  while try_wait(continue_lit, 0.2) and n < 20 do
    tap("cross")
    idle(45)
    n = n + 1
  end
  return n
end

phase("play", { timeout = 180 }, function()
  -- Walk up, right, down, left, firing after each leg; a lesson that opens on the way is read
  -- and closed first, as a player would.
  for i, dir in ipairs({ { 128, 0 }, { 255, 128 }, { 128, 255 }, { 0, 128 } }) do
    if close_lessons() > 0 then
      split(string.format("lesson closed before leg %d", i))
    end
    stick("left", dir[1], dir[2])
    idle(60)
    shot(string.format("walk%d", i))
    idle(60)
    stick("left", 128, 128)
    press("r")
    idle(30)
    shot(string.format("fire%d", i))
    release("r")
    idle(30)
  end
  -- And past it: whatever lesson the walk opened is closed, and the recruit is shown moving
  -- again afterwards - the proof the run is still in control, not parked on a dialog.
  close_lessons()
  stick("left", 255, 128)
  idle(90)
  shot("after-lesson")
  stick("left", 128, 128)
end)
