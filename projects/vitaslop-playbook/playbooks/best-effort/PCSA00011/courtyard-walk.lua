-- GRAVITY RUSH (PCSA00011): boot to the opening apple, tap it until it falls, skip the intro
-- with START/CROSS like a player, read through the first dialogue, and then PLAY: walk Kat out
-- across the courtyard with the cat, past the slide and the swings, left and right.
-- last verified: 2026-10-26, uncommitted next-work tree (desktop headless SMP; shots viewed:
-- the apple, the falling apple, the speech bubble, the courtyard, Kat walking with the cat
-- among the swings)
--
-- HOW THE CUES WERE FOUND: shots every 40-90 frames of exploratory runs.
--
--   apple     the red apple on the branch, top left of centre - tapped at (880, 540), panel
--             coordinates, until it leaves its box
--   bubble    the white speech bubble along the bottom of the screen: present while a line is
--             up, gone when the player has control
--
-- CONTROLS (the title's own): the left stick walks; a dialogue line advances with a touch on
-- its arrow (1666, 952) or CROSS. The gravity shift (R) is not granted yet at this point of
-- the story, so the playbook walks.

local function tap_at(x, y, after)
  touch(x, y)
  idle(6)
  untouch()
  idle(after or 30)
end

local apple = cue.pixels{ x = 380, y = 180, w = 120, h = 120, rgb = { 115, 42, 33 }, tol = 25 }
local bubble = cue.pixels{ x = 300, y = 460, w = 400, h = 40, rgb = { 252, 250, 245 }, tol = 12 }

phase("apple", { timeout = 120 }, function()
  wait(apple:above(3000))
  idle(60)
  shot("apple")
  repeat tap_at(880, 540, 60) until try_wait(apple:below(300), 0.01)
  shot("fell")
end)

phase("intro", { timeout = 400 }, function()
  local n = 0
  repeat
    n = n + 1
    tap("start", 6, 40)
    tap("cross", 6, 40)
    if n % 10 == 0 then shot("intro-" .. n) end
  until try_wait(bubble:above(12000), 0.01)
  split("first dialogue")
  shot("bubble")
end)

phase("dialogue", { timeout = 300 }, function()
  local n = 0
  repeat
    n = n + 1
    tap_at(1666, 952, 40)
    tap("cross", 6, 40)
  until n >= 12 or try_wait(bubble:below(2000), 0.01)
  idle(60)
  split("control")
  shot("control")
end)

phase("the courtyard", { timeout = 900 }, function()
  stick("left", 60, 255) idle(40) stick("left", 128, 128) idle(30)
  tap("circle", 6, 60)
  tap("cross", 6, 60)
  for i = 1, 4 do idle(90) shot("courtyard-" .. i) end
  stick("left", 128, 0)
  for i = 1, 6 do idle(60) shot("up-" .. i) end
  stick("left", 0, 128)
  for i = 1, 4 do idle(60) shot("left-" .. i) end
  stick("left", 255, 128)
  for i = 1, 4 do idle(60) shot("right-" .. i) end
  stick("left", 128, 128)
  idle(60)
  shot("end")
end)
