-- HUSTLE KINGS (PCSA00007): boot through the logos to the title, answer the first-run profile
-- questions (right-handed, the first avatar, YES to the training mode), START GAME, close the
-- BREAKING OFF briefing, read the instructor's lessons, aim into the rack and break - until the
-- training event is PASSED (at least four balls hit a cushion); a failed break is RESTARTed.
-- last verified: 2026-10-23, uncommitted next-work tree (desktop headless, one-baton)
--
-- HOW THE CUES WERE FOUND: `--discover` next to a shot every 60 frames, and shots every frame:
--
--   f1209  sceNearSetGift          the title screen (it moves on by itself to the questions)
--
-- Nothing else the game calls marks a screen, so every later screen is read off the frame
-- (960x544, measured on shots of every screen of the route):
--
--   handed      "Are you left-handed or right-handed?"   1233 white px there, <= 537 elsewhere
--   avatar      the TAKE PHOTO button (SELECT AVATAR)     495 white px, 0 on the other screens
--   start       the black START GAME screen               108000 black px (the whole box)
--   bubble      the instructor's speech bubble text      ~2590 white px, 0 in the overview
--   result      the SAVE REPLAY button (both outcomes)    408 white px, 0 everywhere else
--   passed      the GREEN result ribbon                   1644 green px; 0 on the red FAILED one
--
-- CONTROLS (the title's own, Casual shot style): touch is in PANEL coordinates (the screen
-- doubled). Tap the instructor's bubble to read on; the left stick (or a drag on the table) turns
-- the aim; the cue in the power meter on the right is held, dragged DOWN for power and let go to
-- shoot. The training break's default aim is beside the rack's apex, and a straight shot from it
-- clips the rack and FAILS, so the aim is closed-loop onto the rack (`aim_at_rack`).

local W = { 235, 235, 235 }
local title = cue.call("sceNearSetGift"):happened()
local handed = cue.pixels{ x = 270, y = 260, w = 420, h = 26, rgb = W, tol = 20 }:above(1000)
local take_photo = cue.pixels{ x = 400, y = 500, w = 160, h = 24, rgb = W, tol = 20 }
local avatar = take_photo:above(300)
local avatar_gone = take_photo:below(50)
local centre_black = cue.pixels{ x = 300, y = 100, w = 360, h = 300, rgb = { 0, 0, 0 }, tol = 4 }
local start_screen = centre_black:above(100000)
local not_black = centre_black:below(50000)
local bubble = cue.pixels{ x = 270, y = 40, w = 580, h = 90, rgb = W, tol = 20 }:above(2000)
local result = cue.pixels{ x = 80, y = 500, w = 120, h = 24, rgb = W, tol = 20 }:above(300)
local passed = cue.pixels{ x = 20, y = 140, w = 160, h = 40, rgb = { 60, 140, 60 }, tol = 45 }:above(1000)

-- A tap: touch, hold, lift, and give the screen time to answer.
local function tap_at(x, y, after)
  touch(x, y)
  idle(6)
  untouch()
  idle(after or 30)
end

-- Turn the aim onto the rack: the cloth in a strip under the aim line at the rack's height
-- vanishes when the rack is under it, and the side of the line showing LESS cloth is the side the
-- rack is on. Short stick nudges toward that side until the strip is covered and the two sides
-- balance (MEASURED on a teal cloth: strip 0 px centred, 20-60 off either way; sides 694/812 at
-- the default aim, 866/579 overshot). The CLOTH COLOUR is the title's own run-time choice - teal,
-- purple and blue were all seen - so it is read off a cloth-only patch beside the cue ball first,
-- and the strip cues are made for that colour (once per table: a cue declared in a loop is an error).
local cloth_patch = cue.color{ x = 580, y = 410, w = 80, h = 40 }
local strips = {}

local function strip_cues()
  local rgb = cloth_patch:rgb()
  local key = rgb[1] .. "," .. rgb[2] .. "," .. rgb[3]
  if strips[key] == nil then
    log("aim: cloth rgb(" .. key .. ")")
    strips[key] = {
      mid = cue.pixels{ x = 472, y = 362, w = 16, h = 12, rgb = rgb, tol = 25 },
      left = cue.pixels{ x = 330, y = 355, w = 140, h = 20, rgb = rgb, tol = 25 },
      right = cue.pixels{ x = 490, y = 355, w = 140, h = 20, rgb = rgb, tol = 25 },
    }
    -- A cue has no value until the host reports the next frame.
    idle(2)
  end
  return strips[key]
end

local function aim_at_rack()
  local c = strip_cues()
  for _ = 1, 80 do
    local l, r = c.left:value(), c.right:value()
    if c.mid:value() <= 4 and math.abs(l - r) < 300 then return end
    local lx = l < r and 90 or 166
    stick("left", lx, 128)
    idle(3)
    stick("left", 128, 128)
    idle(6)
  end
  log("aim: the rack never centred under the aim line")
end

-- Hold the power meter's cue, drag it down most of the way (a full-power break often scratches
-- the cue ball into a pocket), then let go.
local function shoot()
  touch(1826, 520)
  idle(15)
  for y = 560, 880, 40 do
    touch(1826, y)
    idle(3)
  end
  untouch()
end

phase("boot", { timeout = 120 }, function()
  wait(title)
  shot("title")
  wait(handed)
  shot("handed")
end)

phase("profile", { timeout = 120 }, function()
  repeat tap_at(1586, 1022) until try_wait(avatar, 2)    -- RIGHT
  shot("avatar")
  repeat tap_at(290, 540) until try_wait(avatar_gone, 2)  -- the first avatar
  idle(30)
  shot("training-offer")
  repeat tap_at(330, 1022) until try_wait(start_screen, 3) -- YES (through a loading screen)
  split("start screen")
end)

phase("hall", { timeout = 120 }, function()
  repeat tap_at(1596, 1030) until try_wait(not_black, 3)  -- START GAME
  idle(120)
  shot("briefing")
  repeat tap_at(1584, 1020, 60) until try_wait(bubble, 2) -- OK on the BREAKING OFF briefing
  idle(60)
  shot("hall")
end)

phase("break", { timeout = 300 }, function()
  for attempt = 1, 6 do
    -- Six lessons, one tap each, the last of them "touch and hold your cue in the power meter".
    for _ = 1, 6 do tap_at(1100, 200, 60) end
    aim_at_rack()
    idle(10)
    shot("aimed-" .. attempt)
    shoot()
    idle(30)
    shot("break-" .. attempt)
    wait(result)
    idle(30)
    shot("result-" .. attempt)
    if try_wait(passed, 0.5) then
      log("training event passed on attempt " .. attempt)
      return
    end
    log("training event failed on attempt " .. attempt .. " - RESTART")
    tap_at(958, 1022, 200)
    wait(bubble)
  end
  error("the training break did not pass in six attempts")
end)
