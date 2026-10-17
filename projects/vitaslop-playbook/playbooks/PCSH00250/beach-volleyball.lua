-- Dead or Alive Xtreme 3 Venus (PCSH00250, Asia): boot through the intro and Day 1's
-- tutorial, pick NIKI BEACH on the activity menu, choose PLAY BEACH VOLLEYBALL, and play.
--
-- Converted from the recipe `beach-volleyball`, which tapped CIRCLE every 150 frames from f800
-- to f4750 and then alternated CIRCLE/CROSS on a fixed grid. This waits on the game instead.
--
-- HOW THE CUES WERE FOUND: `--discover` on the recipe's run (deterministic, 9100 frames) next
-- to a shot every 100 frames. Every file the title reads is a `sceIoOpen` with the path in r0,
-- and each screen loads its own UI layout the moment it opens (frames from that run):
--
--   f812   ui/01_01_title                      the title screen (PRESS ANY BUTTON)
--   f1156  movie/MOV_OPENING.mp4               the intro movie (after Travel to New Zack Island)
--                                              - through sceAvPlayerAddSource (r1), not sceIoOpen
--   f2404  ui/02_12_character_select           CHARACTER SELECT (the cursor is on Marie Rose)
--   f4710  ui/02_04_stage_select               the Day 1 ACTIVITY MENU (location cards)
--   f5001  ui/80_guide_valley                  volleyball was picked; NOW LOADING
--   f5047  camera/activity/BEACHVOLLEY_CAM     ...the match loads behind it
--
-- MEASURED (first passing run): both volleyball loads happen during NOW LOADING; the controls
-- guide (cross = START GAME) only appears once it is done, and nothing marks that. So the guide
-- is shot after a pause, and START is pressed then - a CROSS that lands early is ignored, and
-- the rally loop's own CROSSes would start the game anyway (they did in that run).
--
-- Everything from the title to the activity menu - the autosave notice, PRESS ANY BUTTON,
-- the menu's first entry (Travel to New Zack Island), the movie, character select, the
-- arrival scenes, a Yes/No (Yes), and the tutorial pages - is answered by CIRCLE (confirm on
-- the Asia build). A HELD circle is one press to this title, so every press is a tap.

local function opens(fragment) return cue.call("sceIoOpen", { arg = 0, has = fragment }) end

local title        = opens("01_01_title")
local movie        = cue.call("sceAvPlayerAddSource", { arg = 1, has = "MOV_OPENING.mp4" })
local chara_select = opens("02_12_character_select")
local activity     = opens("02_04_stage_select")
local picked       = opens("80_guide_valley")
local match        = opens("BEACHVOLLEY_CAM")
-- Conditions, declared once: a cue made inside a loop is a new cue every time round.
local movie_seen, chara_seen, activity_seen = movie:happened(), chara_select:happened(), activity:happened()

-- The activity menu's location card: 0 Niki Beach, 1 Tranquil Beach, 2 Pool, 4 Sports Shop.
-- A u32 in the image's own data (its address does not move), found by value search over
-- memory dumps of one run, one dump per RIGHT press. The menu opens on a location the title
-- picks AT RANDOM under SMP, which is why this is a cue and not a count of presses.
local location = cue.u32(0x8178c788)

phase("boot", { timeout = 240 }, function()
  -- The title layout loads only once the autosave notice is answered, so there is nothing to
  -- wait for first: CIRCLE through everything up to the activity menu. A tap a screen is not
  -- ready for is simply ignored and the next one lands.
  title:trace("title screen opened")
  -- MEASURED: waiting the intro movie out with no presses STALLS (a run sat 60 s after it
  -- started and never reached character select) - so the taps go on through it, as the
  -- recipe's did, and the two screens are only marked as they pass.
  local saw_movie, saw_chara = false, false
  repeat
    tap("circle", 6, 24)
    if not saw_movie and movie_seen:value() == 1 then
      saw_movie = true
      split("intro movie")
    end
    if not saw_chara and chara_seen:value() == 1 then
      saw_chara = true
      split("character select")
      -- Its layout loads while the screen is still black; let it fade in.
      idle(90)
      shot("character-select")
    end
  until try_wait(activity_seen, 0.5)
  split("activity menu")
end)

phase("activity", { timeout = 120 }, function()
  -- The menu slides in after its layout loads, and the card is only meaningful once it is up.
  idle(90)
  location:trace("location card")
  shot("activity-menu")
  -- RIGHT steps the card; step until it reads Niki Beach.
  local steps = 0
  while location:value() ~= 0 do
    steps = steps + 1
    if steps > 8 then fail("the location card never reached Niki Beach (reads " .. tostring(location:value()) .. ")") end
    tap("right", 6, 30)
  end
  shot("niki-beach")

  -- CIRCLE opens Niki Beach's list (Relax / Play Beach Volleyball / Play Beach Flags); DOWN
  -- moves to volleyball; CIRCLE picks it, and NOW LOADING comes up.
  tap("circle", 6, 40)
  tap("down", 6, 30)
  tap("circle", 6, 30)
  wait(picked:happened(), { timeout = 30 })
  split("volleyball picked")
  wait(match:happened(), { timeout = 30 })
  split("match loading")
end)

phase("volleyball", { timeout = 200 }, function()
  -- The controls guide follows the load (see the header); CROSS is its START GAME. The serve
  -- intro then takes a few seconds before the ball is in play.
  idle(150)
  shot("controls-guide")
  tap("cross")
  idle(180)
  shot("match-start")
  -- Alternate CIRCLE (receive / toss) and CROSS (spike / block / serve) on a 12-frame
  -- rhythm, as the recipe did - enough to keep rallies going and points scoring. Two
  -- minutes of guest time is several points.
  local _, _, start = now()
  local n = 0
  repeat
    for _ = 1, 50 do
      tap("circle", 5, 7)
      tap("cross", 5, 7)
    end
    n = n + 1
    shot(string.format("rally-%02d", n))
    local _, _, t = now()
  until t - start > 120
end)
