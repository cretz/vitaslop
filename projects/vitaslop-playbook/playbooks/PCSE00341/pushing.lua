-- OlliOlli (PCSE00341): a save-less first launch, through the forced tutorial to the PUSHING
-- lesson, pushed to the goal until the lesson PASSes.
--
-- Converted from the recipe `pushing`, which tapped the touch front end at fixed frames and
-- pressed CROSS every 16 frames for 300 frames. This waits on the lesson's own outcome.
--
-- HOW THE CUES WERE FOUND: `--discover` on the recipe's run (deterministic, 700 frames) next
-- to a shot every 40 frames. The title loads everything it needs in the first few frames, so
-- the front end's screens make NO call of their own - only the lesson's end does:
--
--   f6     sceAppUtilSaveDataDataSave x3     the first-launch save
--   f444   sceAppUtilSaveDataDataSave        the PASS panel ("That was easy right?") - the
--                                            lesson's result is saved the moment it shows
--
-- THE FRONT END IS TOUCH, NOT BUTTONS: menu items are hit-tested on the front panel (panel
-- coordinates are screen x2). Gameplay is BUTTONS: CROSS both pushes on the ground and lands
-- in the air; two pushes reach full speed.
--
-- The menus read the pad every frame, so input frames ARE game frames here and the idles
-- between taps hold at any speed.

-- The skater's height: 60 on the ground, negative in the air, back through 60 on landing.
-- Found by value search (it is the recipe's @watch); the landing signal for the trick lessons.
local vpos   = cue.i32(0x81502058)
local SAVE   = "sceAppUtilSaveDataDataSave"
cue.call(SAVE):trace("savedata written")
-- The recipe's assert: grounded is 60 +-6 (pushing keeps the skater on the deck).
local not_airborne = vpos:atleast(54)
local not_buried   = vpos:atmost(66)

-- A touch-panel tap: down for `n` input frames, then up.
local function touch_tap(x, y, n)
  touch(x, y); idle(n or 7); untouch()
end

phase("front-end", { timeout = 60 }, function()
  idle(12)
  touch_tap(450, 674)             -- "Offline Mode": dismisses "Get yourself connected"
  idle(11)
  touch_tap(230, 230)             -- into the tutorial (a save-less first launch is forced here)
  idle(43)
  shot("lets-learn")
  touch_tap(930, 674)             -- OK on "Let's learn to Skate shall we?"
  idle(25)
  touch_tap(1620, 376)            -- "Pushing" in the tutorial list (its row, right side)
  idle(7)
  shot("pushing-intro")           -- the lesson's card: "X Button = Push & Land", Skip / Next
  touch_tap(630, 870)             -- SKIP on that card: the lesson starts
end)

phase("pushing", { timeout = 60 }, function()
  -- Pushing needs no trick or landing timing (that is The Basics): a steady CROSS cadence
  -- reaches full speed and rolls on to the goal.
  -- The lesson's result is saved as PASS shows. Declared HERE, not at the top: a call cue
  -- counts from its own declaration, so this one cannot see the first-launch saves (f7, f20).
  local passed = cue.call(SAVE):happened()
  local n = 0
  repeat
    tap("cross", 4, 12)
    n = n + 1
    if n == 10 then
      -- Mid-lesson: the skater is on the deck and rolling.
      check(not_airborne, "the skater is grounded while pushing")
      check(not_buried, "the skater is grounded while pushing")
      shot("pushing-rolling")
    end
  until try_wait(passed, 0.01)
  split("pass")
  idle(30)
  shot("pass")
end)
