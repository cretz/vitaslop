-- Hot Shots Golf: World Invitational (PCSA00009): boot, clear the autosave dialog, walk the
-- front end to a STROKE PLAY round at Maple Leaf Golf Club, and play HOLE 1 into the cup.
--
-- Converted from the recipe `hole-one`, which played the hole as ~250 presses at fixed frames.
-- Measured 10-16: that sequence no longer holes out - its drive now lands on a staircase lie
-- ("Cannot hit in this direction") and every later press is ignored for 23,000 frames. This
-- plays each stroke off the game's own stroke state instead.
--
-- HOW THE CUES WERE FOUND: `--discover` on the recipe's run (deterministic, 26000 frames) next
-- to a shot every 200 frames; the stroke state by value search over memory dumps taken every
-- 20 frames across two strokes. Frames are from that run:
--
--   f461   savedata0:data0_US.bin opened      the autosave dialog was answered
--   f479   sceMp4OpenFile mgv_op_a.mp4        the title movie
--   f640   bgm/menu/main.sxd2                 the main menu
--   f861   bgm/menu/stroke.sxd2               STROKE play was chosen (Challenge loads none)
--   f916   player/pc/model_00_cos_00.xb       character select
--   f1055  ca_last_check.sxd                  the Confirmation screen ("Go to Course")
--   f1295  course01/hole01.xb                 HOLE 1 loads
--
-- MENUS ARE PAD-DRIVEN, NOT TOUCH (a tap moves guest state on the title screen and no menu
-- acts on it). This title reads the pad as a 32-sample HISTORY and uses the oldest sample's
-- timestamp, so every press here is held several frames.

local function opens(fragment) return cue.call("sceIoOpen", { arg = 0, has = fragment }) end

-- The stroke phase, a u8 found by value search over dumps every 20 frames across two strokes:
-- 20 while the player addresses and swings, then 1..7 through the strike, the flight, the roll
-- and the result banner, and 20 again on the next turn. (It reads 20 behind the hole's intro
-- panel too, and 2 appears both mid-hole and at the end, so it does not mark the hole-out.)
local STROKE = 0x84aecf00
local stroke = cue.u8(STROKE):trace("stroke phase")

local main_menu  = opens("bgm/menu/main.sxd2"):happened()
local mode_taken = opens("bgm/menu/stroke.sxd2"):happened()
local chara      = opens("model_00_cos_00.xb"):happened()
local confirm    = opens("ca_last_check"):happened()
local hole_load  = opens("hole01.xb")
-- The hole's result: its score jingle (sound/jingle/guts/bogey.sxd2 for a bogey; one per score),
-- then the scorecard's music. (Hole 2's data is no signal: it is prefetched as hole 1
-- starts.)
local jingle     = opens("sound/jingle/")
local scorecard  = opens("bgm/game/score_card"):happened()
local addressing = stroke:is(20)
local in_flight  = stroke:isnt(20)

phase("boot", { timeout = 120 }, function()
  -- The autosave dialog is drawn by the game and makes no call. The recipe cleared it with
  -- cross, circle, cross, circle; CROSS also skips the title movie and takes the title screen.
  -- MEASURED in the browser: a profile that already has the save skips the dialog and the
  -- movie entirely - so the only wait is for the main menu's music, checked after EACH press
  -- (a CIRCLE on the main menu backs out to the title).
  cue.call("sceMp4OpenFile"):trace("title movie")
  while true do
    tap("cross", 4, 16)
    if main_menu:value() == 1 then break end
    tap("circle", 4, 16)
    if main_menu:value() == 1 then break end
  end
  split("main menu")
  idle(30)
  shot("main-menu")
end)

phase("front-end", { timeout = 180 }, function()
  -- The main menu opens on SINGLE PLAYER. (The recipe called its first CROSS here "New Game";
  -- that press was taken by the title screen.) A second CROSS picks CHALLENGE, the mode list's
  -- first entry - a first draft did, and its own loop played a Challenge round.
  tap("cross", 6, 90)
  shot("single-player")
  -- Single Player: Challenge / Stroke / Training, opening on CHALLENGE. One DOWN is STROKE - a
  -- real round, holed out or given up at the stroke limit. (Training, one further, has a Next
  -- Shot menu with retakes and a "Hole Out" that ends the hole without the ball in the cup.)
  -- Stroke's music loads once it is taken; Challenge loads none (and opens on an info page).
  idle(20)
  tap("down", 6, 30)
  repeat tap("cross", 6, 30) until try_wait(mode_taken, 0.01)
  split("stroke play")
  -- The course (the first, Maple Leaf Golf Club), then the character (the default).
  repeat tap("cross", 6, 40) until try_wait(chara, 0.01)
  idle(40)
  shot("character-select")
  tap("cross", 6, 40)
  -- START = Confirm All, onto the Confirmation screen; CROSS = Go to Course.
  repeat tap("start", 6, 40) until try_wait(confirm, 0.01)
  shot("confirmation")
  tap("cross", 6, 40)
  wait(hole_load)
  split("hole 1 loading")
end)

-- One stroke on the shot gauge, in input frames: CROSS starts the swing and the cursor climbs
-- from the impact mark toward 100%; CROSS after `power` sets the power; the cursor then comes
-- back and CROSS after `impact` more must land on the impact mark.
--
-- MEASURED (shots every 2 frames through a swing): the climb's speed is CLUB-DEPENDENT - the
-- 3-iron is at 100% after ~55 frames, and a power press past the end of the climb CANCELS the
-- swing (a practice swing, then "Take the shot" again). From the top the cursor waits, then
-- returns and crosses the impact mark ~56 frames after the power press. No memory cell for the
-- cursor was found (value search over 256 MB of dumps found bones and camera matrices only),
-- so the timings are fixed and the hole is played as full swings:
--   * the DRIVE: power +66, impact +30 - 100% power with a perfect impact (the sparkle).
--   * every later stroke: power +60, impact +56. MEASURED: the 3-iron, an approach wedge and
--     one or two putts hole out - a bogey (one stroke is a penalty) deterministic and at real
--     speed. The recipe's shorter impact (+27) mishit every iron shot (46 yards at 100%).
local function swing(power, impact)
  tap("cross", 4, 0); idle(power - 4)
  tap("cross", 4, 0); idle(impact - 4)
  tap("cross", 4, 4)
end

phase("hole-1", { timeout = 900 }, function()
  -- Stroke play opens on the hole's intro panel over the tee (the stroke phase already reads 20
  -- behind it); CROSS takes it.
  idle(60)
  shot("hole-intro")
  tap("cross", 6, 40)
  local holed_out = jingle:happened()
  local turn_or_done = cue.any(addressing, holed_out)
  local n = 0
  while true do
    -- The player's turn, or the hole is over (its score jingle).
    if wait(turn_or_done) == 2 or holed_out:value() == 1 then break end
    -- The phase reads 20 about 60 frames before the shot is ready ("Take the shot" and the
    -- gauge); a swing started earlier is ignored. MEASURED: it also reads 20 for a moment as
    -- the ball drops, before the score jingle - so a turn is 150 frames of 20 with no jingle,
    -- or the swing's presses would skip the banner and the scorecard into the next hole.
    local held = 0
    while held < 150 and stroke:value() == 20 and holed_out:value() == 0 do
      idle(10); held = held + 10
    end
    if holed_out:value() == 1 then break end
    -- Not a turn after all (the ball is dropping, or a penalty is being applied): wait again.
    if held < 150 then goto continue end
    n = n + 1
    shot(string.format("stroke-%02d", n))
    local tries = 0
    while true do
      if n == 1 then swing(66, 30) else swing(60, 56) end
      if try_wait(in_flight, 4) then break end
      -- The ball did not leave. A lie the game will not play toward ("Cannot hit in this
      -- direction") needs a new aim: rotate left, then right, further each time.
      tries = tries + 1
      if tries >= 8 then fail("the ball never left after 8 swings") end
      if tries % 2 == 0 then
        hold((tries // 2) % 2 == 1 and "left" or "right", 20 * (tries // 2))
        idle(30)
      end
    end
    -- Stroke play gives up a hole at its stroke limit (12 here, +8) and shows the scorecard.
    if n >= 12 then fail("hole 1 was not holed out within 12 strokes") end
    ::continue::
  end
  split("holed out")
  log(string.format("hole 1 holed out in %d strokes", n))
  -- The score banner ("BOGEY") is up as the jingle plays; the scorecard follows on its own.
  idle(40)
  shot("holed-out")
  wait(scorecard)
  idle(60)
  shot("scorecard")
end)
