-- Madden NFL 13 (PCSE00084): boot, PLAY NOW with the defaults, the coin toss, KICK OFF with the
-- right stick, let the return play out, then call a play, snap it and run upfield.
--
-- Converted from the recipe `into-a-game`, which swept every button at fixed frames, pressed
-- CROSS every 120 frames through the front end and timed the tip, the kick and the snap.
--
-- HOW THE CUES WERE FOUND: `--discover` on the recipe's run (deterministic, 4500 frames) next
-- to a shot every 100 frames. This title reads everything through one archive, so its file
-- names are not visible; these calls are (frames from that run):
--
--   f713   sceAppUtilSaveDataSlotGetParam x768   START was taken on the title screen
--   f1240  sceIoOpen *_away_uni_minicamp_big.ast  the match loads (both teams' uniforms)
--   f1471  sceGxmPrecomputedDrawInit (first)      the field is drawn - the coin toss is next
--   f2621  sceMotionReset                         the Motion Sensor Kicking tip was cleared
--
-- FOUR GATES, each non-obvious (from the recipe's own measurements):
--   1. The intro honours a button only after its own 5 s timer, and tests a bitset of its own
--      that is not the SceCtrl order - so START and CROSS are both pressed until the title
--      screen has taken one.
--   2. The front end is a stack of lists (PLAY NOW, the game setup, Advance to Game) whose
--      first entry leads to a match, so CROSS walks it - and CROSS also makes the coin toss.
--   3. The "NEW - Motion Sensor Kicking" tip blocks the kick and CROSS does not dismiss it:
--      its options are CROSS Dismiss / TRIANGLE Disable Tips / SQUARE Controls, and only
--      TRIANGLE was observed to clear it.
--   4. THE KICK IS THE RIGHT STICK: hold it back to raise the power meter, then flick it
--      forward. Every button, both triggers and the d-pad were swept at the kickoff and none
--      of them kicks.

local slots_probed = cue.call("sceAppUtilSaveDataSlotGetParam"):happened()
local match_loads  = cue.call("sceIoOpen", { arg = 0, has = "_uni_minicamp_big.ast" }):happened()
local field_drawn  = cue.call("sceGxmPrecomputedDrawInit"):happened()
local kick_armed   = cue.call("sceMotionReset"):happened()

phase("boot", { timeout = 120 }, function()
  -- Gate 1: START (and CROSS, for the intro's own bitset) until the title screen takes START
  -- and the game probes its save slots.
  repeat
    tap("start", 6, 20); tap("cross", 6, 20)
  until try_wait(slots_probed, 0.01)
  split("title")
  idle(60)
  shot("main-menu")
end)

phase("front-end", { timeout = 180 }, function()
  -- Gate 2: PLAY NOW, the setup screen (Event Type, Quarter, Accel Clock, Stadium, Weather),
  -- Advance to Game - all the first entry.
  repeat tap("cross", 6, 114) until try_wait(match_loads, 0.01)
  split("match loading")
  wait(field_drawn)
  split("field")
  idle(30)
  shot("in-a-game")
end)

phase("kickoff", { timeout = 180 }, function()
  -- The coin toss takes CROSS (the default choices); then the kicking team lines up and the
  -- tip comes up over the kickoff. Gate 3: TRIANGLE clears the tip; CROSS on it does not, and
  -- neither button kicks, so alternating them answers whichever screen is up.
  repeat
    tap("cross", 6, 60); tap("triangle", 6, 60)
  until try_wait(kick_armed, 0.01)
  split("kickoff")
  idle(60)
  shot("at-the-kickoff")
  -- Gate 4: the right stick BACK for 60 display frames (30 input frames) raises the power meter; the FLICK forward only
  -- has to be sampled.
  stick("right", 128, 255); idle(30)
  stick("right", 128, 1); idle(10)
  stick("right", 128, 128)
  idle(10)
  shot("the-kick-is-away")
end)

phase("first-snap", { timeout = 120 }, function()
  -- MEASURED: this title reads the pad on every other display flip, so an input frame is two
  -- display frames and the recipe's display-frame gaps are halved here.
  -- The return plays itself out and the down turns over to 1ST & 10 (the recipe's f3760, about
  -- 860 display frames after the kick). We kicked off, so this is the CPU's offence: its
  -- huddle, its snap. The recipe's three CROSSes 60 display frames apart are kept (a "Hike the
  -- Ball"-style tip takes CROSS as Dismiss; on defence CROSS switches the controlled player),
  -- then the left stick UP moves our player upfield while the play is run.
  idle(430)
  shot("first-and-ten")
  idle(85)
  tap("cross", 3, 27); tap("cross", 3, 27); tap("cross", 3, 2)
  shot("the-snap")
  -- The left stick UP for 190 display frames, until the play is whistled dead.
  stick("left", 128, 0); idle(25)
  shot("running")
  idle(70)
  stick("left", 128, 128)
  idle(20)
  shot("after-the-play")
  -- We kicked off, so the CPU has the ball and snaps on its own play clock (CROSS on defence
  -- switches the player): give it the time for a down to be played.
  idle(400)
  shot("next-down")
end)
