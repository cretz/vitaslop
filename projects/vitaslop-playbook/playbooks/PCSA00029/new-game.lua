-- Uncharted: Golden Abyss (PCSA00029): boot, NEW GAME on Normal, "Start New Campaign?" YES,
-- the opening cinematic, the Journal tutorial gate, the dark prologue (which waits for the
-- stick), the Jumping and Climbing tutorial panel, and FIRST PLAYER CONTROL in the jungle ruins.
--
-- Converted from the recipe `new-game`, which waited fixed frames at every gate (the load into
-- the prologue "ends anywhere in f7700..8500 between runs") and so spent ~4,000 frames idle.
--
-- HOW THE CUES WERE FOUND: `--discover` on the recipe's run (deterministic, 9400 frames) next
-- to a shot every 100 frames (frames from that run):
--
--   f508   sceMp4OpenFile bend_logo3.mp4             the studio logo movie
--   f1302  sceKernelGetTLSAddr r1 "continue_header_1" the main menu's script is up
--   f2310  sceSaveDataDialogInit                     YES on "Start New Campaign?" - the save
--                                                    slot is created
--   f3555  sceIoOpen saves\playerweapons.bin         the prologue level is loaded; the Journal
--                                                    tutorial overlay is up
--   f8418  sceNgsVoicePause (first)                  the Jumping and Climbing panel pauses the
--                                                    game: play started under the stick
--
-- (The title saves often - when the journal is dismissed, at checkpoints - so a save is no
-- screen signal past the new campaign's.)
--
-- EVERY INPUT IS A TOUCH OR THE STICK (panel coordinates are screen x2). The front end is
-- touch-driven; the Journal gate is passed by tapping the journal icon and then the PAGE-CURL
-- icon; the prologue opens on a dark shot that WAITS FOR THE STICK - with no input it holds
-- there for ever, looking like a hang or a black-screen bug; the climbing panel is dismissed
-- with the same page-curl icon.

local main_menu  = cue.call("sceKernelGetTLSAddr", { arg = 1, has = "continue_header" }):happened()
local new_save   = cue.call("sceSaveDataDialogInit"):happened()
local level      = cue.call("sceIoOpen", { arg = 0, has = "playerweapons.bin" })
local panel      = cue.call("sceNgsVoicePause"):happened()

local function touch_tap(x, y)
  touch(x, y); idle(8); untouch()
end

local CURL = { 1844, 830 }      -- the page-curl icon, bottom right

phase("boot", { timeout = 120 }, function()
  -- The title screen ("Touch to Start") is up after the logo movie; tap until the main menu.
  repeat touch_tap(960, 700); idle(52) until try_wait(main_menu, 0.01)
  split("main menu")
  idle(60)
  shot("main-menu")
end)

phase("new-game", { timeout = 120 }, function()
  -- NEW GAME (centre of the menu strip), then NORMAL on the difficulty strip (preselected,
  -- centre), then YES on "Start New Campaign on Normal Difficulty?". A tap that lands before a
  -- screen is ready is repeated by the loop around the last one.
  touch_tap(960, 710); idle(150)
  shot("difficulty")
  touch_tap(960, 440); idle(150)
  shot("start-new-campaign")
  repeat touch_tap(1250, 874); idle(60) until try_wait(new_save, 0.01)
  split("new campaign")
  wait(level, { timeout = 90 })
  split("prologue loaded")
end)

phase("journal-gate", { timeout = 120 }, function()
  -- The cinematic plays out under the load; then "Tutorial: Journal" waits for a tap on the
  -- journal icon and then on the page curl (the CURL passes it).
  idle(60)
  shot("journal-gate")
  touch_tap(1756, 380); idle(60)
  touch_tap(CURL[1], CURL[2]); idle(60)
  split("journal passed")
  -- The prologue opens dark and waits for the stick: push forward until the climbing panel
  -- comes up and pauses the game.
  idle(60)
  shot("prologue-dark")
  stick("left", 128, 0)
  wait(panel, { timeout = 90 })
  stick("left", 128, 128)
  split("prologue")
end)

phase("player-control", { timeout = 60 }, function()
  idle(60)
  shot("tutorial-panel")
  touch_tap(CURL[1], CURL[2])
  idle(60)
  shot("player-control")
  -- The stick walks the player character: gameplay.
  stick("left", 128, 0); idle(50)
  shot("walked")
  stick("left", 128, 128)
  idle(30)
end)
