-- Dead or Alive 5 Plus (PCSE00235): boot through the intro movie and the autosave notice, the
-- title, the play-style prompt, the main menu, FIGHT -> VERSUS -> Solo Match with the defaults
-- (Kasumi vs Ayane), a fresh stage, and THREE whole matches with FIGHT AGAIN between them -
-- each match's result saved.
--
-- Converted from the recipes `boot` (the intro movie and autosave), `versus-fight` (one match,
-- asserting the result save) and `full-fight` (three matches, FIGHT AGAIN between them).
--
-- HOW THE CUES WERE FOUND: `--discover` on `versus-fight`'s run (deterministic, 8000 frames) next
-- to a shot every 150 frames (frames from that run):
--
--   f600   the Team Ninja intro movie ninja_vi.mp4 (sceMp4CloseFile at f1667)
--   f1668  sceMsgDialogInit                       the autosave notice (answered OK)
--   f1997  sceAppUtilSaveDataDataSave             the boot autosave
--   f2054  sceAppUtilSystemParamGetString         the main menu opens (`full-fight` found this)
--   f3259  sceIoOpen app0:./DATA/I1IF_...         the stage loads after stage select
--   f6692  sceAppUtilSaveDataDataSave             the match result ("You got a new title!")
--   later  sceGxmPrecomputedFragmentStateInit     FIGHT AGAIN: the next stage's draw state
--
-- MEASURED in the recipes, kept here: the autosave notice now opens only AFTER START on the
-- title; the play-style prompt's "set to casual" OK can open late (a phone), so the hold for the
-- main menu PULSES CROSS; a direction in the fight loop moved the post-match cursor off FIGHT
-- AGAIN into the options pages for good, so the loop is FACE BUTTONS ONLY - and its CROSS
-- confirms FIGHT AGAIN (the default item) whatever frame a match ends on.
-- Buttons (DOA5+ Vita): triangle PUNCH, circle KICK, square HOLD, cross THROW.

local main_menu = cue.call("sceAppUtilSystemParamGetString"):happened()
local movie_over = cue.call("sceMp4CloseFile"):happened()
local notice    = cue.call("sceMsgDialogInit"):happened()
-- These two are declared FRESH where they are used: a call cue counts from its own
-- declaration, and `:happened()` on a cue declared here would count the boot autosave and the
-- previous match's stage files.
local function stage_load() return cue.call("sceIoOpen", { arg = 0, has = "/DATA/I1IF_" }):happened() end
local function a_save() return cue.call("sceAppUtilSaveDataDataSave"):happened() end

-- One 120-frame cycle of the recipe's fight loop: punches, kicks, a punch+kick, a throw, a hold.
local function cycle()
  tap("triangle", 4, 8); tap("triangle", 4, 8); tap("circle", 4, 12)
  tap("triangle+circle", 4, 12); tap("circle", 4, 12); tap("triangle", 4, 8)
  tap("cross", 4, 12); tap("square", 4, 16)
end

phase("boot", { timeout = 120 }, function()
  -- The Tecmo Koei logo and the intro movie need nothing. Then the autosave notice (CROSS),
  -- "PRESS START" (START) and the play-style prompt (CROSS, and its OK) - in an order that has
  -- differed between engines and devices (the recipes saw the notice both before and after
  -- START), so START and CROSS are both pulsed until the main menu opens.
  wait(movie_over)
  split("intro movie over")
  -- Checked after EACH press: a CROSS once the menu is up opens STORY's submenu (MEASURED).
  while true do
    tap("start", 4, 26)
    if main_menu:value() == 1 then break end
    tap("cross", 4, 26)
    if main_menu:value() == 1 then break end
  end
  if notice:value() == 1 then split("autosave notice was answered") end
  split("main menu")
  idle(40)
  shot("main-menu")
end)

phase("front-end", { timeout = 120 }, function()
  -- The main menu opens on STORY: DOWN to FIGHT, CROSS. Then VERSUS, Solo Match, both
  -- characters, the costume, and stage select all take their default with CROSS, until the
  -- stage loads.
  -- MEASURED: STORY's chapters save as they go - a run that strayed into STORY "won" three
  -- matches in 1,500 frames. The menu settles ~100 frames after its call.
  idle(100)
  tap("down", 4, 56)
  shot("fight-selected")
  local stage = stage_load()
  local n = 0
  repeat
    tap("cross", 4, 116)
    n = n + 1
    if n == 3 then shot("character-select") end
  until try_wait(stage, 0.01)
  split("stage loading")
end)

phase("matches", { timeout = 600 }, function()
  for match = 1, 3 do
    -- The intros; then the fight. The loop runs until this match's result is SAVED (the
    -- recipe's egress assert, as a call: a save after this match began).
    -- (Three matches, so three pairs of cues - declared per match on purpose.)
    idle(240)
    local result = a_save()
    shot(string.format("match-%d", match))
    local n = 0
    repeat
      cycle()
      n = n + 1
      if n == 10 then shot(string.format("match-%d-fight", match)) end
    until try_wait(result, 0.01)
    split(string.format("match %d result saved", match))
    idle(60)
    shot(string.format("match-%d-result", match))
    if match < 3 then
      -- The cycle's CROSS answers the result and confirms FIGHT AGAIN; the next stage loads.
      -- MEASURED: the next match's stage opens no file (its data came with the first load);
      -- it builds its draw state, a burst of sceGxmPrecomputedFragmentStateInit.
      local next_stage = cue.call("sceGxmPrecomputedFragmentStateInit"):happened()
      repeat cycle() until try_wait(next_stage, 0.01)
      split(string.format("match %d", match + 1))
    end
  end
end)
