-- MLB 12 The Show (PCSA00002): boot, accept USER SETTINGS (BEGINNER), the title, GAME MODES ->
-- EXHIBITION -> PLAY NOW, take the ROAD team (Texas, who bat first), walk the uniform, pitcher,
-- lineup, stadium and options screens, load into the game, and SWING at the plate until the
-- ball is put in play.
--
-- Converted from the recipes `boot-user-settings` (boot to the first screen), `into-a-game` and
-- `fast-at-bat` (the menu walk to a live game) and `play-a-game` (the same, as P1, swinging).
-- `swing-and-play` was the CPU playing itself (P1 never assigned) and is superseded.
--
-- HOW THE CUES WERE FOUND: `--discover` on `play-a-game`'s run (deterministic, 13000 frames)
-- next to a shot every 150 frames. Every file is read through one Fios call with its path in
-- r2 (frames from that run):
--
--   f1903  sceAppUtilSaveDataDataSave            USER SETTINGS answered (the first save)
--   f6000  load UNIFORMS/.../FECAPSELECT.TP2     TEAM SELECT advanced to the uniforms
--   f9184  load FRONTEND/STATS/NOTABLE.BIN       OPTIONS confirmed: the game loads
--   f9700  sceGxmDrawInstanced (first)           the stadium is drawn (TODAY'S MATCHUP)
--
-- The other front-end screens make no call, so between these anchors the presses keep the
-- spacing of `fast-at-bat` (150 frames), which walked this menu on a phone at real speed.

local RESOLVE = "_sceFiosKernelOverlayResolveWithRangeSync"
local function loads(fragment) return cue.call(RESOLVE, { arg = 2, has = fragment }):happened() end

local settings  = cue.call("sceAppUtilSaveDataDataSave"):happened()
local uniforms  = loads("FECAPSELECT")
local game_load = loads("STATS/NOTABLE.BIN")
local stadium   = cue.call("sceGxmDrawInstanced"):happened()

phase("boot", { timeout = 120 }, function()
  -- The logo and the load; then USER SETTINGS (BEGINNER / EXPERIENCED / EXPERT) over the photo
  -- background - BEGINNER is the default, CROSS takes it and the settings are saved.
  repeat tap("cross", 10, 50) until try_wait(settings, 0.01)
  split("user settings")
  shot("after-user-settings")
end)

phase("front-end", { timeout = 180 }, function()
  -- The title screen (SWIPE) takes START (MEASURED: the CROSS that answered the settings can
  -- carry on through it): GAME MODES with EXHIBITION highlighted.
  idle(400)
  tap("start", 10, 140)
  shot("game-modes")
  tap("cross", 10, 140)                    -- EXHIBITION
  tap("cross", 10, 140)                    -- PLAY NOW on the EXHIBITION modal: TEAM SELECT
  shot("team-select")
  -- >>> LEFT moves P1 from "unassigned" onto the ROAD team (Texas), who bat first - so the first
  -- at-bat is ours. (`r` here changes the LEAGUE, MLB -> TRIPLE-A; it does not advance.)
  tap("left", 10, 140)
  shot("p1-on-the-road-team")
  repeat tap("cross", 10, 140) until try_wait(uniforms, 0.01)
  split("uniforms")
  tap("cross", 10, 140)                    -- the uniform
  tap("cross", 10, 140)                    -- SELECT PITCHER: READY
  tap("start", 10, 140)                    -- PREGAME LINEUP says PRESS START; CROSS does not leave it
  shot("stadium-select")
  tap("cross", 10, 140)                    -- STADIUM SELECT (Busch Stadium)
  shot("options")
  -- OPTIONS: time of day, cloud cover, innings, CONFIRM - three DOWNs, then CROSS.
  tap("down", 10, 50); tap("down", 10, 50); tap("down", 10, 50)
  repeat tap("cross", 10, 140) until try_wait(game_load, 0.01)
  split("loading the game")
end)

phase("into-the-game", { timeout = 120 }, function()
  wait(stadium)
  split("stadium")
  idle(60)
  shot("in-the-game")
  -- TODAY'S MATCHUP, the broadcast intro: CROSS skips it on to the first pitch.
  for _ = 1, 4 do tap("cross", 10, 140) end
  shot("at-the-plate")
end)

phase("swing", { timeout = 120 }, function()
  -- >>> OURS TO SWING. CROSS is the swing; the CPU pitches on its own schedule and nothing here
  -- can see the pitch, so the swings walk a steady cadence through every phase of the
  -- pitcher's cycle. A miss costs a strike and another pitch; contact only has to happen once.
  for n = 1, 20 do
    tap("cross", 10, 50)
    if n % 5 == 0 then shot(string.format("swing-%02d", n)) end
  end
  -- Nothing is pressed now: a ball in play is the FIELDERS' to resolve, and a button during it
  -- advances a prompt that has not appeared yet.
  idle(300)
  shot("the-ball-is-in-play")
  idle(300)
  shot("after-the-play")
end)
