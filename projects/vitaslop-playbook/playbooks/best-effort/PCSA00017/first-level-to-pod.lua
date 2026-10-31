-- LITTLEBIGPLANET (PCSA00017): boot, skip the intro movie with START, close the PlayStation
-- Network notice, play the opening story level - the monochrome paper stage into the carnival -
-- to its end, then walk on through the story of Carnivalia and the train ride to La Marionetta
-- until the game hands over the POD, the hub every later level starts from, and its narration ends
-- on a lit planet and "Tap the Planet to Begin". Then tap into La Marionetta's first level ("First
-- Lessons In Loco-Motion"), close Colonel Flounder's briefing and play it.
-- last verified: 2026-10-27, uncommitted next-work tree (desktop headless SMP, all phases DONE into
-- the first Pod level, shots viewed); to the Pod also phone browser SMP on 2026-10-22, runner job
-- 20261022203442-004: DONE, 99.2% speed, 0.2% audio underrun, shots viewed
--
-- HOW THE CUES WERE FOUND: `--discover` next to a shot every 100 input frames (frames from one
-- run that walked from the start):
--
--   f316   sceMp4OpenFile                   the intro movie starts
--   f383   sceVideodecTermLibrary           the movie is over (here: skipped by START)
--   f751   sceMotionGetState                the first stage is up; the notice is over it
--   f2798  sceNetCheckDialogTerm            the PlayStation Network notice was closed
--   f5960  sceNpActivityPostStatus          "...I just aced a level in Story Mode!" - the
--                                           opening level is complete
--   f9714  sceLiveAreaUpdateFrameAsync      "Well then, this is your POD." - the hub is up
--
-- The Pod's narration is read off the frame (960x544, measured on every shot of eight runs):
--
--   prompt    the green "Tap the Planet to Begin" pill, bottom centre   624 px there, 0 elsewhere
--   lit       the planet's cyan band (140,238,240 +-40)                 >= 2400 lit, ~160 dark
--
-- >>> HANDS OFF AT THE POD. Input during its narration cuts it short, and the planet is then left
-- DARK and padlocked under the same prompt, refusing every tap (MEASURED: 300 s of taps).
--
-- >>> THE PLANET WAITS FOR THE WELCOME LINE TO END. Taps are ignored while the Pod's introduction
-- is still "playing", and it ends when the voice-over stream reaches its end. (It once never did in
-- the session that reached the Pod: the audio middleware stopped advancing its streams when
-- `sceNgsVoiceGetInfo` left the voice's update count unwritten.) The route on is, in PANEL
-- coordinates (the screen doubled): planet (970,570) -> world badge (950,590) -> the first level's
-- "Tap to Play" (960,490) -> Colonel Flounder's briefing, closed with CIRCLE.
--
-- The notice is the faithful offline answer ("A problem occurred when attempting to connect to
-- the PlayStation Network"); CROSS is its Close.
--
-- CONTROLS (the title's own): the left stick walks, CROSS jumps. The opening level is a walk to
-- the right over low steps and gaps, and the story scenes after it play while the stick is held,
-- so ONE rhythm - right for 50 frames, a jump, let go - carries the whole run; its prompts (the
-- X icons over the first steps) ask for exactly that.

local movie_open = cue.call("sceMp4OpenFile"):happened()
local movie_done = cue.call("sceVideodecTermLibrary"):happened()
local stage = cue.call("sceMotionGetState"):happened()
local notice_closed = cue.call("sceNetCheckDialogTerm"):happened()
local level_done = cue.call("sceNpActivityPostStatus"):happened()
local pod = cue.call("sceLiveAreaUpdateFrameAsync"):happened()

-- Walk right and jump, once.
local function step()
  stick("left", 255, 128)
  idle(50)
  tap("cross", 20, 30)
  stick("left", 128, 128)
end

phase("boot", { timeout = 120 }, function()
  wait(movie_open)
  -- START skips the movie, as it does for a player; pressed until the decoder is let go.
  repeat tap("start", 6, 30) until try_wait(movie_done, 0.01)
  split("movie skipped")
end)

phase("notice", { timeout = 120 }, function()
  wait(stage)
  idle(60)
  shot("notice")
  repeat tap("cross", 6, 30) until try_wait(notice_closed, 0.01)
end)

phase("opening level", { timeout = 600 }, function()
  local n = 0
  repeat
    step()
    n = n + 1
    if n == 6 then shot("monochrome") end
    if n == 15 then shot("carnival") end
  until try_wait(level_done, 0.01)
  -- No shot here: the level ends on a fade to black.
  split("level complete")
end)

phase("to the pod", { timeout = 600 }, function()
  local n = 0
  repeat
    step()
    n = n + 1
    if n == 20 then shot("story") end
  until try_wait(pod, 0.01)
  idle(120)
  shot("pod")
end)

local prompt = cue.pixels{ x = 340, y = 490, w = 280, h = 36, rgb = { 198, 216, 190 }, tol = 20 }:above(300)
local planet_lit = cue.pixels{ x = 380, y = 160, w = 220, h = 240, rgb = { 140, 238, 240 }, tol = 40 }:above(2000)

phase("pod narration", { timeout = 240 }, function()
  wait(prompt)
  wait(planet_lit)
  idle(60)
  shot("planet-lit")
end)

-- A tap in PANEL coordinates: press, hold six frames, let go.
local function tap_panel(x, y)
  touch(x, y)
  idle(6)
  untouch()
end

phase("into the first level", { timeout = 300 }, function()
  -- The planet, the world badge, the first level's "Tap to Play", then Colonel Flounder's
  -- briefing, closed with CIRCLE. Each screen takes a moment to come up, so each tap is followed
  -- by a wait and a shot.
  tap_panel(970, 570)
  idle(240)
  shot("planet")
  tap_panel(950, 590)
  idle(240)
  shot("world")
  tap_panel(960, 490)
  idle(300)
  shot("level-card")
  for _ = 1, 4 do
    tap("circle", 4, 60)
  end
  idle(120)
  shot("briefing-closed")
end)

phase("play", { timeout = 300 }, function()
  for i = 1, 12 do
    step()
    if i % 4 == 0 then shot("play-" .. i) end
  end
end)
