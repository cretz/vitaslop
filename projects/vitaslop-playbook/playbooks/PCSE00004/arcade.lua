-- Ultimate Marvel vs. Capcom 3 (PCSE00004, NTSC): boot, decline PSN, OFFLINE MODE -> ARCADE,
-- take the default team and stage, and fight the first match.
--
-- Converted from the old frame-keyed scripts `into-a-fight` and `arcade-deep`. Those pressed CROSS every 150
-- frames from boot and hoped each press landed on a screen; this waits on what the game DOES.
--
-- HOW THE CUES WERE FOUND: `--discover` on the old script's run (deterministic, 7000 frames)
-- next to a shot every 100 frames. The title loads every file through one Fios call whose
-- third argument (r2) is the path, so "the game started loading X" is the screen signal.
-- Frames are from that run (the old script waited out the intro movie; this skips it):
--
--   f30    sceMp4OpenFile                       the intro movie starts
--   f3148  the title screen reads the pad
--   f3240  sceNpInit (+ the rest of NP)         START taken on the title screen
--   f3712  thread "AutoSavedataThread"          the sign-in prompt was answered NO
--   f4454  sceAppUtilDrmOpen x14                the main menu checks for add-ons as it opens
--   f5022  load nar_battle.spc                  ARCADE chosen
--   f5089  load bgm_gm_030.at9                  character select music
--   f6350  load bgm_gm_032.at9                  the SETTINGS (stage) screen
--   f6592  load /stg/...                        the stage loads after the VS screen
--   f6701  load ga_time35 (the round timer)     the fight HUD - the fight is on
--
-- The dialogs between the title and the main menu are drawn by the game, not the system, and
-- make no call of their own - so they are answered by looping a press until the NEXT cue has
-- happened. The loop is also what makes this run at real speed: a press that lands before a
-- screen is up is simply repeated.
--
-- >>> LOOPS WAIT ON `:happened()`, NOT ON THE CUE. A wait on a call cue fires on the next call
-- after the wait begins; a call made while a tap was down (between two waits) is missed. The
-- first version waited on the bare `cue.call("sceNpInit")` and timed out with NP already up. A
-- `:happened()` condition stays true once the call has been made.

-- Every file the title loads passes through here with its path in r2.
local RESOLVE = "_sceFiosKernelOverlayResolveWithRangeSync"
local function loads(fragment) return cue.call(RESOLVE, { arg = 2, has = fragment }) end

local movie_open    = cue.call("sceMp4OpenFile")
-- MEASURED: a SKIPPED intro movie closes long after the skip (f683 for a skip from f31), and
-- only the title screen says it is over - so the close is traced, not waited on.
local movie_closed  = cue.call("sceMp4CloseFile"):trace("intro movie closed")
-- Each screen as a `:happened()` CONDITION, declared once here: the loops below re-check
-- them every time round, and a cue made inside a loop would be a new cue every time.
local title_taken   = cue.call("sceNpInit"):happened()
local declined_psn  = cue.call("sceKernelCreateThread", { arg = 0, has = "AutoSavedataThread" }):happened()
local main_menu     = cue.call("sceAppUtilDrmOpen"):happened()
local arcade_chosen = loads("nar_battle.spc"):happened()
local chr_select    = loads("bgm_gm_030.at9"):happened()
local settings      = loads("bgm_gm_032.at9"):happened()
local stage_loading = loads("/stg/"):happened()
local fight_hud     = loads("ga_time35"):happened()
-- Every fighter on the screen loads its voice bank; traced so the log shows the teams forming.
loads(".stq"):trace("voice banks loaded")

phase("boot", { timeout = 120 }, function()
  -- The intro movie: START skips it, and the next START is taken by the title screen (PRESS
  -- START), which brings the network library up the moment it does.
  wait(movie_open)
  split("intro movie")
  repeat tap("start") until try_wait(title_taken, 1)
  split("title")
end)

phase("sign-in", { timeout = 120 }, function()
  -- "Would you like to sign in?" opens on YES. DOWN + CROSS answers NO, then CROSS clears
  -- "If you do not sign in ... OK". A CROSS that lands before DOWN took effect answers YES,
  -- which fails offline and asks again - so the whole answer is retried until the autosave
  -- thread (started only on the NO path) appears. The prompt fades in about a second after
  -- NP comes up; the shot waits for it (the answer would be retried anyway).
  idle(60)
  shot("sign-in")
  repeat
    tap("down"); tap("cross"); tap("cross")
  until try_wait(declined_psn, 3)

  -- A first boot then asks "Create new saved data?" (YES), "Save complete" (OK), explains
  -- autosave (OK) and that online play needs PSN (OK) - four CROSSes on screens that make no
  -- call. A run with saved data skips some of them, so: CROSS until the main menu opens.
  repeat tap("cross") until try_wait(main_menu, 2)
  split("main menu")
end)

phase("menus", { timeout = 180 }, function()
  -- OFFLINE MODE is the main menu's first entry and ARCADE the first of its submenu, so
  -- CROSS until the arcade narration loads. The add-on check dialog eats presses while up.
  repeat tap("cross") until try_wait(arcade_chosen, 2)
  split("arcade")

  -- The control type prompt (NORMAL), then three fighters and an assist type each (the
  -- cursor starts on Ryu, and each confirm keeps the default), then the SETTINGS screen.
  -- The select music loads a beat before the control type prompt (on black) fades in, and
  -- the character select takes a couple of seconds to come up behind the confirm (MEASURED:
  -- shots at +0 and +30 frames were black and the prompt respectively).
  wait(chr_select)
  idle(60)
  shot("control-type")
  tap("cross")
  idle(120)
  shot("character-select")
  repeat tap("cross") until try_wait(settings, 2)
  split("settings")
  shot("settings")

  -- Confirm the stage; the VS screen plays, then the stage loads and the HUD comes up.
  repeat tap("cross") until try_wait(stage_loading, 2)
  wait(fight_hud)
  split("fight")
end)

-- One attack cycle facing `fwd` ("right" or "left"): walk in, an L-M-H chain, a Hadoken
-- (down, down-forward, forward + L), a Shinku Hadoken hyper (the same motion + L+M), both
-- assist calls, a jump-in and H, then S (CROSS). Default Vita layout: square L, triangle M,
-- circle H, cross S, L/R assists. Timings are the old script's, in input frames.
local function cycle(fwd)
  hold(fwd, 30)
  tap("square", 4, 4); tap("triangle", 4, 4); tap("circle", 4, 18)
  hold("down", 3); hold("down+" .. fwd, 3); hold(fwd, 2); hold(fwd .. "+square", 4); idle(28)
  hold("down", 3); hold("down+" .. fwd, 3); hold(fwd, 2); hold(fwd .. "+square+triangle", 6); idle(46)
  tap("l", 4, 6); tap("r", 4, 12)
  hold("up+" .. fwd, 4); idle(14)
  tap("circle", 4, 8); tap("cross", 4, 10)
end

phase("fight", { timeout = 150 }, function()
  -- The HUD loads during the fade from black; the fighters' intros play for a few seconds
  -- after it, and take no input.
  idle(150)
  shot("fight-start")
  -- Sides switch mid-fight, so alternate. Ninety guest seconds covers a round or two of
  -- sparks, hypers, assists and KO screens; CROSS also clears a win screen if one comes up.
  local _, _, start = now()
  local n = 0
  repeat
    cycle("right"); cycle("left")
    n = n + 1
    shot(string.format("fight-%02d", n))
    local _, _, t = now()
  until t - start > 90
end)
