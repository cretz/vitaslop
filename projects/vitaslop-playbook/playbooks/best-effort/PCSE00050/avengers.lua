-- ZEN PINBALL 2 (PCSE00050): boot, dismiss the welcome note on MY COLLECTION, open the AVENGERS
-- table (its data ships in the base package; the other tables are add-ons), SINGLE PLAYER, wait out
-- the load and the fly-over, launch, choose an Avenger, plunge, and flip until the first ball
-- drains; then choose, plunge and play the second ball.
-- last verified: 2026-10-22, uncommitted next-work tree (desktop headless, one-baton)
--
-- HOW THE CUES WERE FOUND: `--discover` next to a shot every 30-60 frames (frames from one run):
--
--   f887   sceNetCheckDialogTerm          the network check after boot is over: the menu takes input
--   f1597  sceGxmTextureInitCube          the table finished loading (its first cube map)
--
-- The menus between them make no call of their own and are walked by their buttons. On the table
-- nothing the game calls marks a ball, so the CHOOSE YOUR AVENGER screen - shown before every
-- ball - is read off the frame: its tick button (bottom right) is the only bright white in that
-- box anywhere in play. MEASURED: 150-200 matching pixels on that screen, 0 on every other shot.
--
-- CONTROLS (the title's own): CROSS is the plunger (held to pull, released to fire) and confirms
-- the chosen Avenger; L and R are the flippers, and on the choose screen they step the choice.

local netcheck = cue.call("sceNetCheckDialogTerm"):happened()
local table_up = cue.call("sceGxmTextureInitCube"):happened()
local tick = cue.pixels{ x = 880, y = 235, w = 70, h = 60, rgb = { 235, 235, 235 }, tol = 20 }
local choosing = tick:above(100)
local playing = tick:below(20)

-- Flip both flippers in turn until the choose screen comes back (the ball drained) - a blind
-- player, which keeps a ball in play for tens of seconds on this table.
local function play_ball(name)
  local flips = 0
  repeat
    tap("r", 6, 18)
    tap("l", 6, 18)
    flips = flips + 1
    if flips == 20 then shot(name) end
  until try_wait(choosing, 0.01)
  log(name .. ": drained after " .. flips .. " flip pairs")
end

-- Confirm the Avenger on show, then pull the plunger back and let it go. An Avenger already
-- played is gone from the rack but can still be the one on show, and CROSS does not take it
-- (MEASURED: 107 s of CROSS on a spent choice), so a refused CROSS steps the choice with R.
local function launch()
  repeat
    tap("cross", 8, 30)
    if try_wait(playing, 0.5) then break end
    tap("r", 8, 30)
  until try_wait(playing, 0.5)
  idle(60)
  hold("cross", 90)
end

phase("menu", { timeout = 120 }, function()
  wait(netcheck)
  idle(120)
  shot("collection")
  -- The welcome note's OK, then AVENGERS (the selected tile), then SINGLE PLAYER below BUY PACK.
  tap("cross", 8, 60)
  tap("cross", 8, 120)
  tap("down", 8, 40)
  shot("single-player")
  tap("cross", 8, 20)
  wait(table_up)
  split("table loaded")
end)

phase("first ball", { timeout = 240 }, function()
  -- PRESS LAUNCH TO BEGIN: the plunger, held, until the choose screen answers. The fly-over
  -- after the load does not read it, so it is repeated.
  idle(60)
  shot("press-launch")
  repeat hold("cross", 90) idle(30) until try_wait(choosing, 1)
  split("choose")
  shot("choose")
  launch()
  split("ball 1")
  play_ball("ball-1")
  split("ball 1 drained")
end)

phase("second ball", { timeout = 240 }, function()
  shot("choose-again")
  launch()
  split("ball 2")
  play_ball("ball-2")
  split("ball 2 drained")
end)
