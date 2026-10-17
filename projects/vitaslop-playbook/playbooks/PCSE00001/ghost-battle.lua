-- Ridge Racer (PCSE00001): boot to the title, PLAY, the mandatory first-run wizard (team,
-- ID card, location, region, controls), the main menu, WORLD RACE -> GHOST BATTLE against the
-- built-in Default Ghost on Highland Cliffs, and a live race driven along a recorded line.
--
-- Converted from the recipes `title-screen` (boot, PLAY) and `ghost-battle-race` (the wizard,
-- then a live race); `time-attack.lua` covers `play` and `time-attack-lap` (the TIME card).
-- GHOST BATTLE races POSITION 2/2 against a real opponent, with a lap timer.
--
-- HOW THE CUES WERE FOUND: `--discover` on `ghost-battle-race`'s run (deterministic, 9400
-- frames) next to a shot every 100 frames (frames from that run):
--
--   f1560  sceMp4CloseFile                   PLAY was taken: the title movie stops
--   f6503  sceKernelCreateThread auto_save   SETTINGS COMPLETE was answered (READY)
--   f6581  sceNpTrophyUnlockTrophy           the team trophy + RCTV news ("Welcome to ...")
--   f8121  sceIoOpen Course_A.DAT            READY on the race CONFIRM: the course streams in
--   f8474  sceGxmTextureInitTiled (first)    the race scene is built: a course intro, a look
--                                            at the cars, then the 3-2-1 countdown
--   then   sceNgsVoiceKeyOff x2               the intro ends; the countdown begins
--
-- THE WIZARD MAKES NO CALLS and no screen-id cell was found (a value search over two dumps of
-- each of its ten screens found none constant per screen), so its touches replay the recipe's,
-- in game frames from PLAY. Every coordinate was taken from a shot of the screen it acts on; the
-- wizard is TOUCH ONLY (its buttons are not pad-selectable), panel coordinates are screen x2:
--   * PLANETARY LEAGUE -> SIGN ON (not a PSN sign-in); TEAMS: touch the ticket to sign;
--   * TEAM CONFIRMATION wants the team icon touched and HELD (a tap does nothing);
--   * the ID card and most setup pages advance on the RIGHT-EDGE ARROW, not the cyan OK box
--     (an `explore` sweep: the OK box and the card body do nothing);
--   * "Now creating your ID Card" needs FIND LOCATION ON MAP - the offline path;
--   * COUNTRY/REGION "Other" -> the single entry "RID / Ridge State";
--   * SETTINGS COMPLETE: READY is the big arrow panel at screen (826,276).
-- Judge this title only through the GPU renderer: the software path shows it black.

local movie_closed = cue.call("sceMp4CloseFile"):happened()
local settled      = cue.call("sceKernelCreateThread", { arg = 0, has = "auto_save" }):happened()
local trophy       = cue.call("sceNpTrophyUnlockTrophy"):happened()
local course       = cue.call("sceIoOpen", { arg = 0, has = "Course_A.DAT" }):happened()
local race_built   = cue.call("sceGxmTextureInitTiled"):happened()
-- The countdown has no call of its own. The race's audio is keyed off twice after the scene is
-- built - as the course intro ends and as the 3-2-1 begins (deterministic) - and GO is ~250
-- frames after the second. MEASURED at real speed the two land later in the sequence (GO
-- ~150 frames after the second), so the line starts within ~100 frames of GO either way. (A
-- race clock found by value search - 3000 ticks a second from GO - lives on the heap, and at
-- real speed it was elsewhere: one run read garbage there, the next a constant 0.)
local function key_offs() return cue.call("sceNgsVoiceKeyOff") end
local countdown   -- declared once the race scene is built

-- { frames after the previous anchor, panel x, panel y, frames held }
local WIZARD = {
  {  140,  960,  874,  20 },
  {  220,  960,  874,  20 },
  {  440,  400,  540,  20 },
  {  540,  400,  540,  20 },
  {  840,  960,  556, 200 },
  { 1140, 1870,  516,  40 },
  { 1260, 1870,  516,  40 },
  { 1640,  960,  874,  40 },
  { 1940, 1870,  540,  40 },
  { 2140, 1870,  540,  40 },
  { 2340,  960,  878,  40 },
  { 2440,  960,  878,  40 },
  { 2640,  960,  540,  40 },
  { 2840, 1870,  540,  40 },
  { 3040, 1870,  540,  40 },
  { 3240,  960,  874,  40 },
  { 3440,  960,  874,  40 },
  { 3640,  960,  314,  40 },
  { 3790,  960,  314,  40 },
  { 3990, 1870,  540,  40 },
  { 4190,  960,  874,  40 },
  { 4390, 1870,  540,  40 },
  { 4590,  960,  874,  40 },
  { 4840, 1652,  552,  40 },
  { 5040, 1652,  552,  40 },
}

-- Replay a touch timeline: each row's frame is from the start of the timeline.
local function play_touches(rows)
  local t = 0
  for _, r in ipairs(rows) do
    idle(r[1] - t); touch(r[2], r[3]); idle(r[4]); untouch()
    t = r[1] + r[4]
  end
end

local function touch_tap(x, y)
  touch(x, y); idle(20); untouch()
end

-- THE MENUS HAVE NO CUE EITHER: a value search over two dumps of each menu screen found only a
-- byte that steps every ~160 frames on its own (it matched the dumps by coincidence) and a
-- pointer that moves on unrelated events. So the menu taps keep the recipe's spacing too.

phase("title", { timeout = 120 }, function()
  -- Legal notice, BANDAI NAMCO, namco, CELLIUS, a save-data spinner, then the RIDGE RACER title
  -- with a pulsing PLAY (screen 480,420) over its movie. Tap PLAY until the movie stops.
  repeat touch(960, 840); idle(20); untouch(); idle(40) until try_wait(movie_closed, 0.01)
  split("play")
  idle(100)
  shot("planetary-league")
end)

phase("wizard", { timeout = 240 }, function()
  play_touches(WIZARD)
  wait(settled, { timeout = 30 })
  split("settings complete")
  wait(trophy, { timeout = 30 })
  split("team trophy")
  idle(100)
  shot("news")
  -- The team trophy + RCTV ticker; the right-edge arrow dismisses it to the MAIN MENU - ONCE.
  -- MEASURED: the recipe tapped it at +419 and +719 frames from the trophy. Deterministic, the
  -- first tap came before the BREAKING NEWS panel was ready and the second took it; at real
  -- speed the first took it and the second landed on the main menu, where that spot is the
  -- SPOT RACE card. The panel is ready by ~+420 in both, so: one tap at +500.
  idle(400)
  touch_tap(1870, 540)
  split("main menu")
end)

phase("ghost-battle", { timeout = 120 }, function()
  -- MAIN MENU: WORLD RACE (screen 727,270); RACE SELECTION: GHOST BATTLE (screen 240,270),
  -- the only offline one; GHOST: "Default Ghost" (ships with the title); MACHINE: BISARGENTO,
  -- the only car owned at LV1, confirmed on the right-edge arrow; CONFIRM: Highland Cliffs /
  -- BISARGENTO / Default Ghost, READY at screen (838,275) - which streams the course in.
  -- The main menu is a horizontal scroller that slides in; a tap before it settles lands on
  -- another card (MEASURED at real speed: GARAGE). Settled, WORLD RACE is centred at screen
  -- (727,270). Then the recipe's spacing.
  idle(200)
  shot("main-menu")
  play_touches({
    {  200, 1454,  540,  20 },      -- WORLD RACE
    {  340,  480,  540,  20 },      -- GHOST BATTLE
    {  480, 1144,  544,  20 },      -- Default Ghost
    {  620, 1870,  540,  20 },      -- BISARGENTO, on the right-edge arrow
  })
  idle(60)
  shot("confirm")
  repeat touch_tap(1676, 550); idle(60) until try_wait(course, 0.01)   -- READY
  split("course loading")
  wait(race_built, { timeout = 60 })
  split("race scene")
  countdown = key_offs()
  idle(100)
  shot("race-intro")
end)

-- The recorded line: { frame, button, lx }. CROSS accelerates; lx steers (above 128 toward
-- +X). It is the response of a closed-loop controller that aimed at the far end of the road
-- material ahead, with a carried lap direction. The stick is deadzoned over most of its range
-- (12 units from neutral turns nothing; 42 turns ~10 degrees per 10 frames), so values are
-- 128 or far from it. SQUARE is not a reverse gear.
local LINE = {
  {   0, "cross" , 177 },
  {   4, "cross" , 176 },
  {   8, "cross" , 128 },
  {  12, "cross" , 85 },
  {  16, nil     , 128 },
  {  20, "cross" , 128 },
  {  28, "cross" , 164 },
  {  32, "cross" , 128 },
  {  36, nil     , 128 },
  {  40, "cross" , 128 },
  {  52, nil     , 128 },
  {  56, "cross" , 128 },
  {  68, "cross" , 166 },
  {  72, nil     , 128 },
  {  76, "cross" , 128 },
  {  84, "cross" , 164 },
  {  88, nil     , 128 },
  {  92, "cross" , 128 },
  { 100, "cross" , 169 },
  { 104, nil     , 165 },
  { 108, nil     , 128 },
  { 112, "cross" , 128 },
  { 124, "cross" , 171 },
  { 128, "cross" , 169 },
  { 132, "square", 167 },
  { 136, nil     , 128 },
  { 140, "cross" , 128 },
  { 148, "cross" , 165 },
  { 152, "square", 165 },
  { 156, "square", 85 },
  { 160, "cross" , 76 },
  { 164, "cross" , 79 },
  { 168, "cross" , 128 },
  { 172, nil     , 175 },
  { 176, nil     , 128 },
  { 180, "cross" , 87 },
  { 184, "cross" , 128 },
  { 188, "cross" , 164 },
  { 192, nil     , 128 },
  { 196, nil     , 87 },
  { 200, "cross" , 128 },
  { 204, "cross" , 165 },
  { 208, "cross" , 91 },
  { 212, "cross" , 87 },
  { 216, nil     , 128 },
  { 220, nil     , 169 },
  { 224, "cross" , 128 },
  { 228, "cross" , 90 },
  { 232, "cross" , 128 },
  { 236, "square", 195 },
  { 240, "cross" , 79 },
  { 244, "cross" , 68 },
  { 248, "cross" , 128 },
  { 252, nil     , 170 },
  { 256, nil     , 92 },
  { 260, "cross" , 83 },
  { 264, "cross" , 128 },
  { 268, "cross" , 172 },
  { 272, nil     , 128 },
  { 276, "cross" , 87 },
  { 280, "cross" , 128 },
  { 288, nil     , 128 },
  { 296, "cross" , 128 },
  { 308, "cross" , 90 },
  { 312, nil     , 128 },
  { 316, "square", 201 },
  { 320, "square", 69 },
  { 324, "cross" , 53 },
  { 328, "cross" , 87 },
  { 332, "square", 203 },
  { 336, "cross" , 203 },
  { 344, "square", 75 },
  { 348, "cross" , 72 },
  { 352, "cross" , 128 },
  { 356, "cross" , 187 },
  { 360, "square", 87 },
  { 364, nil     , 165 },
  { 368, "cross" , 179 },
  { 372, "cross" , 81 },
  { 376, "cross" , 128 },
  { 380, "square", 203 },
  { 388, "cross" , 79 },
  { 392, "cross" , 75 },
  { 396, "cross" , 128 },
  { 400, "cross" , 203 },
  { 412, "cross" , 53 },
  { 424, "cross" , 203 },
  { 436, "cross" , 53 },
  { 444, nil     , nil },
}

phase("race", { timeout = 120 }, function()
  -- MEASURED: the intro and the countdown run ~1000 frames after the scene is built (the
  -- recipe's "live by f8800" had gone stale). Throttle may be held through the countdown.
  wait(countdown:atleast(2), { timeout = 60 })
  split("countdown")
  press("cross")
  idle(250)
  split("go")
  shot("race-start")
  -- Drive the line: the start straight, the first right-hander and the long sweep (about 420
  -- frames of racing, the ghost's HUD live). It ends wedged on the tight left at (685,-228) -
  -- completing a lap needs a longer line.
  -- MEASURED: replayed as recorded, the line's CROSS pulses give too little throttle (8 mph
  -- after 8 s - it was recorded when a counted frame bought several frames of simulation), so
  -- CROSS is held throughout and only the steering is replayed.
  press("cross")
  local t = 0
  for i, r in ipairs(LINE) do
    idle(r[1] - t); t = r[1]
    if r[3] then stick("left", r[3], 128) end
    if i == 30 then shot("racing") end
  end
  release()
  stick("left", 128, 128)
  idle(30)
  shot("race-end")
end)
