-- Persona 4 Golden (PCSE00120): boot, the title screen, NEW GAME on NORMAL, both front-end
-- movies, the Velvet Room, NAME ENTRY, Igor's tarot reading, the opening cutscene movie, the
-- arrival in Inaba (Dojima at the station, the car, the gas station) and FIRST PLAYER CONTROL:
-- walking the protagonist down the street in FIELD MODE.
--
-- Converted from the recipes `title-screen` (boot to the title) and `new-game` (everything),
-- which pressed CROSS at fixed frames all the way - ~170 presses spaced 44 to 156 frames apart.
--
-- HOW THE CUES WERE FOUND: `--discover` on `new-game`'s run (deterministic, 24000 frames) next
-- to a shot every 400 frames. The title names its own modes in the semaphores it waits on
-- (frames from that run):
--
--   f748   sceMp4OpenFile P4CTOP3.mp4            the title screen's looping movie
--   f3005  sceMp4OpenFile P4CT001.mp4            NEW GAME + NORMAL were taken: the first movie
--   f4894  sceKernelWaitSemaCB "evtEditLoad"     the Velvet Room event loads (Igor)
--   f8437  sceKernelWaitSemaCB "evtStartScript"  the name was confirmed; the tarot reading
--   f12218 sceMp4OpenFile P4CT002.mp4            the opening cutscene movie (the beach, the
--                                                news desk, the train, the broadcast)
--   f17071 sceMp4CloseFile                       ...it is over: Dojima at Inaba station
--   f22798 sceKernelWaitSemaCB "itf_picture"     FIELD MODE, "Shopping District, South"
--
-- THE ONE THING THAT COSTS A SESSION: this title sits on a BLACK screen for ever with no input,
-- looking perfectly healthy while it does. It runs at its own 30 flips a second.
--
-- HOW THE NAME ENTRY WORKS (two probes read it as broken; it is not): CROSS types the letter
-- under the cursor; on the LAST-name screen START is greyed out and the field is left by
-- FILLING its eight characters, after which the FIRST-name screen comes up on its own; there
-- START ends the name and raises "> Proceed?", a CROSS opens Yes/No and a CROSS takes Yes. So a
-- run of CROSSes with a START among them answers Igor's dialogue AND writes a name.

local function sema(name) return cue.call("sceKernelWaitSemaCB", { arg = 3, has = name }):happened() end
local function movie(file) return cue.call("sceMp4OpenFile", { arg = 0, has = file }):happened() end

local title_movie = movie("P4CTOP3.mp4")
local first_movie = movie("P4CT001.mp4")
local velvet      = sema("evtEditLoad")
local named       = sema("evtStartScript")
local cutscene    = movie("P4CT002.mp4")
local field       = sema("itf_picture")

phase("title", { timeout = 120 }, function()
  -- The boot notices ("connect to PlayStation Network?", which opens on No) take CROSS; the
  -- title's movie loop says the title screen is up.
  repeat tap("cross", 4, 26); tap("start", 4, 26) until try_wait(title_movie, 0.01)
  split("title")
  idle(150)
  shot("title-screen")
end)

phase("new-game", { timeout = 180 }, function()
  -- The "Rescue Request" notice, the title (NEW GAME is the first entry), the difficulty select
  -- (NORMAL preselected) and "Start at NORMAL difficulty?" Yes - all CROSS.
  repeat tap("cross", 4, 74) until try_wait(first_movie, 0.01)
  split("movies")
  -- The first movie (the truck in the fog) and the opening anime play into the Velvet Room.
  idle(300)
  shot("movie-1")
  wait(velvet, { timeout = 120 })
  split("velvet room")
end)

phase("velvet-room", { timeout = 240 }, function()
  idle(150)
  shot("velvet-room")
  -- Igor's dialogue, then name entry: CROSSes (each a line, or a letter) with a START every
  -- eighth (greyed on the last name; ends the first name) until the name is confirmed. MEASURED:
  -- the name is confirmed after ~40 presses; the shots at 24 and 32 land on the keyboard.
  local n = 0
  repeat
    n = n + 1
    if n % 8 == 0 then tap("start", 4, 16) else tap("cross", 4, 16) end
    if n == 24 or n == 32 then shot(string.format("name-entry-%d", n)) end
  until try_wait(named, 0.01)
  split("named")
  idle(60)
  shot("name-confirmed")
end)

phase("tarot", { timeout = 240 }, function()
  -- Igor's tarot reading ("The Moon, in the upright position"), Margaret: CROSS until the
  -- cutscene movie starts.
  local n = 0
  repeat
    tap("cross", 4, 45)
    n = n + 1
    if n == 30 then shot("tarot") end
  until try_wait(cutscene, 0.01)
  split("cutscene")
  -- The cutscene plays to its end with nothing pressed.
  local over = cue.call("sceMp4CloseFile"):happened()
  idle(400)
  shot("intro-movie")
  wait(over, { timeout = 180 })
  split("arrival")
end)

phase("arrival", { timeout = 300 }, function()
  -- Dojima at the station ("Hey! Over here!"), the car ride and the gas station: every line is
  -- a CROSS, until FIELD MODE. In the field a CROSS does nothing.
  idle(60)
  shot("arrival-station")
  repeat tap("cross", 4, 18) until try_wait(field, 0.01)
  split("field")
  -- FIELD MODE opens under a HELP panel ("Examine the blue butterfly to save your progress");
  -- CROSS clears it (the recipe's CROSSes ran on past the field and did).
  idle(60)
  shot("field-help")
  for _ = 1, 3 do tap("cross", 4, 26) end
  idle(30)
  shot("field-control")
  -- The left stick WALKS the protagonist and the camera follows: gameplay.
  stick("left", 0, 128); idle(50)
  stick("left", 128, 255); idle(50)
  stick("left", 128, 128)
  shot("field-walked")
end)
