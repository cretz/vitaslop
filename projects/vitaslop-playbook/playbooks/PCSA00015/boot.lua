-- WipEout 2048 (PCSA00015): the boot ladder with NO INPUT - the baseline every deeper run
-- builds on. Converted from the recipe `boot`.
--
-- Calls, from `--discover` on the campaign recipe's run (deterministic, 10000 frames):
--
--   f545   sceMp4OpenFile data/Videos/intro.mp4   the intro movie (the attract demo)
--   f5151  sceMp4CloseFile                        the movie is over
--   f5160  sceAppUtilSaveDataSlotCreate           the first autosave ("THIS GAME SAVES DATA
--                                                 AUTOMATICALLY..." with its icon), then the
--                                                 title screen, which waits for a button
--
-- A screen that does not advance here is a SAVE or a LOAD before it is an input problem: this
-- title once sat on the autosave notice for 42 minutes of game time because it could not read
-- back the save it had just written. `campaign-race.lua` skips the movie with its first CROSS.

local movie_open  = cue.call("sceMp4OpenFile", { arg = 0, has = "intro.mp4" }):happened()
local movie_close = cue.call("sceMp4CloseFile"):happened()
local autosave    = cue.call("sceAppUtilSaveDataSlotCreate"):happened()

phase("boot", { timeout = 180 }, function()
  -- The SCE / studio splashes, then the intro movie - nothing pressed.
  wait(movie_open)
  split("intro movie")
  idle(300)
  shot("intro-movie")
  wait(movie_close)
  split("movie over")
  wait(autosave)
  split("autosave")
  idle(30)
  shot("autosave-notice")
  -- The title screen follows on its own: PRESS ANY BUTTON TO START.
  idle(300)
  shot("title-screen")
end)
