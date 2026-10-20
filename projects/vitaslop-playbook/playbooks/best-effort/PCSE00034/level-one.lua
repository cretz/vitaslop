-- Tales from Space: Mutant Blobs Attack (PCSE00034): boot, the title, STORY MODE, the intro
-- cutscene, the world map (College), Level 1 "The Morning After", GO, and play the level - the
-- kitchen counter, the pizza boxes, the books, the "Important Documents" box - eating everything
-- on the way until the score passes 2000.
-- last verified: 2026-10-19, uncommitted next-work tree (desktop headless, deterministic and SMP)
--
-- HOW THE CUES WERE FOUND: `--discover` next to a shot every 50 frames, and a value search for
-- the score (`VITASLOP_MEM_SCAN` for 2200 at the end of a run, then each candidate dumped at
-- three frames: two cells went 300 -> 500 -> 2200 with the HUD; a dump around one showed its
-- object, and a scan for the object's address found the one static pointer to it).
--
--   sceKernelCreateThread "DB LoadLevel"   every level load, COUNTED from boot: 1 behind the
--                                          title, 3 for STORY MODE's cutscene, 4 for the world
--                                          map, 5 for GO - Level 1 itself
--   [0x81878d6c] + 0xC4                    the score, as an integer: a field of an object the
--                                          image's own data points at (the object is on the heap
--                                          and moves with the input history; the pointer does
--                                          not). -1 while the level's intro card waits for a
--                                          button, 0 once play begins
--
-- CONTROLS (the title's own): CROSS jumps (again against a wall climbs it), CIRCLE in the air
-- slams; the d-pad moves. Holding RIGHT into a wall and jumping does not climb it - the release
-- between jumps is what lets the blob come off the wall and jump again.
--
-- NOT YET: the end of the level. Past 2200 the blob stops at a tall cabinet that this blind
-- walk-jump-slam cycle does not get over.

local pad = cue.call("sceCtrlPeekBufferPositive")
local loads = cue.call("sceKernelCreateThread", { arg = 0, has = "DB LoadLevel" })
local level_one = loads:atleast(5)
local score = cue.u32(ptr(0x81878d6c, 0xC4))
local playing = score:is(0)

phase("title", { timeout = 120 }, function()
  wait(pad)
  idle(200)
  shot("title")
  tap("start", 10, 90)
  shot("main-menu")
end)

phase("to-level-one", { timeout = 120 }, function()
  -- STORY MODE (highlighted), the intro cutscene (CROSS skips it), the world map with COLLEGE
  -- selected, Level 1's panel, and GO - each one a CROSS. The load is the cue that GO was taken.
  repeat tap("cross", 10, 80) until try_wait(level_one, 0.01)
  split("level 1 loading")
  -- the level's intro card takes a CROSS; the score cell reads 0 once play has begun
  repeat tap("cross", 10, 60) until try_wait(playing, 0.01)
  split("level 1")
  shot("level-one")
end)

phase("play", { timeout = 400 }, function()
  local n = 0
  while score:value() < 2000 do
    n = n + 1
    press("right")
    idle(40)
    if n % 3 == 0 then
      -- a jump, then a SLAM on the way down
      press("right+cross") idle(12) press("right+circle") idle(10) release() idle(30)
    else
      tap("right+cross", 25, 10)
    end
    if n % 6 == 0 then shot(string.format("play-%02d", n)) end
  end
  release()
  split("score " .. score:value())
  shot("score-2000")
end)
