-- LIMBO (PCSE00268): boot, the title, START, wake the boy in the grass, and walk him right
-- through the opening forest - over the fallen log - to the top of the slope that drops into
-- the first spike pit.
-- last verified: 2026-10-19, uncommitted next-work tree (desktop headless, deterministic and SMP)
--
-- HOW THE CUES WERE FOUND: `--discover` next to a shot every 50 frames. The title prints its
-- own script log through a call that leaves the message in r1 of `sceKernelGetTLSAddr` (the
-- next import the logger makes), so the level announces itself (frames from a deterministic
-- run):
--
--   f132   "Resuming from savepoint: 10"           a fresh boot always resumes here
--   f489   "Controls Modified (.../intro.script)"   the boy is up: the player has control
--   f1144  "activating cam script: 3147878..."      the camera zone over the fallen log
--   f1235  "activating cam script: 3147251..."      the zone at the log's end, over the slope
--   f1422  "Boy Spiked Primary (.../spikekiller.script)" / "Load from Save index: 22"
--                                                   walking on dies in the spike pit below
--
-- THE BOY LIES IN THE GRASS AFTER START until a button is pressed; START again would pause the
-- game, so the wake-up is CROSS, repeated until the intro hands over control.
--
-- NOT YET: the drop at the bottom of the slope. Holding right and jumping at any of eight
-- points from the log's end to the bottom (1 to 185 input frames after the slope zone) all
-- die on the spikes, so the way past is a move this playbook does not know yet.

local pad = cue.call("sceCtrlReadBufferPositive")
local LOG = "sceKernelGetTLSAddr"
local function says(s) return cue.call(LOG, { arg = 1, has = s }):happened() end
local control = says("Controls Modified")
local log_cam = says("activating cam script: 3147878")
local slope_cam = says("activating cam script: 3147251")
local respawns = cue.call(LOG, { arg = 1, has = "Load from Save" })

phase("title", { timeout = 120 }, function()
  wait(pad)
  idle(120)
  shot("title")
  tap("start", 10, 200)
  repeat tap("cross", 10, 10) until try_wait(control, 10)
  split("control")
  idle(30)
  shot("awake")
end)

phase("forest", { timeout = 120 }, function()
  press("right")
  wait(log_cam)
  split("the log")
  idle(20)
  shot("on-the-log")
  wait(slope_cam)
  split("the slope")
  shot("top-of-the-slope")
  release()
  idle(60)
  check(respawns:is(0), "the boy is alive at the top of the slope")
  shot("standing")
end)
