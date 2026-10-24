-- LIMBO (PCSE00268): boot, the title, START, wake the boy in the grass, walk him through the
-- opening forest and over the fallen log, JUMP the spike pit at the bottom of the slope, pull the
-- cart out from under the wooden tower, climb it, drop onto the rope and ride it down, walk to the
-- shore, and sail the boat across the lake to the far bank.
-- last verified: 2026-10-20, uncommitted next-work tree (desktop headless, SMP)
--
-- HOW THE CUES WERE FOUND: `--discover` next to a shot every few frames. The title prints its
-- own script log through a call that leaves the message in r1 of `sceKernelGetTLSAddr` (the
-- next import the logger makes), so the level announces itself (frames from one run):
--
--   f132   "Resuming from savepoint: 10"           a fresh boot always resumes here
--   f489   "Controls Modified (.../intro.script)"   the boy is up: the player has control
--   f1144  "activating cam script: 3147878..."      the camera zone over the fallen log
--   f1235  "activating cam script: 3147251..."      the zone at the log's end, over the slope
--   f1543  "SetCurrentSavePoint:  3 ..."            landed past the spike pit
--   f2611  "activating cam script: 3148125..."      down the rope, on the ground below the tower
--   f2772  "activating cam script: 3147073..."      the shore where the boat is moored
--   f3527  "activating cam script: 3146663..."      the far bank of the lake
--   "Boy Spiked Primary (.../spikekiller.script)"   the pit killed him
--   "Load from Save index: N"                       any death: the boy is put back at the save
--
-- CONTROLS (the title's own): CROSS jumps, CIRCLE grabs (held, with a direction, it pulls); the
-- d-pad walks. On the rope the boy lets go of RIGHT and holds DOWN to slide; holding RIGHT on it
-- swings him off into the gorge.
--
-- THE TWO MOVES THAT CAN KILL HIM are timed by frames, so each is retried from the save it puts
-- him back at - which is what the title is built for. MEASURED: the pit jump is cleared by a jump
-- 160 input frames after the slope zone (or 226 after the respawn's first step), and dies about
-- 20 frames either side; the tower jump lands on the beam with RIGHT held 20 frames after it.
--
-- THE BOY LIES IN THE GRASS AFTER START until a button is pressed; START again would pause the
-- game, so the wake-up is CROSS, repeated until the intro hands over control.

local pad = cue.call("sceCtrlReadBufferPositive")
local LOG = "sceKernelGetTLSAddr"
local function says(s) return cue.call(LOG, { arg = 1, has = s }):happened() end
local control = says("Controls Modified")
local log_cam = says("activating cam script: 3147878")
local slope_cam = says("activating cam script: 3147251")
local past_pit = says("SetCurrentSavePoint:  3")
local below_tower = says("activating cam script: 3148125")
local shore = says("activating cam script: 3147073")
local far_bank = says("activating cam script: 3146663")
local respawns = cue.call(LOG, { arg = 1, has = "Load from Save" })

-- After a death the save is loaded (logged twice) and the boy fades back in.
local function back_from(deaths_before)
  wait(respawns:atleast(deaths_before + 1))
  idle(180)
end

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
end)

phase("pit", { timeout = 300 }, function()
  local lead = 160
  for try = 1, 4 do
    local before = respawns:value()
    idle(lead)
    press("right+cross") idle(8) press("right")
    if wait(cue.any(past_pit, respawns:atleast(before + 1)), { timeout = 10 }) == 1 then break end
    release()
    log("the pit, try " .. try .. ": spiked")
    back_from(before)
    press("right")
    lead = 226
  end
  check(past_pit, "the boy is past the spike pit")
  split("past the pit")
  idle(20)
  shot("past-the-pit")
end)

phase("tower", { timeout = 600 }, function()
  for try = 1, 4 do
    local before = respawns:value()
    -- Walking into the cart stops him against it.
    press("right") idle(420) release() idle(10)
    if try == 1 then shot("at-the-cart") end
    press("circle") idle(10) press("circle+left") idle(90)
    release() idle(20)
    -- onto the cart, then up onto the tower
    press("right+cross") idle(8) press("right") idle(60)
    press("right+cross") idle(8) press("right") idle(20)
    release() idle(80)
    if try == 1 then shot("on-the-tower") end
    -- off the beam onto the rope, down it to the platform hanging from it, off that, and on
    press("down") idle(128)
    release() idle(120)
    press("right")
    if wait(cue.any(below_tower, respawns:atleast(before + 1)), { timeout = 15 }) == 1 then break end
    release()
    log("the tower, try " .. try .. ": fell")
    back_from(before)
  end
  check(below_tower, "the boy is down the rope below the tower")
  release()
  split("below the tower")
  shot("below-the-tower")
end)

phase("lake", { timeout = 300 }, function()
  local before = respawns:value()
  press("right")
  wait(shore)
  split("the shore")
  idle(60)
  shot("the-boat")
  wait(far_bank)
  split("the far bank")
  idle(120)
  release()
  shot("the-far-bank")
  check(respawns:is(before), "the boy reached the far bank alive")
end)
