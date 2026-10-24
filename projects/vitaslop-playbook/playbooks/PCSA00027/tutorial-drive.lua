-- MotorStorm RC (PCSA00027): boot, accept the ONLINE TERMS, reach the MAIN SCREEN (a top-down
-- RC car sandbox in a skatepark - the render target), walk the tutorial's intro dialogs, and
-- DRIVE: accelerate, steer, reverse out of a wall, drive off on a new heading.
--
-- Converted from the old frame-keyed scripts `main-screen` (boot to the sandbox), `gameplay-prefix` (the same,
-- plus the dialogs, ending with nothing held) and `tutorial-drive` (the prefix, then the drive).
-- Each old script's end point is a phase here.
--
-- HOW THE CUES WERE FOUND: `--discover` on `tutorial-drive`'s run (deterministic, 2500 frames)
-- next to a shot every 60 frames (frames from that run):
--
--   f660   (no call)                             the ONLINE TERMS panel is up, scrolling
--   f825   sceNpCommerce2StartEmptyStoreCheck    the terms were accepted
--   f874   sceIoOpen track_rc_ms_playground.spvs the sandbox track loads behind the dialogs
--
-- A save-less first boot lands on the terms and waits there for ever; on hardware the accept
-- is saved. CROSS accepts. The legal splash, the logos and a "Communicating with server"
-- spinner before it take no input.
--
-- THE CONTROL SCHEME (got wrong twice, in both directions - see the old script's history):
--   RIGHT stick UP (ry=0) accelerate, RIGHT stick DOWN (ry=255) brake / reverse,
--   LEFT stick X steers as a TURN RATE (lx>128 one way, lx<128 the other); `ly` is NOT read.
-- Measured over 90 frames at full throttle: lx=218 turned about -90 degrees, lx=255 about
-- -254, lx=38 about +80, lx=1 about +267. The legend's L / R / R glyphs are the two STICKS,
-- not the shoulder buttons (`l` and `r` do nothing here).

local accepted = cue.call("sceNpCommerce2StartEmptyStoreCheck"):happened()
local track    = cue.call("sceIoOpen", { arg = 0, has = "track_rc_ms_playground" }):happened()

phase("main-screen", { timeout = 120 }, function()
  -- CROSS until the terms are accepted (a CROSS on the logos does nothing).
  repeat tap("cross", 12, 48) until try_wait(accepted, 0.01)
  split("terms accepted")
  wait(track)
  split("sandbox loading")
  -- The sandbox fades in behind the "Welcome to the MotorStorm RC tutorial" dialog.
  idle(240)
  shot("main-screen")
end)

phase("gameplay-prefix", { timeout = 60 }, function()
  -- Three dialogs walk the tutorial intro ("Welcome...", "There are 2 control schemes...", and
  -- one more). Each waits indefinitely and makes no call, so they are answered 150 frames
  -- apart - far enough that the next is up before the next press. After them the tutorial
  -- reads "Follow the trail to try out the new handling" and the car is the player's.
  for _ = 1, 3 do tap("cross", 12, 138) end
  idle(100)
  shot("gameplay")
end)

phase("drive", { timeout = 60 }, function()
  -- Accelerate away from the start pad, steer left, straighten up.
  stick("right", 128, 0); idle(120)
  shot("drive-accelerated")
  stick("left", 70, 128); idle(60)
  stick("left", 128, 128); idle(120)
  shot("drive-crossed-ground")
  -- The park is enclosed and the car will meet a wall: reverse out, turning, settle, then drive
  -- away on the new heading - the manoeuvre the rest of the tutorial needs.
  stick("right", 128, 255); idle(90)
  stick("left", 40, 128); idle(60)
  stick("right", 128, 128); stick("left", 128, 128); idle(20)
  stick("right", 128, 0); stick("left", 40, 128); idle(90)
  stick("left", 128, 128); idle(90)
  shot("drive-recovered")
  stick("right", 128, 128)
  idle(30)
end)
