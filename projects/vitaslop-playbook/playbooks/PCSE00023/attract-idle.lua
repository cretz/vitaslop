-- Mortal Kombat (PCSE00023): NO INPUT - sit on the title screen and let the attract-mode demo
-- fight run. The negative control for every input-driven run: whatever reaches the screen is
-- what the title chose to show, with nothing pressed. Converted from the recipe `attract-idle`
-- (the user's own repro for stray flat shapes: "stay at main screen and don't press anything").
--
-- The demo fight loads its fighters like a real one (sceIoOpen Asset/CHAR_<name>, then the
-- stage, Asset/BGND_<stage>); it draws no HUD.

local stage = cue.call("sceIoOpen", { arg = 0, has = "Asset/BGND_" }):happened()

phase("attract", { timeout = 240 }, function()
  -- The logos, the title, then the demo.
  wait(stage)
  split("demo fight loading")
  for n = 1, 4 do
    idle(300)
    shot(string.format("demo-%d", n))
  end
end)
