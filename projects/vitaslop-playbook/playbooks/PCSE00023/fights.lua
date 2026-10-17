-- Mortal Kombat (PCSE00023): boot, FIGHT -> ARCADE LADDER with the defaults (Scorpion), the VS
-- screen and BOTH pre-fight intros played out, the first fight fought with combos, specials and
-- X-ray attempts, then on through several more rounds.
--
-- Converted from the recipes `boot` (skip the logos), `fight-with-intros` (the intros played
-- out, then attacks), `fight-hard` (the same attacks) and `walk-several-fights` (CROSS every 150
-- frames through fights - the render rig). `attract-idle.lua` is the no-input control.
--
-- HOW THE CUES WERE FOUND: `--discover` on `walk-several-fights`' run (deterministic, 6000
-- frames) next to a shot every 100 frames; the round timer by value search over dumps at four
-- frames of the first fight. Every file goes through sceIoOpen with its path in r0:
--
--   f201   Movies/WBIntroLogo.mp4            the logo movies (CROSS skips each)
--   f1121  Asset/ui_ps_ladder_ps3            FIGHT -> ARCADE: the ladder and character select
--   f1751  Asset/CHAR_<name>                 both fighters load (the VS screen)
--   f1794  Asset/BGND_<stage>                the stage
--   f1816  Asset/<name>_CharIntro            the intros
--   f2100  (round timer u32 0x81f965bc < 90)  FIGHT - the round clock runs
--
-- Buttons (Vita default): square front punch, triangle back punch, cross front kick, circle back
-- kick, L+R the X-ray. A direction plus a button is a special.

local function loads(fragment) return cue.call("sceIoOpen", { arg = 0, has = fragment }) end

local fighters = loads("Asset/CHAR_"):happened()
local stages   = loads("Asset/BGND_"):trace("stages loaded")
-- 90 on the HUD before each round, counting down once the round is on, and 0 between fights.
-- Found by value search. A round is ON while it reads 1..89.
local timer    = cue.u32(0x81f965bc):trace("round timer")
local function fighting()
  local t = timer:value() or 0
  return t > 0 and t < 90
end

-- One attack string from `fight-hard`: 20-frame steps, each held 10.
local CYCLE = {
  "right+square", "right+triangle", "circle", "down+square", "cross", "left+triangle",
  "square", "triangle", "right+square", "right+triangle", "circle", "down+square", "cross",
  "left+triangle", "square", "l+r", "triangle",
}
local function attack()
  for _, b in ipairs(CYCLE) do tap(b, 10, 10) end
end

phase("boot", { timeout = 120 }, function()
  -- CROSS skips each logo movie and walks the title, MAIN MENU (FIGHT), the FIGHT menu (ARCADE
  -- LADDER), character select (the cursor's default), difficulty and the ladder - every one the
  -- default item - until both fighters load. A CROSS during the VS screen or the intros cuts
  -- them short, so the presses stop there.
  repeat tap("cross", 6, 144) until try_wait(fighters, 0.01)
  split("fighters loading")
  shot("versus")
end)

phase("first-fight", { timeout = 180 }, function()
  -- The VS screen and both fighters' intros play out with nothing pressed.
  idle(120)
  shot("intro")
  while not fighting() do idle(5) end
  split("fight")
  shot("fight-start")
  -- Attack for a minute of the round clock: hit sparks, blood, specials, X-ray attempts.
  local n = 0
  repeat
    attack()
    n = n + 1
    if n % 3 == 0 then shot(string.format("fight-%02d", n // 3)) end
  until timer:value() <= 30 or n >= 12
end)

phase("several-fights", { timeout = 600 }, function()
  -- The walk: the recipe pressed CROSS every 150 frames through fights and result screens alike;
  -- that leaves rounds to run their whole 90-second clock (~5,700 frames each, MEASURED). The
  -- attack string ends them sooner, and its CROSSes still answer the result screens. MEASURED:
  -- a lost fight's continue is a REMATCH on the same stage (no new stage loads) - so the walk is
  -- counted in ROUNDS (the clock back at 90, then running), with a shot as each one starts.
  -- Which opponent and stage come up depends on the run.
  local rounds = 1
  local was_fighting = true
  while rounds < 4 do
    attack()
    local now_fighting = fighting()
    if now_fighting and not was_fighting then
      rounds = rounds + 1
      split(string.format("round %d", rounds))
      idle(60)
      shot(string.format("round-%d", rounds))
    end
    was_fighting = now_fighting
  end
end)
