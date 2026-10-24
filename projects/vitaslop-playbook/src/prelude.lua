-- The playbook API: every global a playbook can call.
--
-- >>> THE VM NEVER CALLS THE HOST. Each API call below yields ONE request (a JSON string) out
-- of the playbook's coroutine; the engine does the work and resumes the coroutine with the
-- reply (also JSON). A Lua implementation therefore needs to do exactly two things for us -
-- run this file, and call a global function with a string and return its string - which is
-- what keeps the VM swappable (see `vm.rs`). Nothing here may depend on anything else.

local yield, resume, status = coroutine.yield, coroutine.resume, coroutine.status

-- Lua 5.4 seeds math.random from the clock at startup; a playbook must replay the same way
-- every run, so it starts from a fixed seed.
math.randomseed(0)

-- "<playbook>:<line>: " for the innermost frame that is not this file, so an error raised
-- from deep inside the API names the playbook line that made the call. Needs the debug
-- library; without it the message carries no location.
local function where()
  if not debug or not debug.getinfo then return "" end
  for level = 2, 32 do
    local d = debug.getinfo(level, "Sl")
    if not d then break end
    if d.source ~= "=prelude" and d.currentline and d.currentline > 0 then
      return d.short_src .. ":" .. d.currentline .. ": "
    end
  end
  return ""
end

---------------------------------------------------------------------------------------------
-- JSON, just enough for the wire: objects, arrays, strings, numbers, booleans, null.
---------------------------------------------------------------------------------------------

local escapes = { ['"'] = '\\"', ['\\'] = '\\\\', ['\n'] = '\\n', ['\r'] = '\\r', ['\t'] = '\\t' }

local function encode(v)
  local t = type(v)
  if t == "nil" then return "null" end
  if t == "boolean" then return tostring(v) end
  if t == "number" then
    if v ~= v or v == math.huge or v == -math.huge then error(where() .. "cannot send a non-finite number", 0) end
    if math.type(v) == "integer" then return string.format("%d", v) end
    return string.format("%.17g", v)
  end
  if t == "string" then
    return '"' .. v:gsub('[%c"\\]', function(c) return escapes[c] or string.format("\\u%04x", c:byte()) end) .. '"'
  end
  if t == "table" then
    -- A cue travels as its id.
    if rawget(v, "__cue") then return string.format("%d", v.__cue) end
    local out = {}
    if v[1] ~= nil then
      for i = 1, #v do out[i] = encode(v[i]) end
      return "[" .. table.concat(out, ",") .. "]"
    end
    for k, x in pairs(v) do out[#out + 1] = encode(tostring(k)) .. ":" .. encode(x) end
    return "{" .. table.concat(out, ",") .. "}"
  end
  error(where() .. "cannot send a " .. t, 0)
end

local function decode(s)
  local i = 1
  local function ws() i = s:find("[^ \t\r\n]", i) or #s + 1 end
  local value
  local function str()
    local out, j = {}, i + 1
    while true do
      local c = s:sub(j, j)
      if c == '"' then i = j + 1; return table.concat(out) end
      if c == "\\" then
        local e = s:sub(j + 1, j + 1)
        if e == "u" then
          out[#out + 1] = utf8.char(tonumber(s:sub(j + 2, j + 5), 16)); j = j + 6
        else
          out[#out + 1] = ({ n = "\n", r = "\r", t = "\t", b = "\b", f = "\f" })[e] or e; j = j + 2
        end
      elseif c == "" then error("unterminated string in a reply")
      else out[#out + 1] = c; j = j + 1 end
    end
  end
  value = function()
    ws()
    local c = s:sub(i, i)
    if c == "{" then
      local o = {}; i = i + 1; ws()
      if s:sub(i, i) == "}" then i = i + 1; return o end
      while true do
        ws(); local k = str(); ws(); i = i + 1 -- ':'
        o[k] = value(); ws()
        local d = s:sub(i, i); i = i + 1
        if d == "}" then return o end
      end
    elseif c == "[" then
      local a = {}; i = i + 1; ws()
      if s:sub(i, i) == "]" then i = i + 1; return a end
      while true do
        a[#a + 1] = value(); ws()
        local d = s:sub(i, i); i = i + 1
        if d == "]" then return a end
      end
    elseif c == '"' then return str()
    elseif s:sub(i, i + 3) == "true" then i = i + 4; return true
    elseif s:sub(i, i + 4) == "false" then i = i + 5; return false
    elseif s:sub(i, i + 3) == "null" then i = i + 4; return nil
    else
      local n = s:match("^-?[%d.eE+-]+", i); i = i + #n
      return math.tointeger(tonumber(n)) or tonumber(n)
    end
  end
  return value()
end

---------------------------------------------------------------------------------------------
-- The request channel.
---------------------------------------------------------------------------------------------

-- Send one request and return the reply. A host-side refusal is raised here, located at the
-- playbook line that made the bad call rather than at this file.
local function req(t)
  local r = decode(yield(encode(t)))
  if type(r) == "table" and r.error then error(where() .. r.error, 0) end
  return r
end

---------------------------------------------------------------------------------------------
-- Cues.
---------------------------------------------------------------------------------------------

local Cue = {}
Cue.__index = Cue
Cue.__tostring = function(c) return "cue#" .. c.__cue end

local function declare(def)
  return setmetatable({ __cue = req({ op = "cue", def = def }).id }, Cue)
end

cue = {}

--- A host call the game makes. `opts.arg` (0-3) with `opts.has` counts only calls whose
--- string argument in that register contains `has` (a file path, a movie name).
function cue.call(name, opts)
  opts = opts or {}
  return declare({ kind = "call", name = name, arg = opts.arg, has = opts.has })
end

--- `ptr(base, off1, off2, ...)`: start at `base`, and for each offset read the pointer
--- there and add the offset - `ptr(0x81c47d30, 0x3c)` is "the field 0x3c into the object
--- whose pointer lives at 0x81c47d30".
function ptr(base, ...)
  return { __ptr = true, base = base, offs = { ... } }
end

for _, ty in ipairs({ "u8", "u16", "u32", "i8", "i16", "i32", "f32" }) do
  cue[ty] = function(at)
    if type(at) == "number" then at = ptr(at) end
    return declare({ kind = "mem", ty = ty, base = at.base, offs = at.offs })
  end
end

--- What the player SEES: how many pixels of a rectangle of the presented frame (960x544
--- panel coordinates) are within `tol` (default 24) of `rgb` on every channel.
--- `cue.pixels{ x = 470, y = 482, w = 6, h = 6, rgb = { 255, 255, 255 } }` - for a HUD element
--- that has no memory cell to find, such as a shot gauge's moving cursor.
function cue.pixels(o)
  return declare({ kind = "pixels", x = o.x, y = o.y, w = o.w or 1, h = o.h or 1, rgb = o.rgb, tol = o.tol or 24 })
end

--- WHERE the first pixel of a colour is in a rectangle (rows top to bottom), for a marker that
--- moves from shot to shot. Its value is `y * 1024 + x`; `:xy()` splits it (nil, nil if none).
function cue.locate(o)
  return declare({ kind = "locate", x = o.x, y = o.y, w = o.w or 1, h = o.h or 1, rgb = o.rgb, tol = o.tol or 24 })
end

--- The AVERAGE colour of a rectangle, packed `0xRRGGBB`; `:rgb()` splits it. For a colour the
--- title picks at run time (a table's cloth), read once and then handed to `cue.pixels`.
function cue.color(o)
  return declare({ kind = "color", x = o.x, y = o.y, w = o.w or 1, h = o.h or 1 })
end

--- Arm-relative durations: `n` input frames, or `s` guest seconds, after the wait begins.
function cue.frames(n) return declare({ kind = "frames", n = n }) end
function cue.seconds(s) return declare({ kind = "seconds", s = s }) end

--- The first of these to fire. A wait on it returns the 1-based INDEX of the one that did.
function cue.any(...) return declare({ kind = "any", of = { ... } }) end
--- Every one of these, each latched once it has fired since the wait began.
function cue.all(...) return declare({ kind = "all", of = { ... } }) end

local function cmp(op)
  return function(self, v) return declare({ kind = "cmp", src = self, op = op, value = v }) end
end
Cue.is, Cue.isnt = cmp("=="), cmp("!=")
Cue.above, Cue.below = cmp(">"), cmp("<")
Cue.atleast, Cue.atmost = cmp(">="), cmp("<=")
--- A call cue has fired at least once since it was declared - a CONDITION, so it stays true.
--- `repeat tap("start") until try_wait(title:happened(), 1)` cannot miss a call made while
--- the tap was down, which `try_wait(title, 1)` (the next call after the wait starts) can.
function Cue:happened() return self:atleast(1) end
--- The value changed on this input frame.
function Cue:changed() return declare({ kind = "changed", src = self }) end
--- The value has not changed for `n` input frames (menus flicker; this debounces).
function Cue:stable(n) return declare({ kind = "stable", src = self, n = n }) end
--- Fires on this cue's first firing after `other` has fired.
function Cue:after(other) return declare({ kind = "after", a = self, b = other }) end
--- The cue's value now: a call's count, a memory value, a condition's truth.
function Cue:value() return req({ op = "value", cue = self }).v end
--- A `cue.color` value as three numbers 0..255 (nil if no frame yet).
function Cue:rgb()
  local v = self:value()
  if v == nil then return nil end
  return { math.floor(v / 65536) % 256, math.floor(v / 256) % 256, v % 256 }
end
--- A `cue.locate`'s position now: x, y in panel coordinates, or nil, nil.
function Cue:xy()
  local v = self:value()
  if v == nil then return nil, nil end
  return v % 1024, v // 1024
end
--- Log every change of this cue's value, under `label`.
function Cue:trace(label) req({ op = "trace", cue = self, label = label or tostring(self) }); return self end

---------------------------------------------------------------------------------------------
-- Phases, waiting, input, observation.
---------------------------------------------------------------------------------------------

--- A named block. `opts.timeout` (guest seconds) bounds the whole phase, and every wait
--- inside it that names no timeout of its own.
function phase(name, opts, fn)
  if type(opts) == "function" then fn, opts = opts, {} end
  req({ op = "phase", name = name, timeout = opts.timeout })
  fn()
  req({ op = "end_phase", name = name })
end

--- Block until `c` fires; returns its value. `opts.timeout` in guest seconds; `opts.soft`
--- returns nil on a timeout instead of failing the run.
function wait(c, opts)
  opts = opts or {}
  return req({ op = "wait", cue = c, timeout = opts.timeout, soft = opts.soft }).v
end

--- `wait` that returns nil instead of failing when `timeout` passes.
function try_wait(c, timeout) return wait(c, { timeout = timeout, soft = true }) end

local function buttons(...)
  local out = {}
  for _, b in ipairs({ ... }) do
    for name in tostring(b):gmatch("[^+]+") do out[#out + 1] = name end
  end
  return out
end

--- Hold these buttons down until released. `press("start+cross")` or `press("l", "r")`.
function press(...) req({ op = "press", buttons = buttons(...) }) end
--- Let these buttons go - all of them when none are named.
function release(...) req({ op = "release", buttons = buttons(...) }) end
--- Hold `b` for `n` input frames, then let go.
function hold(b, n) req({ op = "hold", buttons = buttons(b), frames = n }) end
--- Press and let go: down `down` input frames (default 6), then up `up` (default 6).
function tap(b, down, up) hold(b, down or 6); idle(up or 6) end
--- Do nothing for `n` input frames.
function idle(n) req({ op = "idle", frames = n }) end
--- Set a stick ("left" or "right") to raw axes 0..255; no axes recentres it.
function stick(which, x, y) req({ op = "stick", which = which, x = x or 128, y = y or 128 }) end
--- Put one finger on the front panel at PANEL coordinates (0..1919 x 0..1087): a point read off a
--- 960x544 shot is doubled.
function touch(x, y) req({ op = "touch", x = x, y = y }) end
function untouch() req({ op = "touch" }) end
--- Put one finger on the REAR panel at panel coordinates (0..1919 x 0..889; no screen under it).
function back_touch(x, y) req({ op = "touch", panel = "back", x = x, y = y }) end
function back_untouch() req({ op = "touch", panel = "back" }) end

--- Write a screenshot named `name` of the next rendered frame.
function shot(name) req({ op = "shot", name = name }) end
--- A line in the run's log.
function log(...)
  local parts = {}
  for i = 1, select("#", ...) do parts[i] = tostring(select(i, ...)) end
  req({ op = "log", text = table.concat(parts, " ") })
end
print = log
--- A named point on the run's timeline (input frame, display frame, guest seconds).
function split(name) req({ op = "split", name = name }) end
--- Fail the run unless condition cue `c` holds right now.
function check(c, msg) req({ op = "check", cue = c, msg = msg }) end
--- Fail the run.
function fail(msg) error(msg, 2) end
--- input frame, display frame, guest seconds.
function now() local r = req({ op = "now" }); return r.input, r.display, r.seconds end

---------------------------------------------------------------------------------------------
-- The engine's two entry points. Everything above runs inside the playbook coroutine.
---------------------------------------------------------------------------------------------

local co

-- One string argument, "<name>\n<source>" - every backend call takes exactly one (see vm.rs).
function __playbook_load(arg)
  local cut = arg:find("\n", 1, true)
  local name, source = arg:sub(1, cut - 1), arg:sub(cut + 1)
  local fn, err = load(source, "@" .. name, "t")
  if not fn then return encode({ error = err }) end
  co = coroutine.create(fn)
  return "{}"
end

function __playbook_resume(reply)
  local ok, out = resume(co, reply)
  if not ok then
    local msg = tostring(out)
    if debug and debug.traceback then msg = debug.traceback(co, msg) end
    return encode({ op = "error", message = msg })
  end
  if status(co) == "dead" then return encode({ op = "done" }) end
  return out
end
