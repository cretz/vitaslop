/* An embedded LuaJIT running a fixed script with its JIT compiler switched OFF, so every
 * bytecode runs in LuaJIT's interpreter.
 *
 * That interpreter is the shape the runtime-code path exists for: its opcode handlers are
 * hand-written ARM reached only through a dispatch table LuaJIT builds at start-up from a base
 * address and a table of offsets. No call names a handler, no relocation points at one, so the
 * transpiler cannot discover them ahead of time and the dispatcher translates them as they are
 * first reached. The handlers also end by jumping to the next one's address rather than
 * returning, so a long loop runs on a bounded stack only if those jumps are tail calls.
 *
 * The script prints a deterministic transcript (no table-order dependence) covering the
 * interpreter's main paths: integer and float loops, calls and recursion, closures, tables,
 * strings, sorting with a Lua comparator, metatables, coroutines, errors, varargs, the bit
 * library and goto. The golden is in ../cases/luajit.toml.
 */
#include <stdio.h>
#ifdef __vita__
#include <psp2/kernel/clib.h>
#else
/* The desktop build that produced the golden. */
#define sceClibPrintf printf
#endif
#include <lua.h>
#include <lualib.h>
#include <lauxlib.h>
#include <luajit.h>

static const char SCRIPT[] =
    "local p = print\n"
    "local s = 0\n"
    "for i = 1, 200000 do s = s + i % 7 * 3 - (i % 5) end\n"
    "p('loop', s)\n"
    "local f = 0.0\n"
    "for i = 1, 1000 do f = f + math.sqrt(i) * 0.5 end\n"
    "p('float', string.format('%.6f', f))\n"
    "local function fib(n) if n < 2 then return n end return fib(n - 1) + fib(n - 2) end\n"
    "p('fib', fib(22))\n"
    "local function counter() local c = 0 return function() c = c + 1 return c end end\n"
    "local c1, c2 = counter(), counter()\n"
    "c1() c1() c2()\n"
    "p('closure', c1(), c2())\n"
    "local t = {}\n"
    "for i = 1, 1000 do t[i] = i * i end\n"
    "local sum = 0 for _, v in ipairs(t) do sum = sum + v end\n"
    "p('ipairs', #t, sum)\n"
    "local h = {} for i = 1, 500 do h['k' .. i] = i end\n"
    "local hs, hn = 0, 0 for _, v in pairs(h) do hs = hs + v hn = hn + 1 end\n"
    "p('pairs', hn, hs)\n"
    "local str = string.rep('ab', 5) .. '|' .. string.upper('lua')\n"
    "p('string', str, #str, (str:find('|')), (str:gsub('a', 'x')))\n"
    "p('format', string.format('%d %5.2f %s %x', 42, 3.14159, 'ok', 255))\n"
    "local arr, seed = {}, 12345\n"
    "for i = 1, 300 do seed = (seed * 1103515245 + 12345) % 2147483648 arr[i] = seed % 1000 end\n"
    "table.sort(arr, function(a, b) return a > b end)\n"
    "p('sort', arr[1], arr[150], arr[300])\n"
    "local V = {} V.__index = V\n"
    "V.__add = function(a, b) return setmetatable({x = a.x + b.x}, V) end\n"
    "function V.new(x) return setmetatable({x = x}, V) end\n"
    "function V:get() return self.x end\n"
    "p('meta', (V.new(3) + V.new(4)):get())\n"
    "local co = coroutine.wrap(function(a) local b = coroutine.yield(a + 1) return b * 2 end)\n"
    "p('coroutine', co(1), co(10))\n"
    "local ok, err = pcall(function() error('boom', 0) end)\n"
    "p('pcall', ok, err)\n"
    "local function va(...) return select('#', ...), ... end\n"
    "p('varargs', va(1, nil, 3))\n"
    "p('bit', bit.band(0xff0, 0x0ff), bit.bor(1, 2), bit.lshift(1, 10), bit.tohex(0xdead))\n"
    "local g = 0\n"
    "for i = 1, 10 do if i % 2 == 0 then goto cont end g = g + i ::cont:: end\n"
    "p('goto', g)\n"
    "local parts = {} for i = 1, 100 do parts[#parts + 1] = tostring(i) end\n"
    "p('concat', #table.concat(parts, ','))\n";

/* LuaJIT's `package` and `ffi` libraries can load shared objects; the Vita has no dlopen, and the
 * script loads nothing, so these only have to say no. */
void *dlopen(const char *file, int mode) { (void)file; (void)mode; return NULL; }
void *dlsym(void *handle, const char *name) { (void)handle; (void)name; return NULL; }
int dlclose(void *handle) { (void)handle; return 0; }
char *dlerror(void) { return "no shared objects"; }

/* A ladder of small chunks run before the script, each printing its rung on success: when the
 * interpreter goes wrong, the last rung printed names the first construct it got wrong. */
static const char *const LADDER[] = {
    "",
    "return",
    "local a",
    "local a = 1",
    "local a = 'x'",
    "local a = 1 + 2",
    "local p = print",
    "print('rung')",
    "local t = {} t[1] = 2",
    "local s = 0 for i = 1, 10 do s = s + i end print(s)",
    "local function f(x) return x * 2 end print(f(21))",
    "print(string.format('%d', 7))",
    "print(bit.bnot(0), bit.bswap(1), bit.tobit(5))",
    "print(bit.band(12), bit.band(12, 10), bit.bor(1, 2), bit.lshift(1, 4))",
};

/* Output goes through the system's own `sceClibPrintf` - what a retail title's libc uses - so
 * the transcript depends on the interpreter, not on the C library's stdio. Lua's `print` is
 * replaced with the same: its arguments through `tostring`, tab-separated, then a newline. */
static int print(lua_State *L) {
    int n = lua_gettop(L);
    lua_getglobal(L, "tostring");
    for (int i = 1; i <= n; i++) {
        lua_pushvalue(L, -1);
        lua_pushvalue(L, i);
        lua_call(L, 1, 1);
        sceClibPrintf("%s%s", i > 1 ? "\t" : "", lua_tostring(L, -1));
        lua_pop(L, 1);
    }
    sceClibPrintf("\n");
    return 0;
}

int main(void) {
    sceClibPrintf("start\n");
    lua_State *L = luaL_newstate();
    if (!L) {
        sceClibPrintf("luaL_newstate failed\n");
        return 1;
    }
    luaL_openlibs(L);
    lua_register(L, "print", print);
    /* The interpreter only: no trace is ever recorded, so no machine code is ever made. After
     * the libraries, because opening the `jit` library switches the compiler on. */
    luaJIT_setmode(L, 0, LUAJIT_MODE_ENGINE | LUAJIT_MODE_OFF);
    for (unsigned i = 0; i < sizeof LADDER / sizeof LADDER[0]; i++) {
        if (luaL_loadstring(L, LADDER[i]) || lua_pcall(L, 0, 0, 0)) {
            sceClibPrintf("rung %u: %s\n", i, lua_tostring(L, -1));
            return 1;
        }
        sceClibPrintf("ok %u\n", i);
    }
    if (luaL_loadbuffer(L, SCRIPT, sizeof SCRIPT - 1, "=probe") || lua_pcall(L, 0, 0, 0)) {
        sceClibPrintf("error: %s\n", lua_tostring(L, -1));
        return 1;
    }
    lua_close(L);
    sceClibPrintf("done\n");
    return 0;
}
