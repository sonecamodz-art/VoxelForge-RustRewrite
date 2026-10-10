//! Console variables + commands.
//!
//! A cvar is a named, typed, range-checked tunable. Unlike the C++ engine (one global object
//! per translation unit, registered by its constructor), every cvar here is a plain `static`
//! with a `const` constructor, declared in [`crate::cvars`]; the registry is that module's
//! `ALL` table. Reads are lock-free atomics (bool / int / float), so hot paths can read
//! settings every frame for free.
//!
//! Set from the console (`r.gi.reach 64`), the command line (`+set r.gi.reach 64`) or the
//! environment (`VF_SET="r.gi.reach=64;r.aa=off"`).

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering::Relaxed};

use parking_lot::{Mutex, RwLock};

/// Common interface of every cvar type (the registry and console use it).
pub trait CVar: Sync {
    fn name(&self) -> &'static str;
    fn help(&self) -> &'static str;
    fn default_str(&self) -> String;
    fn str(&self) -> String;
    fn range_str(&self) -> String;
    /// Parse + validate + apply.
    fn set_str(&self, v: &str) -> Result<(), String>;
    /// Bumps on every change (cheap "did it change since I looked" checks).
    fn version(&self) -> u32;
}

fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------
pub struct Bool {
    name: &'static str,
    help: &'static str,
    def: bool,
    v: AtomicBool,
    ver: AtomicU32,
}

impl Bool {
    pub const fn new(name: &'static str, def: bool, help: &'static str) -> Self {
        Self { name, help, def, v: AtomicBool::new(def), ver: AtomicU32::new(0) }
    }
    #[inline]
    pub fn get(&self) -> bool {
        self.v.load(Relaxed)
    }
    pub fn set(&self, v: bool) {
        if self.v.swap(v, Relaxed) != v {
            self.ver.fetch_add(1, Relaxed);
        }
    }
}

impl CVar for Bool {
    fn name(&self) -> &'static str { self.name }
    fn help(&self) -> &'static str { self.help }
    fn default_str(&self) -> String { (self.def as i32).to_string() }
    fn str(&self) -> String { (self.get() as i32).to_string() }
    fn range_str(&self) -> String { "0|1".into() }
    fn set_str(&self, v: &str) -> Result<(), String> {
        let b = parse_bool(v).ok_or_else(|| format!("'{v}' is not a bool (0/1)"))?;
        self.set(b);
        Ok(())
    }
    fn version(&self) -> u32 { self.ver.load(Relaxed) }
}

// ---------------------------------------------------------------------------------------
pub struct Int {
    name: &'static str,
    help: &'static str,
    def: i32,
    lo: i32,
    hi: i32,
    v: AtomicI32,
    ver: AtomicU32,
}

impl Int {
    pub const fn new(name: &'static str, def: i32, help: &'static str, lo: i32, hi: i32) -> Self {
        Self { name, help, def, lo, hi, v: AtomicI32::new(def), ver: AtomicU32::new(0) }
    }
    #[inline]
    pub fn get(&self) -> i32 {
        self.v.load(Relaxed)
    }
    pub fn set(&self, v: i32) {
        let v = v.clamp(self.lo, self.hi);
        if self.v.swap(v, Relaxed) != v {
            self.ver.fetch_add(1, Relaxed);
        }
    }
}

impl CVar for Int {
    fn name(&self) -> &'static str { self.name }
    fn help(&self) -> &'static str { self.help }
    fn default_str(&self) -> String { self.def.to_string() }
    fn str(&self) -> String { self.get().to_string() }
    fn range_str(&self) -> String { format!("[{}, {}]", self.lo, self.hi) }
    fn set_str(&self, v: &str) -> Result<(), String> {
        let n: i32 = v.trim().parse().map_err(|_| format!("'{v}' is not an integer"))?;
        if n < self.lo || n > self.hi {
            return Err(format!("{n} is out of range {}", self.range_str()));
        }
        self.set(n);
        Ok(())
    }
    fn version(&self) -> u32 { self.ver.load(Relaxed) }
}

// ---------------------------------------------------------------------------------------
pub struct Float {
    name: &'static str,
    help: &'static str,
    def: f32,
    lo: f32,
    hi: f32,
    bits: AtomicU32,
    ver: AtomicU32,
}

impl Float {
    pub const fn new(name: &'static str, def: f32, help: &'static str, lo: f32, hi: f32) -> Self {
        Self { name, help, def, lo, hi, bits: AtomicU32::new(def.to_bits()), ver: AtomicU32::new(0) }
    }
    #[inline]
    pub fn get(&self) -> f32 {
        f32::from_bits(self.bits.load(Relaxed))
    }
    pub fn set(&self, v: f32) {
        let v = v.clamp(self.lo, self.hi).to_bits();
        if self.bits.swap(v, Relaxed) != v {
            self.ver.fetch_add(1, Relaxed);
        }
    }
}

fn fmt_float(v: f32) -> String {
    let s = format!("{v}");
    if s.contains('e') { format!("{v:.6}") } else { s }
}

impl CVar for Float {
    fn name(&self) -> &'static str { self.name }
    fn help(&self) -> &'static str { self.help }
    fn default_str(&self) -> String { fmt_float(self.def) }
    fn str(&self) -> String { fmt_float(self.get()) }
    fn range_str(&self) -> String { format!("[{}, {}]", fmt_float(self.lo), fmt_float(self.hi)) }
    fn set_str(&self, v: &str) -> Result<(), String> {
        let f: f32 = v.trim().parse().map_err(|_| format!("'{v}' is not a number"))?;
        if !f.is_finite() || f < self.lo || f > self.hi {
            return Err(format!("{f} is out of range {}", self.range_str()));
        }
        self.set(f);
        Ok(())
    }
    fn version(&self) -> u32 { self.ver.load(Relaxed) }
}

// ---------------------------------------------------------------------------------------
/// A string cvar, optionally restricted to a fixed set of choices (an "enum").
/// Choice cvars also keep the choice's index in an atomic: `is()` / `index()` never lock.
pub struct Str {
    name: &'static str,
    help: &'static str,
    def: &'static str,
    choices: &'static [&'static str],
    v: RwLock<Option<String>>, // None = the default
    idx: AtomicI32,
    ver: AtomicU32,
}

impl Str {
    pub const fn new(name: &'static str, def: &'static str, help: &'static str, choices: &'static [&'static str]) -> Self {
        // Index of the default among the choices (const-evaluated).
        let mut idx = -1;
        let mut i = 0;
        while i < choices.len() {
            if const_str_eq(choices[i], def) {
                idx = i as i32;
            }
            i += 1;
        }
        Self { name, help, def, choices, v: RwLock::new(None), idx: AtomicI32::new(idx), ver: AtomicU32::new(0) }
    }
    pub fn get(&self) -> String {
        self.v.read().clone().unwrap_or_else(|| self.def.to_string())
    }
    /// Lock-free for choice cvars.
    #[inline]
    pub fn is(&self, s: &str) -> bool {
        if !self.choices.is_empty() {
            let i = self.idx.load(Relaxed);
            return i >= 0 && self.choices[i as usize] == s;
        }
        self.get() == s
    }
    /// Index of the current choice (choice cvars).
    #[inline]
    pub fn index(&self) -> usize {
        self.idx.load(Relaxed).max(0) as usize
    }
    pub fn set(&self, v: &str) {
        if !self.choices.is_empty() {
            match self.choices.iter().position(|c| *c == v) {
                Some(i) => self.idx.store(i as i32, Relaxed),
                None => return,
            }
        }
        let mut g = self.v.write();
        let cur = g.clone().unwrap_or_else(|| self.def.to_string());
        if cur != v {
            *g = Some(v.to_string());
            self.ver.fetch_add(1, Relaxed);
        }
    }
}

const fn const_str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

impl CVar for Str {
    fn name(&self) -> &'static str { self.name }
    fn help(&self) -> &'static str { self.help }
    fn default_str(&self) -> String { self.def.to_string() }
    fn str(&self) -> String { self.get() }
    fn range_str(&self) -> String {
        if self.choices.is_empty() { "string".into() } else { self.choices.join("|") }
    }
    fn set_str(&self, v: &str) -> Result<(), String> {
        if !self.choices.is_empty() && !self.choices.contains(&v) {
            return Err(format!("'{v}' is not one of {}", self.range_str()));
        }
        self.set(v);
        Ok(())
    }
    fn version(&self) -> u32 { self.ver.load(Relaxed) }
}

// ---------------------------------------------------------------------------------------
// Registry + console.

pub fn find(name: &str) -> Option<&'static dyn CVar> {
    crate::cvars::ALL.iter().copied().find(|c| c.name() == name)
}

type CmdFn = Box<dyn Fn(&[String]) + Send + Sync>;

struct Cmd {
    help: String,
    f: std::sync::Arc<CmdFn>,
}

static CMDS: Mutex<BTreeMap<String, Cmd>> = Mutex::new(BTreeMap::new());

/// Register a console command (replaces one of the same name).
pub fn cmd_register(name: &str, help: &str, f: impl Fn(&[String]) + Send + Sync + 'static) {
    CMDS.lock().insert(name.to_string(), Cmd { help: help.to_string(), f: std::sync::Arc::new(Box::new(f)) });
}

/// Splits `a b "c d"` -> [a, b, c d].
pub fn tokenize(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    let mut any = false;
    for ch in line.chars() {
        match quote {
            Some(q) if ch == q => quote = None,
            Some(_) => cur.push(ch),
            None if ch == '"' || ch == '\'' => {
                quote = Some(ch);
                any = true;
            }
            None if ch.is_whitespace() => {
                if any || !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    any = false;
                }
            }
            None => cur.push(ch),
        }
    }
    if any || !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// Execute console statements separated by ';': `<cvar>` prints it, `<cvar> <value>` sets
/// it, `<command> args...` runs a command.
pub fn exec(line: &str) {
    for stmt in line.split(';') {
        let toks = tokenize(stmt);
        let Some(head) = toks.first() else { continue };
        if let Some(cv) = find(head) {
            if toks.len() == 1 {
                log::info!(target: "console", "{} = \"{}\"  (default \"{}\", {})  {}", cv.name(), cv.str(),
                           cv.default_str(), cv.range_str(), cv.help());
            } else if let Err(e) = cv.set_str(&toks[1..].join(" ")) {
                log::error!(target: "console", "{}: {e}", cv.name());
            }
            continue;
        }
        let f = CMDS.lock().get(head.as_str()).map(|c| c.f.clone());
        match f {
            Some(f) => f(&toks[1..]),
            None if head == "help" => help(toks.get(1).map(|s| s.as_str()).unwrap_or("")),
            None => log::error!(target: "console", "unknown command or cvar '{head}'"),
        }
    }
}

fn help(filter: &str) {
    for cv in crate::cvars::ALL.iter().filter(|c| c.name().contains(filter)) {
        log::info!(target: "console", "{:28} = {:10} {}", cv.name(), cv.str(), cv.help());
    }
    for (n, c) in CMDS.lock().iter().filter(|(n, _)| n.contains(filter)) {
        log::info!(target: "console", "{n:28}   {}", c.help);
    }
}

/// `name = value` for every cvar (crash reports, the `cvars` command).
pub fn dump_all() -> String {
    let mut s = String::new();
    for cv in crate::cvars::ALL {
        s += &format!("{} = {}\n", cv.name(), cv.str());
    }
    s
}

/// Apply `VF_SET="a=1;b=2"`.
pub fn apply_env() {
    let Ok(s) = std::env::var("VF_SET") else { return };
    for kv in s.split(';') {
        if let Some((k, v)) = kv.split_once('=') {
            match find(k.trim()) {
                Some(cv) => {
                    if let Err(e) = cv.set_str(v.trim()) {
                        log::error!(target: "cvar", "VF_SET {k}: {e}");
                    }
                }
                None => log::error!(target: "cvar", "VF_SET: no cvar named '{k}'"),
            }
        }
    }
}

/// Apply `+set name value` and `+exec "cmd"` argument triples; returns the remaining args.
pub fn apply_args(args: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    let mut execs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "+set" && i + 2 < args.len() {
            match find(&args[i + 1]) {
                Some(cv) => {
                    if let Err(e) = cv.set_str(&args[i + 2]) {
                        log::error!(target: "cvar", "+set {}: {e}", args[i + 1]);
                    }
                }
                None => log::error!(target: "cvar", "+set: no cvar named '{}'", args[i + 1]),
            }
            i += 3;
        } else if args[i] == "+exec" && i + 1 < args.len() {
            execs.push(args[i + 1].clone());
            i += 2;
        } else {
            out.push(args[i].clone());
            i += 1;
        }
    }
    for e in execs {
        exec(&e);
    }
    out
}

/// settings.cfg: one `name "value"` per line ('#' comments).
pub fn save_archive(path: &std::path::Path, names: &[&str]) -> std::io::Result<()> {
    let mut s = String::from("# VoxelForge settings. Lines are console commands; delete the file to reset.\n");
    for n in names {
        if let Some(cv) = find(n) {
            s += &format!("{} \"{}\"\n", cv.name(), cv.str());
        }
    }
    std::fs::write(path, s)
}

pub fn load_archive(path: &std::path::Path) -> bool {
    let Ok(text) = std::fs::read_to_string(path) else { return false };
    let mut n = 0;
    for line in text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#')) {
        exec(line);
        n += 1;
    }
    log::info!(target: "cvar", "loaded {n} setting(s) from {}", path.display());
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer() {
        assert_eq!(tokenize(r#"a b "c d" 'e'"#), vec!["a", "b", "c d", "e"]);
        assert_eq!(tokenize("  x  "), vec!["x"]);
        assert_eq!(tokenize(r#"set """#), vec!["set", ""]);
    }

    #[test]
    fn typed_values() {
        static B: Bool = Bool::new("t.b", true, "");
        static I: Int = Int::new("t.i", 3, "", 0, 10);
        static F: Float = Float::new("t.f", 0.5, "", 0.0, 1.0);
        static S: Str = Str::new("t.s", "b", "", &["a", "b", "c"]);
        assert!(B.get());
        assert!(B.set_str("0").is_ok() && !B.get());
        assert!(I.set_str("11").is_err() && I.get() == 3);
        assert!(F.set_str("0.25").is_ok() && F.get() == 0.25);
        assert!(S.is("b") && S.index() == 1);
        assert!(S.set_str("z").is_err());
        S.set("c");
        assert!(S.is("c") && S.get() == "c" && S.version() == 1);
    }
}
