//! Logger: timestamped `[level][tag] message` lines to stderr and (optionally) a log file.
//!
//! Wraps the `log` crate so every module uses the standard `log::info!` / `warn!` / `error!`
//! macros; the tag is the `target:` (`log::info!(target: "vk", "...")`).

use std::fs::File;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Instant;

use parking_lot::Mutex;

struct Logger {
    start: Instant,
    file: Mutex<Option<File>>,
    path: Mutex<String>,
}

static LOGGER: OnceLock<Logger> = OnceLock::new();

/// Other crates (winit, gpu-allocator, ...) only get to say warnings and errors.
fn wanted(rec: &log::Record) -> bool {
    rec.level() <= log::Level::Warn || rec.module_path().is_none_or(|m| m.starts_with("voxelforge"))
}

impl log::Log for Logger {
    fn enabled(&self, md: &log::Metadata) -> bool {
        md.level() <= log::max_level()
    }

    fn log(&self, rec: &log::Record) {
        if !self.enabled(rec.metadata()) || !wanted(rec) {
            return;
        }
        let t = self.start.elapsed().as_secs_f64();
        let lvl = match rec.level() {
            log::Level::Error => "E",
            log::Level::Warn => "W",
            log::Level::Info => "I",
            log::Level::Debug => "D",
            log::Level::Trace => "T",
        };
        let target = rec.target();
        // Module paths (the default target) shortened to their last segment.
        let tag = target.rsplit("::").next().unwrap_or(target);
        let line = format!("{t:9.3} [{lvl}][{tag}] {}\n", rec.args());
        let _ = std::io::stderr().write_all(line.as_bytes());
        if let Some(f) = self.file.lock().as_mut() {
            let _ = f.write_all(line.as_bytes());
        }
    }

    fn flush(&self) {
        if let Some(f) = self.file.lock().as_mut() {
            let _ = f.flush();
        }
    }
}

/// Install the logger. `dir`: a directory for `voxelforge.log` (None = stderr only).
/// `VF_LOG=trace|debug|info|warn|error` sets the level (default info).
pub fn init(dir: Option<&std::path::Path>) {
    let logger = LOGGER.get_or_init(|| Logger {
        start: Instant::now(),
        file: Mutex::new(None),
        path: Mutex::new(String::new()),
    });
    if let Some(d) = dir {
        if std::fs::create_dir_all(d).is_ok() {
            let p = d.join("voxelforge.log");
            if let Ok(f) = File::create(&p) {
                *logger.file.lock() = Some(f);
                *logger.path.lock() = p.display().to_string();
            }
        }
    }
    let level = match std::env::var("VF_LOG").as_deref() {
        Ok("trace") => log::LevelFilter::Trace,
        Ok("debug") => log::LevelFilter::Debug,
        Ok("warn") => log::LevelFilter::Warn,
        Ok("error") => log::LevelFilter::Error,
        _ => log::LevelFilter::Info,
    };
    let _ = log::set_logger(logger);
    log::set_max_level(level);
}

/// Path of the log file ("" without one).
pub fn file_path() -> String {
    LOGGER.get().map(|l| l.path.lock().clone()).unwrap_or_default()
}

/// Log an error and abort: Vulkan failures and broken invariants are programmer errors.
#[track_caller]
pub fn fatal(tag: &str, msg: &str) -> ! {
    log::error!(target: tag, "FATAL: {msg}");
    log::logger().flush();
    panic!("{tag}: {msg}");
}
