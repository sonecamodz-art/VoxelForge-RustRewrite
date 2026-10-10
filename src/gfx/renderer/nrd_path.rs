//! NVIDIA NRD as a denoiser (r.denoiser nrd): the renderer's side.

use super::record::Rec;
use super::Renderer;

impl Renderer {
    /// NRD can run (built with --features nrd and its instance made).
    pub fn nrd_available(&self) -> bool {
        false
    }

    /// RR's rays -> NRD -> shaded like RR into gb_color_hdr.
    pub fn record_nrd(&mut self, r: &Rec, _frame: u32) {
        self.record_raw(r);
    }
}
