//! Binary layout shared with the GPU (`src/shaders/common/floatexp.wgsl`) and
//! the frontend loader (`src/phenomena/mandelbrot/orbit.ts`). Little endian.
//!
//! Header: 48 × u32
//!   [0] magic "ITRM"   [1] version   [2] request id   [3] orbit length M
//!   [4] flags (bit0: reference escaped, bit1: periodic — returned to 0,
//!       bit2: delta — see below)   [5] K = stored BLA levels
//!   [6] min log2|Z_n| (f32 bits)   [7] precision bits
//!   [8..8+K+1] BLA level offsets in entries (level l = 1..=K; last = total)
//!   [42] log2 |δc|max the table was built for (f32 bits)
//!   [43..46] milliseconds spent on orbit, BLA table, encoding (f32 bits; diagnostics)
//!   [46] delta: orbit length of the base packet   [47] delta: request id of the base
//! Orbit: M × { re: f32, im: f32, e: i32, pad: u32 }      (CFExp, 16 B)
//! BLA:   N × { a: CFExp(12 B), b: CFExp(12 B), r2: f32, r2e: i32 }  (32 B)
//!
//!
//! A delta packet extends the orbit of an earlier packet (same reference,
//! precision and |δc| bound): it holds only orbit points M_base..M and, per
//! stored level, only the entries beyond those the base length has; [8..] then
//! counts these new entries. The count of a level for an orbit of M points is
//! a function of M alone (level 0: M − 2, level l: half of level l − 1,
//! rounded down), so the receiver can place them.
//!
//! Floatexp mantissas follow the frexp convention: max(|re|, |im|) in [0.5, 1).
//! Level 0 of the BLA table (A = 2Z, B = 1) is not stored; the GPU derives it.

use super::bla::BlaEntry;
use super::fexp::{CFe, Fe};

pub const MAGIC: u32 = 0x4D52_5449;
pub const VERSION: u32 = 1;
pub const HEADER_WORDS: usize = 48;
pub const MAX_LEVELS: usize = 33;

pub struct Packet<'a> {
    pub id: u32,
    /// The orbit points in the packet (all of them, or for a delta those from base_len on).
    pub orbit: &'a [CFe],
    /// Delta packets: (base orbit length, base request id).
    pub delta: Option<(u32, u32)>,
    /// Total orbit length (orbit.len() unless delta).
    pub len: u32,
    pub escaped: bool,
    pub periodic: bool,
    pub min_log2_abs: f64,
    pub bits: u32,
    /// BLA levels (level 0 first; for a delta, only the new entries of each level).
    pub bla: &'a [Vec<BlaEntry>],
    pub dc_max_log2: f64,
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn push_cfe(out: &mut Vec<u8>, v: CFe) {
    push_u32(out, (v.re as f32).to_bits());
    push_u32(out, (v.im as f32).to_bits());
    push_u32(out, v.e as u32);
}

fn push_fe(out: &mut Vec<u8>, v: Fe) {
    push_u32(out, (v.m as f32).to_bits());
    push_u32(out, v.e as u32);
}

pub fn encode(p: &Packet) -> Vec<u8> {
    // Stored levels: 1..=K (level 0 is implicit).
    let stored: Vec<_> = p.bla.iter().skip(1).take(MAX_LEVELS - 1).collect();
    let k = stored.len();
    let total_bla: usize = stored.iter().map(|l| l.len()).sum();
    let mut out = Vec::with_capacity(HEADER_WORDS * 4 + p.orbit.len() * 16 + total_bla * 32);

    let mut header = [0u32; HEADER_WORDS];
    header[0] = MAGIC;
    header[1] = VERSION;
    header[2] = p.id;
    header[3] = p.len;
    header[4] = p.escaped as u32 | (p.periodic as u32) << 1 | (p.delta.is_some() as u32) << 2;
    header[5] = k as u32;
    header[6] = (p.min_log2_abs.max(-1e30) as f32).to_bits();
    header[7] = p.bits;
    let mut off = 0u32;
    for (i, level) in stored.iter().enumerate() {
        header[8 + i] = off;
        off += level.len() as u32;
    }
    header[8 + k] = off;
    header[42] = (p.dc_max_log2 as f32).to_bits();
    if let Some((base_len, base_id)) = p.delta {
        header[46] = base_len;
        header[47] = base_id;
    }
    for w in header {
        push_u32(&mut out, w);
    }

    for &z in p.orbit {
        push_cfe(&mut out, z);
        push_u32(&mut out, 0);
    }
    for level in stored {
        for e in level {
            push_cfe(&mut out, e.a);
            push_cfe(&mut out, e.b);
            push_fe(&mut out, e.r.mul(e.r));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepzoom::bla;

    fn f32_at(b: &[u8], word: usize) -> f32 {
        f32::from_le_bytes(b[word * 4..word * 4 + 4].try_into().unwrap())
    }
    fn u32_at(b: &[u8], word: usize) -> u32 {
        u32::from_le_bytes(b[word * 4..word * 4 + 4].try_into().unwrap())
    }

    #[test]
    fn layout_roundtrip() {
        let orbit: Vec<CFe> = (0..20).map(|i| CFe::new(0.1 * i as f64, -0.05 * i as f64, -(i as i32) * 50)).collect();
        let table = bla::build(&orbit, Fe::new(1.0, -200));
        let bytes = encode(&Packet { id: 7, orbit: &orbit, delta: None, len: 20, escaped: true, periodic: false, min_log2_abs: -3.0, bits: 128, bla: &table.levels, dc_max_log2: -200.0 });
        assert_eq!(u32_at(&bytes, 0), MAGIC);
        assert_eq!(u32_at(&bytes, 2), 7);
        assert_eq!(u32_at(&bytes, 3), 20);
        let k = u32_at(&bytes, 5) as usize;
        assert_eq!(k, table.levels.len() - 1);
        let total = u32_at(&bytes, 8 + k) as usize;
        assert_eq!(bytes.len(), HEADER_WORDS * 4 + 20 * 16 + total * 32);
        // Orbit entry 5 decodes to the same value.
        let w = HEADER_WORDS + 5 * 4;
        let (re, im, e) = (f32_at(&bytes, w), f32_at(&bytes, w + 1), u32_at(&bytes, w + 2) as i32);
        let back = CFe::new(re as f64, im as f64, e).to_c64();
        let want = orbit[5].to_c64();
        assert!(((back.0 - want.0) / want.0).abs() < 1e-7);
        assert!(((back.1 - want.1) / want.1).abs() < 1e-7);
        // First stored BLA entry is level-1 entry 0.
        let b0 = HEADER_WORDS + 20 * 4;
        let a = CFe::new(f32_at(&bytes, b0) as f64, f32_at(&bytes, b0 + 1) as f64, u32_at(&bytes, b0 + 2) as i32);
        let want = table.levels[1][0].a;
        assert!((a.abs().log2() - want.abs().log2()).abs() < 1e-6);
    }
}
