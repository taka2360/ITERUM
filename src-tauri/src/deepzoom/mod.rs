//! Deep zoom support: high-precision reference orbits and BLA tables, computed
//! off the UI thread and returned to the WebView as binary packets.

pub mod bla;
// Some helpers exist only for the verification tests.
#[cfg_attr(not(test), allow(dead_code))]
pub mod fexp;
#[cfg_attr(not(test), allow(dead_code))]
pub mod fixed;
pub mod format;
pub mod nucleus;
pub mod reference;

use fexp::{CFe, Fe};
use fixed::Fixed;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tauri::ipc::Response;
use tauri::State;

#[derive(Default)]
pub struct DeepZoomState {
    /// Id of the newest orbit request; older computations abort when it changes.
    latest: Arc<AtomicU32>,
    /// Last completed orbit, kept so the BLA table can be rebuilt for a new |δc| bound.
    cache: Mutex<Option<CachedOrbit>>,
    /// Progress of the running computation (polled by the frontend).
    progress: Arc<Mutex<Progress>>,
}

#[derive(Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub id: u32,
    /// "orbit", "bla", "encode" or "idle".
    pub phase: &'static str,
    pub iterations: u32,
    pub max_iter: u32,
    pub bits: u32,
    pub elapsed_ms: f64,
    #[serde(skip)]
    started: Option<Instant>,
}

fn set_phase(p: &Mutex<Progress>, phase: &'static str) {
    let mut g = p.lock().unwrap();
    g.phase = phase;
    if let Some(t) = g.started {
        g.elapsed_ms = t.elapsed().as_secs_f64() * 1e3;
    }
}

/// Progress of the current (or last) reference computation.
#[tauri::command]
pub fn orbit_progress(state: State<'_, DeepZoomState>) -> Progress {
    let mut p = state.progress.lock().unwrap().clone();
    if let (Some(t), false) = (p.started, p.phase == "idle") {
        p.elapsed_ms = t.elapsed().as_secs_f64() * 1e3;
    }
    p
}

struct CachedOrbit {
    id: u32,
    orbit: Arc<Vec<CFe>>,
    escaped: bool,
    periodic: bool,
    min_log2_abs: f64,
    bits: u32,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrbitRequest {
    pub id: u32,
    /// Coordinates as signed hex integers scaled by 2^frac_bits.
    pub re: String,
    pub im: String,
    pub frac_bits: u32,
    pub bits: u32,
    pub max_iter: u32,
    pub dc_max_log2: f64,
}

fn parse_center(req_re: &str, req_im: &str, frac_bits: u32, bits: u32) -> Result<(Fixed, Fixed), String> {
    let limbs = reference::limbs_for_bits(bits);
    Ok((Fixed::from_hex(req_re, frac_bits, limbs)?, Fixed::from_hex(req_im, frac_bits, limbs)?))
}

#[tauri::command]
pub async fn reference_orbit(state: State<'_, DeepZoomState>, req: OrbitRequest) -> Result<Response, String> {
    state.latest.store(req.id, Ordering::SeqCst);
    let latest = state.latest.clone();
    let (cr, ci) = parse_center(&req.re, &req.im, req.frac_bits, req.bits)?;
    let id = req.id;
    let progress = state.progress.clone();
    *progress.lock().unwrap() = Progress {
        id,
        phase: "orbit",
        iterations: 0,
        max_iter: req.max_iter,
        bits: req.bits,
        elapsed_ms: 0.0,
        started: Some(Instant::now()),
    };
    let result = tauri::async_runtime::spawn_blocking(move || {
        let t0 = Instant::now();
        let orbit = reference::compute(&cr, &ci, req.max_iter, &|n| {
            let mut p = progress.lock().unwrap();
            if p.id == id {
                p.iterations = n;
            }
            drop(p);
            latest.load(Ordering::Relaxed) != id
        })?;
        let t1 = Instant::now();
        progress.lock().unwrap().iterations = orbit.z.len() as u32 - 1;
        set_phase(&progress, "bla");
        let table = bla::build(&orbit.z, Fe::from_log2(req.dc_max_log2));
        let t2 = Instant::now();
        set_phase(&progress, "encode");
        let bytes = format::encode(&format::Packet {
            id,
            orbit: &orbit.z,
            escaped: orbit.escaped,
            periodic: orbit.periodic,
            min_log2_abs: orbit.min_log2_abs,
            bits: req.bits,
            bla: &table,
            dc_max_log2: req.dc_max_log2,
        });
        println!(
            "[iterum] orbit id={id} len={} bits={} escaped={} periodic={} orbit={:.1}ms bla={:.1}ms bytes={}",
            orbit.z.len(),
            req.bits,
            orbit.escaped,
            orbit.periodic,
            (t1 - t0).as_secs_f64() * 1e3,
            (t2 - t1).as_secs_f64() * 1e3,
            bytes.len()
        );
        set_phase(&progress, "idle");
        Some((bytes, orbit))
    })
    .await
    .map_err(|e| e.to_string())?;
    let Some((bytes, orbit)) = result else {
        return Err("cancelled".into());
    };
    *state.cache.lock().unwrap() = Some(CachedOrbit {
        id,
        orbit: Arc::new(orbit.z),
        escaped: orbit.escaped,
        periodic: orbit.periodic,
        min_log2_abs: orbit.min_log2_abs,
        bits: req.bits,
    });
    Ok(Response::new(bytes))
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlaRequest {
    pub orbit_id: u32,
    pub dc_max_log2: f64,
}

/// Rebuilds the BLA table of the cached orbit for a different |δc| bound.
#[tauri::command]
pub async fn bla_table(state: State<'_, DeepZoomState>, req: BlaRequest) -> Result<Response, String> {
    let (orbit, escaped, periodic, min_log2_abs, bits) = {
        let cache = state.cache.lock().unwrap();
        match cache.as_ref() {
            Some(c) if c.id == req.orbit_id => (c.orbit.clone(), c.escaped, c.periodic, c.min_log2_abs, c.bits),
            _ => return Err("orbit not cached".into()),
        }
    };
    let bytes = tauri::async_runtime::spawn_blocking(move || {
        let table = bla::build(&orbit, Fe::from_log2(req.dc_max_log2));
        format::encode(&format::Packet {
            id: req.orbit_id,
            orbit: &orbit,
            escaped,
            periodic,
            min_log2_abs,
            bits,
            bla: &table,
            dc_max_log2: req.dc_max_log2,
        })
    })
    .await
    .map_err(|e| e.to_string())?;
    Ok(Response::new(bytes))
}

#[tauri::command]
pub fn cancel_orbit(state: State<'_, DeepZoomState>) {
    state.latest.store(0, Ordering::SeqCst);
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectRequest {
    pub re: String,
    pub im: String,
    pub frac_bits: u32,
    pub bits: u32,
    pub max_iter: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectResponse {
    pub iterations: u32,
    pub escaped: bool,
    pub final_re: f64,
    pub final_im: f64,
}

/// Verification oracle: iterates one point directly at full precision.
#[tauri::command]
pub async fn direct_iterate(req: DirectRequest) -> Result<DirectResponse, String> {
    let (cr, ci) = parse_center(&req.re, &req.im, req.frac_bits, req.bits)?;
    tauri::async_runtime::spawn_blocking(move || {
        let r = reference::iterate_direct(&cr, &ci, req.max_iter);
        DirectResponse { iterations: r.iterations, escaped: r.escaped, final_re: r.final_re, final_im: r.final_im }
    })
    .await
    .map_err(|e| e.to_string())
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectBatchRequest {
    /// Points as (re, im) signed hex integers scaled by 2^frac_bits.
    pub points: Vec<(String, String)>,
    pub frac_bits: u32,
    pub bits: u32,
    pub max_iter: u32,
}

/// Verification oracle for many points at once, spread over all CPU cores.
#[tauri::command]
pub async fn direct_batch(req: DirectBatchRequest) -> Result<Vec<DirectResponse>, String> {
    let points = req
        .points
        .iter()
        .map(|(re, im)| parse_center(re, im, req.frac_bits, req.bits))
        .collect::<Result<Vec<_>, _>>()?;
    let max_iter = req.max_iter;
    tauri::async_runtime::spawn_blocking(move || {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let chunk = points.len().div_ceil(threads).max(1);
        let mut out: Vec<DirectResponse> = Vec::with_capacity(points.len());
        std::thread::scope(|s| {
            let handles: Vec<_> = points
                .chunks(chunk)
                .map(|ch| {
                    s.spawn(move || {
                        ch.iter()
                            .map(|(cr, ci)| {
                                let r = reference::iterate_direct(cr, ci, max_iter);
                                DirectResponse { iterations: r.iterations, escaped: r.escaped, final_re: r.final_re, final_im: r.final_im }
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for h in handles {
                out.extend(h.join().unwrap());
            }
        });
        out
    })
    .await
    .map_err(|e| e.to_string())
}
