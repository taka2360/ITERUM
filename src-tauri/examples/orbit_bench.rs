//! Reference orbit throughput at several precisions.
//!   cargo run --release --example orbit_bench
//! c = i never escapes or returns to 0, so every run does the full count.

use iterum_lib::deepzoom::fixed::Fixed;
use iterum_lib::deepzoom::reference::{compute, limbs_for_bits};
use std::time::Instant;

fn main() {
    let iters = 200_000;
    println!("bits  limbs  ms/run (best of 3)  ns/iteration");
    for bits in [100u32, 400, 1100, 2100, 3400, 6600] {
        let l = limbs_for_bits(bits);
        let cr = Fixed::zero(l);
        let ci = Fixed::from_f64(1.0, l);
        let mut best = f64::INFINITY;
        for _ in 0..3 {
            let t = Instant::now();
            let o = compute(&cr, &ci, iters, &|_| false).unwrap();
            assert_eq!(o.z.len() as u32, iters + 1);
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("{bits:5} {l:5}  {best:12.1}  {:12.1}", best * 1e6 / iters as f64);
    }
}
