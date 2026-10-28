//! Two-pass risk-set matching at register-study size: 400,000 people, 30,000
//! cases, exact match on sex and municipality, then on sex and region.
//!
//! Run with `cargo bench --bench transition_strata`. Prints the wall time of
//! building the records and matching, over a few repetitions.

#[path = "../tests/common/mod.rs"]
mod common;

use std::time::Instant;

unsafe extern "C" {
    /// Processor time used by the process, in microseconds on macOS and Linux.
    fn clock() -> std::ffi::c_long;
}

fn cpu_seconds() -> f64 {
    // SAFETY: `clock` takes no arguments and has no preconditions.
    let ticks = unsafe { clock() };
    ticks as f64 / 1e6
}

fn main() {
    let people_count: usize = std::env::var("PEOPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(400_000);
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let people = common::cohort(people_count, 75, 98, 2024);
    let cases = people.iter().filter(|p| p.exposed).count();
    println!("people {people_count}, exposed {cases}");

    let mut times = Vec::new();
    let mut cpu_times = Vec::new();
    for run in 0..runs {
        let cpu_start = cpu_seconds();
        let start = Instant::now();
        let (p1, p2) = common::two_pass(&people, 7, false);
        let elapsed = start.elapsed().as_secs_f64();
        let cpu = cpu_seconds() - cpu_start;
        println!(
            "run {run}: wall {elapsed:.3} s, cpu {cpu:.3} s, pass 1 pairs {} matched {} unmatched {}, pass 2 pairs {} matched {}",
            p1.pairs.len(),
            p1.matched_cases,
            p1.unmatched_cases,
            p2.pairs.len(),
            p2.matched_cases
        );
        times.push(elapsed);
        cpu_times.push(cpu);
    }
    times.sort_by(f64::total_cmp);
    cpu_times.sort_by(f64::total_cmp);
    println!(
        "median wall {:.3} s, cpu {:.3} s",
        times[times.len() / 2],
        cpu_times[cpu_times.len() / 2]
    );
}
