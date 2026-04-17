use cohort_matching::{benchmark_ecdf_distance_stats, benchmark_eqq_distance_stats};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn ecdf_distance_stats_reference(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    if case_values.is_empty() || control_values.is_empty() {
        return (0.0, 0.0);
    }

    let mut case_sorted = case_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let mut control_sorted = control_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if case_sorted.is_empty() || control_sorted.is_empty() {
        return (0.0, 0.0);
    }

    case_sorted.sort_by(f64::total_cmp);
    control_sorted.sort_by(f64::total_cmp);
    let mut support = case_sorted
        .iter()
        .chain(control_sorted.iter())
        .copied()
        .collect::<Vec<_>>();
    support.sort_by(f64::total_cmp);
    support.dedup_by(|left, right| left.total_cmp(right).is_eq());

    let case_n = case_sorted.len() as f64;
    let control_n = control_sorted.len() as f64;
    let mut diff_sum = 0.0_f64;
    let mut diff_max = 0.0_f64;

    for point in support.iter().copied() {
        let case_cdf = (case_sorted.partition_point(|value| *value <= point) as f64) / case_n;
        let control_cdf =
            (control_sorted.partition_point(|value| *value <= point) as f64) / control_n;
        let diff = (case_cdf - control_cdf).abs();
        diff_sum += diff;
        diff_max = diff_max.max(diff);
    }

    (diff_sum / (support.len() as f64), diff_max)
}

fn synth_values(len: usize, phase: f64) -> Vec<f64> {
    (0..len)
        .map(|idx| {
            let x = idx as f64;
            (phase.mul_add(x, 0.07 * (idx % 11) as f64)).sin() + 0.15 * ((idx % 5) as f64)
        })
        .collect()
}

fn bench_ecdf_distance_stats(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("balance_ecdf_distance_stats");
    for (case_len, control_len) in [(200usize, 200usize), (2_000, 2_000), (10_000, 10_000)] {
        let case_values = synth_values(case_len, 0.31);
        let control_values = synth_values(control_len, 0.23);
        group.throughput(Throughput::Elements((case_len + control_len) as u64));
        group.bench_function(
            format!("current_stream_merge/case={case_len}/control={control_len}"),
            |bench| {
                bench.iter(|| {
                    let _ = benchmark_ecdf_distance_stats(&case_values, &control_values);
                });
            },
        );
        group.bench_function(
            format!("reference_support_sort/case={case_len}/control={control_len}"),
            |bench| {
                bench.iter(|| {
                    let _ = ecdf_distance_stats_reference(&case_values, &control_values);
                });
            },
        );
    }
    group.finish();
}

fn quantile_from_sorted_reference(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 || denominator == 0 {
        return sorted[0];
    }
    let points = sorted.len() - 1;
    let position = numerator.saturating_mul(points);
    let lower_idx = position / denominator;
    let upper_idx = (lower_idx + 1).min(points);
    let remainder = position % denominator;
    if remainder == 0 {
        sorted[lower_idx]
    } else {
        let weight = (remainder as f64) / (denominator as f64);
        sorted[lower_idx] + weight * (sorted[upper_idx] - sorted[lower_idx])
    }
}

fn eqq_distance_stats_reference(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    let mut case_sorted = case_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let mut control_sorted = control_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if case_sorted.is_empty() || control_sorted.is_empty() {
        return (0.0, 0.0);
    }
    case_sorted.sort_by(f64::total_cmp);
    control_sorted.sort_by(f64::total_cmp);
    let quantile_points = case_sorted.len().max(control_sorted.len()).max(2);
    let denominator = quantile_points - 1;
    let mut diff_sum = 0.0_f64;
    let mut diff_max = 0.0_f64;
    for idx in 0..quantile_points {
        let case_q = quantile_from_sorted_reference(&case_sorted, idx, denominator);
        let control_q = quantile_from_sorted_reference(&control_sorted, idx, denominator);
        let diff = (case_q - control_q).abs();
        diff_sum += diff;
        diff_max = diff_max.max(diff);
    }
    (diff_sum / (quantile_points as f64), diff_max)
}

fn bench_eqq_distance_stats(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("balance_eqq_distance_stats");
    for (case_len, control_len) in [
        (200usize, 200usize),
        (2_000, 2_000),
        (2_000, 3_000),
        (10_000, 10_000),
    ] {
        let case_values = synth_values(case_len, 0.31);
        let control_values = synth_values(control_len, 0.23);
        group.throughput(Throughput::Elements((case_len + control_len) as u64));
        group.bench_function(
            format!("current/case={case_len}/control={control_len}"),
            |bench| {
                bench.iter(|| {
                    let _ = benchmark_eqq_distance_stats(&case_values, &control_values);
                });
            },
        );
        group.bench_function(
            format!("reference/case={case_len}/control={control_len}"),
            |bench| {
                bench.iter(|| {
                    let _ = eqq_distance_stats_reference(&case_values, &control_values);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_ecdf_distance_stats, bench_eqq_distance_stats);
criterion_main!(benches);
