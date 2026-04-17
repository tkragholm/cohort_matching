use chrono::Duration;
use cohort_matching::{
    BaseRecord, DateDistance, DistanceMetric, IdMapMahalanobisDistance,
    IdMapPropensityScoreDistance, date,
};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rapidhash::RapidHashMap;
use std::hint::black_box;

fn synthetic_records(count: usize) -> Vec<BaseRecord> {
    let base_date = date(1980, 1, 1);
    (0..count)
        .map(|idx| {
            let id = format!("r{idx}");
            let day_offset = i64::try_from(idx % 12_000).expect("modulo keeps offset in i64 range");
            let date = base_date + Duration::days(day_offset);
            BaseRecord::new(id, date)
        })
        .collect()
}

fn synthetic_pairs(record_count: usize, pair_count: usize) -> Vec<(usize, usize)> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let modulus = u64::try_from(record_count).expect("record count fits u64");
    let lcg_multiplier = 6_364_136_223_846_793_005_u64;
    let mut pairs = Vec::with_capacity(pair_count);

    for _ in 0..pair_count {
        state = state.wrapping_mul(lcg_multiplier).wrapping_add(1);
        let left = usize::try_from(state % modulus).expect("modulo keeps value in usize range");
        state = state.wrapping_mul(lcg_multiplier).wrapping_add(1);
        let right = usize::try_from(state % modulus).expect("modulo keeps value in usize range");
        pairs.push((left, right));
    }

    pairs
}

fn synthetic_scores(records: &[BaseRecord]) -> RapidHashMap<String, f64> {
    records
        .iter()
        .enumerate()
        .map(|(idx, record)| {
            let score_num =
                u32::try_from(idx % 10_000).expect("modulo keeps score numerator in u32 range");
            let score = f64::from(score_num) / 10_000.0;
            (record.id.clone(), score)
        })
        .collect()
}

fn synthetic_vectors(records: &[BaseRecord], dimension: usize) -> RapidHashMap<String, Vec<f64>> {
    records
        .iter()
        .enumerate()
        .map(|(idx, record)| {
            let vec = (0..dimension)
                .map(|col| {
                    let raw = u32::try_from((idx * (col + 3)) % 997)
                        .expect("modulo keeps vector entry in u32 range");
                    f64::from(raw) / 100.0
                })
                .collect();
            (record.id.clone(), vec)
        })
        .collect()
}

fn inverse_covariance(dimension: usize) -> Vec<f64> {
    (0..dimension)
        .flat_map(|row| {
            (0..dimension).map(move |col| {
                if row == col {
                    let diagonal_offset =
                        u32::try_from(row % 3).expect("modulo keeps diagonal offset in u32 range");
                    1.0 + f64::from(diagonal_offset) * 0.2
                } else {
                    0.01
                }
            })
        })
        .collect()
}

fn pairwise_distance_sum<D>(records: &[BaseRecord], pairs: &[(usize, usize)], metric: &D) -> f64
where
    D: DistanceMetric<BaseRecord>,
{
    pairs
        .iter()
        .filter_map(|(left, right)| metric.distance(&records[*left], &records[*right]))
        .sum()
}

fn bench_distance_channels(c: &mut Criterion) {
    let record_count = 8_000;
    let pair_count = 40_000;
    let records = synthetic_records(record_count);
    let pairs = synthetic_pairs(record_count, pair_count);

    let mut group = c.benchmark_group("distance_channels_pairwise");
    group.throughput(Throughput::Elements(pair_count as u64));

    let date_metric = DateDistance;
    group.bench_function("date", |b| {
        b.iter(|| black_box(pairwise_distance_sum(&records, &pairs, &date_metric)));
    });

    let propensity_scores = synthetic_scores(&records);
    let propensity_metric = IdMapPropensityScoreDistance::new(&propensity_scores);
    group.bench_function("propensity_map", |b| {
        b.iter(|| black_box(pairwise_distance_sum(&records, &pairs, &propensity_metric)));
    });

    for dimension in [4_usize, 8, 16] {
        let vectors = synthetic_vectors(&records, dimension);
        let inverse = inverse_covariance(dimension);
        let metric = IdMapMahalanobisDistance::new(&vectors, &inverse, dimension);

        group.bench_with_input(
            BenchmarkId::new("mahalanobis_map", dimension),
            &dimension,
            |b, _| {
                b.iter(|| black_box(pairwise_distance_sum(&records, &pairs, &metric)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_distance_channels);
criterion_main!(benches);
