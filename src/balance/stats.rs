use crate::matching::to_f64;
use itertools::Itertools;

/// Convert a count to a proportion, guarding against zero denominators.
pub(super) fn proportion(count: usize, total: usize) -> f64 {
    if total == 0 {
        0.0
    } else {
        to_f64(count) / to_f64(total)
    }
}

/// Standardized mean difference for continuous variables.
///
/// Reference:
/// - Cohen J. *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. 1988.
pub(super) fn smd_numeric(
    mean_case: f64,
    mean_control: f64,
    var_case: f64,
    var_control: f64,
) -> f64 {
    let pooled = 0.5 * (var_case + var_control);
    if pooled <= 0.0 {
        0.0
    } else {
        (mean_case - mean_control) / pooled.sqrt()
    }
}

/// Variance ratio (`var_case / var_control`) with finite safeguards.
///
/// Reference:
/// - Stuart EA, Lee BK, Leacy FP. Prognostic score-based balance measures can be
///   a useful diagnostic for propensity score methods in comparative effectiveness research.
///   *J Clin Epidemiol*. 2013.
pub(super) fn variance_ratio(var_case: f64, var_control: f64) -> f64 {
    if !var_case.is_finite() || !var_control.is_finite() || var_case < 0.0 || var_control <= 0.0 {
        0.0
    } else {
        var_case / var_control
    }
}

/// eCDF distance summary (mean and max absolute difference).
///
/// This computes empirical CDFs at pooled unique support points and returns:
/// - mean absolute distance across support points
/// - max absolute distance (Kolmogorov-style sup distance)
///
/// Reference:
/// - Rosenbaum PR, Rubin DB. The central role of the propensity score in observational studies.
///   *Biometrika*. 1983.
pub(super) fn ecdf_distance_stats(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    if case_values.is_empty() || control_values.is_empty() {
        return (0.0, 0.0);
    }

    let mut case_sorted = case_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect_vec();
    let mut control_sorted = control_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect_vec();
    if case_sorted.is_empty() || control_sorted.is_empty() {
        return (0.0, 0.0);
    }

    case_sorted.sort_by(f64::total_cmp);
    control_sorted.sort_by(f64::total_cmp);

    let case_n = to_f64(case_sorted.len());
    let control_n = to_f64(control_sorted.len());
    let mut diff_sum = 0.0_f64;
    let mut diff_max = 0.0_f64;
    let mut support_len = 0usize;
    let mut case_idx = 0usize;
    let mut control_idx = 0usize;

    while case_idx < case_sorted.len() || control_idx < control_sorted.len() {
        let point = match (case_sorted.get(case_idx), control_sorted.get(control_idx)) {
            (Some(case_point), Some(control_point)) => {
                if case_point.total_cmp(control_point).is_le() {
                    *case_point
                } else {
                    *control_point
                }
            }
            (Some(case_point), None) => *case_point,
            (None, Some(control_point)) => *control_point,
            (None, None) => break,
        };

        while case_idx < case_sorted.len() && case_sorted[case_idx].total_cmp(&point).is_le() {
            case_idx += 1;
        }
        while control_idx < control_sorted.len()
            && control_sorted[control_idx].total_cmp(&point).is_le()
        {
            control_idx += 1;
        }

        support_len += 1;
        let case_cdf = to_f64(case_idx) / case_n;
        let control_cdf = to_f64(control_idx) / control_n;
        let diff = (case_cdf - control_cdf).abs();
        diff_sum += diff;
        diff_max = diff_max.max(diff);
    }

    (diff_sum / to_f64(support_len), diff_max)
}

/// eQQ distance summary (mean and max absolute quantile distance).
///
/// Quantiles are computed on an evenly spaced grid between 0 and 1 using
/// linear interpolation between adjacent order statistics.
///
/// Reference:
/// - Ho DE, Imai K, King G, Stuart EA. `MatchIt`: Nonparametric preprocessing for
///   parametric causal inference. *J Stat Softw*. 2011.
pub(super) fn eqq_distance_stats(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    let mut case_sorted = case_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect_vec();
    let mut control_sorted = control_values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect_vec();
    if case_sorted.is_empty() || control_sorted.is_empty() {
        return (0.0, 0.0);
    }

    case_sorted.sort_by(f64::total_cmp);
    control_sorted.sort_by(f64::total_cmp);

    if case_sorted.len() == control_sorted.len() {
        if case_sorted.len() == 1 {
            let diff = (case_sorted[0] - control_sorted[0]).abs();
            return (diff, diff);
        }
        let quantile_points = case_sorted.len().max(2);
        let mut diff_sum = 0.0_f64;
        let mut diff_max = 0.0_f64;
        for (case_q, control_q) in case_sorted.iter().zip(control_sorted.iter()) {
            let diff = (case_q - control_q).abs();
            diff_sum += diff;
            diff_max = diff_max.max(diff);
        }
        return (diff_sum / to_f64(quantile_points), diff_max);
    }

    let quantile_points = case_sorted.len().max(control_sorted.len()).max(2);
    let denominator = quantile_points - 1;
    let mut diff_sum = 0.0_f64;
    let mut diff_max = 0.0_f64;

    for idx in 0..quantile_points {
        let case_q = quantile_from_sorted(&case_sorted, idx, denominator);
        let control_q = quantile_from_sorted(&control_sorted, idx, denominator);
        let diff = (case_q - control_q).abs();
        diff_sum += diff;
        diff_max = diff_max.max(diff);
    }

    (diff_sum / to_f64(quantile_points), diff_max)
}

fn quantile_from_sorted(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
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
        let weight = to_f64(remainder) / to_f64(denominator);
        sorted[lower_idx] + weight * (sorted[upper_idx] - sorted[lower_idx])
    }
}

/// Standardized mean difference for binary proportions.
///
/// Reference:
/// - Austin PC. Balance diagnostics for comparing the distribution of baseline
///   covariates between treatment groups in propensity-score matched samples. *Stat Med*. 2009.
pub(super) fn smd_proportion(p_case: f64, p_control: f64) -> f64 {
    let pooled = 0.5 * p_case.mul_add(1.0 - p_case, p_control * (1.0 - p_control));
    if pooled <= 0.0 {
        0.0
    } else {
        (p_case - p_control) / pooled.sqrt()
    }
}

/// Cramér's V computed from 2xK contingency counts.
///
/// This implementation computes Pearson's chi-square statistic from expected
/// counts and applies `V = sqrt(chi2 / (N * min(r - 1, c - 1)))`.
///
/// References:
/// - Cramér H. *Mathematical Methods of Statistics*. 1946.
/// - Agresti A. *Categorical Data Analysis*, 3rd ed. 2013.
pub(super) fn cramers_v(case_counts: &[usize], control_counts: &[usize]) -> f64 {
    let mut chi2 = 0.0;
    let total_case = case_counts.iter().sum::<usize>();
    let total_control = control_counts.iter().sum::<usize>();
    let total = total_case + total_control;
    if total == 0 {
        return 0.0;
    }

    for (case_count, control_count) in case_counts.iter().zip(control_counts.iter()) {
        let row_total = case_count + control_count;
        if row_total == 0 {
            continue;
        }
        let expected_case = to_f64(row_total) * to_f64(total_case) / to_f64(total);
        let expected_control = to_f64(row_total) * to_f64(total_control) / to_f64(total);

        chi2 += (to_f64(*case_count) - expected_case).powi(2) / expected_case.max(1e-12);
        chi2 += (to_f64(*control_count) - expected_control).powi(2) / expected_control.max(1e-12);
    }

    let rows_minus_one = 1.0_f64; // 2 groups (case/control)
    let cols_minus_one = to_f64(case_counts.len().saturating_sub(1));
    let scale = rows_minus_one.min(cols_minus_one).max(1.0);
    (chi2 / (to_f64(total) * scale)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ecdf_distance_stats_sorted_reference(
        case_values: &[f64],
        control_values: &[f64],
    ) -> (f64, f64) {
        if case_values.is_empty() || control_values.is_empty() {
            return (0.0, 0.0);
        }
        let mut case_sorted = case_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        let mut control_sorted = control_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        if case_sorted.is_empty() || control_sorted.is_empty() {
            return (0.0, 0.0);
        }
        case_sorted.sort_by(f64::total_cmp);
        control_sorted.sort_by(f64::total_cmp);
        let support = case_sorted
            .iter()
            .chain(control_sorted.iter())
            .copied()
            .sorted_by(f64::total_cmp)
            .dedup_by(|left, right| left.total_cmp(right).is_eq())
            .collect_vec();
        let support_len = support.len();

        let case_n = to_f64(case_sorted.len());
        let control_n = to_f64(control_sorted.len());
        let mut diff_sum = 0.0_f64;
        let mut diff_max = 0.0_f64;

        for point in support {
            let case_cdf = to_f64(case_sorted.partition_point(|value| *value <= point)) / case_n;
            let control_cdf =
                to_f64(control_sorted.partition_point(|value| *value <= point)) / control_n;
            let diff = (case_cdf - control_cdf).abs();
            diff_sum += diff;
            diff_max = diff_max.max(diff);
        }

        (diff_sum / to_f64(support_len), diff_max)
    }

    fn eqq_distance_stats_reference(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
        let mut case_sorted = case_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        let mut control_sorted = control_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
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
            let case_q = quantile_from_sorted(&case_sorted, idx, denominator);
            let control_q = quantile_from_sorted(&control_sorted, idx, denominator);
            let diff = (case_q - control_q).abs();
            diff_sum += diff;
            diff_max = diff_max.max(diff);
        }
        (diff_sum / to_f64(quantile_points), diff_max)
    }

    use proptest::prelude::*;
    use rstest::rstest;

    proptest! {
        #[test]
        fn proportion_is_invariant_to_permutation(mut values in prop::collection::vec(0..2u8, 1..100)) {
            let n = values.iter().map(|&value| usize::from(value == 1)).sum::<usize>();
            let total = values.len();
            let p1 = proportion(n, total);
            values.reverse();
            let n2 = values.iter().map(|&value| usize::from(value == 1)).sum::<usize>();
            let p2 = proportion(n2, total);
            prop_assert_eq!(p1.to_bits(), p2.to_bits());
        }

        #[test]
        fn ecdf_distance_is_invariant_to_permutation(
            mut case in prop::collection::vec(-100.0..100.0f64, 1..50),
            mut control in prop::collection::vec(-100.0..100.0f64, 1..50)
        ) {
            let d1 = ecdf_distance_stats(&case, &control);
            case.reverse();
            control.reverse();
            let d2 = ecdf_distance_stats(&case, &control);
            prop_assert_eq!(d1.0.to_bits(), d2.0.to_bits());
            prop_assert_eq!(d1.1.to_bits(), d2.1.to_bits());
        }

        #[test]
        fn eqq_distance_is_invariant_to_permutation(
            mut case in prop::collection::vec(-100.0..100.0f64, 1..50),
            mut control in prop::collection::vec(-100.0..100.0f64, 1..50)
        ) {
            let d1 = eqq_distance_stats(&case, &control);
            case.reverse();
            control.reverse();
            let d2 = eqq_distance_stats(&case, &control);
            prop_assert_eq!(d1.0.to_bits(), d2.0.to_bits());
            prop_assert_eq!(d1.1.to_bits(), d2.1.to_bits());
        }
    }

    #[rstest]
    #[case(3, 0, 0.0)]
    #[case(2, 8, 0.25)]
    fn test_proportion(#[case] n: usize, #[case] total: usize, #[case] expected: f64) {
        assert!((proportion(n, total) - expected).abs() < 1e-12);
    }

    #[rstest]
    #[case(1.0, 3.0, 0.0, 0.0, 0.0)]
    #[case(2.0, 1.0, 1.0, 1.0, 1.0)]
    fn test_smd_numeric(
        #[case] m1: f64,
        #[case] m2: f64,
        #[case] v1: f64,
        #[case] v2: f64,
        #[case] expected: f64,
    ) {
        assert!((smd_numeric(m1, m2, v1, v2) - expected).abs() < 1e-12);
    }

    #[rstest]
    #[case(2.0, 1.0, 2.0)]
    #[case(2.0, 0.0, 0.0)]
    #[case(f64::NAN, 1.0, 0.0)]
    fn test_variance_ratio(#[case] v1: f64, #[case] v2: f64, #[case] expected: f64) {
        assert!((variance_ratio(v1, v2) - expected).abs() < 1e-12);
    }

    #[test]
    fn smd_proportion_handles_degenerate_case() {
        assert!((smd_proportion(0.0, 0.0) - 0.0).abs() < 1e-12);
        let value = smd_proportion(0.5, 0.25);
        assert!(value.is_finite());
        assert!(value > 0.0);
    }

    #[test]
    fn cramers_v_zero_for_identical_distributions() {
        let case_counts = [10, 20, 30];
        let control_counts = [20, 40, 60];
        assert!((cramers_v(&case_counts, &control_counts) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn cramers_v_increases_with_stronger_association() {
        let weak = cramers_v(&[50, 50], &[50, 50]);
        let strong = cramers_v(&[100, 0], &[0, 100]);
        assert!(strong > weak);
        assert!(strong <= 1.0);
    }

    #[test]
    fn ecdf_distance_stats_detects_distribution_difference() {
        let identical = ecdf_distance_stats(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((identical.0 - 0.0).abs() < 1e-12);
        assert!((identical.1 - 0.0).abs() < 1e-12);

        let shifted = ecdf_distance_stats(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]);
        assert!(shifted.0 > 0.0);
        assert!((shifted.1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eqq_distance_stats_detects_quantile_shift() {
        let identical = eqq_distance_stats(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!((identical.0 - 0.0).abs() < 1e-12);
        assert!((identical.1 - 0.0).abs() < 1e-12);

        let shifted = eqq_distance_stats(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]);
        assert!(shifted.0 > 0.0);
        assert!((shifted.1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eqq_distance_stats_handles_missing_inputs() {
        assert_eq!(eqq_distance_stats(&[], &[1.0, 2.0]), (0.0, 0.0));
        assert_eq!(eqq_distance_stats(&[1.0, 2.0], &[]), (0.0, 0.0));
    }

    #[rstest]
    fn ecdf_distance_stats_matches_sorted_reference(
        #[values(1usize, 3, 17, 257)] n_case: usize,
        #[values(1usize, 4, 31, 251)] n_control: usize,
    ) {
        let case_values = (0..n_case)
            .map(|idx| {
                let x = to_f64(idx);
                0.1f64.mul_add(to_f64(idx % 5), (0.31 * x).sin())
            })
            .collect_vec();
        let control_values = (0..n_control)
            .map(|idx| {
                let x = to_f64(idx);
                0.07f64.mul_add(to_f64(idx % 7), (0.23 * x).cos())
            })
            .collect_vec();

        let expected = ecdf_distance_stats_sorted_reference(&case_values, &control_values);
        let actual = ecdf_distance_stats(&case_values, &control_values);
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    #[rstest]
    fn eqq_distance_stats_matches_reference_for_equal_and_unequal_lengths(
        #[values((1usize, 1usize), (4, 3), (31, 31), (251, 257))] lengths: (usize, usize),
    ) {
        let (n_case, n_control) = lengths;
        let case_values = (0..n_case)
            .map(|idx| {
                let x = to_f64(idx);
                0.11f64.mul_add(to_f64(idx % 5), (0.19 * x).sin())
            })
            .collect_vec();
        let control_values = (0..n_control)
            .map(|idx| {
                let x = to_f64(idx);
                0.09f64.mul_add(to_f64(idx % 7), (0.13 * x).cos())
            })
            .collect_vec();
        let expected = eqq_distance_stats_reference(&case_values, &control_values);
        let actual = eqq_distance_stats(&case_values, &control_values);
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }
}
