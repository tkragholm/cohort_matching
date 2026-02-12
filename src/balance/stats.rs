use crate::matching::to_f64;

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

    #[test]
    fn proportion_handles_zero_denominator() {
        assert!((proportion(3, 0) - 0.0).abs() < 1e-12);
        assert!((proportion(2, 8) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn smd_numeric_handles_non_positive_pooled_variance() {
        assert!((smd_numeric(1.0, 3.0, 0.0, 0.0) - 0.0).abs() < 1e-12);
        assert!((smd_numeric(2.0, 1.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
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
}
