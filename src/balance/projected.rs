use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::matching::to_f64;

/// Numeric projected covariate definition for balance summaries.
#[derive(Clone, Copy)]
pub struct NumericCovariateSpec<R> {
    /// Covariate name.
    pub name: &'static str,
    /// Projector from a record to an optional numeric value.
    pub project: fn(&R) -> Option<f64>,
}

/// Categorical projected covariate definition for balance summaries.
#[derive(Clone, Copy)]
pub struct CategoricalCovariateSpec<R> {
    /// Covariate name.
    pub name: &'static str,
    /// Projector from a record to a categorical level label.
    pub project: fn(&R) -> String,
}

/// Compact projected balance row for covariate diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedBalanceRow {
    /// Covariate name.
    pub covariate: String,
    /// Covariate metric type.
    pub metric_type: &'static str,
    /// Pre-match summary metric.
    pub smd_before: f64,
    /// Post-match summary metric.
    pub smd_after: f64,
}

impl<R> NumericCovariateSpec<R> {
    /// Construct a numeric covariate spec.
    #[must_use]
    pub const fn new(name: &'static str, project: fn(&R) -> Option<f64>) -> Self {
        Self { name, project }
    }
}

impl<R> CategoricalCovariateSpec<R> {
    /// Construct a categorical covariate spec.
    #[must_use]
    pub const fn new(name: &'static str, project: fn(&R) -> String) -> Self {
        Self { name, project }
    }
}

/// Sample variance around a supplied mean.
#[must_use]
pub fn variance(vals: &[f64], mean: f64) -> f64 {
    if vals.len() < 2 {
        return 0.0;
    }
    let sum_squares = vals
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>();
    sum_squares / to_f64(vals.len().saturating_sub(1))
}

/// Standardized mean difference from raw value vectors.
#[must_use]
pub fn smd_numeric_from_values(case_vals: &[f64], control_vals: &[f64]) -> f64 {
    if case_vals.is_empty() || control_vals.is_empty() {
        return 0.0;
    }
    let mean_case = case_vals.iter().sum::<f64>() / to_f64(case_vals.len());
    let mean_control = control_vals.iter().sum::<f64>() / to_f64(control_vals.len());
    let var_case = variance(case_vals, mean_case);
    let var_control = variance(control_vals, mean_control);
    let pooled_sd = f64::midpoint(var_case, var_control).sqrt();
    if pooled_sd <= 0.0 || !pooled_sd.is_finite() {
        0.0
    } else {
        (mean_case - mean_control) / pooled_sd
    }
}

/// Maximum absolute difference in category proportions.
#[must_use]
pub fn categorical_max_abs_diff(case_vals: &[String], control_vals: &[String]) -> f64 {
    if case_vals.is_empty() && control_vals.is_empty() {
        return 0.0;
    }

    let mut case_counts: HashMap<&str, usize> = HashMap::new();
    for value in case_vals {
        *case_counts.entry(value.as_str()).or_insert(0) += 1;
    }
    let mut control_counts: HashMap<&str, usize> = HashMap::new();
    for value in control_vals {
        *control_counts.entry(value.as_str()).or_insert(0) += 1;
    }

    let mut levels = case_counts
        .keys()
        .chain(control_counts.keys())
        .copied()
        .collect::<HashSet<_>>();
    let case_denominator = to_f64(case_vals.len().max(1));
    let control_denominator = to_f64(control_vals.len().max(1));
    let mut max_diff = 0.0_f64;
    for level in levels.drain() {
        let case_prop = to_f64(*case_counts.get(level).unwrap_or(&0)) / case_denominator;
        let control_prop = to_f64(*control_counts.get(level).unwrap_or(&0)) / control_denominator;
        max_diff = max_diff.max((case_prop - control_prop).abs());
    }
    max_diff
}

/// Build projected balance rows from pre/post matched cohorts.
#[must_use]
pub fn build_projected_balance_rows<R, SCase: BuildHasher, SControl: BuildHasher>(
    population: &[R],
    cases: &[&R],
    matched_case_ids: &HashSet<String, SCase>,
    matched_control_ids: &HashSet<String, SControl>,
    id: fn(&R) -> &str,
    numeric_specs: &[NumericCovariateSpec<R>],
    categorical_specs: &[CategoricalCovariateSpec<R>],
) -> Vec<ProjectedBalanceRow> {
    let case_id_set = cases.iter().map(|row| id(row)).collect::<HashSet<_>>();
    let controls_before = population
        .iter()
        .filter(|row| !case_id_set.contains(id(row)))
        .collect::<Vec<_>>();
    let cases_after = population
        .iter()
        .filter(|row| matched_case_ids.contains(id(row)))
        .collect::<Vec<_>>();
    let controls_after = population
        .iter()
        .filter(|row| matched_control_ids.contains(id(row)))
        .collect::<Vec<_>>();

    let numeric_rows = numeric_specs.iter().map(|spec| {
        let case_before_values = cases
            .iter()
            .filter_map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        let control_before_values = controls_before
            .iter()
            .filter_map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        let case_after_values = cases_after
            .iter()
            .filter_map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        let control_after_values = controls_after
            .iter()
            .filter_map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        ProjectedBalanceRow {
            covariate: spec.name.to_string(),
            metric_type: "numeric",
            smd_before: smd_numeric_from_values(&case_before_values, &control_before_values),
            smd_after: smd_numeric_from_values(&case_after_values, &control_after_values),
        }
    });

    let categorical_rows = categorical_specs.iter().map(|spec| {
        let case_before_values = cases
            .iter()
            .map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        let control_before_values = controls_before
            .iter()
            .map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        let case_after_values = cases_after
            .iter()
            .map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        let control_after_values = controls_after
            .iter()
            .map(|row| (spec.project)(row))
            .collect::<Vec<_>>();
        ProjectedBalanceRow {
            covariate: spec.name.to_string(),
            metric_type: "categorical_max_abs_diff",
            smd_before: categorical_max_abs_diff(&case_before_values, &control_before_values),
            smd_after: categorical_max_abs_diff(&case_after_values, &control_after_values),
        }
    });

    numeric_rows.chain(categorical_rows).collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{
        CategoricalCovariateSpec, NumericCovariateSpec, build_projected_balance_rows,
        categorical_max_abs_diff, smd_numeric_from_values,
    };

    #[derive(Clone)]
    struct Row {
        id: String,
        birth_year: Option<i32>,
        age: Option<i32>,
        sex: String,
    }

    fn row_id(row: &Row) -> &str {
        row.id.as_str()
    }

    fn birth_year(row: &Row) -> Option<f64> {
        row.birth_year.map(f64::from)
    }

    fn age(row: &Row) -> Option<f64> {
        row.age.map(f64::from)
    }

    fn sex(row: &Row) -> String {
        row.sex.clone()
    }

    #[test]
    fn numeric_smd_returns_zero_for_empty_groups() {
        assert!((smd_numeric_from_values(&[], &[1.0]) - 0.0).abs() < 1e-12);
        assert!((smd_numeric_from_values(&[1.0], &[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn categorical_max_abs_diff_detects_distribution_shift() {
        let case = vec!["m".to_string(), "m".to_string()];
        let control = vec!["f".to_string(), "f".to_string()];
        assert!((categorical_max_abs_diff(&case, &control) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn projected_balance_builds_numeric_and_categorical_rows() {
        let population = vec![
            Row {
                id: "case_a".to_string(),
                birth_year: Some(2010),
                age: Some(2),
                sex: "f".to_string(),
            },
            Row {
                id: "case_b".to_string(),
                birth_year: Some(2011),
                age: Some(3),
                sex: "f".to_string(),
            },
            Row {
                id: "control_a".to_string(),
                birth_year: Some(2010),
                age: Some(4),
                sex: "m".to_string(),
            },
            Row {
                id: "control_b".to_string(),
                birth_year: Some(2012),
                age: Some(5),
                sex: "m".to_string(),
            },
        ];
        let cases = vec![&population[0], &population[1]];
        let matched_case_ids = HashSet::from(["case_a".to_string()]);
        let matched_control_ids = HashSet::from(["control_a".to_string()]);
        let rows = build_projected_balance_rows(
            &population,
            &cases,
            &matched_case_ids,
            &matched_control_ids,
            row_id,
            &[
                NumericCovariateSpec::new("birth_year", birth_year),
                NumericCovariateSpec::new("age", age),
            ],
            &[CategoricalCovariateSpec::new("sex", sex)],
        );

        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].covariate, "birth_year");
        assert_eq!(rows[0].metric_type, "numeric");
        assert_eq!(rows[2].covariate, "sex");
        assert_eq!(rows[2].metric_type, "categorical_max_abs_diff");
    }
}
