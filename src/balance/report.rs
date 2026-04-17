use super::stats::{
    cramers_v, ecdf_distance_stats, eqq_distance_stats, proportion, smd_numeric, smd_proportion,
    variance_ratio,
};
use crate::matching::to_f64;
use crate::types::{
    BalanceRecord, BalanceReport, BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds,
    CategoricalBalance, CategoricalLevelBalance, CovariateValue, MatchOutcome, NumericBalance,
    NumericBalanceThresholdCheck, NumericBalanceTransform,
};
use itertools::Itertools;
use std::collections::HashMap;

/// Compute pre- and post-match covariate balance summaries.
#[must_use]
pub fn balance_report(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    outcome: &MatchOutcome,
) -> BalanceReport {
    balance_report_with_options(cases, controls, outcome, &BalanceReportOptions::default())
}

/// Compute pre- and post-match covariate balance summaries with report options.
#[must_use]
pub fn balance_report_with_options(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    outcome: &MatchOutcome,
    options: &BalanceReportOptions,
) -> BalanceReport {
    let context = BalanceCovariateContext {
        case_supplemental: &options.supplemental_covariates.cases,
        control_supplemental: &options.supplemental_covariates.controls,
    };
    let (numeric_keys, categorical_keys) = split_covariate_keys_by_kind(
        cases,
        controls,
        context.case_supplemental,
        context.control_supplemental,
    );
    let (cases_post, controls_post) = matched_samples(cases, controls, outcome);
    let numeric = build_numeric_balance(
        cases,
        controls,
        &cases_post,
        &controls_post,
        context,
        &numeric_keys,
        options.numeric_transforms,
    );
    let categorical = build_categorical_balance(
        cases,
        controls,
        &cases_post,
        &controls_post,
        context,
        &categorical_keys,
    );

    BalanceReport {
        numeric,
        categorical,
    }
}

/// Evaluate post-match numeric balance diagnostics against configured thresholds.
#[must_use]
pub fn balance_threshold_summary(
    report: &BalanceReport,
    thresholds: &BalanceThresholds,
) -> BalanceThresholdSummary {
    let numeric = report
        .numeric
        .iter()
        .map(|metric| {
            let smd_post_ok = thresholds
                .smd_abs_max
                .map(|threshold| metric.smd_post.abs() <= threshold);
            let var_ratio_post_ok = match (thresholds.var_ratio_min, thresholds.var_ratio_max) {
                (Some(min), Some(max)) => {
                    Some(metric.var_ratio_post >= min && metric.var_ratio_post <= max)
                }
                (Some(min), None) => Some(metric.var_ratio_post >= min),
                (None, Some(max)) => Some(metric.var_ratio_post <= max),
                (None, None) => None,
            };
            let ecdf_max_diff_post_ok = thresholds
                .ecdf_max_diff_max
                .map(|threshold| metric.ecdf_max_diff_post <= threshold);
            let eqq_max_diff_post_ok = thresholds
                .eqq_max_diff_max
                .map(|threshold| metric.eqq_max_diff_post <= threshold);

            let enabled_checks = [
                smd_post_ok,
                var_ratio_post_ok,
                ecdf_max_diff_post_ok,
                eqq_max_diff_post_ok,
            ]
            .into_iter()
            .flatten()
            .collect_vec();
            let all_enabled_checks_ok = enabled_checks.iter().all(|check| *check);

            NumericBalanceThresholdCheck {
                name: metric.name.clone(),
                smd_post_ok,
                var_ratio_post_ok,
                ecdf_max_diff_post_ok,
                eqq_max_diff_post_ok,
                all_enabled_checks_ok,
            }
        })
        .collect_vec();

    let all_enabled_checks_ok = numeric.iter().all(|check| check.all_enabled_checks_ok);
    BalanceThresholdSummary {
        numeric,
        all_enabled_checks_ok,
    }
}

#[derive(Clone, Copy)]
enum CovariateKind {
    Numeric,
    Categorical,
}

#[derive(Debug, Clone)]
enum NumericFeature {
    Raw { key: String },
    Square { key: String },
    Interaction { left: String, right: String },
}

type SupplementalCovariateMap = HashMap<String, HashMap<String, CovariateValue>>;

#[derive(Clone, Copy)]
struct BalanceCovariateContext<'a> {
    case_supplemental: &'a SupplementalCovariateMap,
    control_supplemental: &'a SupplementalCovariateMap,
}

impl NumericFeature {
    fn name(&self) -> String {
        match self {
            Self::Raw { key } => key.clone(),
            Self::Square { key } => format!("{key}^2"),
            Self::Interaction { left, right } => format!("{left} * {right}"),
        }
    }
}

fn covariate_kind(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
    key: &str,
) -> CovariateKind {
    for record in cases {
        if let Some(value) = covariate_value(record, key, case_supplemental) {
            return match value {
                CovariateValue::Numeric(_) => CovariateKind::Numeric,
                _ => CovariateKind::Categorical,
            };
        }
    }
    for record in controls {
        if let Some(value) = covariate_value(record, key, control_supplemental) {
            return match value {
                CovariateValue::Numeric(_) => CovariateKind::Numeric,
                _ => CovariateKind::Categorical,
            };
        }
    }
    CovariateKind::Categorical
}

fn collect_covariate_keys(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
) -> Vec<String> {
    cases
        .iter()
        .flat_map(|record| record.covariates.keys().cloned())
        .chain(
            controls
                .iter()
                .flat_map(|record| record.covariates.keys().cloned()),
        )
        .chain(
            case_supplemental
                .values()
                .flat_map(|values| values.keys().cloned()),
        )
        .chain(
            control_supplemental
                .values()
                .flat_map(|values| values.keys().cloned()),
        )
        .sorted()
        .dedup()
        .collect_vec()
}

fn split_covariate_keys_by_kind(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
) -> (Vec<String>, Vec<String>) {
    collect_covariate_keys(cases, controls, case_supplemental, control_supplemental)
        .into_iter()
        .fold(
            (Vec::new(), Vec::new()),
            |(mut numeric, mut categorical), key| {
                match covariate_kind(
                    cases,
                    controls,
                    case_supplemental,
                    control_supplemental,
                    &key,
                ) {
                    CovariateKind::Numeric => numeric.push(key),
                    CovariateKind::Categorical => categorical.push(key),
                }
                (numeric, categorical)
            },
        )
}

fn build_numeric_balance(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    cases_post: &[BalanceRecord],
    controls_post: &[BalanceRecord],
    context: BalanceCovariateContext<'_>,
    numeric_keys: &[String],
    transforms: NumericBalanceTransform,
) -> Vec<NumericBalance> {
    numeric_feature_specs(numeric_keys, transforms)
        .iter()
        .map(|feature| {
            summarize_numeric_feature(cases, controls, cases_post, controls_post, context, feature)
        })
        .collect_vec()
}

fn summarize_numeric_feature(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    cases_post: &[BalanceRecord],
    controls_post: &[BalanceRecord],
    context: BalanceCovariateContext<'_>,
    feature: &NumericFeature,
) -> NumericBalance {
    let case_pre_values = numeric_feature_values(cases, feature, context.case_supplemental);
    let control_pre_values =
        numeric_feature_values(controls, feature, context.control_supplemental);
    let (mean_case_pre, var_case_pre) = mean_var_numeric(&case_pre_values);
    let (mean_control_pre, var_control_pre) = mean_var_numeric(&control_pre_values);
    let smd_pre = smd_numeric(
        mean_case_pre,
        mean_control_pre,
        var_case_pre,
        var_control_pre,
    );
    let var_ratio_pre = variance_ratio(var_case_pre, var_control_pre);
    let (ecdf_mean_diff_pre, ecdf_max_diff_pre) =
        ecdf_distance_stats(&case_pre_values, &control_pre_values);
    let (eqq_mean_diff_pre, eqq_max_diff_pre) =
        eqq_distance_stats(&case_pre_values, &control_pre_values);

    let case_post_values = numeric_feature_values(cases_post, feature, context.case_supplemental);
    let control_post_values =
        numeric_feature_values(controls_post, feature, context.control_supplemental);
    let (mean_case_post, var_case_post) = mean_var_numeric(&case_post_values);
    let (mean_control_post, var_control_post) = mean_var_numeric(&control_post_values);
    let smd_post = smd_numeric(
        mean_case_post,
        mean_control_post,
        var_case_post,
        var_control_post,
    );
    let var_ratio_post = variance_ratio(var_case_post, var_control_post);
    let (ecdf_mean_diff_post, ecdf_max_diff_post) =
        ecdf_distance_stats(&case_post_values, &control_post_values);
    let (eqq_mean_diff_post, eqq_max_diff_post) =
        eqq_distance_stats(&case_post_values, &control_post_values);

    NumericBalance {
        name: feature.name(),
        mean_case_pre,
        mean_control_pre,
        smd_pre,
        var_ratio_pre,
        ecdf_mean_diff_pre,
        ecdf_max_diff_pre,
        eqq_mean_diff_pre,
        eqq_max_diff_pre,
        mean_case_post,
        mean_control_post,
        smd_post,
        var_ratio_post,
        ecdf_mean_diff_post,
        ecdf_max_diff_post,
        eqq_mean_diff_post,
        eqq_max_diff_post,
    }
}

fn build_categorical_balance(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    cases_post: &[BalanceRecord],
    controls_post: &[BalanceRecord],
    context: BalanceCovariateContext<'_>,
    categorical_keys: &[String],
) -> Vec<CategoricalBalance> {
    categorical_keys
        .iter()
        .map(|key| {
            summarize_categorical_feature(cases, controls, cases_post, controls_post, context, key)
        })
        .collect_vec()
}

fn summarize_categorical_feature(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    cases_post: &[BalanceRecord],
    controls_post: &[BalanceRecord],
    context: BalanceCovariateContext<'_>,
    key: &str,
) -> CategoricalBalance {
    let levels = collect_levels(
        cases,
        controls,
        context.case_supplemental,
        context.control_supplemental,
        key,
    );
    let (counts_case_pre, n_case_pre) =
        counts_categorical(cases, key, &levels, context.case_supplemental);
    let (counts_control_pre, n_control_pre) =
        counts_categorical(controls, key, &levels, context.control_supplemental);
    let (counts_case_post, n_case_post) =
        counts_categorical(cases_post, key, &levels, context.case_supplemental);
    let (counts_control_post, n_control_post) =
        counts_categorical(controls_post, key, &levels, context.control_supplemental);

    let levels = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| {
            let p_case_pre = proportion(counts_case_pre[idx], n_case_pre);
            let p_control_pre = proportion(counts_control_pre[idx], n_control_pre);
            let smd_pre = smd_proportion(p_case_pre, p_control_pre);

            let p_case_post = proportion(counts_case_post[idx], n_case_post);
            let p_control_post = proportion(counts_control_post[idx], n_control_post);
            let smd_post = smd_proportion(p_case_post, p_control_post);

            CategoricalLevelBalance {
                level: level.clone(),
                p_case_pre,
                p_control_pre,
                smd_pre,
                p_case_post,
                p_control_post,
                smd_post,
            }
        })
        .collect_vec();

    CategoricalBalance {
        name: key.to_string(),
        levels,
        cramers_v_pre: cramers_v(&counts_case_pre, &counts_control_pre),
        cramers_v_post: cramers_v(&counts_case_post, &counts_control_post),
    }
}

fn numeric_feature_specs(
    keys: &[String],
    transforms: NumericBalanceTransform,
) -> Vec<NumericFeature> {
    let mut specs = keys
        .iter()
        .cloned()
        .map(|key| NumericFeature::Raw { key })
        .collect_vec();
    if matches!(
        transforms,
        NumericBalanceTransform::Squares | NumericBalanceTransform::SquaresAndPairwiseInteractions
    ) {
        specs.extend(
            keys.iter()
                .cloned()
                .map(|key| NumericFeature::Square { key }),
        );
    }
    if matches!(
        transforms,
        NumericBalanceTransform::SquaresAndPairwiseInteractions
    ) {
        for left_idx in 0..keys.len() {
            for right_idx in (left_idx + 1)..keys.len() {
                specs.push(NumericFeature::Interaction {
                    left: keys[left_idx].clone(),
                    right: keys[right_idx].clone(),
                });
            }
        }
    }
    specs
}

fn matched_samples(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    outcome: &MatchOutcome,
) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let case_map = cases
        .iter()
        .map(|case| (case.core.id.as_str(), case.clone()))
        .collect::<HashMap<_, _>>();
    let control_map = controls
        .iter()
        .map(|control| (control.core.id.as_str(), control.clone()))
        .collect::<HashMap<_, _>>();

    let mut cases_out = Vec::new();
    let mut controls_out = Vec::new();
    for pair in &outcome.pairs {
        if let Some(case) = case_map.get(pair.case_id.as_str()) {
            cases_out.push(case.clone());
        }
        if let Some(control) = control_map.get(pair.control_id.as_str()) {
            controls_out.push(control.clone());
        }
    }
    (cases_out, controls_out)
}

fn numeric_feature_values(
    records: &[BalanceRecord],
    feature: &NumericFeature,
    supplemental: &SupplementalCovariateMap,
) -> Vec<f64> {
    records
        .iter()
        .filter_map(|record| match feature {
            NumericFeature::Raw { key } => numeric_covariate(record, key, supplemental),
            NumericFeature::Square { key } => {
                numeric_covariate(record, key, supplemental).map(|value| value.powi(2))
            }
            NumericFeature::Interaction { left, right } => {
                numeric_covariate(record, left, supplemental)
                    .zip(numeric_covariate(record, right, supplemental))
                    .map(|(left, right)| left * right)
            }
        })
        .collect()
}

fn numeric_covariate(
    record: &BalanceRecord,
    key: &str,
    supplemental: &SupplementalCovariateMap,
) -> Option<f64> {
    match covariate_value(record, key, supplemental) {
        Some(CovariateValue::Numeric(value)) if value.is_finite() => Some(*value),
        _ => None,
    }
}

fn mean_var_numeric(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean = values.iter().sum::<f64>() / to_f64(values.len());
    let var = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / to_f64(values.len());
    (mean, var)
}

fn collect_levels(
    cases: &[BalanceRecord],
    controls: &[BalanceRecord],
    case_supplemental: &SupplementalCovariateMap,
    control_supplemental: &SupplementalCovariateMap,
    key: &str,
) -> Vec<String> {
    let levels = cases
        .iter()
        .filter_map(|record| covariate_value(record, key, case_supplemental))
        .chain(
            controls
                .iter()
                .filter_map(|record| covariate_value(record, key, control_supplemental)),
        )
        .map(map_covariate_to_level)
        .sorted()
        .dedup()
        .collect_vec();

    if levels.is_empty() {
        vec!["missing".to_string()]
    } else {
        levels
    }
}

fn counts_categorical(
    records: &[BalanceRecord],
    key: &str,
    levels: &[String],
    supplemental: &SupplementalCovariateMap,
) -> (Vec<usize>, usize) {
    let level_pos = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| (level.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut counts = vec![0usize; levels.len()];

    for level in records
        .iter()
        .map(|record| map_optional_covariate_to_level(covariate_value(record, key, supplemental)))
    {
        if let Some(pos) = level_pos.get(level.as_str()) {
            counts[*pos] += 1;
        }
    }

    (counts, records.len())
}

fn map_covariate_to_level(value: &CovariateValue) -> String {
    match value {
        CovariateValue::Categorical(value) => value.clone(),
        CovariateValue::Numeric(_) => "numeric".to_string(),
        CovariateValue::Missing => "missing".to_string(),
    }
}

fn map_optional_covariate_to_level(value: Option<&CovariateValue>) -> String {
    value.map_or_else(|| "missing".to_string(), map_covariate_to_level)
}

fn covariate_value<'a>(
    record: &'a BalanceRecord,
    key: &str,
    supplemental: &'a SupplementalCovariateMap,
) -> Option<&'a CovariateValue> {
    record.covariates.get(key).or_else(|| {
        supplemental
            .get(record.core.id.as_str())
            .and_then(|values| values.get(key))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        BalanceRecord, BalanceReportOptions, MatchedPair, NumericBalanceTransform,
        SupplementalBalanceCovariates,
    };
    use crate::{date, test_outcome};
    use std::collections::HashMap;

    fn participant(id: &str) -> BalanceRecord {
        BalanceRecord::new(id, date(2010, 1, 1))
    }

    #[test]
    fn report_includes_numeric_and_categorical_covariates() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(10.0));
        case_a.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("north".to_string()),
        );

        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(14.0));
        case_b.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("south".to_string()),
        );

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(8.0));
        control_a.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("north".to_string()),
        );

        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(16.0));
        control_b.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("south".to_string()),
        );

        let report = balance_report(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
        );

        assert_eq!(report.numeric.len(), 1);
        assert_eq!(report.categorical.len(), 1);
        assert_eq!(report.numeric[0].name, "age");
        assert_eq!(report.categorical[0].name, "region");
    }

    #[test]
    fn report_uses_only_matched_pairs_for_post_values() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(10.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(30.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(20.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(40.0));

        let report = balance_report(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
        );

        let numeric = &report.numeric[0];
        assert!((numeric.mean_case_post - 10.0).abs() < 1e-12);
        assert!((numeric.mean_control_post - 40.0).abs() < 1e-12);
    }

    #[test]
    fn report_with_options_includes_squared_numeric_terms() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(2.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(4.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(1.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(3.0));

        let report = balance_report_with_options(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
            &BalanceReportOptions::builder()
                .numeric_transforms(NumericBalanceTransform::Squares)
                .build(),
        );

        let names = report
            .numeric
            .iter()
            .map(|metric| metric.name.as_str())
            .collect_vec();
        assert_eq!(names, vec!["age", "age^2"]);
    }

    #[test]
    fn report_with_options_includes_pairwise_numeric_interactions() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(2.0));
        case_a
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(10.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(3.0));
        case_b
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(12.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(1.0));
        control_a
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(8.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(2.0));
        control_b
            .covariates
            .insert("income".to_string(), CovariateValue::Numeric(9.0));

        let report = balance_report_with_options(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
            &BalanceReportOptions::builder()
                .numeric_transforms(NumericBalanceTransform::SquaresAndPairwiseInteractions)
                .build(),
        );

        let names = report
            .numeric
            .iter()
            .map(|metric| metric.name.clone())
            .collect_vec();
        assert!(names.contains(&"age".to_string()));
        assert!(names.contains(&"income".to_string()));
        assert!(names.contains(&"age^2".to_string()));
        assert!(names.contains(&"income^2".to_string()));
        assert!(names.contains(&"age * income".to_string()));
    }

    #[test]
    fn helper_functions_handle_missing_and_non_finite_values() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(f64::NAN));
        case_a
            .covariates
            .insert("region".to_string(), CovariateValue::Missing);
        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(12.0));

        let keys = collect_covariate_keys(
            &[case_a.clone()],
            &[control_a.clone()],
            &HashMap::new(),
            &HashMap::new(),
        );
        assert_eq!(keys, vec!["age".to_string(), "region".to_string()]);
        assert!(matches!(
            covariate_kind(
                &[case_a.clone()],
                &[control_a.clone()],
                &HashMap::new(),
                &HashMap::new(),
                "age",
            ),
            CovariateKind::Numeric
        ));
        let raw_age = NumericFeature::Raw {
            key: "age".to_string(),
        };
        assert_eq!(
            numeric_feature_values(&[case_a], &raw_age, &HashMap::new()),
            Vec::<f64>::new()
        );

        let levels = collect_levels(
            &[],
            &[control_a],
            &HashMap::new(),
            &HashMap::new(),
            "region",
        );
        assert_eq!(levels, vec!["missing".to_string()]);
    }

    #[test]
    fn counts_categorical_includes_missing_bucket() {
        let mut case_a = participant("case_a");
        case_a.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical("north".to_string()),
        );
        let case_b = participant("case_b");
        let levels = vec!["missing".to_string(), "north".to_string()];

        let (counts, total) =
            counts_categorical(&[case_a, case_b], "region", &levels, &HashMap::new());
        assert_eq!(total, 2);
        assert_eq!(counts, vec![1, 1]);
    }

    #[test]
    fn report_can_include_supplemental_covariates() {
        let case_a = participant("case_a");
        let control_a = participant("control_a");
        let mut supplemental_cases = HashMap::new();
        supplemental_cases.insert(
            "case_a".to_string(),
            HashMap::from([
                ("age".to_string(), CovariateValue::Numeric(10.0)),
                (
                    "region".to_string(),
                    CovariateValue::Categorical("north".to_string()),
                ),
            ]),
        );
        let mut supplemental_controls = HashMap::new();
        supplemental_controls.insert(
            "control_a".to_string(),
            HashMap::from([
                ("age".to_string(), CovariateValue::Numeric(8.0)),
                (
                    "region".to_string(),
                    CovariateValue::Categorical("south".to_string()),
                ),
            ]),
        );

        let report = balance_report_with_options(
            &[case_a],
            &[control_a],
            &test_outcome(vec![MatchedPair::new("case_a", "control_a")], 1, 0),
            &BalanceReportOptions {
                numeric_transforms: NumericBalanceTransform::None,
                supplemental_covariates: SupplementalBalanceCovariates {
                    cases: supplemental_cases,
                    controls: supplemental_controls,
                },
            },
        );

        assert_eq!(report.numeric.len(), 1);
        assert_eq!(report.numeric[0].name, "age");
        assert_eq!(report.categorical.len(), 1);
        assert_eq!(report.categorical[0].name, "region");
    }

    #[test]
    fn numeric_balance_includes_variance_ratio_and_ecdf_distances() {
        let mut case_a = participant("case_a");
        case_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));
        let mut case_b = participant("case_b");
        case_b
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));

        let mut control_a = participant("control_a");
        control_a
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(1.0));
        let mut control_b = participant("control_b");
        control_b
            .covariates
            .insert("x".to_string(), CovariateValue::Numeric(2.0));

        let report = balance_report(
            &[case_a, case_b],
            &[control_a, control_b],
            &test_outcome(vec![MatchedPair::new("case_a", "control_b")], 1, 0),
        );
        let numeric = &report.numeric[0];
        assert!((numeric.var_ratio_pre - 0.0).abs() < 1e-12);
        assert!((numeric.ecdf_max_diff_pre - 0.5).abs() < 1e-12);
        assert!(numeric.ecdf_mean_diff_pre > 0.0);
        assert!((numeric.eqq_max_diff_pre - 1.0).abs() < 1e-12);
        assert!(numeric.eqq_mean_diff_pre > 0.0);
    }

    #[test]
    fn threshold_summary_reports_pass_and_fail() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "x".to_string(),
                mean_case_pre: 0.0,
                mean_control_pre: 0.0,
                smd_pre: 0.0,
                var_ratio_pre: 1.0,
                ecdf_mean_diff_pre: 0.0,
                ecdf_max_diff_pre: 0.0,
                eqq_mean_diff_pre: 0.0,
                eqq_max_diff_pre: 0.0,
                mean_case_post: 0.0,
                mean_control_post: 0.0,
                smd_post: 0.05,
                var_ratio_post: 1.0,
                ecdf_mean_diff_post: 0.02,
                ecdf_max_diff_post: 0.06,
                eqq_mean_diff_post: 0.03,
                eqq_max_diff_post: 0.40,
            }],
            categorical: Vec::new(),
        };

        let passing = balance_threshold_summary(
            &report,
            &BalanceThresholds {
                eqq_max_diff_max: Some(0.5),
                ..BalanceThresholds::default()
            },
        );
        assert!(passing.all_enabled_checks_ok);
        assert_eq!(passing.numeric[0].smd_post_ok, Some(true));
        assert_eq!(passing.numeric[0].eqq_max_diff_post_ok, Some(true));

        let failing = balance_threshold_summary(
            &report,
            &BalanceThresholds {
                ecdf_max_diff_max: Some(0.05),
                ..BalanceThresholds::default()
            },
        );
        assert!(!failing.all_enabled_checks_ok);
        assert_eq!(failing.numeric[0].ecdf_max_diff_post_ok, Some(false));
    }

    #[test]
    fn threshold_summary_supports_disabling_checks() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "x".to_string(),
                mean_case_pre: 0.0,
                mean_control_pre: 0.0,
                smd_pre: 0.0,
                var_ratio_pre: 1.0,
                ecdf_mean_diff_pre: 0.0,
                ecdf_max_diff_pre: 0.0,
                eqq_mean_diff_pre: 0.0,
                eqq_max_diff_pre: 0.0,
                mean_case_post: 0.0,
                mean_control_post: 0.0,
                smd_post: 5.0,
                var_ratio_post: 5.0,
                ecdf_mean_diff_post: 5.0,
                ecdf_max_diff_post: 5.0,
                eqq_mean_diff_post: 5.0,
                eqq_max_diff_post: 5.0,
            }],
            categorical: Vec::new(),
        };

        let summary = balance_threshold_summary(
            &report,
            &BalanceThresholds {
                smd_abs_max: None,
                var_ratio_min: None,
                var_ratio_max: None,
                ecdf_max_diff_max: None,
                eqq_max_diff_max: None,
            },
        );
        assert!(summary.all_enabled_checks_ok);
        assert_eq!(summary.numeric[0].smd_post_ok, None);
    }
}
