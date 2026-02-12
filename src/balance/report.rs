use super::stats::{cramers_v, proportion, smd_numeric, smd_proportion};
use crate::matching::to_f64;
use crate::types::{
    BalanceReport, CaseRecord, CategoricalBalance, CategoricalLevelBalance, ControlRecord,
    CovariateValue, MatchOutcome, NumericBalance,
};
use itertools::Itertools;
use std::collections::HashMap;

/// Compute pre- and post-match covariate balance summaries.
#[must_use]
pub fn balance_report(
    cases: &[CaseRecord],
    controls: &[ControlRecord],
    outcome: &MatchOutcome,
) -> BalanceReport {
    let covariate_keys = collect_covariate_keys(cases, controls);
    let (cases_post, controls_post) = matched_samples(cases, controls, outcome);

    let mut numeric = Vec::new();
    let mut categorical = Vec::new();

    for key in covariate_keys {
        match covariate_kind(cases, controls, &key) {
            CovariateKind::Numeric => {
                let (mean_case_pre, var_case_pre) = mean_var_numeric(cases, &key);
                let (mean_control_pre, var_control_pre) = mean_var_numeric(controls, &key);
                let smd_pre = smd_numeric(
                    mean_case_pre,
                    mean_control_pre,
                    var_case_pre,
                    var_control_pre,
                );

                let (mean_case_post, var_case_post) = mean_var_numeric(&cases_post, &key);
                let (mean_control_post, var_control_post) = mean_var_numeric(&controls_post, &key);
                let smd_post = smd_numeric(
                    mean_case_post,
                    mean_control_post,
                    var_case_post,
                    var_control_post,
                );

                numeric.push(NumericBalance {
                    name: key,
                    mean_case_pre,
                    mean_control_pre,
                    smd_pre,
                    mean_case_post,
                    mean_control_post,
                    smd_post,
                });
            }
            CovariateKind::Categorical => {
                let levels = collect_levels(cases, controls, &key);
                let (counts_case_pre, n_case_pre) = counts_categorical(cases, &key, &levels);
                let (counts_control_pre, n_control_pre) =
                    counts_categorical(controls, &key, &levels);
                let (counts_case_post, n_case_post) =
                    counts_categorical(&cases_post, &key, &levels);
                let (counts_control_post, n_control_post) =
                    counts_categorical(&controls_post, &key, &levels);

                let mut level_balances = Vec::new();
                for (idx, level) in levels.iter().enumerate() {
                    let p_case_pre = proportion(counts_case_pre[idx], n_case_pre);
                    let p_control_pre = proportion(counts_control_pre[idx], n_control_pre);
                    let smd_pre = smd_proportion(p_case_pre, p_control_pre);

                    let p_case_post = proportion(counts_case_post[idx], n_case_post);
                    let p_control_post = proportion(counts_control_post[idx], n_control_post);
                    let smd_post = smd_proportion(p_case_post, p_control_post);

                    level_balances.push(CategoricalLevelBalance {
                        level: level.clone(),
                        p_case_pre,
                        p_control_pre,
                        smd_pre,
                        p_case_post,
                        p_control_post,
                        smd_post,
                    });
                }

                categorical.push(CategoricalBalance {
                    name: key,
                    levels: level_balances,
                    cramers_v_pre: cramers_v(&counts_case_pre, &counts_control_pre),
                    cramers_v_post: cramers_v(&counts_case_post, &counts_control_post),
                });
            }
        }
    }

    BalanceReport {
        numeric,
        categorical,
    }
}

#[derive(Clone, Copy)]
enum CovariateKind {
    Numeric,
    Categorical,
}

fn covariate_kind(cases: &[CaseRecord], controls: &[ControlRecord], key: &str) -> CovariateKind {
    for record in cases {
        if let Some(value) = record.covariates.get(key) {
            return match value {
                CovariateValue::Numeric(_) => CovariateKind::Numeric,
                _ => CovariateKind::Categorical,
            };
        }
    }
    for record in controls {
        if let Some(value) = record.covariates.get(key) {
            return match value {
                CovariateValue::Numeric(_) => CovariateKind::Numeric,
                _ => CovariateKind::Categorical,
            };
        }
    }
    CovariateKind::Categorical
}

fn collect_covariate_keys(cases: &[CaseRecord], controls: &[ControlRecord]) -> Vec<String> {
    cases
        .iter()
        .flat_map(|record| record.covariates.keys().cloned())
        .chain(
            controls
                .iter()
                .flat_map(|record| record.covariates.keys().cloned()),
        )
        .sorted()
        .dedup()
        .collect_vec()
}

fn matched_samples(
    cases: &[CaseRecord],
    controls: &[ControlRecord],
    outcome: &MatchOutcome,
) -> (Vec<CaseRecord>, Vec<ControlRecord>) {
    let case_map = cases
        .iter()
        .map(|case| (case.core.id.clone(), case.clone()))
        .collect::<HashMap<_, _>>();
    let control_map = controls
        .iter()
        .map(|control| (control.core.id.clone(), control.clone()))
        .collect::<HashMap<_, _>>();

    let mut cases_out = Vec::new();
    let mut controls_out = Vec::new();
    for pair in &outcome.pairs {
        if let Some(case) = case_map.get(&pair.case_id) {
            cases_out.push(case.clone());
        }
        if let Some(control) = control_map.get(&pair.control_id) {
            controls_out.push(control.clone());
        }
    }
    (cases_out, controls_out)
}

fn mean_var_numeric(records: &[CaseRecord], key: &str) -> (f64, f64) {
    let mut values = Vec::new();
    for record in records {
        if let Some(CovariateValue::Numeric(value)) = record.covariates.get(key)
            && value.is_finite()
        {
            values.push(*value);
        }
    }
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

fn collect_levels(cases: &[CaseRecord], controls: &[ControlRecord], key: &str) -> Vec<String> {
    let levels = cases
        .iter()
        .chain(controls.iter())
        .filter_map(|record| record.covariates.get(key))
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

fn counts_categorical(records: &[CaseRecord], key: &str, levels: &[String]) -> (Vec<usize>, usize) {
    let level_pos = levels
        .iter()
        .enumerate()
        .map(|(idx, level)| (level.as_str(), idx))
        .collect::<HashMap<_, _>>();
    let mut counts = vec![0usize; levels.len()];

    for level in records
        .iter()
        .map(|record| map_optional_covariate_to_level(record.covariates.get(key)))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MatchDiagnostics, MatchedPair, ParticipantRecord};
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn participant(id: &str) -> ParticipantRecord {
        ParticipantRecord::new(id, date(2010, 1, 1))
    }

    fn outcome(pairs: Vec<MatchedPair>, matched_cases: usize) -> MatchOutcome {
        MatchOutcome {
            avg_controls_per_case: if matched_cases == 0 {
                0.0
            } else {
                f64::from(u32::try_from(pairs.len()).expect("small test vector length"))
                    / f64::from(u32::try_from(matched_cases).expect("small matched count"))
            },
            used_controls: pairs.len(),
            pairs,
            matched_cases,
            unmatched_cases: 0,
            diagnostics: MatchDiagnostics::default(),
        }
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
            &outcome(vec![MatchedPair::new("case_a", "control_b")], 1),
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
            &outcome(vec![MatchedPair::new("case_a", "control_b")], 1),
        );

        let numeric = &report.numeric[0];
        assert!((numeric.mean_case_post - 10.0).abs() < 1e-12);
        assert!((numeric.mean_control_post - 40.0).abs() < 1e-12);
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

        let keys = collect_covariate_keys(&[case_a.clone()], &[control_a.clone()]);
        assert_eq!(keys, vec!["age".to_string(), "region".to_string()]);
        assert!(matches!(
            covariate_kind(&[case_a.clone()], &[control_a.clone()], "age"),
            CovariateKind::Numeric
        ));
        assert_eq!(mean_var_numeric(&[case_a], "age"), (0.0, 0.0));

        let levels = collect_levels(&[], &[control_a], "region");
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

        let (counts, total) = counts_categorical(&[case_a, case_b], "region", &levels);
        assert_eq!(total, 2);
        assert_eq!(counts, vec![1, 1]);
    }
}
