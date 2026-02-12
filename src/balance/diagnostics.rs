use crate::matching::to_f64;
use crate::types::{BalanceDiagnostics, CaseRecord, ControlRecord, MatchOutcome};
use std::collections::HashMap;

/// Compute cohort-level balance diagnostics from matched pairs.
#[must_use]
pub fn balance_diagnostics(
    cases: &[CaseRecord],
    controls: &[ControlRecord],
    outcome: &MatchOutcome,
    strata_keys: &[String],
) -> BalanceDiagnostics {
    let case_map = cases
        .iter()
        .map(|case| (case.core.id.clone(), case))
        .collect::<HashMap<_, _>>();
    let control_map = controls
        .iter()
        .map(|control| (control.core.id.clone(), control))
        .collect::<HashMap<_, _>>();

    let mut strata_counts = HashMap::new();
    for pair in &outcome.pairs {
        if let Some(case) = case_map.get(&pair.case_id) {
            for key in strata_keys {
                if let Some(value) = case.core.strata.get(key) {
                    let entry = strata_counts
                        .entry(format!("case:{key}={value}"))
                        .or_insert((0, 0));
                    entry.0 += 1;
                }
            }
        }
        if let Some(control) = control_map.get(&pair.control_id) {
            for key in strata_keys {
                if let Some(value) = control.core.strata.get(key) {
                    let entry = strata_counts
                        .entry(format!("control:{key}={value}"))
                        .or_insert((0, 0));
                    entry.1 += 1;
                }
            }
        }
    }

    let match_rate = if cases.is_empty() {
        0.0
    } else {
        to_f64(outcome.matched_cases) / to_f64(cases.len())
    };

    BalanceDiagnostics {
        match_rate,
        matched_cases: outcome.matched_cases,
        unmatched_cases: outcome.unmatched_cases,
        avg_controls_per_case: outcome.avg_controls_per_case,
        strata_counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MatchDiagnostics, MatchedPair, ParticipantRecord};
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn participant(id: &str, municipality: &str) -> ParticipantRecord {
        let mut row = ParticipantRecord::new(id, date(2010, 1, 1));
        row.core
            .strata
            .insert("municipality".to_string(), municipality.to_string());
        row
    }

    fn outcome(
        pairs: Vec<MatchedPair>,
        matched_cases: usize,
        unmatched_cases: usize,
    ) -> MatchOutcome {
        MatchOutcome {
            avg_controls_per_case: if matched_cases == 0 {
                0.0
            } else {
                f64::from(u32::try_from(pairs.len()).expect("small test vector length"))
                    / f64::from(u32::try_from(matched_cases).expect("small matched case count"))
            },
            used_controls: pairs.len(),
            pairs,
            matched_cases,
            unmatched_cases,
            diagnostics: MatchDiagnostics::default(),
        }
    }

    #[test]
    fn diagnostics_counts_strata_for_matched_pairs() {
        let cases = vec![participant("case_a", "0101"), participant("case_b", "0202")];
        let controls = vec![
            participant("control_a", "0101"),
            participant("control_b", "0202"),
        ];
        let pairs = vec![
            MatchedPair::new("case_a", "control_a"),
            MatchedPair::new("case_b", "control_b"),
            MatchedPair::new("missing", "control_a"),
        ];
        let result = balance_diagnostics(
            &cases,
            &controls,
            &outcome(pairs, 2, 0),
            &["municipality".to_string()],
        );

        assert!((result.match_rate - 1.0).abs() < 1e-12);
        assert_eq!(result.matched_cases, 2);
        assert_eq!(
            result.strata_counts.get("case:municipality=0101"),
            Some(&(1, 0))
        );
        assert_eq!(
            result.strata_counts.get("control:municipality=0101"),
            Some(&(0, 2))
        );
        assert_eq!(
            result.strata_counts.get("case:municipality=0202"),
            Some(&(1, 0))
        );
    }

    #[test]
    fn diagnostics_handles_empty_case_cohort() {
        let result = balance_diagnostics(
            &[],
            &[],
            &outcome(Vec::new(), 0, 0),
            &["municipality".to_string()],
        );
        assert!((result.match_rate - 0.0).abs() < 1e-12);
        assert!(result.strata_counts.is_empty());
    }
}
