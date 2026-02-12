use crate::matching::{
    Constraint, MatchEngine, RandomSelection, RoleIndexedRecord, SelectionStrategy, build_outcome,
    invalid_criteria_outcome, unique_value,
};
use crate::types::{
    MatchDiagnostics, MatchOutcome, MatchedPair, MatchingCriteria, RoleTransitionOptions,
};
use chrono::{Months, NaiveDate};
use itertools::Itertools;
use std::collections::HashSet;

/// Risk-set matching with role transitions using neutral terminology.
#[must_use]
pub fn match_with_role_transition<R: RoleIndexedRecord>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    options: &RoleTransitionOptions,
    seed: u64,
) -> MatchOutcome {
    match_with_role_transition_with_strategy_and_constraints(
        cohort,
        criteria,
        options,
        RandomSelection::seeded(seed),
        &[],
    )
}

/// Role-transition matching with explicit candidate selection strategy.
#[must_use]
pub fn match_with_role_transition_with_strategy<R: RoleIndexedRecord, S: SelectionStrategy<R>>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    options: &RoleTransitionOptions,
    strategy: S,
) -> MatchOutcome {
    match_with_role_transition_with_strategy_and_constraints(
        cohort,
        criteria,
        options,
        strategy,
        &[],
    )
}

/// Role-transition matching with explicit strategy and custom constraints.
#[must_use]
pub fn match_with_role_transition_with_strategy_and_constraints<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R>,
>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    options: &RoleTransitionOptions,
    strategy: S,
    extra_constraints: &[&dyn Constraint<R>],
) -> MatchOutcome {
    match_with_role_indexing(
        cohort,
        criteria,
        options.transition_age_limit_years,
        &options.ratio_fallback,
        strategy,
        extra_constraints,
    )
}

pub fn match_with_role_indexing<R: RoleIndexedRecord, S: SelectionStrategy<R>>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    age_limit_years: u8,
    ratio_fallback: &[usize],
    strategy: S,
    extra_constraints: &[&dyn Constraint<R>],
) -> MatchOutcome {
    let validated = match criteria.validate() {
        Ok(validated) => validated,
        Err(err) => return invalid_criteria_outcome(cohort.len(), err),
    };

    let ratios = resolved_ratios(validated.as_ref(), ratio_fallback);
    let case_indices = eligible_case_indices(cohort, age_limit_years);
    let mut engine = MatchEngine::new(cohort, validated.as_ref(), strategy);

    let mut pairs = Vec::new();
    let mut used_controls = HashSet::new();
    let mut used_unique = HashSet::new();
    let mut matched_cases = 0usize;
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: case_indices.len(),
        ..MatchDiagnostics::default()
    };

    for case_idx in &case_indices {
        let case = &cohort[*case_idx];
        let Some(case_event_date) = case.event_date() else {
            continue;
        };

        let mut candidates = engine.candidate_pool(
            case,
            &used_controls,
            &used_unique,
            |_idx, control| control_is_at_risk(control, case_event_date),
            extra_constraints,
            &mut diagnostics,
        );

        if candidates.is_empty() {
            diagnostics.anchors_with_no_candidates += 1;
            continue;
        }

        let Some(target_ratio) = best_ratio(candidates.len(), &ratios) else {
            diagnostics.anchors_below_required_ratio += 1;
            continue;
        };

        let mut local_matches = 0usize;
        for _ in 0..target_ratio {
            let Some(selected_idx) = engine.select_control(case, &mut candidates) else {
                break;
            };
            let control = &engine.controls()[selected_idx];
            pairs.push(MatchedPair::new(case.id(), control.id()));
            local_matches += 1;

            if !validated.allow_replacement {
                used_controls.insert(selected_idx);
            }
            if let Some(value) = unique_value(control, validated.as_ref()) {
                used_unique.insert(value.to_string());
            }
        }

        if local_matches > 0 {
            matched_cases += 1;
        }
    }

    diagnostics.matched_anchors = matched_cases;
    diagnostics.pairs_selected = pairs.len();
    build_outcome(
        pairs,
        matched_cases,
        case_indices.len(),
        used_controls.len(),
        diagnostics,
    )
}

fn eligible_case_indices<R: RoleIndexedRecord>(cohort: &[R], age_limit_years: u8) -> Vec<usize> {
    cohort
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            if is_case_before_age_limit(row, age_limit_years) {
                Some(idx)
            } else {
                None
            }
        })
        .sorted_by(|left_idx, right_idx| {
            let left = &cohort[*left_idx];
            let right = &cohort[*right_idx];
            left.event_date()
                .cmp(&right.event_date())
                .then_with(|| left.id().cmp(right.id()))
        })
        .collect_vec()
}

fn is_case_before_age_limit<R: RoleIndexedRecord>(row: &R, age_limit_years: u8) -> bool {
    let Some(event_date) = row.event_date() else {
        return false;
    };
    let age_limit_months = u32::from(age_limit_years) * 12;
    let Some(age_limit_date) = row
        .birth_date()
        .checked_add_months(Months::new(age_limit_months))
    else {
        return false;
    };

    event_date < age_limit_date
}

fn control_is_at_risk<R: RoleIndexedRecord>(control: &R, case_event_date: NaiveDate) -> bool {
    control
        .event_date()
        .is_none_or(|event_date| event_date > case_event_date)
}

fn resolved_ratios(criteria: &MatchingCriteria, ratio_fallback: &[usize]) -> Vec<usize> {
    if ratio_fallback.is_empty() {
        vec![criteria.match_ratio]
    } else {
        ratio_fallback.to_vec()
    }
    .into_iter()
    .filter(|ratio| *ratio > 0)
    .sorted_by(|left, right| right.cmp(left))
    .unique()
    .collect_vec()
}

fn best_ratio(available_controls: usize, ratios: &[usize]) -> Option<usize> {
    ratios
        .iter()
        .copied()
        .find(|ratio| available_controls >= *ratio)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::DeterministicSelection;
    use crate::types::{BaseRecord, RoleTransitionRecord};
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    #[test]
    fn transition_matching_is_generic_over_record_shape() {
        #[derive(Clone)]
        struct DemoRecord {
            id: String,
            birth_date: NaiveDate,
            strata: std::collections::HashMap<String, String>,
            unique_key: Option<String>,
        }

        impl crate::MatchingRecord for DemoRecord {
            fn id(&self) -> &str {
                &self.id
            }

            fn birth_date(&self) -> NaiveDate {
                self.birth_date
            }

            fn strata(&self) -> &std::collections::HashMap<String, String> {
                &self.strata
            }

            fn unique_key(&self) -> Option<&str> {
                self.unique_key.as_deref()
            }
        }

        impl DemoRecord {
            fn new(id: &str, birth_date: NaiveDate) -> Self {
                Self {
                    id: id.to_string(),
                    birth_date,
                    strata: std::collections::HashMap::new(),
                    unique_key: None,
                }
            }
        }

        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: 6,
            ratio_fallback: vec![1],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                DemoRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(DemoRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = match_with_role_transition_with_strategy(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }

    #[test]
    fn transition_works_with_base_record_default_type() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: 6,
            ratio_fallback: vec![1],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = match_with_role_transition(&cohort, &criteria, &options, 9);
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }
}
