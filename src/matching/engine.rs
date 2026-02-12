use super::constraints::{
    Constraint, ConstraintContext, DateCaliperConstraint, ExactMatchConstraint,
    NoSelfMatchConstraint, ReplacementConstraint, UniqueKeyConstraint, build_strata_values,
    unique_value,
};
use super::index::CandidateIndex;
use super::records::MatchingRecord;
use super::selection::{RandomSelection, SelectionStrategy};
use crate::types::{
    CriteriaValidationError, MatchDiagnostics, MatchOutcome, MatchedPair, MatchingCriteria,
};
use itertools::Itertools;
use std::collections::HashSet;

/// Reusable matching engine with precomputed lookups and pluggable selection.
pub struct MatchEngine<'a, R: MatchingRecord, S: SelectionStrategy<R>> {
    controls: &'a [R],
    criteria: &'a MatchingCriteria,
    precomputed: MatchingPrecomputed<'a>,
    candidate_index: CandidateIndex<'a>,
    selector: S,
}

/// Explicit run configuration for [`MatchEngine`].
#[derive(Debug, Clone, Copy)]
pub struct EngineRunConfig {
    /// Requested number of controls per anchor.
    pub target_ratio: usize,
}

impl<'a, R: MatchingRecord, S: SelectionStrategy<R>> MatchEngine<'a, R, S> {
    /// Construct a reusable engine for a fixed control pool and criteria.
    #[must_use]
    pub fn new(controls: &'a [R], criteria: &'a MatchingCriteria, selector: S) -> Self {
        let control_strata_values = precompute_strata_values(controls, &criteria.required_strata);
        let precomputed = MatchingPrecomputed {
            criteria,
            control_strata_values,
        };

        Self {
            controls,
            criteria,
            precomputed,
            candidate_index: CandidateIndex::new(controls, &criteria.required_strata),
            selector,
        }
    }

    #[must_use]
    pub const fn controls(&self) -> &'a [R] {
        self.controls
    }

    #[must_use]
    pub const fn criteria(&self) -> &'a MatchingCriteria {
        self.criteria
    }

    pub fn candidate_pool<F>(
        &self,
        case: &R,
        used_controls: &HashSet<usize>,
        used_unique: &HashSet<String>,
        mut additional_filter: F,
        extra_constraints: &[&dyn Constraint<R>],
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<usize>
    where
        F: FnMut(usize, &R) -> bool,
    {
        let case_strata_values = if self.criteria.required_strata.is_empty() {
            None
        } else {
            Some(build_strata_values(
                case.strata(),
                &self.criteria.required_strata,
            ))
        };

        let candidate_indices = self.candidate_index.candidate_indices(
            case.birth_date(),
            i64::from(self.criteria.birth_date_window_days),
            case_strata_values.as_deref(),
        );
        let mut eligible = Vec::new();

        let replacement_constraint = ReplacementConstraint;
        let no_self_match_constraint = NoSelfMatchConstraint;
        let strata_constraint = ExactMatchConstraint;
        let unique_constraint = UniqueKeyConstraint;
        let date_caliper_constraint = DateCaliperConstraint;
        let builtin_constraints: [&dyn Constraint<R>; 5] = [
            &replacement_constraint,
            &no_self_match_constraint,
            &strata_constraint,
            &unique_constraint,
            &date_caliper_constraint,
        ];

        for idx in candidate_indices {
            let control = &self.controls[idx];

            if !additional_filter(idx, control) {
                bump_count(diagnostics, "additional_filter");
                continue;
            }

            let control_strata_values = self
                .precomputed
                .control_strata_values
                .as_deref()
                .and_then(|values| values.get(idx).map(Vec::as_slice));
            let context = ConstraintContext {
                criteria: self.precomputed.criteria,
                used_controls,
                used_unique,
                control_idx: idx,
                case_strata_values: case_strata_values.as_deref(),
                control_strata_values,
            };

            let mut blocked_reason = None;
            for constraint in builtin_constraints
                .iter()
                .copied()
                .chain(extra_constraints.iter().copied())
            {
                if !constraint.allows(case, control, &context) {
                    blocked_reason = Some(constraint.reason());
                    break;
                }
            }
            if let Some(reason) = blocked_reason {
                bump_count(diagnostics, reason);
                continue;
            }

            eligible.push(idx);
        }

        eligible
    }

    pub fn select_control(
        &mut self,
        case: &R,
        candidate_indices: &mut Vec<usize>,
    ) -> Option<usize> {
        let selected_pos = self
            .selector
            .select_position(case, self.controls, candidate_indices)?;
        Some(candidate_indices.swap_remove(selected_pos))
    }

    /// Run simple anchor matching with this engine's fixed control pool.
    #[must_use]
    pub fn run_anchor_matching(
        &mut self,
        anchors: &[R],
        config: EngineRunConfig,
        extra_constraints: &[&dyn Constraint<R>],
    ) -> MatchOutcome {
        let mut pairs = Vec::new();
        let mut used_controls = HashSet::new();
        let mut used_unique = HashSet::new();
        let mut matched_cases = 0usize;
        let mut diagnostics = MatchDiagnostics {
            total_anchors_evaluated: anchors.len(),
            ..MatchDiagnostics::default()
        };

        for case in anchors {
            let mut candidates = self.candidate_pool(
                case,
                &used_controls,
                &used_unique,
                |_idx, _control| true,
                extra_constraints,
                &mut diagnostics,
            );
            if candidates.is_empty() {
                diagnostics.anchors_with_no_candidates += 1;
                continue;
            }
            if candidates.len() < config.target_ratio {
                diagnostics.anchors_below_required_ratio += 1;
            }

            let mut local_matches = 0usize;
            for _ in 0..config.target_ratio {
                let Some(selected_idx) = self.select_control(case, &mut candidates) else {
                    break;
                };
                let control = &self.controls[selected_idx];
                pairs.push(MatchedPair::new(case.id(), control.id()));
                local_matches += 1;

                if !self.criteria.allow_replacement {
                    used_controls.insert(selected_idx);
                }
                if let Some(value) = unique_value(control, self.criteria) {
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
            anchors.len(),
            used_controls.len(),
            diagnostics,
        )
    }
}

/// Run caliper-based matching using neutral anchor/candidate terminology.
#[must_use]
pub fn match_anchors_to_candidates<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    criteria: &MatchingCriteria,
    seed: u64,
) -> MatchOutcome {
    match_anchors_to_candidates_with_strategy(
        anchors,
        candidates,
        criteria,
        RandomSelection::seeded(seed),
    )
}

/// Run anchor/candidate matching with an explicit control selection strategy.
#[must_use]
pub fn match_anchors_to_candidates_with_strategy<R: MatchingRecord, S: SelectionStrategy<R>>(
    anchors: &[R],
    candidates: &[R],
    criteria: &MatchingCriteria,
    strategy: S,
) -> MatchOutcome {
    match_anchors_to_candidates_with_strategy_and_constraints(
        anchors,
        candidates,
        criteria,
        strategy,
        &[],
    )
}

/// Run anchor/candidate matching with random selection and custom constraints.
#[must_use]
pub fn match_anchors_to_candidates_with_constraints<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    criteria: &MatchingCriteria,
    seed: u64,
    extra_constraints: &[&dyn Constraint<R>],
) -> MatchOutcome {
    match_anchors_to_candidates_with_strategy_and_constraints(
        anchors,
        candidates,
        criteria,
        RandomSelection::seeded(seed),
        extra_constraints,
    )
}

#[must_use]
pub fn match_anchors_to_candidates_with_strategy_and_constraints<
    R: MatchingRecord,
    S: SelectionStrategy<R>,
>(
    anchors: &[R],
    candidates: &[R],
    criteria: &MatchingCriteria,
    strategy: S,
    extra_constraints: &[&dyn Constraint<R>],
) -> MatchOutcome {
    let validated = match criteria.validate() {
        Ok(validated) => validated,
        Err(err) => return invalid_criteria_outcome(anchors.len(), err),
    };

    let mut engine = MatchEngine::new(candidates, validated.as_ref(), strategy);
    engine.run_anchor_matching(
        anchors,
        EngineRunConfig {
            target_ratio: validated.match_ratio,
        },
        extra_constraints,
    )
}

pub struct MatchingPrecomputed<'a> {
    pub(crate) criteria: &'a MatchingCriteria,
    pub(crate) control_strata_values: Option<Vec<Vec<Option<&'a str>>>>,
}

pub fn precompute_strata_values<'a, R: MatchingRecord>(
    controls: &'a [R],
    required_strata: &[String],
) -> Option<Vec<Vec<Option<&'a str>>>> {
    if required_strata.is_empty() {
        None
    } else {
        Some(
            controls
                .iter()
                .map(|control| build_strata_values(control.strata(), required_strata))
                .collect_vec(),
        )
    }
}

pub fn build_outcome(
    pairs: Vec<MatchedPair>,
    matched_cases: usize,
    eligible_cases: usize,
    used_controls_len: usize,
    diagnostics: MatchDiagnostics,
) -> MatchOutcome {
    let matched = pairs.len();
    let avg_controls = if matched_cases > 0 {
        to_f64(matched) / to_f64(matched_cases)
    } else {
        0.0
    };

    MatchOutcome {
        pairs,
        unmatched_cases: eligible_cases.saturating_sub(matched_cases),
        used_controls: used_controls_len,
        matched_cases,
        avg_controls_per_case: avg_controls,
        diagnostics,
    }
}

fn bump_count(diagnostics: &mut MatchDiagnostics, reason: &str) {
    *diagnostics
        .exclusion_counts
        .entry(reason.to_string())
        .or_insert(0) += 1;
}

pub fn invalid_criteria_outcome(anchor_count: usize, err: CriteriaValidationError) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics
        .exclusion_counts
        .insert(format!("invalid_criteria:{err:?}"), 1);

    build_outcome(Vec::new(), 0, anchor_count, 0, diagnostics)
}

pub fn to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::DeterministicSelection;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn record(id: &str, birth_date: NaiveDate) -> BaseRecord {
        BaseRecord::new(id, birth_date)
    }

    #[test]
    fn matches_within_birth_window() {
        let criteria = MatchingCriteria::default();
        let anchor = record("anchor", date(2010, 1, 1));
        let candidate = record("candidate", date(2010, 1, 15));

        let outcome = match_anchors_to_candidates(&[anchor], &[candidate], &criteria, 42);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.unmatched_cases, 0);
    }

    #[test]
    fn unmatched_cases_uses_case_count_not_pair_count() {
        let criteria = MatchingCriteria {
            match_ratio: 2,
            ..MatchingCriteria::default()
        };

        let anchor_a = record("anchor_a", date(2010, 1, 1));
        let anchor_b = record("anchor_b", date(2010, 1, 2));

        let candidate_a = record("candidate_a", date(2010, 1, 1));
        let candidate_b = record("candidate_b", date(2010, 1, 3));

        let outcome = match_anchors_to_candidates(
            &[anchor_a, anchor_b],
            &[candidate_a, candidate_b],
            &criteria,
            1,
        );

        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 2);
        assert_eq!(outcome.unmatched_cases, 1);
    }

    #[test]
    fn neutral_wrapper_matches() {
        let criteria = MatchingCriteria::default();
        let anchor = record("anchor", date(2010, 1, 1));
        let candidate = record("candidate", date(2010, 1, 2));

        let outcome = match_anchors_to_candidates(&[anchor], &[candidate], &criteria, 11);
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }

    #[test]
    fn deterministic_strategy_is_stable() {
        let criteria = MatchingCriteria::default();

        let anchor = record("anchor", date(2010, 1, 1));
        let c1 = record("c1", date(2010, 1, 2));
        let c2 = record("c2", date(2010, 1, 3));

        let outcome = match_anchors_to_candidates_with_strategy(
            &[anchor],
            &[c1, c2],
            &criteria,
            DeterministicSelection,
        );

        assert_eq!(outcome.pairs[0].comparator_id(), "c1");
    }

    #[test]
    fn custom_constraints_are_applied() {
        struct NeverAllow;

        impl Constraint<BaseRecord> for NeverAllow {
            fn reason(&self) -> &'static str {
                "never_allow"
            }

            fn allows(
                &self,
                _case: &BaseRecord,
                _control: &BaseRecord,
                _ctx: &ConstraintContext<'_>,
            ) -> bool {
                false
            }
        }

        let criteria = MatchingCriteria::default();
        let anchor = record("anchor", date(2010, 1, 1));
        let candidate = record("candidate", date(2010, 1, 2));
        let never_allow = NeverAllow;

        let outcome = match_anchors_to_candidates_with_constraints(
            &[anchor],
            &[candidate],
            &criteria,
            1,
            &[&never_allow],
        );

        assert!(outcome.pairs.is_empty());
        assert_eq!(
            outcome
                .diagnostics
                .exclusion_counts
                .get("never_allow")
                .copied(),
            Some(1)
        );
    }
}
