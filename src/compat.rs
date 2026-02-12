//! Compatibility API for case/control and diagnosis-oriented naming.

use crate::matching::{
    Constraint, ConstraintContext, MatchingRecord, RandomSelection, SelectionStrategy,
    build_outcome, match_anchors_to_candidates_with_constraints,
};
use crate::types::{MatchDiagnostics, MatchOutcome, MatchingCriteria, ParentMatching};
use chrono::NaiveDate;
use itertools::Itertools;

pub use crate::types::{
    CaseRecord, ControlRecord, ParticipantAttributes, ParticipantConstraintOptions,
    ParticipantRecord, RoleSwitchingOptions, RoleSwitchingRecord,
};

/// Compatibility alias for transition records built on [`ParticipantRecord`].
pub type RoleTransitionRecord = crate::types::RoleTransitionRecord<ParticipantRecord>;

/// Project-specific gender constraint used by compatibility wrappers.
pub struct GenderConstraint {
    /// Enforce exact gender matching when set to `true`.
    pub require_same_gender: bool,
}

impl<R: MatchingRecord + ParticipantAttributes> Constraint<R> for GenderConstraint {
    fn reason(&self) -> &'static str {
        "gender_mismatch"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        !self.require_same_gender || case.gender() == control.gender()
    }
}

/// Project-specific parent date constraint used by compatibility wrappers.
pub struct ParentDateConstraint {
    /// Parent matching strategy.
    pub matching: ParentMatching,
    /// Absolute parent date difference window in days.
    pub window_days: i32,
    /// Require all requested parent dates to be present.
    pub require_both_parents: bool,
}

impl ParentDateConstraint {
    #[must_use]
    pub const fn from_options(options: &ParticipantConstraintOptions) -> Self {
        Self {
            matching: options.parent_matching,
            window_days: options.parent_birth_date_window_days,
            require_both_parents: options.require_both_parents,
        }
    }
}

impl<R: MatchingRecord + ParticipantAttributes> Constraint<R> for ParentDateConstraint {
    fn reason(&self) -> &'static str {
        "parent_date_mismatch"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        if matches!(self.matching, ParentMatching::Disabled) {
            return true;
        }

        let window = i64::from(self.window_days);
        let mother_ok = parent_date_within_window(
            case.mother_birth_date(),
            control.mother_birth_date(),
            window,
            self.require_both_parents,
        );

        if matches!(self.matching, ParentMatching::MotherOnly) {
            return mother_ok;
        }

        let father_ok = parent_date_within_window(
            case.father_birth_date(),
            control.father_birth_date(),
            window,
            self.require_both_parents,
        );

        mother_ok && father_ok
    }
}

/// Backward-compatible case/control wrapper.
#[must_use]
pub fn match_cases_to_controls(
    cases: &[CaseRecord],
    controls: &[ControlRecord],
    criteria: &MatchingCriteria,
    seed: u64,
) -> MatchOutcome {
    match_cases_to_controls_with_options(
        cases,
        controls,
        criteria,
        &ParticipantConstraintOptions::default(),
        seed,
    )
}

/// Case/control wrapper with explicit participant constraints.
#[must_use]
pub fn match_cases_to_controls_with_options(
    cases: &[CaseRecord],
    controls: &[ControlRecord],
    criteria: &MatchingCriteria,
    participant_options: &ParticipantConstraintOptions,
    seed: u64,
) -> MatchOutcome {
    if participant_options.parent_birth_date_window_days < 0 {
        return invalid_compat_options_outcome(cases.len(), "negative_parent_birth_date_window");
    }

    let gender_constraint = GenderConstraint {
        require_same_gender: participant_options.require_same_gender,
    };
    let parent_constraint = ParentDateConstraint::from_options(participant_options);
    let constraints = [
        &gender_constraint as &dyn Constraint<CaseRecord>,
        &parent_constraint,
    ];
    match_anchors_to_candidates_with_constraints(cases, controls, criteria, seed, &constraints)
}

/// Backward-compatible role-switching wrapper.
#[must_use]
pub fn match_with_role_switching(
    cohort: &[RoleSwitchingRecord],
    criteria: &MatchingCriteria,
    options: &RoleSwitchingOptions,
    seed: u64,
) -> MatchOutcome {
    match_with_role_switching_with_strategy_and_constraints(
        cohort,
        criteria,
        options,
        RandomSelection::seeded(seed),
        &[],
    )
}

/// Backward-compatible role-switching wrapper with explicit selection strategy.
#[must_use]
pub fn match_with_role_switching_with_strategy<S: SelectionStrategy<RoleSwitchingRecord>>(
    cohort: &[RoleSwitchingRecord],
    criteria: &MatchingCriteria,
    options: &RoleSwitchingOptions,
    strategy: S,
) -> MatchOutcome {
    match_with_role_switching_with_strategy_and_constraints(
        cohort,
        criteria,
        options,
        strategy,
        &[],
    )
}

/// Backward-compatible role-switching wrapper with strategy and custom constraints.
#[must_use]
pub fn match_with_role_switching_with_strategy_and_constraints<
    S: SelectionStrategy<RoleSwitchingRecord>,
>(
    cohort: &[RoleSwitchingRecord],
    criteria: &MatchingCriteria,
    options: &RoleSwitchingOptions,
    strategy: S,
    extra_constraints: &[&dyn Constraint<RoleSwitchingRecord>],
) -> MatchOutcome {
    if options
        .participant_constraints
        .parent_birth_date_window_days
        < 0
    {
        return invalid_compat_options_outcome(cohort.len(), "negative_parent_birth_date_window");
    }

    let gender_constraint = GenderConstraint {
        require_same_gender: options.participant_constraints.require_same_gender,
    };
    let parent_constraint = ParentDateConstraint::from_options(&options.participant_constraints);
    let constraints = std::iter::once(&gender_constraint as &dyn Constraint<RoleSwitchingRecord>)
        .chain(std::iter::once(
            &parent_constraint as &dyn Constraint<RoleSwitchingRecord>,
        ))
        .chain(extra_constraints.iter().copied())
        .collect_vec();

    crate::role_transition::match_with_role_indexing(
        cohort,
        criteria,
        options.diagnosis_age_limit_years,
        &options.ratio_fallback,
        strategy,
        &constraints,
    )
}

fn parent_date_within_window(
    case_date: Option<NaiveDate>,
    control_date: Option<NaiveDate>,
    window_days: i64,
    require_both: bool,
) -> bool {
    match (case_date, control_date) {
        (Some(case), Some(control)) => (control - case).num_days().abs() <= window_days,
        _ => !require_both,
    }
}

fn invalid_compat_options_outcome(anchor_count: usize, reason: &str) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics
        .exclusion_counts
        .insert(format!("invalid_compat_options:{reason}"), 1);
    build_outcome(Vec::new(), 0, anchor_count, 0, diagnostics)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::DeterministicSelection;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn participant(id: &str, birth_date: NaiveDate) -> ParticipantRecord {
        let mut row = ParticipantRecord::new(id, birth_date);
        row.gender = Some("F".to_string());
        row
    }

    #[test]
    fn case_control_uses_compat_parent_constraint() {
        let criteria = MatchingCriteria::default();
        let mut options = ParticipantConstraintOptions {
            require_same_gender: false,
            parent_matching: ParentMatching::MotherOnly,
            parent_birth_date_window_days: 10,
            require_both_parents: true,
        };

        let mut case = participant("case", date(2010, 1, 1));
        case.mother_birth_date = Some(date(1980, 1, 1));
        let mut control = participant("control", date(2010, 1, 2));
        control.mother_birth_date = Some(date(1990, 1, 1));

        let outcome =
            match_cases_to_controls_with_options(&[case], &[control], &criteria, &options, 7);
        assert!(outcome.pairs.is_empty());
        assert_eq!(
            outcome
                .diagnostics
                .exclusion_counts
                .get("parent_date_mismatch")
                .copied(),
            Some(1)
        );

        options.parent_matching = ParentMatching::Disabled;
        let mut case = participant("case", date(2010, 1, 1));
        case.mother_birth_date = Some(date(1980, 1, 1));
        let mut control = participant("control", date(2010, 1, 2));
        control.mother_birth_date = Some(date(1990, 1, 1));
        let outcome =
            match_cases_to_controls_with_options(&[case], &[control], &criteria, &options, 7);
        assert_eq!(outcome.matched_cases, 1);
    }

    #[test]
    fn role_switching_remains_available_in_compat() {
        let criteria = MatchingCriteria::default();
        let options = RoleSwitchingOptions {
            diagnosis_age_limit_years: 6,
            ratio_fallback: vec![1],
            participant_constraints: ParticipantConstraintOptions {
                require_same_gender: false,
                ..ParticipantConstraintOptions::default()
            },
        };

        let cohort = vec![
            RoleSwitchingRecord::from_participant(
                participant("case_early", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleSwitchingRecord::from_participant(
                participant("switcher", date(2010, 1, 2)),
                Some(date(2015, 1, 1)),
            ),
            RoleSwitchingRecord::from_participant(participant("never", date(2010, 1, 7)), None),
        ];

        let outcome = match_with_role_switching_with_strategy(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
        );
        assert_eq!(outcome.matched_cases, 2);
        assert_eq!(outcome.pairs.len(), 2);
    }
}
