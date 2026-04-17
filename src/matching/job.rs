use crate::matching::engine::{StandardMatchRequest, match_standard};
use crate::matching::{
    Constraint, ConstraintGroup, DistanceConfig, MatchingRecord, RandomSelection,
    ResidentAtIndexRecord, RoleIndexedRecord, SelectionStrategy,
};
use crate::role_transition::{
    DefaultRiskSetPolicy, RiskSetPolicy, TransitionMatchRequest, match_transition,
};
use crate::types::{
    AgeLimitYears, BirthDateWindowDays, Estimand, MatchOutcome, MatchRatio, MatchingCriteria,
    RoleTransitionOptions,
};
use std::marker::PhantomData;

/// Marker and data for standard matching.
pub struct StandardMode<'a, R: MatchingRecord> {
    anchors: &'a [R],
    candidates: &'a [R],
    ratio_fallback: Vec<MatchRatio>,
    distance_config: Option<DistanceConfig>,
}

/// Marker and data for role-transition matching.
pub struct TransitionMode<'a, R: RoleIndexedRecord> {
    cohort: &'a [R],
    options: RoleTransitionOptions,
    distance_config: Option<DistanceConfig>,
}

type AliveSelector<R> = fn(&R) -> Option<chrono::NaiveDate>;
type ResidentSelector<R> = fn(&R, chrono::NaiveDate) -> bool;
type WithAliveConstraint<Mode, R, S, G, P> =
    MatchJob<Mode, R, S, (G, crate::constraints::MustBeAlive<AliveSelector<R>>), P>;
type WithResidentConstraint<Mode, R, S, G, P> =
    MatchJob<Mode, R, S, (G, crate::constraints::MustBeResident<ResidentSelector<R>>), P>;

/// Unified entrypoint for matching jobs.
///
/// `MatchJob` provides a fluent, statically-typed builder API for configuring and executing cohort matching.
/// It uses a typestate pattern to ensure that standard matching and role-transition matching
/// expose only their valid methods and execution paths.
///
/// ### Example (Standard)
/// ```rust,ignore
/// use cohort_matching::prelude::*;
///
/// let job = MatchJob::new_standard(&anchors, &candidates, 42)
///     .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
///     .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
///     .with_gender_match();
///
/// let result = job.run();
/// ```
///
/// ### Example (Role-Transition)
/// ```rust,ignore
/// use cohort_matching::prelude::*;
///
/// let job = MatchJob::new_transition(&records, 42)
///     .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
///     .with_age_limit(AgeLimitYears::new(6).expect("positive age limit"))
///     .with_alive_check();
///
/// let result = job.run();
/// ```
pub struct MatchJob<Mode, R, S, G = (), P = DefaultRiskSetPolicy> {
    mode: Mode,
    criteria: MatchingCriteria,
    strategy: S,
    constraints: G,
    risk_set_policy: P,
    record: PhantomData<R>,
}

impl<'a, R: RoleIndexedRecord>
    MatchJob<TransitionMode<'a, R>, R, RandomSelection, (), DefaultRiskSetPolicy>
{
    /// Create a new matching job for role-transition risk-set matching.
    #[must_use]
    pub fn new_transition(cohort: &'a [R], seed: u64) -> Self {
        Self {
            mode: TransitionMode {
                cohort,
                options: RoleTransitionOptions::default(),
                distance_config: None,
            },
            criteria: MatchingCriteria::default(),
            strategy: RandomSelection::seeded(seed),
            constraints: (),
            risk_set_policy: DefaultRiskSetPolicy,
            record: PhantomData,
        }
    }
}

impl<'a, R: MatchingRecord>
    MatchJob<StandardMode<'a, R>, R, RandomSelection, (), DefaultRiskSetPolicy>
{
    /// Create a new matching job for standard anchor-to-candidate matching.
    #[must_use]
    pub fn new_standard(anchors: &'a [R], candidates: &'a [R], seed: u64) -> Self {
        Self {
            mode: StandardMode {
                anchors,
                candidates,
                ratio_fallback: Vec::new(),
                distance_config: None,
            },
            criteria: MatchingCriteria::default(),
            strategy: RandomSelection::seeded(seed),
            constraints: (),
            risk_set_policy: DefaultRiskSetPolicy,
            record: PhantomData,
        }
    }
}

impl<Mode, R: MatchingRecord, S, G, P> MatchJob<Mode, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Set the matching ratio (number of controls per case).
    #[must_use]
    pub const fn with_ratio(mut self, ratio: MatchRatio) -> Self {
        self.criteria.match_ratio = ratio.get();
        self
    }

    /// Set the birth date caliper window in days.
    ///
    /// Candidates must have a birth date within `+/- days` of the case.
    #[must_use]
    pub const fn with_birth_window(mut self, days: BirthDateWindowDays) -> Self {
        self.criteria.birth_date_window_days = days.get();
        self
    }

    /// Add a required exact-match strata key.
    ///
    /// The engine will only consider candidates that have the same value for this
    /// strata key as the case.
    #[must_use]
    pub fn with_exact_match(mut self, key: impl Into<String>) -> Self {
        self.criteria.required_strata.push(key.into());
        self
    }

    /// Add multiple required exact-match strata keys.
    ///
    /// Equivalent to calling [`Self::with_exact_match`] once per key.
    #[must_use]
    pub fn with_exact_matches<I, K>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = K>,
        K: Into<String>,
    {
        self.criteria
            .required_strata
            .extend(keys.into_iter().map(Into::into));
        self
    }

    /// Enable or disable replacement (candidate reuse).
    ///
    /// If `true`, a candidate can be matched to multiple cases.
    #[must_use]
    pub const fn with_replacement(mut self, allow: bool) -> Self {
        self.criteria.allow_replacement = allow;
        self
    }

    /// Set optional uniqueness by a strata key.
    ///
    /// When set, controls sharing the same value for this key are treated as mutually exclusive.
    #[must_use]
    pub fn with_unique_by_key(mut self, key: Option<String>) -> Self {
        self.criteria.unique_by_key = key;
        self
    }

    /// Set the target estimand (e.g., ATT, ATE).
    #[must_use]
    pub const fn with_estimand(mut self, estimand: Estimand) -> Self {
        self.criteria.estimand = estimand;
        self
    }

    /// Set a custom selection strategy.
    #[must_use]
    pub fn with_strategy<S2>(self, strategy: S2) -> MatchJob<Mode, R, S2, G, P> {
        MatchJob::<Mode, R, S2, G, P> {
            mode: self.mode,
            criteria: self.criteria,
            strategy,
            constraints: self.constraints,
            risk_set_policy: self.risk_set_policy,
            record: PhantomData,
        }
    }

    /// Add a matching constraint.
    #[must_use]
    pub fn with_constraint<C>(self, constraint: C) -> MatchJob<Mode, R, S, (G, C), P>
    where
        C: Constraint<R>,
    {
        MatchJob::<Mode, R, S, (G, C), P> {
            mode: self.mode,
            criteria: self.criteria,
            strategy: self.strategy,
            constraints: (self.constraints, constraint),
            risk_set_policy: self.risk_set_policy,
            record: PhantomData,
        }
    }

    /// Add a gender match constraint.
    ///
    /// Convenience helper that adds a [`GenderMatch`] constraint using the "sex" strata key.
    #[must_use]
    pub fn with_gender_match(
        self,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::GenderMatch), P> {
        use crate::constraints::GenderMatch;
        self.with_constraint(GenderMatch::on_key("sex"))
    }
}

impl<Mode, R: RoleIndexedRecord, S, G, P> MatchJob<Mode, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Add a constraint ensuring controls are alive at each case's index date.
    ///
    /// This is equivalent to:
    /// `with_constraint(MustBeAlive::at_index_date(|record: &R| record.death_date()))`.
    ///
    /// Matching behavior for each case/control pair:
    /// - If the case has no `event_date`, the pair is allowed by this constraint.
    /// - If the control has no `death_date`, the pair is allowed by this constraint.
    /// - Otherwise, the pair is allowed only when `control.death_date > case.event_date`.
    #[must_use]
    pub fn with_alive_check(self) -> WithAliveConstraint<Mode, R, S, G, P> {
        use crate::constraints::MustBeAlive;
        fn death_selector<R: RoleIndexedRecord>(record: &R) -> Option<chrono::NaiveDate> {
            record.death_date()
        }
        self.with_constraint(MustBeAlive::at_index_date(death_selector::<R>))
    }

    /// Add a custom alive-at-index constraint.
    ///
    /// Use this when death dates are provided by an external map.
    #[must_use]
    pub fn with_alive_check_by<F>(
        self,
        selector: F,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::MustBeAlive<F>), P>
    where
        F: Fn(&R) -> Option<chrono::NaiveDate> + Send + Sync,
    {
        use crate::constraints::MustBeAlive;
        self.with_constraint(MustBeAlive::at_index_date(selector))
    }

    /// Add a constraint ensuring controls are resident at each case's index date.
    ///
    /// This uses [`ResidentAtIndexRecord::is_resident_at`] as the residency source.
    #[must_use]
    pub fn with_resident_check(self) -> WithResidentConstraint<Mode, R, S, G, P>
    where
        R: ResidentAtIndexRecord,
    {
        use crate::constraints::MustBeResident;
        fn resident_selector<R: ResidentAtIndexRecord>(
            record: &R,
            index_date: chrono::NaiveDate,
        ) -> bool {
            record.is_resident_at(index_date)
        }
        self.with_constraint(MustBeResident::at_index_date(resident_selector::<R>))
    }

    /// Add a custom residency-at-index constraint.
    ///
    /// Use this when residency is derived from study-specific fields.
    #[must_use]
    pub fn with_resident_check_by<F>(
        self,
        check: F,
    ) -> MatchJob<Mode, R, S, (G, crate::constraints::MustBeResident<F>), P>
    where
        F: Fn(&R, chrono::NaiveDate) -> bool + Send + Sync,
    {
        use crate::constraints::MustBeResident;
        self.with_constraint(MustBeResident::at_index_date(check))
    }
}

impl<'a, R: RoleIndexedRecord, S, G, P> MatchJob<TransitionMode<'a, R>, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Set the transition age limit in years (for role-transition matching).
    #[must_use]
    pub const fn with_age_limit(mut self, years: AgeLimitYears) -> Self {
        self.mode.options.transition_age_limit_years = years;
        self
    }

    /// Set optional descending fallback ratios for role-transition matching.
    #[must_use]
    pub fn with_ratio_fallback(mut self, fallback: Vec<MatchRatio>) -> Self {
        self.mode.options.ratio_fallback = fallback;
        self
    }

    /// Set a custom distance metric configuration.
    #[must_use]
    pub fn with_distance_config(mut self, config: DistanceConfig) -> Self {
        self.mode.distance_config = Some(config);
        self
    }

    /// Set a custom risk-set policy.
    #[must_use]
    pub fn with_risk_set_policy<P2>(
        self,
        risk_set_policy: P2,
    ) -> MatchJob<TransitionMode<'a, R>, R, S, G, P2> {
        MatchJob::<TransitionMode<'a, R>, R, S, G, P2> {
            mode: self.mode,
            criteria: self.criteria,
            strategy: self.strategy,
            constraints: self.constraints,
            risk_set_policy,
            record: PhantomData,
        }
    }

    /// Execute role-transition matching.
    pub fn run(self) -> MatchOutcome
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
        P: RiskSetPolicy<R> + Send + Sync,
    {
        match_transition(
            self.mode.cohort,
            TransitionMatchRequest {
                criteria: &self.criteria,
                options: &self.mode.options,
                strategy: self.strategy,
                constraints: &self.constraints,
                distance_config: self.mode.distance_config.as_ref(),
                risk_set_policy: &self.risk_set_policy,
            },
        )
    }

    /// Execute role-transition matching and compute balance diagnostics in one call.
    pub fn run_with_balance(
        self,
        cases: &[crate::types::BalanceRecord],
        controls: &[crate::types::BalanceRecord],
    ) -> (MatchOutcome, crate::types::BalanceReport)
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
        P: RiskSetPolicy<R> + Send + Sync,
    {
        let outcome = self.run();
        let report = crate::balance_report(cases, controls, &outcome);
        (outcome, report)
    }
}

impl<R: MatchingRecord, S, G, P> MatchJob<StandardMode<'_, R>, R, S, G, P>
where
    G: ConstraintGroup<R>,
{
    /// Set optional descending fallback ratios for standard matching.
    #[must_use]
    pub fn with_ratio_fallback(mut self, fallback: Vec<MatchRatio>) -> Self {
        self.mode.ratio_fallback = fallback;
        self
    }

    /// Set a custom distance metric configuration.
    #[must_use]
    pub fn with_distance_config(mut self, config: DistanceConfig) -> Self {
        self.mode.distance_config = Some(config);
        self
    }

    /// Execute standard matching.
    pub fn run(self) -> MatchOutcome
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
    {
        match_standard(
            self.mode.anchors,
            self.mode.candidates,
            StandardMatchRequest {
                criteria: &self.criteria,
                strategy: self.strategy,
                constraints: &self.constraints,
                ratio_fallback: &self.mode.ratio_fallback,
                distance_config: self.mode.distance_config.as_ref(),
            },
        )
    }

    /// Execute standard matching and compute balance diagnostics in one call.
    pub fn run_with_balance(
        self,
        cases: &[crate::types::BalanceRecord],
        controls: &[crate::types::BalanceRecord],
    ) -> (MatchOutcome, crate::types::BalanceReport)
    where
        S: SelectionStrategy<R> + Clone + Send + Sync,
    {
        let outcome = self.run();
        let report = crate::balance_report(cases, controls, &outcome);
        (outcome, report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::types::BaseRecord;

    #[test]
    fn match_job_runs_transition() {
        use crate::types::RoleTransitionRecord;

        let records = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("a1", date(2010, 1, 1)),
                Some(date(2012, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("c1", date(2010, 1, 2)), None),
        ];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs.len(), 1);
    }

    #[test]
    fn match_job_runs_standard() {
        let anchors = vec![BaseRecord::new("a1", date(2010, 1, 1))];
        let candidates = vec![BaseRecord::new("c1", date(2010, 1, 2))];

        let result = MatchJob::new_standard(&anchors, &candidates, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs.len(), 1);
    }

    #[test]
    fn match_job_with_gender_match_helper() {
        let mut a1 = BaseRecord::new("a1", date(2010, 1, 1));
        a1.strata.insert("sex".to_string(), "M".to_string());

        let mut c1_m = BaseRecord::new("c1_m", date(2010, 1, 1));
        c1_m.strata.insert("sex".to_string(), "M".to_string());

        let mut c2_f = BaseRecord::new("c2_f", date(2010, 1, 1));
        c2_f.strata.insert("sex".to_string(), "F".to_string());

        let anchors = vec![a1];
        let candidates = vec![c1_m, c2_f];

        let result = MatchJob::new_standard(&anchors, &candidates, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_gender_match()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c1_m");
    }

    #[test]
    fn match_job_with_exact_matches_helper() {
        let mut a1 = BaseRecord::new("a1", date(2010, 1, 1));
        a1.strata.insert("sex".to_string(), "M".to_string());
        a1.strata.insert("region".to_string(), "A".to_string());

        let mut c1_match = BaseRecord::new("c1_match", date(2010, 1, 1));
        c1_match.strata.insert("sex".to_string(), "M".to_string());
        c1_match
            .strata
            .insert("region".to_string(), "A".to_string());

        let mut c2_wrong_region = BaseRecord::new("c2_wrong_region", date(2010, 1, 1));
        c2_wrong_region
            .strata
            .insert("sex".to_string(), "M".to_string());
        c2_wrong_region
            .strata
            .insert("region".to_string(), "B".to_string());

        let anchors = vec![a1];
        let candidates = vec![c1_match, c2_wrong_region];

        let result = MatchJob::new_standard(&anchors, &candidates, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_exact_matches(["sex", "region"])
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c1_match");
    }

    #[test]
    fn match_job_with_alive_check_helper() {
        use crate::types::RoleTransitionRecord;

        let a1 = RoleTransitionRecord::from_record(
            BaseRecord::new("a1", date(2010, 1, 1)),
            Some(date(2012, 1, 1)),
        );

        let mut c1_dead = BaseRecord::new("c1_dead", date(2010, 1, 1));
        c1_dead.death_date = Some(date(2011, 1, 1)); // dead before index
        let c1_dead = RoleTransitionRecord::from_record(c1_dead, None);

        let mut c2_alive = BaseRecord::new("c2_alive", date(2010, 1, 1));
        c2_alive.death_date = Some(date(2015, 1, 1)); // alive at index
        let c2_alive = RoleTransitionRecord::from_record(c2_alive, None);

        let records = vec![a1, c1_dead, c2_alive];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_alive_check()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_alive");
    }

    #[test]
    fn match_job_with_resident_check_helper() {
        use crate::types::RoleTransitionRecord;
        use chrono::NaiveDate;
        use std::collections::HashMap;

        #[derive(Clone)]
        struct ResidencyRecord {
            inner: RoleTransitionRecord<BaseRecord>,
            resident_until: Option<NaiveDate>,
        }

        impl crate::matching::MatchingRecord for ResidencyRecord {
            crate::delegate_matching_record!(inner);
        }

        impl crate::matching::RoleIndexedRecord for ResidencyRecord {
            fn event_date(&self) -> Option<NaiveDate> {
                self.inner.transition_date
            }
        }

        impl crate::matching::ResidentAtIndexRecord for ResidencyRecord {
            fn is_resident_at(&self, index_date: NaiveDate) -> bool {
                self.resident_until.is_none_or(|until| until > index_date)
            }
        }

        let a1 = ResidencyRecord {
            inner: RoleTransitionRecord::from_record(
                BaseRecord::new("a1", date(2010, 1, 1)),
                Some(date(2012, 1, 1)),
            ),
            resident_until: None,
        };
        let c1_non_resident = ResidencyRecord {
            inner: RoleTransitionRecord::from_record(
                BaseRecord::new("c1_non_resident", date(2010, 1, 1)),
                None,
            ),
            resident_until: Some(date(2011, 1, 1)),
        };
        let c2_resident = ResidencyRecord {
            inner: RoleTransitionRecord::from_record(
                BaseRecord::new("c2_resident", date(2010, 1, 1)),
                None,
            ),
            resident_until: None,
        };

        let records = vec![a1, c1_non_resident, c2_resident];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_resident_check()
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_resident");
    }

    #[test]
    fn match_job_with_resident_check_by_helper() {
        use crate::types::RoleTransitionRecord;

        let a1 = RoleTransitionRecord::from_record(
            BaseRecord::new("a1", date(2010, 1, 1)),
            Some(date(2012, 1, 1)),
        );
        let c1_non_resident = RoleTransitionRecord::from_record(
            BaseRecord::new("c1_non_resident", date(2010, 1, 1)),
            None,
        );
        let c2_resident = RoleTransitionRecord::from_record(
            BaseRecord::new("c2_resident", date(2010, 1, 1)),
            None,
        );

        let records = vec![a1, c1_non_resident, c2_resident];

        let result = MatchJob::new_transition(&records, 42)
            .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
            .with_resident_check_by(|record: &RoleTransitionRecord<BaseRecord>, _index_date| {
                record.record.id == "c2_resident"
            })
            .run();

        assert_eq!(result.matched_cases, 1);
        assert_eq!(result.pairs[0].control_id, "c2_resident");
    }
}
