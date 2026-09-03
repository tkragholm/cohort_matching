use crate::matching::{Constraint, ConstraintContext, MatchingRecord, RoleIndexedRecord};
use chrono::NaiveDate;
use std::marker::PhantomData;

/// Constraint that ensures case and control have the same value for a categorical field.
///
/// By default, it uses the "gender" key in the strata map.
pub struct GenderMatch {
    pub key: String,
    pub allow_unknown: bool,
}

impl GenderMatch {
    /// Create a constraint that requires the same gender as the case.
    /// Uses "gender" as the default strata key.
    #[must_use]
    pub fn same_as_case() -> Self {
        Self {
            key: "gender".to_string(),
            allow_unknown: false,
        }
    }

    /// Create a constraint that requires the same gender as the case or matches if unknown.
    #[must_use]
    pub fn same_as_case_or_unknown() -> Self {
        Self {
            key: "gender".to_string(),
            allow_unknown: true,
        }
    }

    /// Create a constraint for a custom strata key.
    #[must_use]
    pub fn on_key(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            allow_unknown: false,
        }
    }
}

impl<R: MatchingRecord> Constraint<R> for GenderMatch {
    fn reason(&self) -> &'static str {
        "gender_mismatch"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        let case_val = case.strata().get(&self.key);
        let control_val = control.strata().get(&self.key);
        match (case_val, control_val) {
            (Some(c), Some(ctrl)) => c == ctrl,
            _ => self.allow_unknown,
        }
    }
}

/// Constraint that ensures a numeric field is within a certain caliper window.
///
/// This is a generic version of [`DateWindow`] that works on any numeric value.
/// What a caliper does when a record does not carry the value.
///
/// There is no universally right answer, which is why it is a choice rather
/// than a default buried in the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingPolicy {
    /// Refuse the pair. An unverifiable constraint has not been satisfied.
    ///
    /// The safe reading, and the default: a caliper is a claim about how close
    /// two records are, and a value nobody recorded supports no such claim.
    #[default]
    Refuse,
    /// Two records both missing the value satisfy the caliper; one missing and
    /// one present does not.
    ///
    /// For a matched-cohort design this is often the honest reading rather than
    /// the lax one, because MISSINGNESS IS ITSELF A STRATUM. A study that
    /// exact-matches on a categorical covariate routinely gives "unknown" its
    /// own level, so that a record the register does not cover matches another
    /// the register does not cover -- stricter than dropping the constraint,
    /// and more honest than imputing a value. This is that convention for a
    /// numeric field. Two children with no recorded father are alike in that
    /// respect; pairing one of them with a child whose father IS recorded is
    /// the comparison the caliper was meant to prevent, and it still is.
    ///
    /// It does not make missingness free: a record missing the value can only
    /// match another missing it, so the constraint still binds, and it binds on
    /// a smaller pool.
    MatchMissing,
}

pub struct Caliper<R, F> {
    selector: F,
    window: f64,
    reason: &'static str,
    missing: MissingPolicy,
    _marker: std::marker::PhantomData<R>,
}

impl<R, F> Caliper<R, F>
where
    F: Fn(&R) -> Option<f64> + Send + Sync,
{
    /// Create a caliper constraint on a specific numeric field.
    pub const fn on(selector: F, window: f64) -> Self {
        Self {
            selector,
            window,
            reason: "caliper_mismatch",
            missing: MissingPolicy::Refuse,
            _marker: std::marker::PhantomData,
        }
    }

    /// Choose what happens when a record does not carry the value.
    #[must_use]
    pub const fn on_missing(mut self, policy: MissingPolicy) -> Self {
        self.missing = policy;
        self
    }

    crate::impl_with_reason!();
}

/// A caliper on a named numeric field, for callers configuring N of them.
///
/// [`Caliper::on`] takes a selector, which means a caller wiring calipers from
/// configuration has to write one closure per field at compile time. This is the
/// same constraint addressed by NAME, so a list of `(field, window)` pairs read
/// from a config file becomes a list of constraints.
///
/// The field is read through [`MatchingRecord::numeric`], and a record that does
/// not carry it refuses the pair — an unverifiable constraint has not been
/// satisfied. [`Caliper::on_missing`] changes that when a study's convention is
/// that unknown matches unknown.
///
/// The returned caliper owns its field name, so `use<R>` states that the opaque
/// type captures only `R`. Without it, edition 2024 captures every lifetime in
/// scope -- including the one behind `impl Into<String>` -- and a caller passing
/// a `&str` gets back a caliper borrowing it, which cannot be returned from the
/// helper that built it. That helper is the whole point: matching rules that
/// come from configuration are built somewhere and used somewhere else.
#[must_use]
pub fn caliper_on_field<R: MatchingRecord>(
    field: &str,
    window: f64,
) -> Caliper<R, impl Fn(&R) -> Option<f64> + Send + Sync + use<R>> {
    let field = field.to_owned();
    Caliper::on(move |record: &R| record.numeric(&field), window)
}

impl<R: MatchingRecord, F> Constraint<R> for Caliper<R, F>
where
    F: Fn(&R) -> Option<f64> + Send + Sync,
{
    fn reason(&self) -> &'static str {
        self.reason
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        let case_val = (self.selector)(case);
        let control_val = (self.selector)(control);
        match (case_val, control_val) {
            (Some(c), Some(ctrl)) => (c - ctrl).abs() <= self.window,
            // Both missing: alike in the one respect the caliper can see, if the
            // caller has said so. One missing and one present is refused either
            // way -- that is the comparison a caliper exists to prevent.
            (None, None) => self.missing == MissingPolicy::MatchMissing,
            _ => false,
        }
    }
}

/// Constraint that ensures a date field is within a certain window.
pub struct DateWindow<R, F> {
    selector: F,
    window_days: i64,
    reason: &'static str,
    _marker: PhantomData<R>,
}

impl<R, F> DateWindow<R, F>
where
    F: Fn(&R) -> Option<NaiveDate> + Send + Sync,
{
    /// Create a date window constraint on a specific field.
    pub const fn on(selector: F, window_days: i64) -> Self {
        Self {
            selector,
            window_days,
            reason: "date_window_mismatch",
            _marker: PhantomData,
        }
    }

    crate::impl_with_reason!();
}

impl<R: MatchingRecord, F> Constraint<R> for DateWindow<R, F>
where
    F: Fn(&R) -> Option<NaiveDate> + Send + Sync,
{
    fn reason(&self) -> &'static str {
        self.reason
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        let case_date = (self.selector)(case);
        let control_date = (self.selector)(control);
        match (case_date, control_date) {
            (Some(c), Some(ctrl)) => (c - ctrl).num_days().abs() <= self.window_days,
            _ => false,
        }
    }
}

/// Constraint that ensures the control is alive at the case's index date.
pub enum MustBeAlive<F = fn(&dyn MatchingRecord) -> Option<NaiveDate>> {
    /// Use a custom closure to select the death date.
    Selector(F),
    /// Use the record's own [`MatchingRecord::death_date`] method.
    RecordMethod,
}

impl<F> MustBeAlive<F> {
    /// Create a constraint that checks if the control is alive using a custom selector.
    pub const fn at_index_date(death_date_selector: F) -> Self {
        Self::Selector(death_date_selector)
    }
}

impl<R: MatchingRecord> MustBeAlive<fn(&R) -> Option<NaiveDate>> {
    /// Create a constraint that uses the record's own [`MatchingRecord::death_date`] method.
    #[must_use]
    pub const fn new() -> Self {
        Self::RecordMethod
    }
}

impl<R: MatchingRecord> Default for MustBeAlive<fn(&R) -> Option<NaiveDate>> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RoleIndexedRecord, F> Constraint<R> for MustBeAlive<F>
where
    F: Fn(&R) -> Option<NaiveDate> + Send + Sync,
{
    fn reason(&self) -> &'static str {
        "control_not_alive_at_index_date"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        let Some(index_date) = case.event_date() else {
            return true;
        };
        let death_date = match self {
            Self::Selector(selector) => selector(control),
            Self::RecordMethod => control.death_date(),
        };
        death_date.is_none_or(|d| d > index_date)
    }
}

/// Constraint that ensures the control is resident at the case's index date.
pub struct MustBeResident<F> {
    check: F,
}

impl<F> MustBeResident<F> {
    /// Create a constraint that checks if the control is resident at the case's index date.
    ///
    /// Requires that the record implements [`RoleIndexedRecord`] to provide the index date.
    pub const fn at_index_date(check: F) -> Self {
        Self { check }
    }
}

impl<R: RoleIndexedRecord, F> Constraint<R> for MustBeResident<F>
where
    F: Fn(&R, NaiveDate) -> bool + Send + Sync,
{
    fn reason(&self) -> &'static str {
        "control_non_resident_at_index_date"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        let Some(index_date) = case.event_date() else {
            return true;
        };
        (self.check)(control, index_date)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{ConstraintContext, UsedControlsVec};
    use crate::types::{BaseRecord, ControlIdx};
    use rapidhash::RapidHashMap;
    use rustc_hash::FxHashSet;

    use crate::{date, record};

    #[test]
    fn gender_match_constraint_requires_exact_strata_match() {
        let constraint = GenderMatch::same_as_case();
        let mut case = record("case", date(2010, 1, 1));
        case.strata.insert("gender".to_string(), "M".to_string());

        let mut control_m = record("control_m", date(2010, 1, 2));
        control_m
            .strata
            .insert("gender".to_string(), "M".to_string());

        let mut control_f = record("control_f", date(2010, 1, 3));
        control_f
            .strata
            .insert("gender".to_string(), "F".to_string());

        let criteria = crate::types::MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let unique_interner = RapidHashMap::default();

        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &unique_interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };

        assert!(constraint.allows(&case, &control_m, &ctx));
        assert!(!constraint.allows(&case, &control_f, &ctx));
    }

    #[test]
    fn gender_match_can_allow_unknown() {
        let constraint = GenderMatch::same_as_case_or_unknown();
        let mut case = record("case", date(2010, 1, 1));
        case.strata.insert("gender".to_string(), "M".to_string());

        let control_no_gender = record("control_none", date(2010, 1, 2));

        let criteria = crate::types::MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let unique_interner = RapidHashMap::default();

        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &unique_interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };

        assert!(constraint.allows(&case, &control_no_gender, &ctx));
    }

    #[test]
    fn date_window_constraint_checks_field_distance() {
        let constraint = DateWindow::on(|r: &BaseRecord| Some(r.birth_date), 5);
        let case = record("case", date(2010, 1, 1));
        let control_near = record("near", date(2010, 1, 5));
        let control_far = record("far", date(2010, 1, 10));

        let criteria = crate::types::MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let unique_interner = RapidHashMap::default();

        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &unique_interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };

        assert!(constraint.allows(&case, &control_near, &ctx));
        assert!(!constraint.allows(&case, &control_far, &ctx));
    }

    #[test]
    fn must_be_alive_constraint_checks_death_date_relative_to_index() {
        use crate::types::RoleTransitionRecord;
        use std::collections::HashMap;

        let death_dates: HashMap<String, NaiveDate> = [
            ("ctrl_alive".to_string(), date(2025, 1, 1)),
            ("ctrl_dead".to_string(), date(2015, 1, 1)),
        ]
        .into_iter()
        .collect();

        let constraint = MustBeAlive::at_index_date(move |r: &RoleTransitionRecord<BaseRecord>| {
            death_dates.get(r.id()).copied()
        });

        let case = RoleTransitionRecord::from_record(
            record("case", date(2010, 1, 1)),
            Some(date(2020, 1, 1)), // index date
        );

        let ctrl_alive =
            RoleTransitionRecord::from_record(record("ctrl_alive", date(2010, 1, 1)), None);

        let ctrl_dead =
            RoleTransitionRecord::from_record(record("ctrl_dead", date(2010, 1, 1)), None);

        let criteria = crate::types::MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let unique_interner = RapidHashMap::default();

        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &unique_interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };

        assert!(constraint.allows(&case, &ctrl_alive, &ctx));
        assert!(!constraint.allows(&case, &ctrl_dead, &ctx));
    }

    #[test]
    fn must_be_alive_constraint_uses_record_death_date_by_default() {
        use crate::types::RoleTransitionRecord;

        let constraint =
            MustBeAlive::at_index_date(|r: &RoleTransitionRecord<BaseRecord>| r.death_date);

        let case = RoleTransitionRecord::from_record(
            record("case", date(2010, 1, 1)),
            Some(date(2020, 1, 1)),
        );

        let mut ctrl_dead = BaseRecord::new("dead", date(2010, 1, 1));
        ctrl_dead.death_date = Some(date(2015, 1, 1));
        let ctrl_dead = RoleTransitionRecord::from_record(ctrl_dead, None);

        let mut ctrl_alive = BaseRecord::new("alive", date(2010, 1, 1));
        ctrl_alive.death_date = Some(date(2025, 1, 1));
        let ctrl_alive = RoleTransitionRecord::from_record(ctrl_alive, None);

        let criteria = crate::types::MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let unique_interner = RapidHashMap::default();

        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &unique_interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };

        assert!(!constraint.allows(&case, &ctrl_dead, &ctx));
        assert!(constraint.allows(&case, &ctrl_alive, &ctx));
    }

    #[test]
    fn must_be_resident_constraint_checks_closure_at_index() {
        use crate::types::RoleTransitionRecord;

        let constraint =
            MustBeResident::at_index_date(|r: &RoleTransitionRecord<BaseRecord>, index_date| {
                // Mock logic: resident if id starts with 'res' or index_date is early
                r.id().starts_with("res") || index_date < date(2015, 1, 1)
            });

        let case_late = RoleTransitionRecord::from_record(
            record("case", date(2010, 1, 1)),
            Some(date(2020, 1, 1)),
        );

        let res_ctrl =
            RoleTransitionRecord::from_record(record("res_ctrl", date(2010, 1, 1)), None);

        let non_res_ctrl =
            RoleTransitionRecord::from_record(record("non_res_ctrl", date(2010, 1, 1)), None);

        let criteria = crate::types::MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique = FxHashSet::default();
        let unique_interner = RapidHashMap::default();

        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &unique_interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };

        assert!(constraint.allows(&case_late, &res_ctrl, &ctx));
        assert!(!constraint.allows(&case_late, &non_res_ctrl, &ctx));
    }
}

#[cfg(test)]
mod named_caliper_tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn day(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid date")
    }

    fn record(id: &str, value: Option<f64>) -> BaseRecord {
        let base = BaseRecord::new(id, day(2000, 1, 1));
        match value {
            Some(v) => base.with_numeric("parent_birth_year", v),
            None => base,
        }
    }

    /// A context the caliper does not read.
    ///
    /// `Caliper::allows` takes `_ctx` and ignores it -- the constraint is a
    /// comparison of two records and nothing else -- but the trait's signature
    /// requires one, so it is built rather than mocked.
    fn allows_field<R: MatchingRecord>(field: &str, case: &R, control: &R, window: f64) -> bool {
        use crate::matching::UsedControlsVec;
        use crate::types::{ControlIdx, MatchingCriteria, UniqueValueId};
        use rapidhash::RapidHashMap;
        use rustc_hash::FxHashSet;

        let criteria = MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique: FxHashSet<UniqueValueId> = FxHashSet::default();
        let interner: RapidHashMap<String, UniqueValueId> = RapidHashMap::default();
        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };
        let constraint = caliper_on_field::<R>(field, window);
        constraint.allows(case, control, &ctx)
    }

    fn allows(case: &BaseRecord, control: &BaseRecord, window: f64) -> bool {
        allows_field("parent_birth_year", case, control, window)
    }

    #[test]
    fn a_pair_inside_the_window_is_allowed() {
        // The rule this exists for: "parental birth years (+/- 1 year)".
        assert!(allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1981.0)),
            1.0
        ));
        assert!(allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1979.0)),
            1.0
        ));
        assert!(allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1980.0)),
            1.0
        ));
    }

    #[test]
    fn a_pair_outside_the_window_is_refused() {
        assert!(!allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1982.0)),
            1.0
        ));
        // The gap the first production run measured at the 95th percentile.
        assert!(!allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1996.0)),
            1.0
        ));
    }

    #[test]
    fn the_window_is_inclusive_at_its_edge() {
        // A caliper of "+/- 1 year" admits exactly one year, which is the
        // difference between the rule as written and the rule as coded.
        assert!(allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1981.0)),
            1.0
        ));
        assert!(!allows(
            &record("a", Some(1980.0)),
            &record("b", Some(1981.5)),
            1.0
        ));
    }

    #[test]
    fn a_record_without_the_field_refuses() {
        // An unverifiable constraint has not been satisfied. A child with no
        // recorded parent is 1.1% of mothers and 3.7% of fathers on the real
        // cohort, and admitting them would be silently dropping the constraint
        // for exactly the records that cannot support it.
        assert!(!allows(&record("a", Some(1980.0)), &record("b", None), 1.0));
        assert!(!allows(&record("a", None), &record("b", Some(1980.0)), 1.0));
        assert!(!allows(&record("a", None), &record("b", None), 1.0));
    }

    #[test]
    fn calipers_on_different_fields_are_independent() {
        // The point of naming them: N constraints from a list, not one baked
        // into the record type.
        let case = BaseRecord::new("a", day(2000, 1, 1))
            .with_numeric("mother_birth_year", 1980.0)
            .with_numeric("father_birth_year", 1975.0);
        let control = BaseRecord::new("b", day(2000, 1, 1))
            .with_numeric("mother_birth_year", 1981.0)
            .with_numeric("father_birth_year", 1990.0);
        assert!(allows_field("mother_birth_year", &case, &control, 1.0));
        assert!(!allows_field("father_birth_year", &case, &control, 1.0));
    }

    #[test]
    fn a_record_carries_no_numerics_unless_given_one() {
        // The field is `#[serde(default)]`, so a cohort serialised before it
        // existed deserialises with an empty map -- and every caliper over it
        // then refuses rather than passes, which is the safe direction.
        let record = BaseRecord::new("a", day(2000, 1, 1));
        assert!(record.numerics.is_empty());
        assert_eq!(record.numeric("parent_birth_year"), None);
    }

    #[test]
    fn caliper_reads_through_a_role_transition_record() {
        // `numeric()` is defaulted to `None`, so a wrapper that forgets to
        // forward it COMPILES and then refuses every pair on a field the inner
        // record carries. Risk-set matching is the only mode this crate offers
        // for incidence-density designs, and `RoleTransitionRecord` is the
        // record type it takes, so a missing forward here would mean named
        // calipers never match anything at all.
        use crate::types::RoleTransitionRecord;

        let wrap = |id: &str, year: f64| {
            RoleTransitionRecord::from_record(
                BaseRecord::new(id, day(2000, 1, 1)).with_numeric("parent_birth_year", year),
                Some(day(2010, 1, 1)),
            )
        };

        assert!(allows_field(
            "parent_birth_year",
            &wrap("a", 1980.0),
            &wrap("b", 1981.0),
            1.0
        ));
        assert!(!allows_field(
            "parent_birth_year",
            &wrap("a", 1980.0),
            &wrap("b", 1982.0),
            1.0
        ));
    }

    #[test]
    fn caliper_reads_through_a_balance_record() {
        use crate::types::BalanceRecord;

        let wrap = |id: &str, year: f64| {
            let mut record = BalanceRecord::new(id, day(2000, 1, 1));
            record
                .core
                .numerics
                .insert("parent_birth_year".to_string(), year);
            record
        };

        assert!(allows_field(
            "parent_birth_year",
            &wrap("a", 1980.0),
            &wrap("b", 1981.0),
            1.0
        ));
        assert!(!allows_field(
            "parent_birth_year",
            &wrap("a", 1980.0),
            &wrap("b", 1982.0),
            1.0
        ));
    }

    #[test]
    fn a_missing_value_refuses_by_default() {
        // The safe reading: a caliper is a claim about how close two records
        // are, and a value nobody recorded supports no such claim.
        assert!(!allows(&record("a", None), &record("b", Some(1980.0)), 1.0));
        assert!(!allows(&record("a", Some(1980.0)), &record("b", None), 1.0));
        assert!(!allows(&record("a", None), &record("b", None), 1.0));
    }

    /// As `allows_field`, with a missing-value policy.
    fn allows_with_policy<R: MatchingRecord>(
        field: &str,
        case: &R,
        control: &R,
        window: f64,
        policy: MissingPolicy,
    ) -> bool {
        use crate::matching::UsedControlsVec;
        use crate::types::{ControlIdx, MatchingCriteria, UniqueValueId};
        use rapidhash::RapidHashMap;
        use rustc_hash::FxHashSet;

        let criteria = MatchingCriteria::default();
        let used_controls = UsedControlsVec::with_capacity(0);
        let used_unique: FxHashSet<UniqueValueId> = FxHashSet::default();
        let interner: RapidHashMap<String, UniqueValueId> = RapidHashMap::default();
        let ctx = ConstraintContext {
            criteria: &criteria,
            used_controls: &used_controls,
            used_unique: &used_unique,
            unique_interner: &interner,
            control_idx: ControlIdx::new(0),
            case_strata_values: None,
            control_strata_values: None,
        };
        caliper_on_field::<R>(field, window)
            .on_missing(policy)
            .allows(case, control, &ctx)
    }

    #[test]
    fn two_missing_values_match_when_the_caller_says_so() {
        // Missingness as its own stratum, which is what a study already does
        // when it exact-matches on a categorical covariate with an "unknown"
        // level. Two children with no recorded father are alike in that respect.
        assert!(allows_with_policy(
            "parent_birth_year",
            &record("a", None),
            &record("b", None),
            1.0,
            MissingPolicy::MatchMissing,
        ));
    }

    #[test]
    fn one_missing_and_one_present_is_refused_under_either_policy() {
        // The comparison a caliper exists to prevent, and neither policy allows
        // it: "unknown matches unknown" is not "unknown matches anything".
        for policy in [MissingPolicy::Refuse, MissingPolicy::MatchMissing] {
            assert!(!allows_with_policy(
                "parent_birth_year",
                &record("a", None),
                &record("b", Some(1980.0)),
                1.0,
                policy,
            ));
            assert!(!allows_with_policy(
                "parent_birth_year",
                &record("a", Some(1980.0)),
                &record("b", None),
                1.0,
                policy,
            ));
        }
    }

    #[test]
    fn the_window_still_binds_when_both_values_are_present() {
        // The policy governs missing values and nothing else: a permissive
        // policy must not loosen the caliper for records that have the value.
        assert!(!allows_with_policy(
            "parent_birth_year",
            &record("a", Some(1980.0)),
            &record("b", Some(1996.0)),
            1.0,
            MissingPolicy::MatchMissing,
        ));
        assert!(allows_with_policy(
            "parent_birth_year",
            &record("a", Some(1980.0)),
            &record("b", Some(1981.0)),
            1.0,
            MissingPolicy::MatchMissing,
        ));
    }
}
