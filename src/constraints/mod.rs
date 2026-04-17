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
pub struct Caliper<R, F> {
    selector: F,
    window: f64,
    reason: &'static str,
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
            _marker: std::marker::PhantomData,
        }
    }

    crate::impl_with_reason!();
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
