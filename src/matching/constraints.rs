use super::distance::DistanceMetric;
use super::records::MatchingRecord;
use crate::types::{ControlIdx, MatchingCriteria, UniqueValueId};
use rapidhash::RapidHashMap;
use rustc_hash::FxHashSet;
use std::collections::HashMap;
use std::marker::PhantomData;

/// Compact used-control tracking backed by a flat `Vec<bool>`.
///
/// Indexed by raw control index; `contains(idx)` is a single array lookup
/// instead of a hash probe, which gives better cache behaviour when the
/// control pool is large (2 M+ entries).
pub struct UsedControlsVec {
    bits: Vec<bool>,
    count: usize,
}

impl UsedControlsVec {
    #[must_use]
    pub fn with_capacity(n: usize) -> Self {
        Self {
            bits: vec![false; n],
            count: 0,
        }
    }

    #[inline]
    #[must_use]
    pub fn contains(&self, idx: ControlIdx) -> bool {
        self.bits.get(idx.get()).copied().unwrap_or(false)
    }

    /// Mark a control as used. Returns `true` if it was not already marked.
    #[inline]
    pub fn insert(&mut self, idx: ControlIdx) -> bool {
        let Some(slot) = self.bits.get_mut(idx.get()) else {
            return false;
        };
        if *slot {
            return false;
        }
        *slot = true;
        self.count += 1;
        true
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.count
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.count == 0
    }
}

pub trait Constraint<R: MatchingRecord>: Send + Sync {
    fn reason(&self) -> &'static str;
    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool;
}

/// Statically-dispatched collection of constraints.
///
/// This enables callers to compose constraints as tuples (heterogeneous)
/// or slices/arrays (homogeneous) without requiring `dyn Constraint`.
pub trait ConstraintGroup<R: MatchingRecord>: Send + Sync {
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str>;
}

impl<R: MatchingRecord> ConstraintGroup<R> for () {
    fn first_blocking_reason(
        &self,
        _case: &R,
        _control: &R,
        _ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        None
    }
}

impl<R: MatchingRecord, G: ConstraintGroup<R> + ?Sized> ConstraintGroup<R> for &G {
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        (*self).first_blocking_reason(case, control, ctx)
    }
}

impl<R: MatchingRecord> ConstraintGroup<R> for [&dyn Constraint<R>] {
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        self.iter().find_map(|constraint| {
            (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
        })
    }
}

impl<R: MatchingRecord, C, const N: usize> ConstraintGroup<R> for [C; N]
where
    C: Constraint<R>,
{
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        self.iter().find_map(|constraint| {
            (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
        })
    }
}

impl<R: MatchingRecord, G, C> ConstraintGroup<R> for (G, C)
where
    G: ConstraintGroup<R>,
    C: Constraint<R>,
{
    fn first_blocking_reason(
        &self,
        case: &R,
        control: &R,
        ctx: &ConstraintContext<'_>,
    ) -> Option<&'static str> {
        let (group, constraint) = self;
        if let Some(reason) = group.first_blocking_reason(case, control, ctx) {
            return Some(reason);
        }
        (!constraint.allows(case, control, ctx)).then_some(constraint.reason())
    }
}

macro_rules! impl_constraint_group_tuple {
    ($( $ty:ident : $value:ident ),+ $(,)?) => {
        impl<R: MatchingRecord, $($ty),+> ConstraintGroup<R> for ($($ty,)+)
        where
            $($ty: Constraint<R>,)+
        {
            fn first_blocking_reason(
                &self,
                case: &R,
                control: &R,
                ctx: &ConstraintContext<'_>,
            ) -> Option<&'static str> {
                let ($($value,)+) = self;
                $(
                    if !$value.allows(case, control, ctx) {
                        return Some($value.reason());
                    }
                )+
                None
            }
        }
    };
}

impl_constraint_group_tuple!(C1: c1);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, C6: c6);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, C6: c6, C7: c7);
impl_constraint_group_tuple!(C1: c1, C2: c2, C3: c3, C4: c4, C5: c5, C6: c6, C7: c7, C8: c8);

pub struct ConstraintContext<'a> {
    pub criteria: &'a MatchingCriteria,
    pub used_controls: &'a UsedControlsVec,
    pub used_unique: &'a FxHashSet<UniqueValueId>,
    pub unique_interner: &'a RapidHashMap<String, UniqueValueId>,
    pub control_idx: ControlIdx,
    pub case_strata_values: Option<&'a [Option<&'a str>]>,
    pub control_strata_values: Option<&'a [Option<&'a str>]>,
}

pub struct ReplacementConstraint;

impl<R: MatchingRecord> Constraint<R> for ReplacementConstraint {
    fn reason(&self) -> &'static str {
        "used_control_replacement_disabled"
    }

    fn allows(&self, _case: &R, _control: &R, ctx: &ConstraintContext<'_>) -> bool {
        ctx.criteria.allow_replacement || !ctx.used_controls.contains(ctx.control_idx)
    }
}

pub struct NoSelfMatchConstraint;

impl<R: MatchingRecord> Constraint<R> for NoSelfMatchConstraint {
    fn reason(&self) -> &'static str {
        "self_match"
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        case.id() != control.id()
    }
}

pub struct ExactMatchConstraint;

impl<R: MatchingRecord> Constraint<R> for ExactMatchConstraint {
    fn reason(&self) -> &'static str {
        "strata_mismatch"
    }

    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool {
        strata_match_fast(
            case,
            control,
            &ctx.criteria.required_strata,
            ctx.case_strata_values,
            ctx.control_strata_values,
        )
    }
}

pub type StrataExactConstraint = ExactMatchConstraint;

pub struct UniqueKeyConstraint;

impl<R: MatchingRecord> Constraint<R> for UniqueKeyConstraint {
    fn reason(&self) -> &'static str {
        "unique_key_reused"
    }

    fn allows(&self, _case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool {
        if ctx.criteria.unique_by_key.is_none() {
            return true;
        }

        let value = unique_value(control, ctx.criteria);
        value.is_none_or(|value| {
            !ctx.unique_interner
                .get(value)
                .is_some_and(|id| ctx.used_unique.contains(id))
        })
    }
}

/// Generic caliper constraint over a distance channel.
pub struct CaliperConstraint<'a, R: MatchingRecord, D: DistanceMetric<R>> {
    pub metric: &'a D,
    pub max_distance: f64,
    pub reason: &'static str,
    marker: PhantomData<R>,
}

impl<'a, R: MatchingRecord, D: DistanceMetric<R>> CaliperConstraint<'a, R, D> {
    #[must_use]
    pub const fn new(metric: &'a D, max_distance: f64, reason: &'static str) -> Self {
        Self {
            metric,
            max_distance,
            reason,
            marker: PhantomData,
        }
    }
}

impl<R: MatchingRecord, D: DistanceMetric<R> + Sync> Constraint<R> for CaliperConstraint<'_, R, D> {
    fn reason(&self) -> &'static str {
        self.reason
    }

    fn allows(&self, case: &R, control: &R, _ctx: &ConstraintContext<'_>) -> bool {
        if !self.max_distance.is_finite() || self.max_distance < 0.0 {
            return false;
        }
        self.metric
            .distance(case, control)
            .is_some_and(|distance| distance.is_finite() && distance <= self.max_distance)
    }
}

pub fn unique_value<'a, R: MatchingRecord>(
    record: &'a R,
    criteria: &'a MatchingCriteria,
) -> Option<&'a str> {
    criteria
        .unique_by_key
        .as_ref()
        .and_then(|key| record.strata().get(key).map(String::as_str))
        .or_else(|| record.unique_key())
}

pub fn build_strata_values<'a>(
    strata: &'a HashMap<String, String>,
    required: &[String],
) -> Vec<Option<&'a str>> {
    required
        .iter()
        .map(|key| strata.get(key).map(String::as_str))
        .collect()
}

pub(super) fn strata_match_fast<R: MatchingRecord>(
    case: &R,
    control: &R,
    required: &[String],
    case_values: Option<&[Option<&str>]>,
    control_values: Option<&[Option<&str>]>,
) -> bool {
    if required.is_empty() {
        return true;
    }
    let Some(case_values) = case_values else {
        return strata_match(case, control, required);
    };
    let Some(control_values) = control_values else {
        return strata_match(case, control, required);
    };
    if case_values.len() != control_values.len() {
        return false;
    }

    case_values
        .iter()
        .zip(control_values.iter())
        .all(|(case_val, control_val)| case_val == control_val)
}

fn strata_match<R: MatchingRecord>(case: &R, control: &R, required: &[String]) -> bool {
    required
        .iter()
        .all(|key| case.strata().get(key) == control.strata().get(key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::DateDistance;
    use crate::types::MatchingCriteria;
    use crate::{date, record};

    fn context<'a>(
        criteria: &'a MatchingCriteria,
        used_controls: &'a UsedControlsVec,
        used_unique: &'a FxHashSet<UniqueValueId>,
        unique_interner: &'a RapidHashMap<String, UniqueValueId>,
        control_idx: ControlIdx,
        case_strata_values: Option<&'a [Option<&'a str>]>,
        control_strata_values: Option<&'a [Option<&'a str>]>,
    ) -> ConstraintContext<'a> {
        ConstraintContext {
            criteria,
            used_controls,
            used_unique,
            unique_interner,
            control_idx,
            case_strata_values,
            control_strata_values,
        }
    }

    fn empty_used_controls() -> UsedControlsVec {
        UsedControlsVec::with_capacity(0)
    }

    #[test]
    fn replacement_constraint_blocks_reuse_without_replacement() {
        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let case = record("a", date(2010, 1, 1));
        let control = record("b", date(2010, 1, 2));
        let mut used_controls = UsedControlsVec::with_capacity(3);
        used_controls.insert(ControlIdx::new(2));
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &used_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(2),
            None,
            None,
        );
        assert!(!ReplacementConstraint.allows(&case, &control, &ctx));

        let criteria = MatchingCriteria::builder()
            .allow_replacement(true)
            .build()
            .validate()
            .unwrap();
        let ctx = context(
            &criteria,
            &used_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(2),
            None,
            None,
        );
        assert!(ReplacementConstraint.allows(&case, &control, &ctx));
    }

    #[test]
    fn no_self_match_constraint_blocks_self_pairs() {
        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let row = record("same", date(2010, 1, 1));
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(!NoSelfMatchConstraint.allows(&row, &row, &ctx));
    }

    #[test]
    fn exact_match_constraint_uses_strata_paths() {
        let criteria = MatchingCriteria::builder()
            .required_strata(vec!["municipality".to_string()])
            .build()
            .validate()
            .unwrap();
        let mut case = record("case", date(2010, 1, 1));
        case.strata
            .insert("municipality".to_string(), "0101".to_string());
        let mut control = record("control", date(2010, 1, 2));
        control
            .strata
            .insert("municipality".to_string(), "0202".to_string());
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(!ExactMatchConstraint.allows(&case, &control, &ctx));

        let case_vals = build_strata_values(&case.strata, &criteria.required_strata);
        let control_vals = build_strata_values(&case.strata, &criteria.required_strata);
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            Some(&case_vals),
            Some(&control_vals),
        );
        assert!(ExactMatchConstraint.allows(&case, &control, &ctx));
    }

    #[test]
    fn unique_key_constraint_blocks_reused_unique_values() {
        let criteria = MatchingCriteria::builder()
            .unique_by_key("family".to_string())
            .build()
            .validate()
            .unwrap();
        let mut control = record("control", date(2010, 1, 1));
        control
            .strata
            .insert("family".to_string(), "F1".to_string());

        let mut unique_interner = RapidHashMap::default();
        unique_interner.insert("F1".to_string(), UniqueValueId::new(1));
        let mut used_unique = FxHashSet::default();
        used_unique.insert(UniqueValueId::new(1));
        let empty_controls = empty_used_controls();
        let ctx = context(
            &criteria,
            &empty_controls,
            &used_unique,
            &unique_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(!UniqueKeyConstraint.allows(&control, &control, &ctx));

        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let ctx = context(
            &criteria,
            &empty_controls,
            &used_unique,
            &unique_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(UniqueKeyConstraint.allows(&control, &control, &ctx));
    }

    #[test]
    fn date_caliper_constraint_uses_birth_window() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(5)
            .build()
            .validate()
            .unwrap();
        let case = record("case", date(2010, 1, 1));
        let control_close = record("control_close", date(2010, 1, 5));
        let control_far = record("control_far", date(2010, 1, 10));
        let metric = DateDistance;
        let caliper = CaliperConstraint::new(
            &metric,
            f64::from(criteria.birth_date_window_days),
            "date_caliper",
        );
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(caliper.allows(&case, &control_close, &ctx));
        assert!(!caliper.allows(&case, &control_far, &ctx));
    }

    #[test]
    fn generic_caliper_constraint_works_with_custom_metric() {
        let criteria = MatchingCriteria::builder().build().validate().unwrap();
        let case = record("case", date(2010, 1, 1));
        let near = record("near", date(2010, 1, 2));
        let far = record("far", date(2010, 1, 10));
        let metric = DateDistance;
        let caliper = CaliperConstraint::new(&metric, 3.0, "date_caliper");
        let empty_controls = empty_used_controls();
        let empty_unique = FxHashSet::default();
        let empty_interner = RapidHashMap::default();
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            &empty_interner,
            ControlIdx::new(0),
            None,
            None,
        );
        assert!(caliper.allows(&case, &near, &ctx));
        assert!(!caliper.allows(&case, &far, &ctx));
    }

    #[test]
    fn unique_value_prefers_strata_key_then_fallback_key() {
        let criteria = MatchingCriteria {
            unique_by_key: Some("family".to_string()),
            ..MatchingCriteria::default()
        };
        let mut record_with_strata = record("a", date(2010, 1, 1));
        record_with_strata
            .strata
            .insert("family".to_string(), "fam_a".to_string());
        assert_eq!(unique_value(&record_with_strata, &criteria), Some("fam_a"));

        let mut fallback = record("b", date(2010, 1, 1));
        fallback.unique_key = Some("uk_b".to_string());
        assert_eq!(unique_value(&fallback, &criteria), Some("uk_b"));
    }
}
