use super::records::MatchingRecord;
use crate::types::MatchingCriteria;
use std::collections::{HashMap, HashSet};

pub trait Constraint<R: MatchingRecord> {
    fn reason(&self) -> &'static str;
    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool;
}

pub struct ConstraintContext<'a> {
    pub criteria: &'a MatchingCriteria,
    pub used_controls: &'a HashSet<usize>,
    pub used_unique: &'a HashSet<String>,
    pub control_idx: usize,
    pub case_strata_values: Option<&'a [Option<&'a str>]>,
    pub control_strata_values: Option<&'a [Option<&'a str>]>,
}

pub struct ReplacementConstraint;

impl<R: MatchingRecord> Constraint<R> for ReplacementConstraint {
    fn reason(&self) -> &'static str {
        "used_control_replacement_disabled"
    }

    fn allows(&self, _case: &R, _control: &R, ctx: &ConstraintContext<'_>) -> bool {
        ctx.criteria.allow_replacement || !ctx.used_controls.contains(&ctx.control_idx)
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
        value.is_none_or(|value| !ctx.used_unique.contains(value))
    }
}

pub struct DateCaliperConstraint;

impl<R: MatchingRecord> Constraint<R> for DateCaliperConstraint {
    fn reason(&self) -> &'static str {
        "date_caliper"
    }

    fn allows(&self, case: &R, control: &R, ctx: &ConstraintContext<'_>) -> bool {
        let diff_days = (control.birth_date() - case.birth_date()).num_days().abs();
        diff_days <= i64::from(ctx.criteria.birth_date_window_days)
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
    use crate::types::{BaseRecord, MatchingCriteria};
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn record(id: &str, birth_date: NaiveDate) -> BaseRecord {
        BaseRecord::new(id, birth_date)
    }

    fn context<'a>(
        criteria: &'a MatchingCriteria,
        used_controls: &'a HashSet<usize>,
        used_unique: &'a HashSet<String>,
        control_idx: usize,
        case_strata_values: Option<&'a [Option<&'a str>]>,
        control_strata_values: Option<&'a [Option<&'a str>]>,
    ) -> ConstraintContext<'a> {
        ConstraintContext {
            criteria,
            used_controls,
            used_unique,
            control_idx,
            case_strata_values,
            control_strata_values,
        }
    }

    #[test]
    fn replacement_constraint_blocks_reuse_without_replacement() {
        let criteria = MatchingCriteria::default();
        let case = record("a", date(2010, 1, 1));
        let control = record("b", date(2010, 1, 2));
        let used_controls = HashSet::from([2]);
        let empty_unique = HashSet::new();
        let ctx = context(&criteria, &used_controls, &empty_unique, 2, None, None);
        assert!(!ReplacementConstraint.allows(&case, &control, &ctx));

        let criteria = MatchingCriteria {
            allow_replacement: true,
            ..MatchingCriteria::default()
        };
        let ctx = context(&criteria, &used_controls, &empty_unique, 2, None, None);
        assert!(ReplacementConstraint.allows(&case, &control, &ctx));
    }

    #[test]
    fn no_self_match_constraint_blocks_self_pairs() {
        let criteria = MatchingCriteria::default();
        let row = record("same", date(2010, 1, 1));
        let empty_controls = HashSet::new();
        let empty_unique = HashSet::new();
        let ctx = context(&criteria, &empty_controls, &empty_unique, 0, None, None);
        assert!(!NoSelfMatchConstraint.allows(&row, &row, &ctx));
    }

    #[test]
    fn exact_match_constraint_uses_strata_paths() {
        let criteria = MatchingCriteria {
            required_strata: vec!["municipality".to_string()],
            ..MatchingCriteria::default()
        };
        let mut case = record("case", date(2010, 1, 1));
        case.strata
            .insert("municipality".to_string(), "0101".to_string());
        let mut control = record("control", date(2010, 1, 2));
        control
            .strata
            .insert("municipality".to_string(), "0202".to_string());
        let empty_controls = HashSet::new();
        let empty_unique = HashSet::new();
        let ctx = context(&criteria, &empty_controls, &empty_unique, 0, None, None);
        assert!(!ExactMatchConstraint.allows(&case, &control, &ctx));

        let case_vals = build_strata_values(&case.strata, &criteria.required_strata);
        let control_vals = build_strata_values(&case.strata, &criteria.required_strata);
        let ctx = context(
            &criteria,
            &empty_controls,
            &empty_unique,
            0,
            Some(&case_vals),
            Some(&control_vals),
        );
        assert!(ExactMatchConstraint.allows(&case, &control, &ctx));
    }

    #[test]
    fn unique_key_constraint_blocks_reused_unique_values() {
        let criteria = MatchingCriteria {
            unique_by_key: Some("family".to_string()),
            ..MatchingCriteria::default()
        };
        let mut control = record("control", date(2010, 1, 1));
        control
            .strata
            .insert("family".to_string(), "F1".to_string());

        let used_unique = HashSet::from(["F1".to_string()]);
        let empty_controls = HashSet::new();
        let ctx = context(&criteria, &empty_controls, &used_unique, 0, None, None);
        assert!(!UniqueKeyConstraint.allows(&control, &control, &ctx));

        let criteria = MatchingCriteria::default();
        let ctx = context(&criteria, &empty_controls, &used_unique, 0, None, None);
        assert!(UniqueKeyConstraint.allows(&control, &control, &ctx));
    }

    #[test]
    fn date_caliper_constraint_uses_birth_window() {
        let criteria = MatchingCriteria {
            birth_date_window_days: 5,
            ..MatchingCriteria::default()
        };
        let case = record("case", date(2010, 1, 1));
        let control_close = record("control_close", date(2010, 1, 5));
        let control_far = record("control_far", date(2010, 1, 10));
        let empty_controls = HashSet::new();
        let empty_unique = HashSet::new();
        let ctx = context(&criteria, &empty_controls, &empty_unique, 0, None, None);
        assert!(DateCaliperConstraint.allows(&case, &control_close, &ctx));
        assert!(!DateCaliperConstraint.allows(&case, &control_far, &ctx));
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
