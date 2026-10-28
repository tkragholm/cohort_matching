use crate::types::{BalanceRecord, BaseRecord, CovariateValue, RoleTransitionRecord};
use chrono::NaiveDate;
use std::collections::HashMap;

pub trait MatchingRecord: Send + Sync {
    /// Unique identifier for the record.
    fn id(&self) -> &str;
    /// The individual's birth date (used for caliper matching).
    fn birth_date(&self) -> NaiveDate;
    /// Map of categorical values used for exact strata matching.
    fn strata(&self) -> &HashMap<String, String>;
    /// A named numeric field, for [`crate::constraints::Caliper`] constraints.
    ///
    /// The crate has had a generic numeric caliper since 0.1, and no way to
    /// address a field by name: `strata()` carries arbitrary CATEGORICAL values
    /// for exact matching and there was no numeric counterpart, so a caller with
    /// two numeric constraints had to bake the second one into its record type.
    /// A study matching on a child's birth date and a parent's birth year is the
    /// case that found it.
    ///
    /// `None` means "this record has no such value", and every caliper treats
    /// that as a refusal rather than a pass: an unverifiable constraint has not
    /// been satisfied. Defaulted so existing implementors are unaffected.
    ///
    /// A WRAPPER TYPE MUST FORWARD THIS, exactly as it forwards `strata()`. The
    /// default makes a missing forward compile, and the pair is then refused on
    /// a field the inner record actually carries -- which looks like a caliper
    /// that is far too tight rather than like a bug. `RoleTransitionRecord` and
    /// `BalanceRecord` forward it, and `caliper_reads_through_*` pins that.
    fn numeric(&self, _name: &str) -> Option<f64> {
        None
    }
    /// Optional key for ensuring uniqueness (e.g., family ID).
    fn unique_key(&self) -> Option<&str>;
    /// Optional death date. If provided, used by [`MustBeAlive`] constraints.
    fn death_date(&self) -> Option<NaiveDate> {
        None
    }
    /// Optional emigration date. If provided, used to derive residency at index
    /// (see [`ResidentAtIndexRecord`]): a record is non-resident from this date
    /// onward.
    fn emigration_date(&self) -> Option<NaiveDate> {
        None
    }
}

pub trait RoleIndexedRecord: MatchingRecord {
    fn event_date(&self) -> Option<NaiveDate>;
}

/// Record type that can determine residency at a given index date.
pub trait ResidentAtIndexRecord: RoleIndexedRecord {
    fn is_resident_at(&self, index_date: NaiveDate) -> bool;
}

pub trait CovariateRecord: MatchingRecord {
    fn covariates(&self) -> &HashMap<String, CovariateValue>;
}

impl MatchingRecord for BaseRecord {
    fn id(&self) -> &str {
        &self.id
    }

    fn birth_date(&self) -> NaiveDate {
        self.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.strata
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.numerics.get(name).copied()
    }

    fn unique_key(&self) -> Option<&str> {
        self.unique_key.as_deref()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.death_date
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        self.emigration_date
    }
}

impl MatchingRecord for BalanceRecord {
    fn id(&self) -> &str {
        self.core.id.as_str()
    }

    fn birth_date(&self) -> NaiveDate {
        self.core.birth_date
    }

    fn strata(&self) -> &HashMap<String, String> {
        &self.core.strata
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.core.numerics.get(name).copied()
    }

    fn unique_key(&self) -> Option<&str> {
        self.core.unique_key.as_deref()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.core.death_date
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        self.core.emigration_date
    }
}

impl CovariateRecord for BalanceRecord {
    fn covariates(&self) -> &HashMap<String, CovariateValue> {
        &self.covariates
    }
}

impl<R: MatchingRecord> MatchingRecord for RoleTransitionRecord<R> {
    fn id(&self) -> &str {
        self.record.id()
    }

    fn birth_date(&self) -> NaiveDate {
        self.record.birth_date()
    }

    fn strata(&self) -> &HashMap<String, String> {
        self.record.strata()
    }

    fn numeric(&self, name: &str) -> Option<f64> {
        self.record.numeric(name)
    }

    fn unique_key(&self) -> Option<&str> {
        self.record.unique_key()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.record.death_date()
    }

    fn emigration_date(&self) -> Option<NaiveDate> {
        self.record.emigration_date()
    }
}

impl<R: MatchingRecord> RoleIndexedRecord for RoleTransitionRecord<R> {
    fn event_date(&self) -> Option<NaiveDate> {
        self.transition_date
    }
}

impl<R: MatchingRecord> ResidentAtIndexRecord for RoleTransitionRecord<R> {
    /// Resident at `index_date` unless an emigration date on or before it is known.
    ///
    /// Mirrors [`MatchJob::with_alive_check`](crate::matching::MatchJob::with_alive_check)
    /// semantics: a control with no emigration date is treated as resident, and
    /// one that emigrated strictly after the case index date is still resident.
    fn is_resident_at(&self, index_date: NaiveDate) -> bool {
        self.emigration_date()
            .is_none_or(|emigration| emigration > index_date)
    }
}

impl<R: CovariateRecord> CovariateRecord for RoleTransitionRecord<R> {
    fn covariates(&self) -> &HashMap<String, CovariateValue> {
        self.record.covariates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::caliper_on_field;
    use crate::test_utils::date;
    use crate::{MatchJob, MatchRatio};
    use std::collections::BTreeSet;

    struct Wrapper<R> {
        inner: R,
    }

    impl<R: MatchingRecord> MatchingRecord for Wrapper<R> {
        crate::delegate_matching_record!(inner);
    }

    impl<R: MatchingRecord> RoleIndexedRecord for Wrapper<RoleTransitionRecord<R>> {
        fn event_date(&self) -> Option<NaiveDate> {
            self.inner.transition_date
        }
    }

    impl<R: MatchingRecord> ResidentAtIndexRecord for Wrapper<RoleTransitionRecord<R>> {
        fn is_resident_at(&self, index_date: NaiveDate) -> bool {
            self.emigration_date()
                .is_none_or(|emigration| emigration > index_date)
        }
    }

    /// Names of the `fn` items declared between the first `{` after `start`
    /// and the matching closing brace at column zero or four.
    fn fn_names(source: &str, start: &str, end: &str) -> BTreeSet<String> {
        let from = source.find(start).expect("block start present");
        let body = &source[from..];
        let to = body.find(end).expect("block end present");
        body[..to]
            .lines()
            .filter_map(|line| {
                let line = line.trim_start();
                let rest = line.strip_prefix("fn ")?;
                let name = rest.split(['(', '<']).next()?;
                Some(name.to_owned())
            })
            .collect()
    }

    #[test]
    fn delegate_macro_forwards_every_trait_method() {
        let trait_fns = fn_names(
            include_str!("records.rs"),
            "pub trait MatchingRecord",
            "pub trait RoleIndexedRecord",
        );
        let macro_fns = fn_names(
            include_str!("../lib.rs"),
            "macro_rules! delegate_matching_record",
            "mod balance;",
        );
        assert!(!trait_fns.is_empty());
        assert_eq!(trait_fns, macro_fns);
    }

    #[test]
    fn wrapper_returns_what_its_inner_record_returns() {
        let inner = BaseRecord::new("r1", date(2010, 3, 4))
            .with_numeric("mother_birth_year", 1980.0)
            .with_death_date(date(2030, 1, 1))
            .with_emigration_date(date(2025, 6, 1));
        let mut inner = inner;
        inner.strata.insert("sex".to_owned(), "F".to_owned());
        inner.unique_key = Some("family".to_owned());
        let wrapped = Wrapper {
            inner: inner.clone(),
        };

        assert_eq!(wrapped.id(), inner.id());
        assert_eq!(wrapped.birth_date(), inner.birth_date());
        assert_eq!(wrapped.strata(), inner.strata());
        assert_eq!(wrapped.unique_key(), Some("family"));
        assert_eq!(wrapped.death_date(), Some(date(2030, 1, 1)));
        assert_eq!(wrapped.emigration_date(), Some(date(2025, 6, 1)));
        assert_eq!(wrapped.numeric("mother_birth_year"), Some(1980.0));
        assert_eq!(wrapped.numeric("absent"), None);
    }

    fn base(id: &str, mother: f64, emigrated: Option<NaiveDate>) -> BaseRecord {
        let record =
            BaseRecord::new(id, date(2010, 1, 1)).with_numeric("mother_birth_year", mother);
        match emigrated {
            Some(day) => record.with_emigration_date(day),
            None => record,
        }
    }

    #[test]
    fn matching_run_is_the_same_for_wrapper_and_inner_record() {
        let rows = [
            ("case", 1980.0, None, Some(date(2012, 1, 1))),
            ("far", 1996.0, None, None),
            ("gone", 1980.2, Some(date(2011, 1, 1)), None),
            ("near", 1980.5, Some(date(2015, 1, 1)), None),
        ];
        let inner: Vec<_> = rows
            .iter()
            .map(|&(id, mother, emigrated, transition)| {
                RoleTransitionRecord::from_record(base(id, mother, emigrated), transition)
            })
            .collect();
        let wrapped: Vec<_> = rows
            .iter()
            .map(|&(id, mother, emigrated, transition)| Wrapper {
                inner: RoleTransitionRecord::from_record(base(id, mother, emigrated), transition),
            })
            .collect();

        let ratio = MatchRatio::new(1).expect("non-zero ratio");
        let direct = MatchJob::new_transition(&inner, 42)
            .with_ratio(ratio)
            .with_constraints(vec![caliper_on_field("mother_birth_year", 1.0)])
            .with_resident_check()
            .run();
        let through_macro = MatchJob::new_transition(&wrapped, 42)
            .with_ratio(ratio)
            .with_constraints(vec![caliper_on_field("mother_birth_year", 1.0)])
            .with_resident_check()
            .run();

        assert_eq!(direct.pairs.len(), 1);
        assert_eq!(direct.pairs[0].control_id, "near");
        assert_eq!(through_macro.pairs.len(), direct.pairs.len());
        assert_eq!(
            through_macro.pairs[0].anchor_id(),
            direct.pairs[0].anchor_id()
        );
        assert_eq!(
            through_macro.pairs[0].comparator_id(),
            direct.pairs[0].comparator_id()
        );
        assert_eq!(through_macro.matched_cases, direct.matched_cases);
        assert_eq!(through_macro.unmatched_cases, direct.unmatched_cases);
    }
}
