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
