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
    /// Optional key for ensuring uniqueness (e.g., family ID).
    fn unique_key(&self) -> Option<&str>;
    /// Optional death date. If provided, used by [`MustBeAlive`] constraints.
    fn death_date(&self) -> Option<NaiveDate> {
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

    fn unique_key(&self) -> Option<&str> {
        self.unique_key.as_deref()
    }

    fn death_date(&self) -> Option<NaiveDate> {
        self.death_date
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
}

impl<R: MatchingRecord> RoleIndexedRecord for RoleTransitionRecord<R> {
    fn event_date(&self) -> Option<NaiveDate> {
        self.transition_date
    }
}

impl<R: CovariateRecord> CovariateRecord for RoleTransitionRecord<R> {
    fn covariates(&self) -> &HashMap<String, CovariateValue> {
        self.record.covariates()
    }
}
