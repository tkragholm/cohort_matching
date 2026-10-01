//! Record types the matchers and balance diagnostics read.

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

/// Covariate value used in balance diagnostics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CovariateValue {
    /// Continuous covariate.
    Numeric(f64),
    /// String-valued covariate.
    Categorical(String),
    /// Explicit missing marker.
    Missing,
}

/// Generic matching record with minimal core fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseRecord {
    /// Stable record identifier.
    pub id: String,
    /// Reference date used for date-caliper matching.
    pub birth_date: NaiveDate,
    /// Exact-match strata fields.
    pub strata: HashMap<String, String>,
    /// Named numeric fields, for caliper constraints beyond the birth date.
    ///
    /// `#[serde(default)]` so a record serialised before this field existed
    /// still deserialises: the map is then empty and every caliper over it
    /// refuses, which is the safe direction for a constraint.
    #[serde(default)]
    pub numerics: HashMap<String, f64>,
    /// Optional generic uniqueness key.
    pub unique_key: Option<String>,
    /// Optional death date.
    pub death_date: Option<NaiveDate>,
    /// Optional emigration date (used to derive residency at index).
    #[serde(default)]
    pub emigration_date: Option<NaiveDate>,
}

impl BaseRecord {
    /// Construct a record with empty optional fields.
    #[must_use]
    pub fn new(id: impl Into<String>, birth_date: NaiveDate) -> Self {
        Self {
            id: id.into(),
            birth_date,
            strata: HashMap::new(),
            numerics: HashMap::new(),
            unique_key: None,
            death_date: None,
            emigration_date: None,
        }
    }

    /// Attach a named numeric value, for a [`crate::constraints::Caliper`].
    #[must_use]
    pub fn with_numeric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.numerics.insert(name.into(), value);
        self
    }

    /// Set an optional death date for the record.
    #[must_use]
    pub const fn with_death_date(mut self, date: NaiveDate) -> Self {
        self.death_date = Some(date);
        self
    }

    /// Set an optional emigration date for the record.
    #[must_use]
    pub const fn with_emigration_date(mut self, date: NaiveDate) -> Self {
        self.emigration_date = Some(date);
        self
    }
}

/// Neutral alias for an index/anchor group record.
pub type AnchorRecord = BaseRecord;

/// Neutral alias for candidate comparison records.
pub type CandidateRecord = BaseRecord;

/// Generic record carrying covariates for balance reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceRecord {
    /// Core record fields used by matching primitives.
    #[serde(flatten)]
    pub core: BaseRecord,
    /// Optional covariates used for balance checks.
    pub covariates: HashMap<String, CovariateValue>,
}

impl BalanceRecord {
    /// Construct a balance record with empty covariates.
    #[must_use]
    pub fn new(id: impl Into<String>, birth_date: NaiveDate) -> Self {
        Self {
            core: BaseRecord::new(id, birth_date),
            covariates: HashMap::new(),
        }
    }

    /// Construct a builder for declarative record creation.
    pub fn builder(id: impl Into<String>, birth_date: NaiveDate) -> BalanceRecordBuilder {
        BalanceRecordBuilder::new(id, birth_date)
    }
}

/// Builder for [`BalanceRecord`] with declarative covariate policies.
pub struct BalanceRecordBuilder {
    id: String,
    birth_date: NaiveDate,
    covariates: HashMap<String, CovariateValue>,
}

impl BalanceRecordBuilder {
    fn new(id: impl Into<String>, birth_date: NaiveDate) -> Self {
        Self {
            id: id.into(),
            birth_date,
            covariates: HashMap::new(),
        }
    }

    /// Add a numeric covariate with canonical missing/non-finite policy.
    #[must_use]
    pub fn numeric(mut self, name: impl Into<String>, value: impl Into<Option<f64>>) -> Self {
        let name = name.into();
        let value = value.into();
        self.covariates.insert(
            name,
            value
                .filter(|v| v.is_finite())
                .map_or(CovariateValue::Missing, CovariateValue::Numeric),
        );
        self
    }

    /// Add a categorical covariate with canonical missing policy.
    #[must_use]
    pub fn categorical(
        mut self,
        name: impl Into<String>,
        value: impl Into<Option<String>>,
    ) -> Self {
        let name = name.into();
        let value = value.into();
        self.covariates.insert(
            name,
            value.map_or(CovariateValue::Missing, CovariateValue::Categorical),
        );
        self
    }

    /// Add a categorical covariate with a specific domain-specified ordering.
    ///
    /// Currently, the ordering is treated as a hint for reporting tools.
    #[must_use]
    pub fn categorical_ordered(
        self,
        name: impl Into<String>,
        value: impl Into<Option<String>>,
        _ordering: Vec<String>,
    ) -> Self {
        // For now, we reuse categorical logic. In a future update, we can store
        // ordering metadata in the BalanceRecord if needed.
        self.categorical(name, value)
    }

    /// Consume builder and return record.
    #[must_use]
    pub fn build(self) -> BalanceRecord {
        BalanceRecord {
            core: BaseRecord::new(self.id, self.birth_date),
            covariates: self.covariates,
        }
    }
}

/// Generic record for transition-based role logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleTransitionRecord<R = BaseRecord> {
    /// Shared core record attributes used for matching constraints.
    #[serde(flatten)]
    pub record: R,
    /// Date when the record transitions from comparison risk set to anchor group.
    pub transition_date: Option<NaiveDate>,
}

impl<R> RoleTransitionRecord<R> {
    /// Construct a transition record from an arbitrary record type.
    #[must_use]
    pub const fn from_record(record: R, transition_date: Option<NaiveDate>) -> Self {
        Self {
            record,
            transition_date,
        }
    }
}

impl<R> Deref for RoleTransitionRecord<R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        &self.record
    }
}

impl<R> DerefMut for RoleTransitionRecord<R> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.record
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;

    #[test]
    fn role_transition_record_constructor_sets_fields() {
        let record = BaseRecord::new("a", date(2010, 1, 1));
        let transition_date = Some(date(2014, 1, 1));
        let row = RoleTransitionRecord::from_record(record, transition_date);
        assert_eq!(row.record.id, "a");
        assert_eq!(row.transition_date, transition_date);
    }

    #[test]
    fn role_transition_record_deref_allows_direct_core_field_access() {
        let mut row =
            RoleTransitionRecord::from_record(BaseRecord::new("a", date(2010, 1, 1)), None);
        row.death_date = Some(date(2020, 1, 1));
        let selector = |r: &RoleTransitionRecord<BaseRecord>| r.death_date;
        assert_eq!(row.id, "a");
        assert_eq!(selector(&row), Some(date(2020, 1, 1)));
    }
}
