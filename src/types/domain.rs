use super::{BaseRecord, CovariateValue, RoleTransitionOptions, RoleTransitionRecord};
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for matching on parent birth dates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParentMatching {
    /// Disable parent matching.
    Disabled,
    /// Match on both maternal and paternal parent birth dates.
    #[default]
    BothParents,
    /// Match only on maternal parent birth date.
    MotherOnly,
}

/// Domain-specific participant constraint settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantConstraintOptions {
    /// Require exact gender match.
    pub require_same_gender: bool,
    /// Require all parent dates used by [`ParentMatching`] to be present.
    pub require_both_parents: bool,
    /// Parent matching strategy.
    pub parent_matching: ParentMatching,
    /// Maximum absolute difference in days between parent birth dates.
    pub parent_birth_date_window_days: i32,
}

impl Default for ParticipantConstraintOptions {
    fn default() -> Self {
        Self {
            require_same_gender: true,
            require_both_parents: false,
            parent_matching: ParentMatching::default(),
            parent_birth_date_window_days: 365,
        }
    }
}

/// Generic participant record used by compatibility APIs and diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticipantRecord {
    /// Core record fields.
    #[serde(flatten)]
    pub core: BaseRecord,
    /// Optional gender/sex category.
    pub gender: Option<String>,
    /// Optional mother birth date.
    pub mother_birth_date: Option<NaiveDate>,
    /// Optional father birth date.
    pub father_birth_date: Option<NaiveDate>,
    /// Optional covariates used for balance checks.
    pub covariates: HashMap<String, CovariateValue>,
}

impl ParticipantRecord {
    /// Construct a participant with empty optional fields.
    #[must_use]
    pub fn new(id: impl Into<String>, birth_date: NaiveDate) -> Self {
        Self {
            core: BaseRecord::new(id, birth_date),
            gender: None,
            mother_birth_date: None,
            father_birth_date: None,
            covariates: HashMap::new(),
        }
    }
}

/// Accessor trait for compatibility-only participant attributes.
pub trait ParticipantAttributes {
    fn gender(&self) -> Option<&str>;
    fn mother_birth_date(&self) -> Option<NaiveDate>;
    fn father_birth_date(&self) -> Option<NaiveDate>;
}

impl ParticipantAttributes for ParticipantRecord {
    fn gender(&self) -> Option<&str> {
        self.gender.as_deref()
    }

    fn mother_birth_date(&self) -> Option<NaiveDate> {
        self.mother_birth_date
    }

    fn father_birth_date(&self) -> Option<NaiveDate> {
        self.father_birth_date
    }
}

/// Compatibility alias for case records.
///
/// This name is maintained for migration and may be deprecated in a future release.
pub type CaseRecord = ParticipantRecord;

/// Compatibility alias for control records.
///
/// This name is maintained for migration and may be deprecated in a future release.
pub type ControlRecord = ParticipantRecord;

/// Input record for risk-set matching with role switching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleSwitchingRecord {
    /// Shared participant attributes used for compatibility constraints.
    #[serde(flatten)]
    pub participant: ParticipantRecord,
    /// First diagnosis/index date for exposed re-entry. `None` means never exposed.
    pub diagnosis_date: Option<NaiveDate>,
}

impl RoleSwitchingRecord {
    /// Construct a role-switching record from a participant row.
    #[must_use]
    pub const fn from_participant(
        participant: ParticipantRecord,
        diagnosis_date: Option<NaiveDate>,
    ) -> Self {
        Self {
            participant,
            diagnosis_date,
        }
    }
}

impl ParticipantAttributes for RoleSwitchingRecord {
    fn gender(&self) -> Option<&str> {
        self.participant.gender.as_deref()
    }

    fn mother_birth_date(&self) -> Option<NaiveDate> {
        self.participant.mother_birth_date
    }

    fn father_birth_date(&self) -> Option<NaiveDate> {
        self.participant.father_birth_date
    }
}

/// Options specific to compatibility role-switching risk-set matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleSwitchingOptions {
    /// Include exposed cases diagnosed strictly before this age threshold in years.
    pub diagnosis_age_limit_years: u8,
    /// Optional descending fallback ratios, for example `[4, 3, 2]`.
    /// When empty, [`crate::MatchingCriteria::match_ratio`] is used.
    pub ratio_fallback: Vec<usize>,
    /// Domain-specific participant constraints.
    #[serde(default)]
    pub participant_constraints: ParticipantConstraintOptions,
}

impl Default for RoleSwitchingOptions {
    fn default() -> Self {
        Self {
            diagnosis_age_limit_years: 6,
            ratio_fallback: Vec::new(),
            participant_constraints: ParticipantConstraintOptions::default(),
        }
    }
}

impl From<RoleTransitionOptions> for RoleSwitchingOptions {
    fn from(value: RoleTransitionOptions) -> Self {
        Self {
            diagnosis_age_limit_years: value.transition_age_limit_years,
            ratio_fallback: value.ratio_fallback,
            participant_constraints: ParticipantConstraintOptions::default(),
        }
    }
}

impl From<RoleSwitchingOptions> for RoleTransitionOptions {
    fn from(value: RoleSwitchingOptions) -> Self {
        Self {
            transition_age_limit_years: value.diagnosis_age_limit_years,
            ratio_fallback: value.ratio_fallback,
        }
    }
}

impl From<RoleTransitionRecord<ParticipantRecord>> for RoleSwitchingRecord {
    fn from(value: RoleTransitionRecord<ParticipantRecord>) -> Self {
        Self {
            participant: value.record,
            diagnosis_date: value.transition_date,
        }
    }
}

impl From<RoleSwitchingRecord> for RoleTransitionRecord<ParticipantRecord> {
    fn from(value: RoleSwitchingRecord) -> Self {
        Self {
            record: value.participant,
            transition_date: value.diagnosis_date,
        }
    }
}

impl ParticipantAttributes for RoleTransitionRecord<ParticipantRecord> {
    fn gender(&self) -> Option<&str> {
        self.record.gender.as_deref()
    }

    fn mother_birth_date(&self) -> Option<NaiveDate> {
        self.record.mother_birth_date
    }

    fn father_birth_date(&self) -> Option<NaiveDate> {
        self.record.father_birth_date
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    #[test]
    fn participant_constraint_options_defaults_are_stable() {
        let defaults = ParticipantConstraintOptions::default();
        assert!(defaults.require_same_gender);
        assert!(!defaults.require_both_parents);
        assert!(matches!(
            defaults.parent_matching,
            ParentMatching::BothParents
        ));
        assert_eq!(defaults.parent_birth_date_window_days, 365);
    }

    #[test]
    fn participant_attribute_accessors_work_for_all_compat_records() {
        let mut participant = ParticipantRecord::new("a", date(2010, 1, 1));
        participant.gender = Some("F".to_string());
        participant.mother_birth_date = Some(date(1980, 1, 1));
        participant.father_birth_date = Some(date(1978, 1, 1));
        assert_eq!(participant.gender(), Some("F"));
        assert_eq!(participant.mother_birth_date(), Some(date(1980, 1, 1)));
        assert_eq!(participant.father_birth_date(), Some(date(1978, 1, 1)));

        let switching =
            RoleSwitchingRecord::from_participant(participant.clone(), Some(date(2014, 1, 1)));
        assert_eq!(switching.gender(), Some("F"));
        assert_eq!(switching.mother_birth_date(), Some(date(1980, 1, 1)));
        assert_eq!(switching.father_birth_date(), Some(date(1978, 1, 1)));

        let transition = RoleTransitionRecord::from_record(participant, Some(date(2014, 1, 1)));
        assert_eq!(transition.gender(), Some("F"));
        assert_eq!(transition.mother_birth_date(), Some(date(1980, 1, 1)));
        assert_eq!(transition.father_birth_date(), Some(date(1978, 1, 1)));
    }

    #[test]
    fn options_and_record_conversions_round_trip() {
        let switching_options = RoleSwitchingOptions {
            diagnosis_age_limit_years: 7,
            ratio_fallback: vec![4, 3, 2],
            participant_constraints: ParticipantConstraintOptions::default(),
        };
        let transition_options = RoleTransitionOptions::from(switching_options);
        assert_eq!(transition_options.transition_age_limit_years, 7);
        assert_eq!(transition_options.ratio_fallback, vec![4, 3, 2]);

        let switching_from_transition = RoleSwitchingOptions::from(transition_options);
        assert_eq!(switching_from_transition.diagnosis_age_limit_years, 7);
        assert_eq!(switching_from_transition.ratio_fallback, vec![4, 3, 2]);
        assert_eq!(
            switching_from_transition
                .participant_constraints
                .parent_birth_date_window_days,
            365
        );

        let participant = ParticipantRecord::new("child", date(2010, 1, 1));
        let transition = RoleTransitionRecord::from_record(participant, Some(date(2014, 1, 1)));
        let switching = RoleSwitchingRecord::from(transition.clone());
        assert_eq!(switching.diagnosis_date, Some(date(2014, 1, 1)));
        let transition_again = RoleTransitionRecord::from(switching);
        assert_eq!(transition_again.transition_date, transition.transition_date);
    }
}
