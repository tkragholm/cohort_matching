//! Matching criteria, their validation, and transition options.

use super::ids::{AgeLimitYears, BirthDateWindowDays, DistanceCaliper, MatchRatio};
use serde::{Deserialize, Serialize};
use std::ops::Deref;

/// Target estimand for matched analyses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Estimand {
    /// Average treatment effect in the treated (anchor group).
    #[default]
    Att,
    /// Average treatment effect in the control (candidate group).
    Atc,
    /// Average treatment effect in the overall population.
    Ate,
    /// Average treatment effect in the matched sample.
    Atm,
}

/// Common-support trimming policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommonSupport {
    /// Discard anchors outside the overlap region.
    Treated,
    /// Discard candidates outside the overlap region.
    Control,
    /// Discard both anchors and candidates outside overlap.
    Both,
}

/// Matching criteria used by the core engine.
#[derive(bon::Builder, Debug, Clone, Serialize, Deserialize)]
pub struct MatchingCriteria {
    /// Maximum absolute difference in days between record dates.
    #[builder(default = 30)]
    pub birth_date_window_days: i32,
    /// Requested number of candidates per anchor.
    #[builder(default = 1)]
    pub match_ratio: usize,
    /// Required exact-match strata keys.
    #[builder(default)]
    pub required_strata: Vec<String>,
    /// Optional strata key for control uniqueness (fallbacks to `unique_key`).
    pub unique_by_key: Option<String>,
    /// Allow reusing candidates across anchors.
    #[builder(default = false)]
    pub allow_replacement: bool,
    /// Requested target estimand.
    #[builder(default)]
    #[serde(default)]
    pub estimand: Estimand,
    /// Optional common-support trimming policy.
    #[serde(default)]
    pub common_support: Option<CommonSupport>,
}

/// Errors returned by matching criteria validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CriteriaValidationError {
    /// Birth date window must be non-negative.
    NegativeBirthDateWindow,
    /// Match ratio must be at least one.
    ZeroMatchRatio,
}

impl Default for MatchingCriteria {
    fn default() -> Self {
        Self {
            birth_date_window_days: 30,
            match_ratio: 1,
            required_strata: Vec::new(),
            unique_by_key: None,
            allow_replacement: false,
            estimand: Estimand::default(),
            common_support: None,
        }
    }
}

impl MatchingCriteria {
    /// Validate criteria and return an immutable validated wrapper.
    ///
    /// # Errors
    ///
    /// Returns [`CriteriaValidationError`] when one or more criteria values are invalid.
    pub fn validate(&self) -> Result<ValidatedMatchingCriteria, CriteriaValidationError> {
        self.clone().build()
    }

    /// Build and validate matching criteria.
    ///
    /// # Errors
    ///
    /// Returns [`CriteriaValidationError`] when one or more criteria values are invalid.
    pub fn build(self) -> Result<ValidatedMatchingCriteria, CriteriaValidationError> {
        if self.typed_birth_date_window().is_none() {
            return Err(CriteriaValidationError::NegativeBirthDateWindow);
        }
        if self.match_ratio == 0 {
            return Err(CriteriaValidationError::ZeroMatchRatio);
        }
        Ok(ValidatedMatchingCriteria { inner: self })
    }

    /// Typed non-negative birth-date window when criteria is valid.
    #[must_use]
    pub const fn typed_birth_date_window(&self) -> Option<BirthDateWindowDays> {
        BirthDateWindowDays::new(self.birth_date_window_days)
    }

    /// Typed non-zero match ratio when criteria is valid.
    #[must_use]
    pub const fn typed_match_ratio(&self) -> Option<MatchRatio> {
        MatchRatio::new(self.match_ratio)
    }

    /// Typed non-negative day-window caliper when criteria is valid.
    #[must_use]
    pub fn typed_birth_date_caliper(&self) -> Option<DistanceCaliper> {
        self.typed_birth_date_window()
            .and_then(|days| DistanceCaliper::new(f64::from(days.get())))
    }
}

/// Validated matching criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedMatchingCriteria {
    inner: MatchingCriteria,
}

impl ValidatedMatchingCriteria {
    /// Access wrapped criteria.
    #[must_use]
    pub const fn criteria(&self) -> &MatchingCriteria {
        &self.inner
    }

    /// Consume wrapper and return raw criteria.
    #[must_use]
    pub fn into_inner(self) -> MatchingCriteria {
        self.inner
    }
}

impl Deref for ValidatedMatchingCriteria {
    type Target = MatchingCriteria;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl AsRef<MatchingCriteria> for ValidatedMatchingCriteria {
    fn as_ref(&self) -> &MatchingCriteria {
        &self.inner
    }
}

/// Generalized options for transition-based risk-set matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleTransitionOptions {
    /// Include records transitioning strictly before this age threshold in years.
    pub transition_age_limit_years: AgeLimitYears,
    /// Optional descending fallback ratios, for example `[4, 3, 2]`.
    /// When empty, [`MatchingCriteria::match_ratio`] is used.
    pub ratio_fallback: Vec<MatchRatio>,
}

impl Default for RoleTransitionOptions {
    fn default() -> Self {
        Self {
            transition_age_limit_years: AgeLimitYears(6),
            ratio_fallback: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_criteria_validation_rejects_invalid_values() {
        let negative_window = MatchingCriteria::builder()
            .birth_date_window_days(-1)
            .build();
        assert!(matches!(
            negative_window.validate(),
            Err(CriteriaValidationError::NegativeBirthDateWindow)
        ));

        let zero_ratio = MatchingCriteria::builder().match_ratio(0).build();
        assert!(matches!(
            zero_ratio.validate(),
            Err(CriteriaValidationError::ZeroMatchRatio)
        ));
    }

    #[test]
    fn builder_sets_fields_and_builds_validated_criteria() {
        let validated = MatchingCriteria::builder()
            .birth_date_window_days(10)
            .match_ratio(2)
            .required_strata(vec!["municipality".to_string()])
            .unique_by_key("family".to_string())
            .allow_replacement(true)
            .estimand(Estimand::Ate)
            .common_support(CommonSupport::Both)
            .build()
            .validate()
            .expect("valid criteria");

        assert_eq!(validated.birth_date_window_days, 10);
        assert_eq!(validated.match_ratio, 2);
        assert_eq!(validated.required_strata, vec!["municipality".to_string()]);
        assert_eq!(validated.unique_by_key, Some("family".to_string()));
        assert!(validated.allow_replacement);
        assert_eq!(validated.estimand, Estimand::Ate);
        assert_eq!(validated.common_support, Some(CommonSupport::Both));
    }
}
