use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::ops::Deref;

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
    /// Optional generic uniqueness key.
    pub unique_key: Option<String>,
}

impl BaseRecord {
    /// Construct a record with empty optional fields.
    #[must_use]
    pub fn new(id: impl Into<String>, birth_date: NaiveDate) -> Self {
        Self {
            id: id.into(),
            birth_date,
            strata: HashMap::new(),
            unique_key: None,
        }
    }
}

/// Neutral alias for an index/anchor group record.
pub type AnchorRecord = BaseRecord;

/// Neutral alias for candidate comparison records.
pub type CandidateRecord = BaseRecord;

/// Matching criteria used by the core engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchingCriteria {
    /// Maximum absolute difference in days between record dates.
    pub birth_date_window_days: i32,
    /// Requested number of candidates per anchor.
    pub match_ratio: usize,
    /// Required exact-match strata keys.
    pub required_strata: Vec<String>,
    /// Optional strata key for control uniqueness (fallbacks to `unique_key`).
    pub unique_by_key: Option<String>,
    /// Allow reusing candidates across anchors.
    pub allow_replacement: bool,
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
        if self.birth_date_window_days < 0 {
            return Err(CriteriaValidationError::NegativeBirthDateWindow);
        }
        if self.match_ratio == 0 {
            return Err(CriteriaValidationError::ZeroMatchRatio);
        }
        Ok(ValidatedMatchingCriteria {
            inner: self.clone(),
        })
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

/// Builder for [`MatchingCriteria`] with built-in validation.
#[derive(Debug, Clone, Default)]
pub struct MatchingCriteriaBuilder {
    inner: MatchingCriteria,
}

impl MatchingCriteriaBuilder {
    /// Create a new builder from default criteria.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub const fn birth_date_window_days(mut self, value: i32) -> Self {
        self.inner.birth_date_window_days = value;
        self
    }

    #[must_use]
    pub const fn match_ratio(mut self, value: usize) -> Self {
        self.inner.match_ratio = value;
        self
    }

    #[must_use]
    pub fn required_strata(mut self, value: Vec<String>) -> Self {
        self.inner.required_strata = value;
        self
    }

    #[must_use]
    pub fn unique_by_key(mut self, value: Option<String>) -> Self {
        self.inner.unique_by_key = value;
        self
    }

    #[must_use]
    pub const fn allow_replacement(mut self, value: bool) -> Self {
        self.inner.allow_replacement = value;
        self
    }

    /// Build and validate matching criteria.
    ///
    /// # Errors
    ///
    /// Returns [`CriteriaValidationError`] when one or more criteria values are invalid.
    pub fn build(self) -> Result<ValidatedMatchingCriteria, CriteriaValidationError> {
        self.inner.validate()
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

/// Generalized options for transition-based risk-set matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoleTransitionOptions {
    /// Include records transitioning strictly before this age threshold in years.
    pub transition_age_limit_years: u8,
    /// Optional descending fallback ratios, for example `[4, 3, 2]`.
    /// When empty, [`MatchingCriteria::match_ratio`] is used.
    pub ratio_fallback: Vec<usize>,
}

impl Default for RoleTransitionOptions {
    fn default() -> Self {
        Self {
            transition_age_limit_years: 6,
            ratio_fallback: Vec::new(),
        }
    }
}

/// Matched anchor/candidate pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedPair {
    /// Anchor identifier.
    pub case_id: String,
    /// Candidate identifier.
    pub control_id: String,
}

impl MatchedPair {
    /// Construct a pair using neutral anchor/comparator naming.
    #[must_use]
    pub fn new(anchor_id: impl Into<String>, comparator_id: impl Into<String>) -> Self {
        Self {
            case_id: anchor_id.into(),
            control_id: comparator_id.into(),
        }
    }

    /// Neutral accessor for the index/anchor identifier.
    #[must_use]
    pub fn anchor_id(&self) -> &str {
        self.case_id.as_str()
    }

    /// Neutral accessor for the comparator identifier.
    #[must_use]
    pub fn comparator_id(&self) -> &str {
        self.control_id.as_str()
    }
}

/// Matching summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchOutcome {
    /// Selected anchor/candidate pairs.
    pub pairs: Vec<MatchedPair>,
    /// Number of unmatched eligible anchors.
    pub unmatched_cases: usize,
    /// Number of unique candidates used (0 when matching with replacement).
    pub used_controls: usize,
    /// Number of anchors with at least one match.
    pub matched_cases: usize,
    /// Average number of selected candidates among matched anchors.
    pub avg_controls_per_case: f64,
    /// Structured run diagnostics.
    #[serde(default)]
    pub diagnostics: MatchDiagnostics,
}

/// Run diagnostics for matching.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatchDiagnostics {
    /// Total anchors evaluated by the engine.
    pub total_anchors_evaluated: usize,
    /// Anchors with no eligible candidates after filtering.
    pub anchors_with_no_candidates: usize,
    /// Anchors that had candidates but failed required ratio.
    pub anchors_below_required_ratio: usize,
    /// Matched anchor count.
    pub matched_anchors: usize,
    /// Number of selected pairs.
    pub pairs_selected: usize,
    /// Count of exclusions by reason.
    pub exclusion_counts: BTreeMap<String, usize>,
}

impl MatchOutcome {
    /// Neutral accessor for matched anchor count.
    #[must_use]
    pub const fn matched_anchors(&self) -> usize {
        self.matched_cases
    }

    /// Neutral accessor for unmatched anchor count.
    #[must_use]
    pub const fn unmatched_anchors(&self) -> usize {
        self.unmatched_cases
    }

    /// Neutral accessor for average comparators per matched anchor.
    #[must_use]
    pub const fn avg_comparators_per_anchor(&self) -> f64 {
        self.avg_controls_per_case
    }

    /// Neutral accessor for unique comparator usage count.
    #[must_use]
    pub const fn used_comparators(&self) -> usize {
        self.used_controls
    }
}

/// Cohort-level balance diagnostics from match results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceDiagnostics {
    /// Proportion of anchors matched.
    pub match_rate: f64,
    /// Number of matched anchors.
    pub matched_cases: usize,
    /// Number of unmatched anchors.
    pub unmatched_cases: usize,
    /// Average candidates per matched anchor.
    pub avg_controls_per_case: f64,
    /// Counts by strata key (`anchor_count`, `candidate_count`).
    pub strata_counts: HashMap<String, (usize, usize)>,
}

impl BalanceDiagnostics {
    /// Neutral accessor for matched anchor count.
    #[must_use]
    pub const fn matched_anchors(&self) -> usize {
        self.matched_cases
    }

    /// Neutral accessor for unmatched anchor count.
    #[must_use]
    pub const fn unmatched_anchors(&self) -> usize {
        self.unmatched_cases
    }

    /// Neutral accessor for average comparators per matched anchor.
    #[must_use]
    pub const fn avg_comparators_per_anchor(&self) -> f64 {
        self.avg_controls_per_case
    }
}

/// Numeric covariate balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericBalance {
    /// Covariate name.
    pub name: String,
    /// Pre-match anchor mean.
    pub mean_case_pre: f64,
    /// Pre-match candidate mean.
    pub mean_control_pre: f64,
    /// Pre-match standardized mean difference.
    pub smd_pre: f64,
    /// Post-match anchor mean.
    pub mean_case_post: f64,
    /// Post-match candidate mean.
    pub mean_control_post: f64,
    /// Post-match standardized mean difference.
    pub smd_post: f64,
}

/// Categorical level balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalLevelBalance {
    /// Level name.
    pub level: String,
    /// Pre-match anchor proportion.
    pub p_case_pre: f64,
    /// Pre-match candidate proportion.
    pub p_control_pre: f64,
    /// Pre-match standardized mean difference.
    pub smd_pre: f64,
    /// Post-match anchor proportion.
    pub p_case_post: f64,
    /// Post-match candidate proportion.
    pub p_control_post: f64,
    /// Post-match standardized mean difference.
    pub smd_post: f64,
}

/// Categorical covariate balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalBalance {
    /// Covariate name.
    pub name: String,
    /// Per-level balance statistics.
    pub levels: Vec<CategoricalLevelBalance>,
    /// Pre-match Cramer's V.
    pub cramers_v_pre: f64,
    /// Post-match Cramer's V.
    pub cramers_v_post: f64,
}

/// Full balance report across numeric and categorical covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReport {
    /// Numeric covariate summaries.
    pub numeric: Vec<NumericBalance>,
    /// Categorical covariate summaries.
    pub categorical: Vec<CategoricalBalance>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    #[test]
    fn matching_criteria_validation_rejects_invalid_values() {
        let negative_window = MatchingCriteria {
            birth_date_window_days: -1,
            ..MatchingCriteria::default()
        };
        assert!(matches!(
            negative_window.validate(),
            Err(CriteriaValidationError::NegativeBirthDateWindow)
        ));

        let zero_ratio = MatchingCriteria {
            match_ratio: 0,
            ..MatchingCriteria::default()
        };
        assert!(matches!(
            zero_ratio.validate(),
            Err(CriteriaValidationError::ZeroMatchRatio)
        ));
    }

    #[test]
    fn builder_sets_fields_and_builds_validated_criteria() {
        let validated = MatchingCriteriaBuilder::new()
            .birth_date_window_days(10)
            .match_ratio(2)
            .required_strata(vec!["municipality".to_string()])
            .unique_by_key(Some("family".to_string()))
            .allow_replacement(true)
            .build()
            .expect("valid criteria");

        assert_eq!(validated.birth_date_window_days, 10);
        assert_eq!(validated.match_ratio, 2);
        assert_eq!(validated.required_strata, vec!["municipality".to_string()]);
        assert_eq!(validated.unique_by_key, Some("family".to_string()));
        assert!(validated.allow_replacement);
    }

    #[test]
    fn role_transition_record_constructor_sets_fields() {
        let record = BaseRecord::new("a", date(2010, 1, 1));
        let transition_date = Some(date(2014, 1, 1));
        let row = RoleTransitionRecord::from_record(record, transition_date);
        assert_eq!(row.record.id, "a");
        assert_eq!(row.transition_date, transition_date);
    }

    #[test]
    fn matched_pair_and_outcome_accessors_are_consistent() {
        let pair = MatchedPair::new("anchor", "candidate");
        assert_eq!(pair.anchor_id(), "anchor");
        assert_eq!(pair.comparator_id(), "candidate");

        let outcome = MatchOutcome {
            pairs: vec![pair],
            unmatched_cases: 1,
            used_controls: 1,
            matched_cases: 1,
            avg_controls_per_case: 1.0,
            diagnostics: MatchDiagnostics::default(),
        };
        assert_eq!(outcome.matched_anchors(), 1);
        assert_eq!(outcome.unmatched_anchors(), 1);
        assert_eq!(outcome.used_comparators(), 1);
        assert!((outcome.avg_comparators_per_anchor() - 1.0).abs() < 1e-12);
    }
}
