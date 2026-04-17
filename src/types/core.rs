use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt::{Display, Formatter};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};

/// Strictly positive requested matching ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MatchRatio(NonZeroUsize);

impl MatchRatio {
    /// Build a validated ratio.
    #[must_use]
    pub const fn new(value: usize) -> Option<Self> {
        match NonZeroUsize::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Access the raw ratio value.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// Validated non-negative birth-date window (days).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BirthDateWindowDays(i32);

impl BirthDateWindowDays {
    #[must_use]
    pub const fn new(value: i32) -> Option<Self> {
        if value >= 0 { Some(Self(value)) } else { None }
    }

    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// Validated age limit (years) for transition-case eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgeLimitYears(u8);

impl AgeLimitYears {
    #[must_use]
    pub const fn new(value: u8) -> Option<Self> {
        if value > 0 { Some(Self(value)) } else { None }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Non-negative finite distance-caliper value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct DistanceCaliper(f64);

impl DistanceCaliper {
    /// Build a validated caliper value.
    #[must_use]
    pub fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value >= 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Access the inner caliper distance.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Typed index into the control slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ControlIdx(usize);

impl ControlIdx {
    /// Wrap a raw control index.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Access the wrapped control index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Typed identifier for deduplicated unique-key values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UniqueValueId(usize);

impl UniqueValueId {
    /// Wrap a raw unique-value index.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Access the wrapped unique-value index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

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
    /// Optional death date.
    pub death_date: Option<NaiveDate>,
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
            death_date: None,
        }
    }

    /// Set an optional death date for the record.
    #[must_use]
    pub const fn with_death_date(mut self, date: NaiveDate) -> Self {
        self.death_date = Some(date);
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

/// Canonical exclusion-reason key for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExclusionReason {
    AdditionalFilter,
    Constraint(ConstraintReason),
    DistanceCaliper(DistanceCaliperReason),
    InvalidCriteria(InvalidCriteriaReason),
    CommonSupportFailure(CommonSupportFailureReason),
}

/// Typed estimand drift reason recorded by matching diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EstimandDriftReason {
    CommonSupportTrimming,
    UnmatchedAnchors,
    RatioShortfall,
    DistanceCaliperExclusion,
    CommonSupportCaliperInteraction,
}

impl Display for EstimandDriftReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::CommonSupportTrimming => "common_support_trimming",
            Self::UnmatchedAnchors => "unmatched_anchors",
            Self::RatioShortfall => "ratio_shortfall",
            Self::DistanceCaliperExclusion => "distance_caliper_exclusion",
            Self::CommonSupportCaliperInteraction => "common_support_caliper_interaction",
        };
        f.write_str(key)
    }
}

/// Typed constraint exclusion reason.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConstraintReason {
    ReplacementDisallowed,
    NoSelfMatch,
    MissingRequiredStrata,
    UniqueKeyReused,
    DistanceCaliper,
    GenderMismatch,
    CaliperExceeded,
    DateWindowExceeded,
    ParentDateMismatch,
    SameFamily,
    ControlAlreadyUsedInPrimaryTier,
    NotAliveAtIndex,
    NotResidentAtIndex,
    Custom(String),
}

impl ConstraintReason {
    #[must_use]
    pub fn from_reason_str(reason: &str) -> Self {
        match reason {
            "replacement_disallowed" => Self::ReplacementDisallowed,
            "no_self_match" => Self::NoSelfMatch,
            "missing_required_strata" => Self::MissingRequiredStrata,
            "unique_key_reused" => Self::UniqueKeyReused,
            "distance_caliper" => Self::DistanceCaliper,
            "gender_mismatch" => Self::GenderMismatch,
            "caliper_exceeded" => Self::CaliperExceeded,
            "date_window_exceeded" => Self::DateWindowExceeded,
            "parent_date_mismatch" => Self::ParentDateMismatch,
            "same_family" => Self::SameFamily,
            "control_already_used_in_primary_tier" => Self::ControlAlreadyUsedInPrimaryTier,
            "not_alive_at_index" => Self::NotAliveAtIndex,
            "not_resident_at_index" => Self::NotResidentAtIndex,
            _ => Self::Custom(reason.to_string()),
        }
    }
}

impl Display for ConstraintReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::ReplacementDisallowed => "replacement_disallowed",
            Self::NoSelfMatch => "no_self_match",
            Self::MissingRequiredStrata => "missing_required_strata",
            Self::UniqueKeyReused => "unique_key_reused",
            Self::DistanceCaliper => "distance_caliper",
            Self::GenderMismatch => "gender_mismatch",
            Self::CaliperExceeded => "caliper_exceeded",
            Self::DateWindowExceeded => "date_window_exceeded",
            Self::ParentDateMismatch => "parent_date_mismatch",
            Self::SameFamily => "same_family",
            Self::ControlAlreadyUsedInPrimaryTier => "control_already_used_in_primary_tier",
            Self::NotAliveAtIndex => "not_alive_at_index",
            Self::NotResidentAtIndex => "not_resident_at_index",
            Self::Custom(reason) => reason.as_str(),
        };
        f.write_str(key)
    }
}

/// Typed distance-caliper exclusion reason.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DistanceCaliperReason {
    DistanceCaliper,
    Custom(String),
}

impl DistanceCaliperReason {
    #[must_use]
    pub fn from_reason_str(reason: &str) -> Self {
        if reason == "distance_caliper" {
            Self::DistanceCaliper
        } else {
            Self::Custom(reason.to_string())
        }
    }
}

impl Display for DistanceCaliperReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DistanceCaliper => f.write_str("distance_caliper"),
            Self::Custom(reason) => f.write_str(reason),
        }
    }
}

/// Typed invalid-criteria diagnostics code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InvalidCriteriaReason {
    NegativeBirthDateWindow,
    ZeroMatchRatio,
    StudyOptions,
}

impl Display for InvalidCriteriaReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::NegativeBirthDateWindow => "negative_birth_date_window",
            Self::ZeroMatchRatio => "zero_match_ratio",
            Self::StudyOptions => "study_options",
        };
        f.write_str(key)
    }
}

impl From<CriteriaValidationError> for InvalidCriteriaReason {
    fn from(err: CriteriaValidationError) -> Self {
        match err {
            CriteriaValidationError::NegativeBirthDateWindow => Self::NegativeBirthDateWindow,
            CriteriaValidationError::ZeroMatchRatio => Self::ZeroMatchRatio,
        }
    }
}

/// Typed common-support failure diagnostics code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommonSupportFailureReason {
    RequiresPropensityScoreMap,
    NoOverlap,
}

impl Display for CommonSupportFailureReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::RequiresPropensityScoreMap => "requires_propensity_score_map",
            Self::NoOverlap => "no_overlap",
        };
        f.write_str(key)
    }
}

impl Display for ExclusionReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdditionalFilter => f.write_str("additional_filter"),
            Self::Constraint(reason) => Display::fmt(reason, f),
            Self::DistanceCaliper(reason) => Display::fmt(reason, f),
            Self::InvalidCriteria(reason) => write!(f, "invalid_criteria:{reason}"),
            Self::CommonSupportFailure(reason) => write!(f, "common_support_failure:{reason}"),
        }
    }
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
        if self.typed_birth_date_window().is_none() {
            return Err(CriteriaValidationError::NegativeBirthDateWindow);
        }
        if self.match_ratio == 0 {
            return Err(CriteriaValidationError::ZeroMatchRatio);
        }
        Ok(ValidatedMatchingCriteria {
            inner: self.clone(),
        })
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

/// Matched anchor/candidate pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub const fn anchor_id(&self) -> &str {
        self.case_id.as_str()
    }

    /// Neutral accessor for the comparator identifier.
    #[must_use]
    pub const fn comparator_id(&self) -> &str {
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

impl MatchDiagnostics {
    /// Merge another diagnostics object into this one.
    pub fn merge(&mut self, other: Self) {
        self.total_anchors_evaluated += other.total_anchors_evaluated;
        self.anchors_with_no_candidates += other.anchors_with_no_candidates;
        self.anchors_below_required_ratio += other.anchors_below_required_ratio;
        self.matched_anchors += other.matched_anchors;
        self.pairs_selected += other.pairs_selected;
        self.common_support_trimmed_anchors += other.common_support_trimmed_anchors;
        self.common_support_trimmed_candidates += other.common_support_trimmed_candidates;

        for (reason, count) in other.exclusion_counts {
            *self.exclusion_counts.entry(reason).or_insert(0) += count;
        }
        for reason in other.estimand_drift_reasons {
            if !self.estimand_drift_reasons.contains(&reason) {
                self.estimand_drift_reasons.push(reason);
            }
        }
    }
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
    pub exclusion_counts: BTreeMap<ExclusionReason, usize>,
    /// Requested estimand from criteria.
    #[serde(default)]
    pub requested_estimand: Estimand,
    /// Realized estimand after trimming/matching exclusions.
    #[serde(default)]
    pub realized_estimand: Estimand,
    /// Number of anchors discarded by common-support trimming.
    #[serde(default)]
    pub common_support_trimmed_anchors: usize,
    /// Number of candidates discarded by common-support trimming.
    #[serde(default)]
    pub common_support_trimmed_candidates: usize,
    /// Applied overlap interval from common-support trimming when available.
    #[serde(default)]
    pub common_support_overlap: Option<(f64, f64)>,
    /// Applied common-support policy when configured.
    #[serde(default)]
    pub common_support_policy: Option<CommonSupport>,
    /// Anchor score bounds used to compute common-support overlap when available.
    #[serde(default)]
    pub common_support_anchor_score_bounds: Option<(f64, f64)>,
    /// Candidate score bounds used to compute common-support overlap when available.
    #[serde(default)]
    pub common_support_candidate_score_bounds: Option<(f64, f64)>,
    /// Structured machine-readable drift reasons.
    #[serde(default)]
    pub estimand_drift_reasons: Vec<EstimandDriftReason>,
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericBalance {
    /// Covariate name.
    pub name: String,
    /// Pre-match anchor mean.
    pub mean_case_pre: f64,
    /// Pre-match candidate mean.
    pub mean_control_pre: f64,
    /// Pre-match standardized mean difference.
    pub smd_pre: f64,
    /// Pre-match variance ratio (`anchor / candidate`).
    #[serde(default)]
    pub var_ratio_pre: f64,
    /// Pre-match mean absolute eCDF distance.
    #[serde(default)]
    pub ecdf_mean_diff_pre: f64,
    /// Pre-match max absolute eCDF distance.
    #[serde(default)]
    pub ecdf_max_diff_pre: f64,
    /// Pre-match mean absolute eQQ distance.
    #[serde(default)]
    pub eqq_mean_diff_pre: f64,
    /// Pre-match max absolute eQQ distance.
    #[serde(default)]
    pub eqq_max_diff_pre: f64,
    /// Post-match anchor mean.
    pub mean_case_post: f64,
    /// Post-match candidate mean.
    pub mean_control_post: f64,
    /// Post-match standardized mean difference.
    pub smd_post: f64,
    /// Post-match variance ratio (`anchor / candidate`).
    #[serde(default)]
    pub var_ratio_post: f64,
    /// Post-match mean absolute eCDF distance.
    #[serde(default)]
    pub ecdf_mean_diff_post: f64,
    /// Post-match max absolute eCDF distance.
    #[serde(default)]
    pub ecdf_max_diff_post: f64,
    /// Post-match mean absolute eQQ distance.
    #[serde(default)]
    pub eqq_mean_diff_post: f64,
    /// Post-match max absolute eQQ distance.
    #[serde(default)]
    pub eqq_max_diff_post: f64,
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

/// Optional transformed numeric covariates included in balance summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NumericBalanceTransform {
    /// Include only observed covariates.
    #[default]
    None,
    /// Include squared numeric terms (for example `age^2`).
    Squares,
    /// Include squared terms and pairwise interactions (for example `age * income`).
    SquaresAndPairwiseInteractions,
}

/// Configuration for balance reporting behavior.
#[derive(bon::Builder, Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReportOptions {
    /// Which transformed numeric terms to include.
    #[builder(default)]
    #[serde(default)]
    pub numeric_transforms: NumericBalanceTransform,
    /// Optional supplemental covariates for balance-only diagnostics.
    #[builder(default)]
    #[serde(default)]
    pub supplemental_covariates: SupplementalBalanceCovariates,
}

impl Default for BalanceReportOptions {
    fn default() -> Self {
        Self {
            numeric_transforms: NumericBalanceTransform::None,
            supplemental_covariates: SupplementalBalanceCovariates::default(),
        }
    }
}

/// Supplemental covariates used only for balance diagnostics.
///
/// This enables `addlvariables`-style workflows where diagnostics include
/// variables not stored in the matching records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SupplementalBalanceCovariates {
    /// Supplemental covariates for anchor/case records keyed by record id.
    #[serde(default)]
    pub cases: HashMap<String, HashMap<String, CovariateValue>>,
    /// Supplemental covariates for candidate/control records keyed by record id.
    #[serde(default)]
    pub controls: HashMap<String, HashMap<String, CovariateValue>>,
}

/// Threshold configuration for post-match balance checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceThresholds {
    /// Maximum allowed absolute post-match SMD.
    pub smd_abs_max: Option<f64>,
    /// Minimum allowed post-match variance ratio.
    pub var_ratio_min: Option<f64>,
    /// Maximum allowed post-match variance ratio.
    pub var_ratio_max: Option<f64>,
    /// Maximum allowed post-match eCDF max distance.
    pub ecdf_max_diff_max: Option<f64>,
    /// Maximum allowed post-match eQQ max distance.
    pub eqq_max_diff_max: Option<f64>,
}

impl Default for BalanceThresholds {
    fn default() -> Self {
        Self {
            smd_abs_max: Some(0.1),
            var_ratio_min: Some(0.8),
            var_ratio_max: Some(1.25),
            ecdf_max_diff_max: Some(0.1),
            eqq_max_diff_max: None,
        }
    }
}

impl BalanceThresholds {
    /// Strict threshold profile for diagnostics-sensitive applications.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            smd_abs_max: Some(0.05),
            var_ratio_min: Some(0.9),
            var_ratio_max: Some(1.11),
            ecdf_max_diff_max: Some(0.05),
            eqq_max_diff_max: None,
        }
    }

    /// Moderate threshold profile aligned with crate defaults.
    #[must_use]
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Lenient threshold profile for exploratory checks.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            smd_abs_max: Some(0.15),
            var_ratio_min: Some(0.67),
            var_ratio_max: Some(1.5),
            ecdf_max_diff_max: Some(0.15),
            eqq_max_diff_max: None,
        }
    }
}

/// Threshold-check result for one numeric covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericBalanceThresholdCheck {
    /// Covariate name.
    pub name: String,
    /// Whether post-match SMD is within threshold (or `None` when disabled).
    pub smd_post_ok: Option<bool>,
    /// Whether post-match variance ratio is within threshold (or `None` when disabled).
    pub var_ratio_post_ok: Option<bool>,
    /// Whether post-match eCDF max distance is within threshold (or `None` when disabled).
    pub ecdf_max_diff_post_ok: Option<bool>,
    /// Whether post-match eQQ max distance is within threshold (or `None` when disabled).
    pub eqq_max_diff_post_ok: Option<bool>,
    /// Aggregate pass/fail over enabled checks.
    pub all_enabled_checks_ok: bool,
}

/// Summary of threshold checks across numeric covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceThresholdSummary {
    /// Per-covariate threshold checks.
    pub numeric: Vec<NumericBalanceThresholdCheck>,
    /// Aggregate pass/fail over all numeric covariates and enabled checks.
    pub all_enabled_checks_ok: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;

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

    #[test]
    fn balance_thresholds_default_is_reasonable() {
        let thresholds = BalanceThresholds::default();
        assert_eq!(thresholds.smd_abs_max, Some(0.1));
        assert_eq!(thresholds.var_ratio_min, Some(0.8));
        assert_eq!(thresholds.var_ratio_max, Some(1.25));
        assert_eq!(thresholds.ecdf_max_diff_max, Some(0.1));
        assert_eq!(thresholds.eqq_max_diff_max, None);
    }

    #[test]
    fn balance_report_options_defaults_to_no_transforms() {
        let options = BalanceReportOptions::default();
        assert_eq!(options.numeric_transforms, NumericBalanceTransform::None);
        assert!(options.supplemental_covariates.cases.is_empty());
        assert!(options.supplemental_covariates.controls.is_empty());
    }

    #[test]
    fn balance_threshold_presets_are_ordered_by_strictness() {
        let strict = BalanceThresholds::strict();
        let moderate = BalanceThresholds::moderate();
        let lenient = BalanceThresholds::lenient();

        assert!(strict.smd_abs_max < moderate.smd_abs_max);
        assert!(moderate.smd_abs_max < lenient.smd_abs_max);
        assert!(strict.ecdf_max_diff_max < moderate.ecdf_max_diff_max);
        assert!(moderate.ecdf_max_diff_max < lenient.ecdf_max_diff_max);
    }
}
