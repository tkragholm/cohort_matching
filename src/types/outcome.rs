//! Match outcomes, diagnostics and the typed reasons they count.

use super::criteria::{CommonSupport, CriteriaValidationError, Estimand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

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

#[cfg(test)]
mod tests {
    use super::*;

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
