//! # `cohort_matching`
//!
//! Caliper-based cohort matching utilities for observational cohort studies.
//!
//! The crate provides a high-level, research-ready API for creating matched cohorts using
//! both standard anchor-to-candidate matching and longitudinal role-transition (risk-set)
//! sampling.
//!
//! ## Primary API: `MatchJob`
//!
//! The [`MatchJob`] builder is the recommended entry point for configuring and executing
//! matching. It supports:
//!
//! - **Standard Matching**: Fixed sets of anchors and candidates.
//! - **Role-Transition Matching**: Risk-set sampling within a longitudinal cohort.
//! - **Flexible Metrics**: Birth date calipers, exact match strata, Propensity Scores,
//!   and Mahalanobis distance.
//! - **Declarative Constraints**: Built-in constraints for gender, residency, and more.
//!
//! ```rust,ignore
//! use cohort_matching::prelude::*;
//!
//! // Standard matching example
//! let outcome = MatchJob::new_standard(&anchors, &candidates, 42)
//!     .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
//!     .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
//!     .with_gender_match()
//!     .run();
//! ```
//!
//! For more details, see the [GUIDE.md](https://github.com/tkragholm/cohort_matching/blob/main/GUIDE.md).

// ---------------------------------------------------------------------------
// Shared builder-setter macros — eliminates identical 3-line setter bodies
// that arise from the builder pattern on structs with a `reason` or
// `distance_config` field.
// ---------------------------------------------------------------------------

/// Generates a `with_reason` builder setter for any struct that has a
/// `reason: &'static str` field.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_with_reason {
    () => {
        /// Override the default reason string used in diagnostics or rejection messages.
        #[must_use]
        pub const fn with_reason(mut self, reason: &'static str) -> Self {
            self.reason = reason;
            self
        }
    };
}

/// Generates a `with_distance_config` builder setter for structs that carry
/// `distance_config: Option<&'a DistanceConfig>`.  The invoking `impl` block
/// must have a lifetime `'a` in scope and `DistanceConfig` must be imported.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_with_distance_config {
    () => {
        #[must_use]
        pub const fn with_distance_config(mut self, distance_config: &'a DistanceConfig) -> Self {
            self.distance_config = Some(distance_config);
            self
        }
    };
}

/// Generates a full `MatchingRecord` impl that delegates every method to an
/// inner field.  Used in test fixtures that wrap a `BaseRecord` or
/// `RoleTransitionRecord<BaseRecord>` to avoid repeating identical 3-line
/// getter bodies.
///
/// ```ignore
/// impl crate::MatchingRecord for MyWrapper {
///     crate::delegate_matching_record!(inner);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! delegate_matching_record {
    ($field:ident) => {
        fn id(&self) -> &str {
            self.$field.id()
        }
        fn birth_date(&self) -> chrono::NaiveDate {
            self.$field.birth_date()
        }
        fn strata(&self) -> &std::collections::HashMap<String, String> {
            self.$field.strata()
        }
        fn unique_key(&self) -> Option<&str> {
            self.$field.unique_key()
        }
        fn death_date(&self) -> Option<chrono::NaiveDate> {
            self.$field.death_date()
        }
    };
}

mod balance;
pub mod constraints;
mod matching;
#[cfg(feature = "reporting")]
pub mod reporting;
mod role_transition;
mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
#[cfg(any(test, feature = "test-utils"))]
pub use test_utils::{date, record, test_outcome};

pub use rapidhash::RapidHashMap;

pub use balance::{
    CategoricalCovariateSpec, NumericCovariateSpec, ProjectedBalanceRow, balance_diagnostics,
    balance_report, balance_report_with_options, balance_threshold_summary,
    build_projected_balance_rows, categorical_max_abs_diff, smd_numeric_from_values, variance,
};
pub use matching::{
    CaliperConstraint, Constraint, ConstraintContext, ConstraintGroup, CovariateEncodingConfig,
    CovariateMatrix, CovariateRecord, DateDistance, DeterministicSelection, DistanceChannel,
    DistanceConfig, DistanceMetric, ElasticNetLogisticConfig, EstimationError,
    ExactMatchConstraint, IdMapMahalanobisDistance, IdMapPropensityScoreDistance,
    LogisticRegressionConfig, MahalanobisCovarianceStrategy, MahalanobisDistance,
    MahalanobisDistancePreparation, MahalanobisError, MahalanobisPreparationConfig,
    MahalanobisTransform, MatchJob, MatchWeightMethod, MatchingRecord, MissingValuePolicy,
    NearestBirthDateSelection, PairWeight, PairWeightRow, PairWeightSet, PairWeightTable,
    PropensityDistancePreparation, PropensityEstimator, PropensityMatchedOutcome,
    PropensityScoreConfig, PropensityScoreDistance, PropensityScoreEstimate,
    PropensityScoreOutputScale, RandomSelection, ReconstructedEpisode, ReconstructedMatchedPair,
    ReconstructedMatchingOptions, ReconstructedMatchingOutput, ReconstructedMatchingTier,
    ResidentAtIndexRecord, RoleIndexedRecord, SelectionStrategy, StandardMatchRequest,
    StrataExactConstraint, SubclassReferenceGroup, SubclassSummary, SubclassificationConfig,
    SubclassificationOutcome, UnitRole, UnitWeightRow, UnitWeightSet, UnitWeightTable,
    UsedControlsVec, effective_sample_size, estimate_propensity_and_match, match_standard,
    match_weights_from_pairs, pair_weights_from_pairs, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, reconstruct_case_control_pairs,
    subclassify_by_propensity_score_map,
};
pub use role_transition::{
    DefaultRiskSetPolicy, RiskSetPolicy, TransitionMatchRequest, match_transition,
};
pub use types::{
    AgeLimitYears, AnchorRecord, BalanceDiagnostics, BalanceRecord, BalanceRecordBuilder,
    BalanceReport, BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds, BaseRecord,
    BirthDateWindowDays, CandidateRecord, CategoricalBalance, CategoricalLevelBalance,
    CommonSupport, CommonSupportFailureReason, ConstraintReason, ControlIdx, CovariateValue,
    CriteriaValidationError, DistanceCaliper, DistanceCaliperReason, Estimand, EstimandDriftReason,
    ExclusionReason, InvalidCriteriaReason, MatchDiagnostics, MatchOutcome, MatchRatio,
    MatchedPair, MatchingCriteria, NumericBalance, NumericBalanceThresholdCheck,
    NumericBalanceTransform, RoleTransitionOptions, RoleTransitionRecord,
    SupplementalBalanceCovariates, UniqueValueId, ValidatedMatchingCriteria,
};

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_ecdf_distance_stats(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    balance::benchmark_ecdf_distance_stats(case_values, control_values)
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_eqq_distance_stats(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    balance::benchmark_eqq_distance_stats(case_values, control_values)
}

/// Common imports for consumers of this crate.
pub mod prelude {
    pub use crate::{
        AgeLimitYears, AnchorRecord, BalanceDiagnostics, BalanceRecord, BalanceRecordBuilder,
        BalanceReport, BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds,
        BaseRecord, BirthDateWindowDays, CaliperConstraint, CandidateRecord, CategoricalBalance,
        CategoricalCovariateSpec, CategoricalLevelBalance, CommonSupport,
        CommonSupportFailureReason, Constraint, ConstraintContext, ConstraintGroup,
        ConstraintReason, ControlIdx, CovariateEncodingConfig, CovariateMatrix, CovariateRecord,
        CovariateValue, CriteriaValidationError, DateDistance, DefaultRiskSetPolicy,
        DistanceCaliper, DistanceCaliperReason, DistanceChannel, DistanceConfig, DistanceMetric,
        ElasticNetLogisticConfig, Estimand, EstimandDriftReason, EstimationError,
        ExactMatchConstraint, ExclusionReason, IdMapMahalanobisDistance,
        IdMapPropensityScoreDistance, InvalidCriteriaReason, LogisticRegressionConfig,
        MahalanobisCovarianceStrategy, MahalanobisDistance, MahalanobisDistancePreparation,
        MahalanobisError, MahalanobisPreparationConfig, MahalanobisTransform, MatchDiagnostics,
        MatchJob, MatchOutcome, MatchRatio, MatchWeightMethod, MatchedPair, MatchingCriteria,
        MatchingRecord, MissingValuePolicy, NumericBalance, NumericBalanceThresholdCheck,
        NumericBalanceTransform, NumericCovariateSpec, PairWeight, PairWeightRow, PairWeightSet,
        PairWeightTable, ProjectedBalanceRow, PropensityDistancePreparation, PropensityEstimator,
        PropensityMatchedOutcome, PropensityScoreConfig, PropensityScoreDistance,
        PropensityScoreEstimate, PropensityScoreOutputScale, ReconstructedEpisode,
        ReconstructedMatchedPair, ReconstructedMatchingOptions, ReconstructedMatchingOutput,
        ReconstructedMatchingTier, ResidentAtIndexRecord, RiskSetPolicy, RoleIndexedRecord,
        RoleTransitionOptions, RoleTransitionRecord, StandardMatchRequest, StrataExactConstraint,
        SubclassReferenceGroup, SubclassSummary, SubclassificationConfig, SubclassificationOutcome,
        SupplementalBalanceCovariates, TransitionMatchRequest, UniqueValueId, UnitRole,
        UnitWeightRow, UnitWeightSet, UnitWeightTable, ValidatedMatchingCriteria,
        balance_diagnostics, balance_report, balance_report_with_options,
        balance_threshold_summary, build_projected_balance_rows, categorical_max_abs_diff,
        effective_sample_size, estimate_propensity_and_match, match_standard, match_transition,
        match_weights_from_pairs, pair_weights_from_pairs, prepare_mahalanobis_distance_config,
        prepare_propensity_distance_config, reconstruct_case_control_pairs,
        smd_numeric_from_values, subclassify_by_propensity_score_map,
    };

    pub mod constraints {
        pub use crate::constraints::{
            Caliper, DateWindow, GenderMatch, MustBeAlive, MustBeResident,
        };
    }
}
