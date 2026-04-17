mod constraints;
mod distance;
mod engine;
mod estimation;
mod index;
pub mod job;
pub mod ratio;
mod reconstructed;
mod records;
mod selection;
mod subclassification;
mod weights;

pub use constraints::{
    CaliperConstraint, Constraint, ConstraintContext, ConstraintGroup, ExactMatchConstraint,
    StrataExactConstraint, UsedControlsVec,
};
pub use distance::{
    DateDistance, DistanceChannel, DistanceConfig, DistanceMetric, IdMapMahalanobisDistance,
    IdMapPropensityScoreDistance, MahalanobisDistance, MahalanobisError, PropensityScoreDistance,
};
pub use engine::{StandardMatchRequest, match_standard};
pub use estimation::{
    CovariateEncodingConfig, CovariateMatrix, ElasticNetLogisticConfig, EstimationError,
    LogisticRegressionConfig, MahalanobisCovarianceStrategy, MahalanobisDistancePreparation,
    MahalanobisPreparationConfig, MahalanobisTransform, MissingValuePolicy,
    PropensityDistancePreparation, PropensityEstimator, PropensityMatchedOutcome,
    PropensityScoreConfig, PropensityScoreEstimate, PropensityScoreOutputScale,
    estimate_propensity_and_match, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config,
};
pub use reconstructed::{
    ReconstructedEpisode, ReconstructedMatchedPair, ReconstructedMatchingOptions,
    ReconstructedMatchingOutput, ReconstructedMatchingTier, reconstruct_case_control_pairs,
};
pub use records::{CovariateRecord, MatchingRecord, ResidentAtIndexRecord, RoleIndexedRecord};
pub use selection::{
    DeterministicSelection, NearestBirthDateSelection, RandomSelection, SelectionStrategy,
};
pub use subclassification::{
    SubclassReferenceGroup, SubclassSummary, SubclassificationConfig, SubclassificationOutcome,
    subclassify_by_propensity_score_map,
};
pub use weights::{
    MatchWeightMethod, PairWeight, PairWeightRow, PairWeightSet, PairWeightTable, UnitRole,
    UnitWeightRow, UnitWeightSet, UnitWeightTable, effective_sample_size, match_weights_from_pairs,
    pair_weights_from_pairs,
};

pub use constraints::unique_value;
pub use engine::{build_outcome, invalid_criteria_outcome, to_f64};

pub use engine::CandidatePoolRequest;
pub use engine::{MatchEngine, finalize_estimand_diagnostics, invalid_common_support_outcome};
pub use job::MatchJob;
