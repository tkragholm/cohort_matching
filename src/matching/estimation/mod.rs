//! Propensity scores and Mahalanobis configurations estimated from covariate records.

mod config;
mod covariance;
mod encoding;
mod linalg;
mod logistic;

pub use config::*;
pub use encoding::CovariateMatrix;

use super::constraints::ConstraintGroup;
use super::distance::DistanceConfig;
use super::engine::{StandardMatchRequest, match_standard};
use super::records::{CovariateRecord, MatchingRecord};
use super::selection::SelectionStrategy;
use crate::types::{DistanceCaliper, MatchOutcome, MatchRatio, MatchingCriteria};
use covariance::{covariance_matrix, rank_transform_columns};
use encoding::build_covariate_matrix;
use itertools::Itertools;
use linalg::invert_matrix_with_ridge;
use logistic::{fit_elastic_net_logistic, fit_logistic_regression};
use rapidhash::RapidHashMap;

/// Propensity-score estimation output.
#[derive(Debug, Clone)]
pub struct PropensityScoreEstimate {
    /// Probability-scale propensity scores by id.
    pub probabilities: RapidHashMap<String, f64>,
    /// Linear predictor values by id.
    pub linear_predictor: RapidHashMap<String, f64>,
    /// Distance-scale scores (selected by [`PropensityScoreOutputScale`]).
    pub distance_scores: RapidHashMap<String, f64>,
    /// Fitted coefficients aligned with `column_names`.
    pub coefficients: Vec<f64>,
    /// Design-matrix column names.
    pub column_names: Vec<String>,
    /// Estimator used for this run.
    pub estimator: &'static str,
    /// Whether IRLS converged within max iterations.
    pub converged: bool,
    /// Iterations executed.
    pub iterations: usize,
}

/// Combined propensity output and distance config.
#[derive(Debug, Clone)]
pub struct PropensityDistancePreparation {
    /// Prepared distance config for matching APIs.
    pub distance_config: DistanceConfig,
    /// Estimation details and diagnostics.
    pub estimate: PropensityScoreEstimate,
}

/// In-crate Mahalanobis preprocessing result.
#[derive(Debug, Clone)]
pub struct MahalanobisDistancePreparation {
    /// Prepared distance config for matching APIs.
    pub distance_config: DistanceConfig,
    /// Encoded vectors by id.
    pub vectors: RapidHashMap<String, Vec<f64>>,
    /// Inverse covariance in row-major format.
    pub inverse_covariance: Vec<f64>,
    /// Vector dimension.
    pub dimension: usize,
    /// Encoded vector column names.
    pub column_names: Vec<String>,
}

/// Matched output combined with in-crate propensity estimation details.
#[derive(Debug, Clone)]
pub struct PropensityMatchedOutcome {
    /// Matching outcome.
    pub outcome: MatchOutcome,
    /// Propensity estimation output used to build matching distance.
    pub propensity: PropensityScoreEstimate,
}

/// Errors for in-crate preprocessing and estimation.
#[derive(Debug, Clone)]
pub enum EstimationError {
    EmptyInput,
    MissingValuesNotAllowed {
        covariate: String,
    },
    NonFiniteNumericCovariate {
        record_id: String,
        covariate: String,
    },
    MixedCovariateType {
        covariate: String,
    },
    NoObservedValuesForCovariate {
        covariate: String,
    },
    NoCovariatesAvailable,
    InvalidElasticNetConfig,
    LinearSystemSolveFailed,
    InvalidCovarianceMatrix,
    MahalanobisConfigBuildFailed,
}

/// Build propensity-score distance config directly from covariate records.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or model fitting fails.
pub fn prepare_propensity_distance_config<R: CovariateRecord>(
    anchors: &[R],
    candidates: &[R],
    config: &PropensityScoreConfig,
) -> Result<PropensityDistancePreparation, EstimationError> {
    if anchors.is_empty() || candidates.is_empty() {
        return Err(EstimationError::EmptyInput);
    }

    let records = anchors.iter().chain(candidates.iter()).collect_vec();
    let matrix = build_covariate_matrix(&records, &config.encoding)?;

    let labels = std::iter::repeat_n(1.0_f64, anchors.len())
        .chain(std::iter::repeat_n(0.0_f64, candidates.len()))
        .collect_vec();
    let (fit, estimator_name) = match &config.estimator {
        PropensityEstimator::GlmLogit => (
            fit_logistic_regression(&matrix, &labels, &config.logistic)?,
            "glm_logit",
        ),
        PropensityEstimator::ElasticNetLogit(elastic_net) => (
            fit_elastic_net_logistic(&matrix, &labels, elastic_net)?,
            "elastic_net_logit",
        ),
    };

    let probabilities = matrix
        .row_ids
        .iter()
        .cloned()
        .zip(fit.probabilities.iter().copied())
        .collect::<RapidHashMap<_, _>>();
    let linear_predictor = matrix
        .row_ids
        .iter()
        .cloned()
        .zip(fit.linear_predictor.iter().copied())
        .collect::<RapidHashMap<_, _>>();
    let distance_scores = match config.output_scale {
        PropensityScoreOutputScale::Probability => probabilities.clone(),
        PropensityScoreOutputScale::LinearPredictor => linear_predictor.clone(),
    };

    let estimate = PropensityScoreEstimate {
        probabilities,
        linear_predictor,
        distance_scores: distance_scores.clone(),
        coefficients: fit.coefficients,
        column_names: matrix.column_names,
        estimator: estimator_name,
        converged: fit.converged,
        iterations: fit.iterations,
    };
    let distance_config = DistanceConfig::propensity_score_map(distance_scores, config.caliper);

    Ok(PropensityDistancePreparation {
        distance_config,
        estimate,
    })
}

/// Prepare Mahalanobis vectors and inverse covariance directly from covariates.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or covariance inversion fails.
pub fn prepare_mahalanobis_distance_config<R: CovariateRecord>(
    anchors: &[R],
    candidates: &[R],
    caliper: Option<DistanceCaliper>,
    config: &MahalanobisPreparationConfig,
) -> Result<MahalanobisDistancePreparation, EstimationError> {
    if anchors.is_empty() || candidates.is_empty() {
        return Err(EstimationError::EmptyInput);
    }

    let mut encoding = config.encoding.clone();
    encoding.include_intercept = false;
    let records = anchors.iter().chain(candidates.iter()).collect_vec();
    let mut matrix = build_covariate_matrix(&records, &encoding)?;
    if matches!(config.transform, MahalanobisTransform::Rank) {
        rank_transform_columns(&mut matrix);
    }
    let covariance = covariance_matrix(
        &matrix,
        anchors.len(),
        candidates.len(),
        config.covariance_strategy,
    )?;
    let inverse_covariance =
        invert_matrix_with_ridge(&covariance, config.ridge, config.max_ridge_attempts)?;

    let vectors = matrix
        .row_ids
        .iter()
        .enumerate()
        .map(|(row_idx, id)| (id.clone(), matrix.row(row_idx).to_vec()))
        .collect::<RapidHashMap<_, _>>();
    let dimension = matrix.n_cols;
    let distance_config = DistanceConfig::mahalanobis_map(
        vectors.clone(),
        inverse_covariance.clone(),
        dimension,
        caliper,
    )
    .map_err(|_| EstimationError::MahalanobisConfigBuildFailed)?;

    Ok(MahalanobisDistancePreparation {
        distance_config,
        vectors,
        inverse_covariance,
        dimension,
        column_names: matrix.column_names,
    })
}

/// Run matching with propensity scores estimated internally from covariates.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or model fitting fails.
pub fn estimate_propensity_and_match<
    R: MatchingRecord + CovariateRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
>(
    anchors: &[R],
    candidates: &[R],
    criteria: &MatchingCriteria,
    strategy: S,
    ratio_fallback: &[MatchRatio],
    extra_constraints: &G,
    propensity_config: &PropensityScoreConfig,
) -> Result<PropensityMatchedOutcome, EstimationError> {
    let prepared = prepare_propensity_distance_config(anchors, candidates, propensity_config)?;
    let outcome = match_standard(
        anchors,
        candidates,
        StandardMatchRequest {
            criteria,
            strategy,
            constraints: extra_constraints,
            ratio_fallback,
            distance_config: Some(&prepared.distance_config),
        },
    );

    Ok(PropensityMatchedOutcome {
        outcome,
        propensity: prepared.estimate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::matching::DeterministicSelection;
    use crate::types::{BalanceRecord, CovariateValue, MatchDiagnostics, MatchedPair};
    use chrono::NaiveDate;

    fn record(id: &str, birth_date: NaiveDate, age: f64, region: &str) -> BalanceRecord {
        let mut row = BalanceRecord::new(id, birth_date);
        row.covariates
            .insert("age".to_string(), CovariateValue::Numeric(age));
        row.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical(region.to_string()),
        );
        row
    }

    #[test]
    fn propensity_distance_preparation_returns_finite_scores() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "south"),
        ];

        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("propensity estimation should succeed");
        assert_eq!(prepared.estimate.estimator, "glm_logit");
        assert_eq!(prepared.estimate.distance_scores.len(), 4);
        assert!(
            prepared
                .estimate
                .distance_scores
                .values()
                .all(|value| value.is_finite())
        );

        let anchor_mean = f64::midpoint(
            prepared.estimate.probabilities["a1"],
            prepared.estimate.probabilities["a2"],
        );
        let candidate_mean = f64::midpoint(
            prepared.estimate.probabilities["c1"],
            prepared.estimate.probabilities["c2"],
        );
        assert!(anchor_mean > candidate_mean);
    }

    #[test]
    fn elastic_net_propensity_estimator_runs_and_shrinks_coefficients() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
            record("a3", date(2010, 1, 3), 12.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 4), 1.0, "south"),
            record("c2", date(2010, 1, 5), 2.0, "south"),
            record("c3", date(2010, 1, 6), 3.0, "south"),
        ];

        let glm_prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("glm propensity estimation should succeed");
        let elastic_net_prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::builder()
                .estimator(PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::builder()
                        .lambda(0.5)
                        .alpha(1.0)
                        .build(),
                ))
                .build(),
        )
        .expect("elastic-net propensity estimation should succeed");

        assert_eq!(elastic_net_prepared.estimate.estimator, "elastic_net_logit");
        let glm_abs_sum = glm_prepared
            .estimate
            .coefficients
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                (glm_prepared.estimate.column_names[idx] != "intercept").then_some(value.abs())
            })
            .sum::<f64>();
        let elastic_abs_sum = elastic_net_prepared
            .estimate
            .coefficients
            .iter()
            .enumerate()
            .filter_map(|(idx, value)| {
                (elastic_net_prepared.estimate.column_names[idx] != "intercept")
                    .then_some(value.abs())
            })
            .sum::<f64>();
        assert!(elastic_abs_sum <= glm_abs_sum + 1e-8);
    }

    #[test]
    fn elastic_net_respects_linear_predictor_output_scale() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "south"),
        ];

        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::builder()
                .estimator(PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::default(),
                ))
                .output_scale(PropensityScoreOutputScale::LinearPredictor)
                .build(),
        )
        .expect("elastic-net propensity estimation should succeed");
        assert_eq!(prepared.estimate.estimator, "elastic_net_logit");
        for (id, score) in &prepared.estimate.distance_scores {
            assert!(
                (*score - prepared.estimate.linear_predictor[id]).abs() < 1e-12,
                "distance score should equal linear predictor for {id}"
            );
        }
    }

    #[test]
    fn elastic_net_rejects_invalid_config() {
        let anchors = vec![record("a1", date(2010, 1, 1), 10.0, "north")];
        let candidates = vec![record("c1", date(2010, 1, 2), 1.0, "south")];

        let err = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::builder()
                .estimator(PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::builder()
                        .probability_clip(0.75)
                        .build(),
                ))
                .build(),
        )
        .expect_err("invalid elastic-net configuration should fail");
        assert!(matches!(err, EstimationError::InvalidElasticNetConfig));
    }

    #[test]
    fn estimated_propensity_matching_wrapper_runs() {
        let anchors = vec![record("a1", date(2010, 1, 1), 10.0, "north")];
        let candidates = vec![
            record("c1", date(2010, 1, 1), 9.5, "north"),
            record("c2", date(2010, 1, 2), 1.5, "south"),
        ];
        let criteria = MatchingCriteria::default();
        let matched = estimate_propensity_and_match(
            &anchors,
            &candidates,
            &criteria,
            DeterministicSelection,
            &[],
            &(),
            &PropensityScoreConfig::builder()
                .caliper(DistanceCaliper::new(10.0).expect("valid positive caliper"))
                .build(),
        )
        .expect("matching wrapper should succeed");

        assert_eq!(matched.propensity.distance_scores.len(), 3);
        assert_eq!(matched.outcome.matched_cases, 1);
        assert_eq!(matched.outcome.pairs.len(), 1);
    }

    #[test]
    fn mahalanobis_preparation_produces_config() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "south"),
        ];

        let prepared = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            Some(DistanceCaliper::new(5.0).expect("valid positive caliper")),
            &MahalanobisPreparationConfig::default(),
        )
        .expect("mahalanobis preparation should succeed");
        assert_eq!(prepared.dimension, prepared.column_names.len());
        assert_eq!(prepared.vectors.len(), 4);
        assert_eq!(
            prepared.inverse_covariance.len(),
            prepared.dimension * prepared.dimension
        );
    }

    #[test]
    fn mahalanobis_preparation_supports_rank_transform_and_full_sample_covariance() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 10.0, "south"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "north"),
        ];

        let prepared = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            Some(DistanceCaliper::new(5.0).expect("valid positive caliper")),
            &MahalanobisPreparationConfig::builder()
                .covariance_strategy(MahalanobisCovarianceStrategy::FullSample)
                .transform(MahalanobisTransform::Rank)
                .build(),
        )
        .expect("rank + full-sample config should succeed");
        assert_eq!(prepared.vectors.len(), 4);
        assert!(
            prepared
                .inverse_covariance
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn covariance_strategy_changes_covariance_matrix() {
        let matrix = CovariateMatrix {
            row_ids: vec![
                "a1".to_string(),
                "a2".to_string(),
                "c1".to_string(),
                "c2".to_string(),
            ],
            column_names: vec!["x".to_string()],
            values: vec![0.0, 2.0, 10.0, 12.0],
            n_rows: 4,
            n_cols: 1,
        };

        let pooled = covariance_matrix(
            &matrix,
            2,
            2,
            MahalanobisCovarianceStrategy::PooledWithinGroups,
        )
        .expect("pooled covariance should compute");
        let full = covariance_matrix(&matrix, 2, 2, MahalanobisCovarianceStrategy::FullSample)
            .expect("full covariance should compute");

        assert!((pooled[0] - full[0]).abs() > 1e-12);
        assert!((pooled[0] - 2.0).abs() < 1e-12);
        assert!((full[0] - 34.666_666_666_666_664).abs() < 1e-12);
    }

    #[test]
    fn rank_transform_columns_uses_average_ranks_for_ties() {
        let mut matrix = CovariateMatrix {
            row_ids: vec!["r1".to_string(), "r2".to_string(), "r3".to_string()],
            column_names: vec!["x".to_string(), "y".to_string()],
            values: vec![
                5.0, 10.0, //
                5.0, 30.0, //
                9.0, 20.0, //
            ],
            n_rows: 3,
            n_cols: 2,
        };

        rank_transform_columns(&mut matrix);

        assert!((matrix.values[0] - 1.5).abs() < 1e-12);
        assert!((matrix.values[2] - 1.5).abs() < 1e-12);
        assert!((matrix.values[4] - 3.0).abs() < 1e-12);
        assert!((matrix.values[1] - 1.0).abs() < 1e-12);
        assert!((matrix.values[3] - 3.0).abs() < 1e-12);
        assert!((matrix.values[5] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn missing_value_policy_error_rejects_missing_covariate() {
        let mut anchor = BalanceRecord::new("a1", date(2010, 1, 1));
        anchor
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(5.0));
        let candidate = BalanceRecord::new("c1", date(2010, 1, 1));

        let config = PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".to_string()])
                    .include_intercept(true)
                    .build(),
            )
            .build();

        let err = prepare_propensity_distance_config(&[anchor], &[candidate], &config)
            .expect_err("missing values should fail");
        assert!(matches!(
            err,
            EstimationError::MissingValuesNotAllowed { .. }
        ));
    }

    #[test]
    fn impute_policy_handles_missing_values() {
        let mut anchor = BalanceRecord::new("a1", date(2010, 1, 1));
        anchor
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(5.0));
        let candidate = BalanceRecord::new("c1", date(2010, 1, 1));

        let config = PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".to_string()])
                    .include_intercept(true)
                    .missing_value_policy(MissingValuePolicy::Impute)
                    .build(),
            )
            .build();

        let prepared = prepare_propensity_distance_config(&[anchor], &[candidate], &config)
            .expect("impute mode should succeed");
        assert_eq!(prepared.estimate.distance_scores.len(), 2);
    }

    #[test]
    fn helper_type_is_constructible_for_external_use() {
        let _unused = PropensityMatchedOutcome {
            outcome: MatchOutcome {
                pairs: vec![MatchedPair::new("a", "c")],
                unmatched_cases: 0,
                used_controls: 1,
                matched_cases: 1,
                avg_controls_per_case: 1.0,
                diagnostics: MatchDiagnostics::default(),
            },
            propensity: PropensityScoreEstimate {
                probabilities: RapidHashMap::default(),
                linear_predictor: RapidHashMap::default(),
                distance_scores: RapidHashMap::default(),
                coefficients: Vec::new(),
                column_names: Vec::new(),
                estimator: "glm_logit",
                converged: true,
                iterations: 0,
            },
        };
    }
}
