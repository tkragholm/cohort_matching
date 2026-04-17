use super::constraints::ConstraintGroup;
use super::distance::DistanceConfig;
use super::engine::{StandardMatchRequest, match_standard};
use super::records::{CovariateRecord, MatchingRecord};
use super::selection::SelectionStrategy;
use crate::types::{CovariateValue, DistanceCaliper, MatchOutcome, MatchRatio, MatchingCriteria};
use itertools::Itertools;
use rapidhash::RapidHashMap;
use std::collections::BTreeSet;

const MISSING_LEVEL: &str = "__missing__";

/// Missing-value handling for in-crate covariate preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MissingValuePolicy {
    /// Reject missing values.
    #[default]
    Error,
    /// Impute numeric columns with observed means and map categorical missing values
    /// to a dedicated level.
    Impute,
}

/// Covariate encoding options for design-matrix construction.
#[derive(bon::Builder, Debug, Clone)]
pub struct CovariateEncodingConfig {
    /// Explicit covariate keys. Empty means all observed keys in deterministic order.
    #[builder(default)]
    pub covariate_keys: Vec<String>,
    /// Include an intercept column.
    #[builder(default = false)]
    pub include_intercept: bool,
    /// Drop the first categorical level for each encoded factor.
    #[builder(default = true)]
    pub drop_first_categorical_level: bool,
    /// Missing-value policy.
    #[builder(default)]
    pub missing_value_policy: MissingValuePolicy,
}

impl Default for CovariateEncodingConfig {
    fn default() -> Self {
        Self {
            covariate_keys: Vec::new(),
            include_intercept: false,
            drop_first_categorical_level: true,
            missing_value_policy: MissingValuePolicy::Error,
        }
    }
}

/// Logistic-regression settings used for propensity estimation.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct LogisticRegressionConfig {
    /// Maximum IRLS iterations.
    #[builder(default = 100_usize)]
    pub max_iter: usize,
    /// Convergence tolerance on coefficient max absolute change.
    #[builder(default = 1e-8_f64)]
    pub tolerance: f64,
    /// L2 penalty added to non-intercept coefficients.
    #[builder(default = 1e-6_f64)]
    pub l2_penalty: f64,
    /// Numeric stability clipping for predicted probabilities.
    #[builder(default = 1e-8_f64)]
    pub probability_clip: f64,
}

impl Default for LogisticRegressionConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-8,
            l2_penalty: 1e-6,
            probability_clip: 1e-8,
        }
    }
}

/// Elastic-net logistic settings for in-crate propensity estimation.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct ElasticNetLogisticConfig {
    /// Maximum proximal-gradient iterations.
    #[builder(default = 500_usize)]
    pub max_iter: usize,
    /// Convergence tolerance on coefficient max absolute change.
    #[builder(default = 1e-7_f64)]
    pub tolerance: f64,
    /// Elastic-net penalty strength.
    #[builder(default = 1e-2_f64)]
    pub lambda: f64,
    /// Elastic-net mixing parameter (0 = ridge, 1 = lasso).
    #[builder(default = 0.5_f64)]
    pub alpha: f64,
    /// Step-size scaling factor.
    #[builder(default = 1.0_f64)]
    pub step_scale: f64,
    /// Numeric stability clipping for predicted probabilities.
    #[builder(default = 1e-8_f64)]
    pub probability_clip: f64,
}

impl Default for ElasticNetLogisticConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tolerance: 1e-7,
            lambda: 1e-2,
            alpha: 0.5,
            step_scale: 1.0,
            probability_clip: 1e-8,
        }
    }
}

/// In-crate propensity estimator selection.
#[derive(Debug, Clone, Default)]
pub enum PropensityEstimator {
    /// Logistic regression (`glm`-style IRLS).
    #[default]
    GlmLogit,
    /// Elastic-net penalized logistic regression.
    ElasticNetLogit(ElasticNetLogisticConfig),
}

/// Output scale used for propensity-score distance values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropensityScoreOutputScale {
    /// Use probability scale.
    #[default]
    Probability,
    /// Use linear predictor (`logit` when logistic link is used).
    LinearPredictor,
}

/// Combined configuration for in-crate propensity score estimation.
#[derive(bon::Builder, Debug, Clone)]
pub struct PropensityScoreConfig {
    /// Covariate encoding options.
    #[builder(default = CovariateEncodingConfig::builder().include_intercept(true).build())]
    pub encoding: CovariateEncodingConfig,
    /// Estimator configuration.
    #[builder(default)]
    pub estimator: PropensityEstimator,
    /// Solver options.
    #[builder(default)]
    pub logistic: LogisticRegressionConfig,
    /// Output scale used for matching distance.
    #[builder(default)]
    pub output_scale: PropensityScoreOutputScale,
    /// Optional caliper on selected output scale.
    pub caliper: Option<DistanceCaliper>,
}

impl Default for PropensityScoreConfig {
    fn default() -> Self {
        Self {
            encoding: CovariateEncodingConfig::builder()
                .include_intercept(true)
                .build(),
            estimator: PropensityEstimator::default(),
            logistic: LogisticRegressionConfig::default(),
            output_scale: PropensityScoreOutputScale::Probability,
            caliper: None,
        }
    }
}

/// Mahalanobis preprocessing configuration using in-crate covariate encoding.
#[derive(bon::Builder, Debug, Clone)]
pub struct MahalanobisPreparationConfig {
    /// Covariate encoding options.
    #[builder(default)]
    pub encoding: CovariateEncodingConfig,
    /// Covariance estimation strategy.
    #[builder(default = MahalanobisCovarianceStrategy::PooledWithinGroups)]
    pub covariance_strategy: MahalanobisCovarianceStrategy,
    /// Optional covariate transformation before covariance estimation.
    #[builder(default = MahalanobisTransform::Raw)]
    pub transform: MahalanobisTransform,
    /// Initial ridge added to covariance diagonal before inversion.
    #[builder(default = 1e-8_f64)]
    pub ridge: f64,
    /// Number of ridge escalation attempts.
    #[builder(default = 8_usize)]
    pub max_ridge_attempts: usize,
}

impl Default for MahalanobisPreparationConfig {
    fn default() -> Self {
        Self {
            encoding: CovariateEncodingConfig::default(),
            covariance_strategy: MahalanobisCovarianceStrategy::PooledWithinGroups,
            transform: MahalanobisTransform::Raw,
            ridge: 1e-8,
            max_ridge_attempts: 8,
        }
    }
}

/// Covariance strategy used for Mahalanobis preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahalanobisCovarianceStrategy {
    /// Average treatment-group covariance matrices (pooled within groups).
    PooledWithinGroups,
    /// Covariance over the full sample after encoding.
    FullSample,
}

/// Covariate transformation used before Mahalanobis covariance estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahalanobisTransform {
    /// No transformation.
    Raw,
    /// Rank-transform each column (average ranks for ties).
    Rank,
}

/// Row-major dense covariate matrix.
#[derive(Debug, Clone)]
pub struct CovariateMatrix {
    /// Row ids aligned with matrix rows.
    pub row_ids: Vec<String>,
    /// Column names aligned with matrix columns.
    pub column_names: Vec<String>,
    /// Row-major values with length `n_rows * n_cols`.
    pub values: Vec<f64>,
    /// Number of rows.
    pub n_rows: usize,
    /// Number of columns.
    pub n_cols: usize,
}

impl CovariateMatrix {
    fn row(&self, row_idx: usize) -> &[f64] {
        let start = row_idx * self.n_cols;
        let end = start + self.n_cols;
        &self.values[start..end]
    }
}

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

#[derive(Debug, Clone)]
enum CovariateSchema {
    Numeric {
        key: String,
        mean: f64,
    },
    Categorical {
        key: String,
        emitted_levels: Vec<String>,
    },
}

#[derive(Debug, Clone)]
struct LogisticFit {
    coefficients: Vec<f64>,
    linear_predictor: Vec<f64>,
    probabilities: Vec<f64>,
    converged: bool,
    iterations: usize,
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

fn build_covariate_matrix<R: CovariateRecord>(
    records: &[&R],
    config: &CovariateEncodingConfig,
) -> Result<CovariateMatrix, EstimationError> {
    if records.is_empty() {
        return Err(EstimationError::EmptyInput);
    }

    let keys = resolve_covariate_keys(records, &config.covariate_keys);
    let mut schemas = Vec::new();
    for key in keys {
        let Some(schema) = build_schema_for_key(records, &key, config)? else {
            continue;
        };
        schemas.push(schema);
    }
    if schemas.is_empty() && !config.include_intercept {
        return Err(EstimationError::NoCovariatesAvailable);
    }

    let mut column_names = Vec::new();
    if config.include_intercept {
        column_names.push("intercept".to_string());
    }
    for schema in &schemas {
        match schema {
            CovariateSchema::Numeric { key, .. } => column_names.push(key.clone()),
            CovariateSchema::Categorical {
                key,
                emitted_levels,
                ..
            } => {
                column_names.extend(emitted_levels.iter().map(|level| format!("{key}={level}")));
            }
        }
    }

    let n_rows = records.len();
    let n_cols = column_names.len();
    let mut values = vec![0.0_f64; n_rows.saturating_mul(n_cols)];
    let row_ids = records
        .iter()
        .map(|record| record.id().to_string())
        .collect_vec();

    for (row_idx, record_ref) in records.iter().enumerate() {
        let record = *record_ref;
        let mut col_idx = 0usize;
        if config.include_intercept {
            values[row_idx * n_cols + col_idx] = 1.0;
            col_idx += 1;
        }

        for schema in &schemas {
            match schema {
                CovariateSchema::Numeric { key, mean } => {
                    let value =
                        encode_numeric_value(record, key, *mean, config.missing_value_policy)?;
                    values[row_idx * n_cols + col_idx] = value;
                    col_idx += 1;
                }
                CovariateSchema::Categorical {
                    key,
                    emitted_levels,
                } => {
                    let level = encode_categorical_level(record, key, config.missing_value_policy)?;
                    for emitted in emitted_levels {
                        values[row_idx * n_cols + col_idx] =
                            if level == emitted.as_str() { 1.0 } else { 0.0 };
                        col_idx += 1;
                    }
                }
            }
        }
    }

    let mut matrix = CovariateMatrix {
        row_ids,
        column_names,
        values,
        n_rows,
        n_cols,
    };
    drop_near_constant_columns(&mut matrix, 1e-12)?;
    Ok(matrix)
}

fn resolve_covariate_keys<R: CovariateRecord>(records: &[&R], requested: &[String]) -> Vec<String> {
    if !requested.is_empty() {
        return requested.to_vec();
    }

    records
        .iter()
        .flat_map(|record| record.covariates().keys().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect_vec()
}

fn build_schema_for_key<R: CovariateRecord>(
    records: &[&R],
    key: &str,
    config: &CovariateEncodingConfig,
) -> Result<Option<CovariateSchema>, EstimationError> {
    let mut numeric_sum = 0.0_f64;
    let mut numeric_count = 0usize;
    let mut categorical_levels = BTreeSet::new();
    let mut observed_numeric = false;
    let mut observed_categorical = false;
    let mut saw_missing = false;

    for record in records {
        match record.covariates().get(key) {
            Some(CovariateValue::Numeric(value)) => {
                if !value.is_finite() {
                    return Err(EstimationError::NonFiniteNumericCovariate {
                        record_id: record.id().to_string(),
                        covariate: key.to_string(),
                    });
                }
                observed_numeric = true;
                numeric_sum += *value;
                numeric_count += 1;
            }
            Some(CovariateValue::Categorical(value)) => {
                observed_categorical = true;
                categorical_levels.insert(value.clone());
            }
            Some(CovariateValue::Missing) | None => {
                saw_missing = true;
            }
        }
    }

    if observed_numeric && observed_categorical {
        return Err(EstimationError::MixedCovariateType {
            covariate: key.to_string(),
        });
    }

    if !observed_numeric && !observed_categorical {
        if config.covariate_keys.is_empty() {
            return Ok(None);
        }
        return Err(EstimationError::NoObservedValuesForCovariate {
            covariate: key.to_string(),
        });
    }

    if observed_numeric {
        if saw_missing && matches!(config.missing_value_policy, MissingValuePolicy::Error) {
            return Err(EstimationError::MissingValuesNotAllowed {
                covariate: key.to_string(),
            });
        }
        let mean = if numeric_count == 0 {
            0.0
        } else {
            numeric_sum / to_f64(numeric_count)
        };
        return Ok(Some(CovariateSchema::Numeric {
            key: key.to_string(),
            mean,
        }));
    }

    if saw_missing && matches!(config.missing_value_policy, MissingValuePolicy::Impute) {
        categorical_levels.insert(MISSING_LEVEL.to_string());
    }
    if saw_missing && matches!(config.missing_value_policy, MissingValuePolicy::Error) {
        return Err(EstimationError::MissingValuesNotAllowed {
            covariate: key.to_string(),
        });
    }
    let levels = categorical_levels.into_iter().collect_vec();
    if levels.is_empty() {
        return Err(EstimationError::NoObservedValuesForCovariate {
            covariate: key.to_string(),
        });
    }

    let emitted_levels = if config.drop_first_categorical_level && levels.len() > 1 {
        levels.into_iter().skip(1).collect_vec()
    } else {
        levels
    };
    Ok(Some(CovariateSchema::Categorical {
        key: key.to_string(),
        emitted_levels,
    }))
}

fn encode_numeric_value<R: CovariateRecord>(
    record: &R,
    key: &str,
    mean: f64,
    policy: MissingValuePolicy,
) -> Result<f64, EstimationError> {
    match record.covariates().get(key) {
        Some(CovariateValue::Numeric(value)) => {
            if !value.is_finite() {
                return Err(EstimationError::NonFiniteNumericCovariate {
                    record_id: record.id().to_string(),
                    covariate: key.to_string(),
                });
            }
            Ok(*value)
        }
        Some(CovariateValue::Categorical(_)) => Err(EstimationError::MixedCovariateType {
            covariate: key.to_string(),
        }),
        Some(CovariateValue::Missing) | None => match policy {
            MissingValuePolicy::Error => Err(EstimationError::MissingValuesNotAllowed {
                covariate: key.to_string(),
            }),
            MissingValuePolicy::Impute => Ok(mean),
        },
    }
}

fn encode_categorical_level<R: CovariateRecord>(
    record: &R,
    key: &str,
    policy: MissingValuePolicy,
) -> Result<String, EstimationError> {
    match record.covariates().get(key) {
        Some(CovariateValue::Categorical(value)) => Ok(value.clone()),
        Some(CovariateValue::Numeric(_)) => Err(EstimationError::MixedCovariateType {
            covariate: key.to_string(),
        }),
        Some(CovariateValue::Missing) | None => match policy {
            MissingValuePolicy::Error => Err(EstimationError::MissingValuesNotAllowed {
                covariate: key.to_string(),
            }),
            MissingValuePolicy::Impute => Ok(MISSING_LEVEL.to_string()),
        },
    }
}

fn drop_near_constant_columns(
    matrix: &mut CovariateMatrix,
    tolerance: f64,
) -> Result<(), EstimationError> {
    if matrix.n_cols == 0 {
        return Err(EstimationError::NoCovariatesAvailable);
    }

    let keep = (0..matrix.n_cols)
        .filter(|col_idx| {
            if matrix.column_names[*col_idx] == "intercept" {
                return true;
            }
            let mut min_value = f64::INFINITY;
            let mut max_value = f64::NEG_INFINITY;
            for row_idx in 0..matrix.n_rows {
                let value = matrix.values[row_idx * matrix.n_cols + *col_idx];
                min_value = min_value.min(value);
                max_value = max_value.max(value);
            }
            (max_value - min_value).abs() > tolerance
        })
        .collect_vec();

    if keep.is_empty() {
        return Err(EstimationError::NoCovariatesAvailable);
    }
    if keep.len() == matrix.n_cols {
        return Ok(());
    }

    let mut reduced = vec![0.0_f64; matrix.n_rows * keep.len()];
    for row_idx in 0..matrix.n_rows {
        for (new_col_idx, old_col_idx) in keep.iter().enumerate() {
            reduced[row_idx * keep.len() + new_col_idx] =
                matrix.values[row_idx * matrix.n_cols + *old_col_idx];
        }
    }

    matrix.column_names = keep
        .iter()
        .map(|idx| matrix.column_names[*idx].clone())
        .collect_vec();
    matrix.values = reduced;
    matrix.n_cols = matrix.column_names.len();
    Ok(())
}

fn fit_logistic_regression(
    matrix: &CovariateMatrix,
    labels: &[f64],
    config: &LogisticRegressionConfig,
) -> Result<LogisticFit, EstimationError> {
    if labels.len() != matrix.n_rows {
        return Err(EstimationError::LinearSystemSolveFailed);
    }

    let mut coefficients = vec![0.0_f64; matrix.n_cols];
    let mut linear_predictor = vec![0.0_f64; matrix.n_rows];
    let mut probabilities = vec![0.5_f64; matrix.n_rows];
    let mut converged = false;
    let mut iterations = 0usize;

    for iter_idx in 0..config.max_iter {
        iterations = iter_idx + 1;
        for row_idx in 0..matrix.n_rows {
            let row = matrix.row(row_idx);
            let linear = row
                .iter()
                .zip(coefficients.iter())
                .map(|(x, beta)| x * beta)
                .sum::<f64>();
            linear_predictor[row_idx] = linear;
            probabilities[row_idx] = sigmoid_clipped(linear, config.probability_clip);
        }

        let mut normal_matrix = vec![0.0_f64; matrix.n_cols.saturating_mul(matrix.n_cols)];
        let mut weighted_target = vec![0.0_f64; matrix.n_cols];
        for row_idx in 0..matrix.n_rows {
            let probability = probabilities[row_idx];
            let weight = (probability * (1.0 - probability)).max(config.probability_clip);
            let adjusted_response =
                linear_predictor[row_idx] + (labels[row_idx] - probability) / weight;
            let row = matrix.row(row_idx);

            for left in 0..matrix.n_cols {
                let x_left = row[left];
                weighted_target[left] =
                    (weight * x_left).mul_add(adjusted_response, weighted_target[left]);
                for (right, x_right) in row.iter().enumerate().take(left + 1) {
                    let idx = left * matrix.n_cols + right;
                    normal_matrix[idx] = (weight * x_left).mul_add(*x_right, normal_matrix[idx]);
                }
            }
        }
        for left in 0..matrix.n_cols {
            for right in (left + 1)..matrix.n_cols {
                let upper = left * matrix.n_cols + right;
                let lower = right * matrix.n_cols + left;
                normal_matrix[upper] = normal_matrix[lower];
            }
        }

        for diag in 0..matrix.n_cols {
            if matrix.column_names[diag] == "intercept" {
                continue;
            }
            normal_matrix[diag * matrix.n_cols + diag] += config.l2_penalty.max(0.0);
        }

        let updated = solve_with_ridge_retry(&normal_matrix, &weighted_target, matrix.n_cols)?;
        let max_delta = updated
            .iter()
            .zip(coefficients.iter())
            .map(|(next, prev)| (next - prev).abs())
            .fold(0.0_f64, f64::max);
        coefficients = updated;

        if max_delta <= config.tolerance {
            converged = true;
            break;
        }
    }

    for row_idx in 0..matrix.n_rows {
        let row = matrix.row(row_idx);
        let linear = row
            .iter()
            .zip(coefficients.iter())
            .map(|(x, beta)| x * beta)
            .sum::<f64>();
        linear_predictor[row_idx] = linear;
        probabilities[row_idx] = sigmoid_clipped(linear, config.probability_clip);
    }

    Ok(LogisticFit {
        coefficients,
        linear_predictor,
        probabilities,
        converged,
        iterations,
    })
}

fn fit_elastic_net_logistic(
    matrix: &CovariateMatrix,
    labels: &[f64],
    config: &ElasticNetLogisticConfig,
) -> Result<LogisticFit, EstimationError> {
    if labels.len() != matrix.n_rows {
        return Err(EstimationError::LinearSystemSolveFailed);
    }
    if !config.alpha.is_finite()
        || !config.lambda.is_finite()
        || !config.tolerance.is_finite()
        || !config.step_scale.is_finite()
        || !config.probability_clip.is_finite()
        || config.max_iter == 0
        || config.tolerance <= 0.0
        || config.step_scale <= 0.0
        || config.probability_clip <= 0.0
        || config.probability_clip >= 0.5
    {
        return Err(EstimationError::InvalidElasticNetConfig);
    }

    let alpha = config.alpha.clamp(0.0, 1.0);
    let lambda = config.lambda.max(0.0);
    let l1 = lambda * alpha;
    let l2 = lambda * (1.0 - alpha);
    let step_size = compute_elastic_net_step_size(matrix, l2, config.step_scale)
        .ok_or(EstimationError::InvalidElasticNetConfig)?;
    let n_inv = 1.0 / to_f64(matrix.n_rows);

    let mut coefficients = vec![0.0_f64; matrix.n_cols];
    let mut linear_predictor = vec![0.0_f64; matrix.n_rows];
    let mut probabilities = vec![0.5_f64; matrix.n_rows];
    let mut converged = false;
    let mut iterations = 0usize;

    for iter_idx in 0..config.max_iter {
        iterations = iter_idx + 1;
        for row_idx in 0..matrix.n_rows {
            let row = matrix.row(row_idx);
            let linear = row
                .iter()
                .zip(coefficients.iter())
                .map(|(x, beta)| x * beta)
                .sum::<f64>();
            linear_predictor[row_idx] = linear;
            probabilities[row_idx] = sigmoid_clipped(linear, config.probability_clip);
        }

        let mut gradients = vec![0.0_f64; matrix.n_cols];
        for row_idx in 0..matrix.n_rows {
            let row = matrix.row(row_idx);
            let diff = probabilities[row_idx] - labels[row_idx];
            for col_idx in 0..matrix.n_cols {
                gradients[col_idx] = (row[col_idx] * diff).mul_add(n_inv, gradients[col_idx]);
            }
        }

        let previous = coefficients.clone();
        for col_idx in 0..matrix.n_cols {
            if matrix.column_names[col_idx] == "intercept" {
                coefficients[col_idx] -= step_size * gradients[col_idx];
                continue;
            }

            let candidate = coefficients[col_idx]
                - step_size * (gradients[col_idx] + l2 * coefficients[col_idx]);
            coefficients[col_idx] = soft_threshold(candidate, step_size * l1);
        }

        let max_delta = coefficients
            .iter()
            .zip(previous.iter())
            .map(|(next, prev)| (next - prev).abs())
            .fold(0.0_f64, f64::max);
        if max_delta <= config.tolerance {
            converged = true;
            break;
        }
    }

    for row_idx in 0..matrix.n_rows {
        let row = matrix.row(row_idx);
        let linear = row
            .iter()
            .zip(coefficients.iter())
            .map(|(x, beta)| x * beta)
            .sum::<f64>();
        linear_predictor[row_idx] = linear;
        probabilities[row_idx] = sigmoid_clipped(linear, config.probability_clip);
    }

    Ok(LogisticFit {
        coefficients,
        linear_predictor,
        probabilities,
        converged,
        iterations,
    })
}

fn compute_elastic_net_step_size(
    matrix: &CovariateMatrix,
    ridge_penalty: f64,
    step_scale: f64,
) -> Option<f64> {
    if matrix.n_rows == 0 {
        return None;
    }

    let max_row_sq = (0..matrix.n_rows)
        .map(|row_idx| {
            matrix
                .row(row_idx)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
        })
        .fold(0.0_f64, f64::max);
    let lipschitz = 0.25_f64.mul_add(max_row_sq, ridge_penalty.max(0.0));
    let denom = lipschitz.max(1e-8);
    Some(step_scale / denom)
}

fn soft_threshold(value: f64, threshold: f64) -> f64 {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        0.0
    }
}

fn sigmoid_clipped(value: f64, clip: f64) -> f64 {
    let probability = if value >= 0.0 {
        let exp_neg = (-value).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = value.exp();
        exp_pos / (1.0 + exp_pos)
    };
    probability.clamp(clip, 1.0 - clip)
}

fn solve_with_ridge_retry(
    matrix: &[f64],
    rhs: &[f64],
    dimension: usize,
) -> Result<Vec<f64>, EstimationError> {
    let mut ridge = 0.0_f64;
    for attempt in 0..8 {
        let mut adjusted = matrix.to_vec();
        if attempt > 0 {
            ridge = if ridge == 0.0 { 1e-8 } else { ridge * 10.0 };
            for idx in 0..dimension {
                adjusted[idx * dimension + idx] += ridge;
            }
        }
        if let Some(solution) = solve_linear_system(&adjusted, rhs, dimension) {
            return Ok(solution);
        }
    }
    Err(EstimationError::LinearSystemSolveFailed)
}

fn solve_linear_system(matrix: &[f64], rhs: &[f64], dimension: usize) -> Option<Vec<f64>> {
    if matrix.len() != dimension.saturating_mul(dimension) || rhs.len() != dimension {
        return None;
    }
    if dimension == 0 {
        return Some(Vec::new());
    }

    let width = dimension + 1;
    let mut augmented = vec![0.0_f64; dimension * width];
    for row in 0..dimension {
        for col in 0..dimension {
            augmented[row * width + col] = matrix[row * dimension + col];
        }
        augmented[row * width + dimension] = rhs[row];
    }

    for pivot in 0..dimension {
        let mut pivot_row = pivot;
        let mut pivot_abs = augmented[pivot * width + pivot].abs();
        for row in (pivot + 1)..dimension {
            let candidate = augmented[row * width + pivot].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= 1e-14 {
            return None;
        }
        if pivot_row != pivot {
            for col in 0..width {
                augmented.swap(pivot * width + col, pivot_row * width + col);
            }
        }

        let pivot_value = augmented[pivot * width + pivot];
        for col in pivot..width {
            augmented[pivot * width + col] /= pivot_value;
        }

        for row in 0..dimension {
            if row == pivot {
                continue;
            }
            let factor = augmented[row * width + pivot];
            if factor.abs() <= 1e-18 {
                continue;
            }
            for col in pivot..width {
                augmented[row * width + col] -= factor * augmented[pivot * width + col];
            }
        }
    }

    Some(
        (0..dimension)
            .map(|row| augmented[row * width + dimension])
            .collect_vec(),
    )
}

fn pooled_covariance_matrix(
    matrix: &CovariateMatrix,
    n_anchor: usize,
    n_candidate: usize,
) -> Result<Vec<f64>, EstimationError> {
    if matrix.n_cols == 0 || matrix.n_rows != n_anchor.saturating_add(n_candidate) {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }
    if n_anchor == 0 || n_candidate == 0 {
        return Err(EstimationError::EmptyInput);
    }

    let anchor_cov = covariance_for_slice(matrix, 0, n_anchor);
    let candidate_cov = covariance_for_slice(matrix, n_anchor, n_candidate);
    let mut pooled = vec![0.0_f64; matrix.n_cols * matrix.n_cols];
    for idx in 0..pooled.len() {
        pooled[idx] = 0.5 * (anchor_cov[idx] + candidate_cov[idx]);
    }
    Ok(pooled)
}

fn covariance_matrix(
    matrix: &CovariateMatrix,
    n_anchor: usize,
    n_candidate: usize,
    strategy: MahalanobisCovarianceStrategy,
) -> Result<Vec<f64>, EstimationError> {
    match strategy {
        MahalanobisCovarianceStrategy::PooledWithinGroups => {
            pooled_covariance_matrix(matrix, n_anchor, n_candidate)
        }
        MahalanobisCovarianceStrategy::FullSample => {
            if matrix.n_cols == 0 || matrix.n_rows != n_anchor.saturating_add(n_candidate) {
                return Err(EstimationError::InvalidCovarianceMatrix);
            }
            if matrix.n_rows <= 1 {
                return Err(EstimationError::EmptyInput);
            }
            Ok(covariance_for_slice(matrix, 0, matrix.n_rows))
        }
    }
}

fn covariance_for_slice(matrix: &CovariateMatrix, start_row: usize, len: usize) -> Vec<f64> {
    let mut means = vec![0.0_f64; matrix.n_cols];
    for row_idx in start_row..(start_row + len) {
        let row = matrix.row(row_idx);
        for col_idx in 0..matrix.n_cols {
            means[col_idx] += row[col_idx];
        }
    }
    for mean in &mut means {
        *mean /= to_f64(len);
    }

    let mut cov = vec![0.0_f64; matrix.n_cols * matrix.n_cols];
    if len <= 1 {
        return cov;
    }
    for row_idx in start_row..(start_row + len) {
        let row = matrix.row(row_idx);
        for left in 0..matrix.n_cols {
            let left_centered = row[left] - means[left];
            for right in 0..=left {
                cov[left * matrix.n_cols + right] = left_centered
                    .mul_add(row[right] - means[right], cov[left * matrix.n_cols + right]);
            }
        }
    }
    let denom = to_f64(len - 1);
    for left in 0..matrix.n_cols {
        for right in 0..=left {
            let value = cov[left * matrix.n_cols + right] / denom;
            cov[left * matrix.n_cols + right] = value;
            cov[right * matrix.n_cols + left] = value;
        }
    }
    cov
}

fn rank_transform_columns(matrix: &mut CovariateMatrix) {
    if matrix.n_rows == 0 || matrix.n_cols == 0 {
        return;
    }
    let mut transformed = matrix.values.clone();

    for col_idx in 0..matrix.n_cols {
        let mut entries = (0..matrix.n_rows)
            .map(|row_idx| (row_idx, matrix.values[row_idx * matrix.n_cols + col_idx]))
            .collect_vec();
        entries.sort_by(|left, right| left.1.total_cmp(&right.1));

        let mut cursor = 0usize;
        while cursor < entries.len() {
            let mut end = cursor + 1;
            while end < entries.len() && entries[end].1.total_cmp(&entries[cursor].1).is_eq() {
                end += 1;
            }

            let low_rank = cursor + 1;
            let high_rank = end;
            let avg_rank = 0.5 * (to_f64(low_rank) + to_f64(high_rank));
            for (row_idx, _) in entries.iter().take(end).skip(cursor) {
                transformed[*row_idx * matrix.n_cols + col_idx] = avg_rank;
            }
            cursor = end;
        }
    }

    matrix.values = transformed;
}

fn invert_matrix_with_ridge(
    matrix: &[f64],
    ridge: f64,
    max_attempts: usize,
) -> Result<Vec<f64>, EstimationError> {
    let dimension = to_dimension(matrix.len()).ok_or(EstimationError::InvalidCovarianceMatrix)?;
    if dimension == 0 {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }

    let mut current_ridge = ridge.max(0.0);
    for attempt in 0..max_attempts.max(1) {
        let mut adjusted = matrix.to_vec();
        if current_ridge > 0.0 {
            for idx in 0..dimension {
                adjusted[idx * dimension + idx] += current_ridge;
            }
        }
        if let Some(inverse) = invert_matrix(&adjusted, dimension) {
            return Ok(inverse);
        }
        if attempt == 0 && current_ridge == 0.0 {
            current_ridge = 1e-8;
        } else {
            current_ridge *= 10.0;
        }
    }

    Err(EstimationError::InvalidCovarianceMatrix)
}

fn invert_matrix(matrix: &[f64], dimension: usize) -> Option<Vec<f64>> {
    if matrix.len() != dimension.saturating_mul(dimension) {
        return None;
    }

    let width = dimension * 2;
    let mut augmented = vec![0.0_f64; dimension * width];
    for row in 0..dimension {
        for col in 0..dimension {
            augmented[row * width + col] = matrix[row * dimension + col];
        }
        augmented[row * width + dimension + row] = 1.0;
    }

    for pivot in 0..dimension {
        let mut pivot_row = pivot;
        let mut pivot_abs = augmented[pivot * width + pivot].abs();
        for row in (pivot + 1)..dimension {
            let candidate = augmented[row * width + pivot].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs <= 1e-14 {
            return None;
        }
        if pivot_row != pivot {
            for col in 0..width {
                augmented.swap(pivot * width + col, pivot_row * width + col);
            }
        }

        let pivot_value = augmented[pivot * width + pivot];
        for col in 0..width {
            augmented[pivot * width + col] /= pivot_value;
        }

        for row in 0..dimension {
            if row == pivot {
                continue;
            }
            let factor = augmented[row * width + pivot];
            if factor.abs() <= 1e-18 {
                continue;
            }
            for col in 0..width {
                augmented[row * width + col] -= factor * augmented[pivot * width + col];
            }
        }
    }

    let mut inverse = vec![0.0_f64; dimension * dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            inverse[row * dimension + col] = augmented[row * width + dimension + col];
        }
    }
    Some(inverse)
}

fn to_dimension(length: usize) -> Option<usize> {
    if length == 0 {
        return Some(0);
    }
    let mut root = 1usize;
    while root.saturating_mul(root) < length {
        root += 1;
    }
    (root.saturating_mul(root) == length).then_some(root)
}

use super::to_f64;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::matching::DeterministicSelection;
    use crate::types::{BalanceRecord, MatchDiagnostics, MatchedPair};
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
