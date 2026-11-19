//! Configuration for covariate encoding, propensity estimation and Mahalanobis preparation.

use crate::types::DistanceCaliper;

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
    /// Convergence tolerance on the largest variance-weighted squared coefficient change.
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
///
/// The fit minimises the mean log-loss plus
/// `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)`, with the intercept
/// unpenalised and the covariates used as given, not standardised. This is the
/// objective of R's `glmnet(family = "binomial", standardize = FALSE)`, and the
/// coefficients agree with it to about 1e-7
/// (`tests/glmnet_reference.rs`). `glmnet`'s default `standardize = TRUE`
/// penalises the standardised coefficients and gives different estimates.
///
/// The optimiser is IRLS with cyclic coordinate descent on the penalised
/// weighted least-squares approximation, as in `glmnet` (Friedman, Hastie and
/// Tibshirani 2010, doi:10.18637/jss.v033.i01). It stops when the largest
/// change of a coefficient, squared and weighted by the weighted variance of its
/// column, is at most `tolerance`, so the rule does not depend on covariate
/// scale. With a column standardised to unit weighted variance, the default
/// 1e-20 corresponds to a coefficient change of about 1e-10. The defaults
/// converge on correlated designs and on covariates of large scale. [`PropensityScoreEstimate::converged`](crate::PropensityScoreEstimate::converged)
/// is `false` when `max_iter` IRLS steps are not enough.
#[derive(bon::Builder, Debug, Clone, Copy)]
pub struct ElasticNetLogisticConfig {
    /// Maximum IRLS steps.
    #[builder(default = 100_usize)]
    pub max_iter: usize,
    /// Convergence tolerance on the largest variance-weighted squared coefficient change.
    #[builder(default = 1e-20_f64)]
    pub tolerance: f64,
    /// Elastic-net penalty strength.
    #[builder(default = 1e-2_f64)]
    pub lambda: f64,
    /// Elastic-net mixing parameter (0 = ridge, 1 = lasso).
    #[builder(default = 0.5_f64)]
    pub alpha: f64,
    /// Numeric stability clipping for predicted probabilities.
    #[builder(default = 1e-8_f64)]
    pub probability_clip: f64,
}

impl Default for ElasticNetLogisticConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tolerance: 1e-20,
            lambda: 1e-2,
            alpha: 0.5,
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
