//! Logistic propensity models: IRLS with a ridge penalty, and elastic net by coordinate descent.

use super::config::{ElasticNetLogisticConfig, LogisticRegressionConfig};
use super::encoding::CovariateMatrix;
use super::linalg::{is_singular, solve_with_ridge_retry};
use super::{EstimationError, RidgeReport};
use epistat_glm::{ElasticNet, Family};

#[derive(Debug, Clone)]
pub(super) struct LogisticFit {
    pub(super) coefficients: Vec<f64>,
    pub(super) linear_predictor: Vec<f64>,
    pub(super) probabilities: Vec<f64>,
    pub(super) converged: bool,
    pub(super) iterations: usize,
    /// The ridge of the linear solves; `None` for the elastic net, which solves none.
    pub(super) ridge: Option<RidgeReport>,
}

pub(super) fn fit_logistic_regression(
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
    let mut ridge = RidgeReport::default();

    for iter_idx in 0..config.max_iter {
        iterations = iter_idx + 1;
        refresh_predictions(
            matrix,
            &coefficients,
            config.probability_clip,
            &mut linear_predictor,
            &mut probabilities,
        );

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

        if iter_idx == 0 {
            // At the start every weight is 1/4, so this tests the design alone.
            ridge.singular_without_ridge = is_singular(&normal_matrix, matrix.n_cols);
        }
        for diag in 0..matrix.n_cols {
            if matrix.column_names[diag] == "intercept" {
                continue;
            }
            normal_matrix[diag * matrix.n_cols + diag] += config.l2_penalty.max(0.0);
        }

        let (updated, added) =
            solve_with_ridge_retry(&normal_matrix, &weighted_target, matrix.n_cols)?;
        ridge.ridge = ridge.ridge.max(added);
        ridge.singular_without_ridge |= added > 0.0;
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

    refresh_predictions(
        matrix,
        &coefficients,
        config.probability_clip,
        &mut linear_predictor,
        &mut probabilities,
    );

    Ok(LogisticFit {
        ridge: Some(ridge),
        coefficients,
        linear_predictor,
        probabilities,
        converged,
        iterations,
    })
}

/// Elastic-net logistic regression by IRLS with cyclic coordinate descent,
/// fitted by [`epistat_glm::ElasticNet`].
///
/// The objective is the mean log-loss plus
/// `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)`, with the intercept
/// unpenalised and the covariates used as given. It is the objective of R's
/// `glmnet(family = "binomial", standardize = FALSE)`. The column named
/// `intercept` has penalty factor 0 and every other column 1. `alpha` is
/// clamped to `[0, 1]` and a negative `lambda` is taken as 0.
///
/// The coordinate pass stops when `max_j v_j * (change in b_j)^2` is at most
/// `tolerance`, the rule of `glmnet`'s `thresh`, and the outer loop stops on the
/// same quantity measured over a whole IRLS step. `converged` is false when
/// `max_iter` steps or the pass limit were reached first.
///
/// Reference: Friedman, Hastie and Tibshirani (2010), Regularization paths for
/// generalized linear models via coordinate descent, Journal of Statistical
/// Software 33(1), doi:10.18637/jss.v033.i01.
pub(super) fn fit_elastic_net_logistic(
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
        || !config.probability_clip.is_finite()
        || config.max_iter == 0
        || config.tolerance <= 0.0
        || config.probability_clip <= 0.0
        || config.probability_clip >= 0.5
        || matrix.n_rows == 0
    {
        return Err(EstimationError::InvalidElasticNetConfig);
    }

    let design = faer::MatRef::from_row_major_slice(&matrix.values, matrix.n_rows, matrix.n_cols);
    let penalty_factor: Vec<f64> = matrix
        .column_names
        .iter()
        .map(|name| if name == "intercept" { 0.0 } else { 1.0 })
        .collect();
    let mut model = ElasticNet::new(
        Family::Binomial,
        config.lambda.max(0.0),
        config.alpha.clamp(0.0, 1.0),
    );
    model.tolerance = config.tolerance;
    model.max_iterations = config.max_iter;
    model.probability_clip = config.probability_clip;
    let fit = model
        .fit(design, labels, None, None, Some(&penalty_factor), None)
        .map_err(|_| EstimationError::InvalidElasticNetConfig)?;

    Ok(LogisticFit {
        ridge: None,
        coefficients: fit.coefficients,
        linear_predictor: fit.linear_predictor,
        probabilities: fit.fitted,
        converged: fit.converged,
        iterations: fit.iterations,
    })
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

/// Recompute the linear predictor and the clipped probabilities in place.
fn refresh_predictions(
    matrix: &CovariateMatrix,
    coefficients: &[f64],
    probability_clip: f64,
    linear_predictor: &mut [f64],
    probabilities: &mut [f64],
) {
    for row_idx in 0..matrix.n_rows {
        let linear = matrix
            .row(row_idx)
            .iter()
            .zip(coefficients.iter())
            .map(|(x, beta)| x * beta)
            .sum::<f64>();
        linear_predictor[row_idx] = linear;
        probabilities[row_idx] = sigmoid_clipped(linear, probability_clip);
    }
}
