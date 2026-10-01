//! Logistic propensity models: IRLS with a ridge penalty, and elastic net by proximal gradient.

use super::EstimationError;
use super::config::{ElasticNetLogisticConfig, LogisticRegressionConfig};
use super::encoding::CovariateMatrix;
use super::linalg::solve_with_ridge_retry;
use crate::matching::to_f64;

#[derive(Debug, Clone)]
pub(super) struct LogisticFit {
    pub(super) coefficients: Vec<f64>,
    pub(super) linear_predictor: Vec<f64>,
    pub(super) probabilities: Vec<f64>,
    pub(super) converged: bool,
    pub(super) iterations: usize,
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

    refresh_predictions(
        matrix,
        &coefficients,
        config.probability_clip,
        &mut linear_predictor,
        &mut probabilities,
    );

    Ok(LogisticFit {
        coefficients,
        linear_predictor,
        probabilities,
        converged,
        iterations,
    })
}

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
        refresh_predictions(
            matrix,
            &coefficients,
            config.probability_clip,
            &mut linear_predictor,
            &mut probabilities,
        );

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

    refresh_predictions(
        matrix,
        &coefficients,
        config.probability_clip,
        &mut linear_predictor,
        &mut probabilities,
    );

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

/// Recompute the linear predictor and the clipped probabilities in place.
///
/// The same nine lines appeared four times: twice inside an IRLS loop and
/// twice after one, to leave the caller holding predictions that match the
/// coefficients it is about to return. Four copies of a numeric kernel is four
/// places for a clip or a dot product to drift apart.
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
