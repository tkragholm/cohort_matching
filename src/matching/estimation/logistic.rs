//! Logistic propensity models: IRLS with a ridge penalty, and elastic net by coordinate descent.

use super::config::{ElasticNetLogisticConfig, LogisticRegressionConfig};
use super::encoding::CovariateMatrix;
use super::linalg::{is_singular, solve_with_ridge_retry};
use super::{EstimationError, RidgeReport};
use crate::matching::to_f64;

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

/// Inner coordinate-descent passes allowed per IRLS step.
const MAX_INNER_PASSES: usize = 10_000;
/// Step halvings allowed when an IRLS step raises the penalised objective.
const MAX_HALVINGS: usize = 30;

/// Elastic-net logistic regression by IRLS with cyclic coordinate descent.
///
/// The objective is the mean log-loss plus
/// `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)`, with the intercept
/// unpenalised and the covariates used as given. It is the objective of R's
/// `glmnet(family = "binomial", standardize = FALSE)`.
///
/// Each outer step takes the quadratic approximation of the log-loss at the
/// current coefficients: weights `w = p (1 - p)` and working response
/// `z = eta + (y - p) / w`. The penalised weighted least-squares problem
/// `1 / (2 n) * sum w (z - x b)^2 + penalty` is then solved by cyclic
/// coordinate descent. For coordinate `j`, with `v_j = mean(w x_j^2)`,
///
/// ```text
/// b_j = S(mean(w x_j (z - x b + x_j b_j)), lambda * alpha) / (v_j + lambda * (1 - alpha))
/// ```
///
/// where `S` is soft-thresholding; the intercept is updated without it and
/// without the ridge term. The residual is updated in place after each
/// coordinate, so a pass costs `O(n p)`. If a step raises the penalised
/// objective it is halved towards the previous coefficients.
///
/// A coordinate pass stops when `max_j v_j * (change in b_j)^2` is at most
/// `tolerance`, the rule of `glmnet`'s `thresh`. The weighting makes the
/// criterion independent of the scale of a column. The outer loop stops on the
/// same quantity measured over a whole IRLS step. Because the penalised
/// objective is strictly convex in the quadratic approximation, the pass count
/// does not depend on the scale of the covariates and grows only mildly with
/// their correlation.
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

    let alpha = config.alpha.clamp(0.0, 1.0);
    let lambda = config.lambda.max(0.0);
    let l1 = lambda * alpha;
    let l2 = lambda * (1.0 - alpha);
    let n = matrix.n_rows;
    let p = matrix.n_cols;
    let n_inv = 1.0 / to_f64(n);

    // Column-major copy, so a coordinate update runs over contiguous memory.
    let mut columns = vec![0.0_f64; n * p];
    for row_idx in 0..n {
        for (col_idx, value) in matrix.row(row_idx).iter().enumerate() {
            columns[col_idx * n + row_idx] = *value;
        }
    }
    let is_intercept: Vec<bool> = matrix
        .column_names
        .iter()
        .map(|name| name == "intercept")
        .collect();

    let mut coefficients = vec![0.0_f64; p];
    let mut linear_predictor = vec![0.0_f64; n];
    let mut probabilities = vec![0.5_f64; n];
    let mut weights = vec![0.0_f64; n];
    let mut residual = vec![0.0_f64; n];
    let mut variances = vec![0.0_f64; p];
    let mut converged = false;
    let mut iterations = 0usize;

    refresh_predictions(
        matrix,
        &coefficients,
        config.probability_clip,
        &mut linear_predictor,
        &mut probabilities,
    );
    let mut current_objective = penalised_objective(
        &linear_predictor,
        labels,
        &coefficients,
        &is_intercept,
        l1,
        l2,
    );

    for iter_idx in 0..config.max_iter {
        iterations = iter_idx + 1;

        for row_idx in 0..n {
            let probability = probabilities[row_idx];
            let weight = (probability * (1.0 - probability)).max(config.probability_clip);
            weights[row_idx] = weight;
            residual[row_idx] = (labels[row_idx] - probability) / weight;
        }
        for col_idx in 0..p {
            let column = &columns[col_idx * n..(col_idx + 1) * n];
            variances[col_idx] = column
                .iter()
                .zip(&weights)
                .map(|(x, w)| w * x * x)
                .sum::<f64>()
                * n_inv;
        }

        let start = coefficients.clone();
        let mut inner_converged = false;
        for _ in 0..MAX_INNER_PASSES {
            let mut max_change = 0.0_f64;
            for col_idx in 0..p {
                let variance = variances[col_idx];
                if variance <= 0.0 {
                    continue;
                }
                let column = &columns[col_idx * n..(col_idx + 1) * n];
                let previous = coefficients[col_idx];
                let gradient = column
                    .iter()
                    .zip(&weights)
                    .zip(&residual)
                    .map(|((x, w), r)| w * x * r)
                    .sum::<f64>()
                    * n_inv;
                let target = variance.mul_add(previous, gradient);
                let updated = if is_intercept[col_idx] {
                    target / variance
                } else {
                    soft_threshold(target, l1) / (variance + l2)
                };
                let change = updated - previous;
                if change != 0.0 {
                    for (r, x) in residual.iter_mut().zip(column) {
                        *r -= change * x;
                    }
                    coefficients[col_idx] = updated;
                }
                max_change = max_change.max(variance * change * change);
            }
            if max_change <= config.tolerance {
                inner_converged = true;
                break;
            }
        }

        // Halve the step towards the previous coefficients until the penalised
        // objective does not rise.
        let full_step = coefficients.clone();
        let mut accepted = false;
        let mut factor = 1.0_f64;
        for _ in 0..=MAX_HALVINGS {
            for col_idx in 0..p {
                coefficients[col_idx] =
                    start[col_idx] + factor * (full_step[col_idx] - start[col_idx]);
            }
            refresh_predictions(
                matrix,
                &coefficients,
                config.probability_clip,
                &mut linear_predictor,
                &mut probabilities,
            );
            let candidate = penalised_objective(
                &linear_predictor,
                labels,
                &coefficients,
                &is_intercept,
                l1,
                l2,
            );
            if candidate <= current_objective + 1e-14 * current_objective.abs().max(1.0) {
                current_objective = candidate;
                accepted = true;
                break;
            }
            factor *= 0.5;
        }
        if !accepted {
            coefficients.copy_from_slice(&start);
            refresh_predictions(
                matrix,
                &coefficients,
                config.probability_clip,
                &mut linear_predictor,
                &mut probabilities,
            );
            break;
        }

        let outer_change = (0..p)
            .map(|col_idx| {
                let change = coefficients[col_idx] - start[col_idx];
                variances[col_idx] * change * change
            })
            .fold(0.0_f64, f64::max);
        if outer_change <= config.tolerance && inner_converged {
            converged = true;
            break;
        }
    }

    Ok(LogisticFit {
        ridge: None,
        coefficients,
        linear_predictor,
        probabilities,
        converged,
        iterations,
    })
}

/// Mean log-loss plus the elastic-net penalty on the non-intercept coefficients.
fn penalised_objective(
    linear_predictor: &[f64],
    labels: &[f64],
    coefficients: &[f64],
    is_intercept: &[bool],
    l1: f64,
    l2: f64,
) -> f64 {
    let loss = linear_predictor
        .iter()
        .zip(labels)
        .map(|(eta, y)| eta.max(0.0) + (-eta.abs()).exp().ln_1p() - y * eta)
        .sum::<f64>()
        / to_f64(linear_predictor.len());
    let penalty = coefficients
        .iter()
        .zip(is_intercept)
        .filter(|(_, intercept)| !**intercept)
        .map(|(b, _)| l1.mul_add(b.abs(), 0.5 * l2 * b * b))
        .sum::<f64>();
    loss + penalty
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
