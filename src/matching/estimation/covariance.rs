//! Covariance matrices and the rank transform for Mahalanobis preparation.

use super::EstimationError;
use super::config::MahalanobisCovarianceStrategy;
use super::encoding::CovariateMatrix;
use crate::matching::to_f64;
use itertools::Itertools;

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

pub(super) fn covariance_matrix(
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

pub(super) fn rank_transform_columns(matrix: &mut CovariateMatrix) {
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
