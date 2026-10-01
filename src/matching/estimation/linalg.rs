//! Dense solves and inversions, with a ridge added when the system is singular.

use super::EstimationError;
use itertools::Itertools;

pub(super) fn solve_with_ridge_retry(
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

pub(super) fn invert_matrix_with_ridge(
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
    // The side of the square this flattened matrix is, or None if it is not
    // square. Was a linear search upward for the root; `isqrt` is the same
    // answer, including for 0.
    let root = length.isqrt();
    (root * root == length).then_some(root)
}
