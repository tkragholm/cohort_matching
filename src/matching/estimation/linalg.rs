//! Dense solves and inversions, with a ridge added when the system is singular.

use super::{EstimationError, RidgeReport};
use epistat_linalg::{Attempt, Method, Ridge, inverse, solve};
use faer::{Mat, MatRef};

/// A pivot at most this in absolute value makes a factorisation count as failed.
const MIN_PIVOT: f64 = 1e-14;

/// First ridge of the escalation: later ridges grow tenfold each.
const FIRST_RIDGE: f64 = 1e-8;

/// Solves `matrix x = rhs` for a row-major square `matrix`: once as given, then
/// with a ridge of 1e-8 added to the diagonal, growing tenfold, seven times.
/// Returns the solution and the ridge that was added (0 when none was).
pub(super) fn solve_with_ridge_retry(
    matrix: &[f64],
    rhs: &[f64],
    dimension: usize,
) -> Result<(Vec<f64>, f64), EstimationError> {
    if matrix.len() != dimension.saturating_mul(dimension) || rhs.len() != dimension {
        return Err(EstimationError::LinearSystemSolveFailed);
    }
    if dimension == 0 {
        return Ok((Vec::new(), 0.0));
    }
    let attempt = Attempt::ridged(
        Method::PartialPivLu,
        Ridge {
            start: FIRST_RIDGE,
            growth: 10.0,
            tries: 7,
        },
    )
    .with_min_pivot(MIN_PIVOT);
    let a = MatRef::from_row_major_slice(matrix, dimension, dimension);
    let b = MatRef::from_column_major_slice(rhs, dimension, 1);
    let solution = solve(a, b, &[attempt]).map_err(|_| EstimationError::LinearSystemSolveFailed)?;
    let x = (0..dimension).map(|row| solution.x[(row, 0)]).collect();
    Ok((x, solution.ridge))
}

/// Whether a plain factorisation of the row-major square `matrix` finds a pivot
/// at most 1e-14 in absolute value.
pub(super) fn is_singular(matrix: &[f64], dimension: usize) -> bool {
    if dimension == 0 || matrix.len() != dimension.saturating_mul(dimension) {
        return false;
    }
    let a = MatRef::from_row_major_slice(matrix, dimension, dimension);
    let plain = [Attempt::plain(Method::PartialPivLu).with_min_pivot(MIN_PIVOT)];
    inverse(a, &plain).is_err()
}

/// Inverts a row-major square `matrix` after adding `ridge` to the diagonal.
///
/// Up to `max_attempts` inversions are tried (at least one). A positive `ridge`
/// is added from the first try and grows tenfold after each failure. A ridge of
/// zero gives a plain inversion first, then 1e-8, growing tenfold.
///
/// Returns the inverse and a [`RidgeReport`]: the ridge that was finally added,
/// and whether the matrix without any ridge had a pivot at most 1e-14.
pub(super) fn invert_matrix_with_ridge(
    matrix: &[f64],
    ridge: f64,
    max_attempts: usize,
) -> Result<(Vec<f64>, RidgeReport), EstimationError> {
    let dimension = to_dimension(matrix.len()).ok_or(EstimationError::InvalidCovarianceMatrix)?;
    if dimension == 0 {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }

    let attempts = max_attempts.max(1);
    let start = ridge.max(0.0);
    let a = MatRef::from_row_major_slice(matrix, dimension, dimension);
    let plain = [Attempt::plain(Method::PartialPivLu).with_min_pivot(MIN_PIVOT)];
    if start > 0.0 {
        let singular_without_ridge = inverse(a, &plain).is_err();
        let mut current = start;
        for _ in 0..attempts {
            let mut adjusted = a.to_owned();
            for idx in 0..dimension {
                adjusted[(idx, idx)] += current;
            }
            if let Ok(solution) = inverse(adjusted.as_ref(), &plain) {
                let report = RidgeReport {
                    ridge: current,
                    singular_without_ridge,
                };
                return Ok((row_major(&solution.x, dimension), report));
            }
            current *= 10.0;
        }
        return Err(EstimationError::InvalidCovarianceMatrix);
    }

    let tries = u32::try_from(attempts - 1).unwrap_or(u32::MAX);
    let attempt = Attempt::ridged(
        Method::PartialPivLu,
        Ridge {
            start: FIRST_RIDGE,
            growth: 10.0,
            tries,
        },
    )
    .with_min_pivot(MIN_PIVOT);
    inverse(a, &[attempt])
        .map(|solution| {
            let report = RidgeReport {
                ridge: solution.ridge,
                singular_without_ridge: solution.ridge > 0.0,
            };
            (row_major(&solution.x, dimension), report)
        })
        .map_err(|_| EstimationError::InvalidCovarianceMatrix)
}

fn row_major(matrix: &Mat<f64>, dimension: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(dimension * dimension);
    for row in 0..dimension {
        for col in 0..dimension {
            out.push(matrix[(row, col)]);
        }
    }
    out
}

fn to_dimension(length: usize) -> Option<usize> {
    // The side of the square this flattened matrix is, or None if it is not
    // square.
    let root = length.isqrt();
    (root * root == length).then_some(root)
}
