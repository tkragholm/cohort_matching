//! Encoding covariate records into a numeric design matrix.

use super::EstimationError;
use super::config::{CovariateEncodingConfig, MissingValuePolicy};
use crate::matching::{CovariateRecord, to_f64};
use crate::types::CovariateValue;
use itertools::Itertools;
use std::collections::BTreeSet;

const MISSING_LEVEL: &str = "__missing__";

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
    pub(super) fn row(&self, row_idx: usize) -> &[f64] {
        let start = row_idx * self.n_cols;
        let end = start + self.n_cols;
        &self.values[start..end]
    }
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

pub(super) fn build_covariate_matrix<R: CovariateRecord>(
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
