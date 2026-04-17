mod diagnostics;
mod projected;
mod report;
mod stats;

pub use diagnostics::balance_diagnostics;
pub use projected::{
    CategoricalCovariateSpec, NumericCovariateSpec, ProjectedBalanceRow,
    build_projected_balance_rows, categorical_max_abs_diff, smd_numeric_from_values, variance,
};
pub use report::{balance_report, balance_report_with_options, balance_threshold_summary};

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_ecdf_distance_stats(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    stats::ecdf_distance_stats(case_values, control_values)
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_eqq_distance_stats(case_values: &[f64], control_values: &[f64]) -> (f64, f64) {
    stats::eqq_distance_stats(case_values, control_values)
}
