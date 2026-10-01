//! Balance report, threshold and diagnostic types.

use super::records::CovariateValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cohort-level balance diagnostics from match results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceDiagnostics {
    /// Proportion of anchors matched.
    pub match_rate: f64,
    /// Number of matched anchors.
    pub matched_cases: usize,
    /// Number of unmatched anchors.
    pub unmatched_cases: usize,
    /// Average candidates per matched anchor.
    pub avg_controls_per_case: f64,
    /// Counts by strata key (`anchor_count`, `candidate_count`).
    pub strata_counts: HashMap<String, (usize, usize)>,
}

impl BalanceDiagnostics {
    /// Neutral accessor for matched anchor count.
    #[must_use]
    pub const fn matched_anchors(&self) -> usize {
        self.matched_cases
    }

    /// Neutral accessor for unmatched anchor count.
    #[must_use]
    pub const fn unmatched_anchors(&self) -> usize {
        self.unmatched_cases
    }

    /// Neutral accessor for average comparators per matched anchor.
    #[must_use]
    pub const fn avg_comparators_per_anchor(&self) -> f64 {
        self.avg_controls_per_case
    }
}

/// Numeric covariate balance summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericBalance {
    /// Covariate name.
    pub name: String,
    /// Pre-match anchor mean.
    pub mean_case_pre: f64,
    /// Pre-match candidate mean.
    pub mean_control_pre: f64,
    /// Pre-match standardized mean difference.
    pub smd_pre: f64,
    /// Pre-match variance ratio (`anchor / candidate`).
    #[serde(default)]
    pub var_ratio_pre: f64,
    /// Pre-match mean absolute eCDF distance.
    #[serde(default)]
    pub ecdf_mean_diff_pre: f64,
    /// Pre-match max absolute eCDF distance.
    #[serde(default)]
    pub ecdf_max_diff_pre: f64,
    /// Pre-match mean absolute eQQ distance.
    #[serde(default)]
    pub eqq_mean_diff_pre: f64,
    /// Pre-match max absolute eQQ distance.
    #[serde(default)]
    pub eqq_max_diff_pre: f64,
    /// Post-match anchor mean.
    pub mean_case_post: f64,
    /// Post-match candidate mean.
    pub mean_control_post: f64,
    /// Post-match standardized mean difference.
    pub smd_post: f64,
    /// Post-match variance ratio (`anchor / candidate`).
    #[serde(default)]
    pub var_ratio_post: f64,
    /// Post-match mean absolute eCDF distance.
    #[serde(default)]
    pub ecdf_mean_diff_post: f64,
    /// Post-match max absolute eCDF distance.
    #[serde(default)]
    pub ecdf_max_diff_post: f64,
    /// Post-match mean absolute eQQ distance.
    #[serde(default)]
    pub eqq_mean_diff_post: f64,
    /// Post-match max absolute eQQ distance.
    #[serde(default)]
    pub eqq_max_diff_post: f64,
}

/// Categorical level balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalLevelBalance {
    /// Level name.
    pub level: String,
    /// Pre-match anchor proportion.
    pub p_case_pre: f64,
    /// Pre-match candidate proportion.
    pub p_control_pre: f64,
    /// Pre-match standardized mean difference.
    pub smd_pre: f64,
    /// Post-match anchor proportion.
    pub p_case_post: f64,
    /// Post-match candidate proportion.
    pub p_control_post: f64,
    /// Post-match standardized mean difference.
    pub smd_post: f64,
}

/// Categorical covariate balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalBalance {
    /// Covariate name.
    pub name: String,
    /// Per-level balance statistics.
    pub levels: Vec<CategoricalLevelBalance>,
    /// Pre-match Cramer's V.
    pub cramers_v_pre: f64,
    /// Post-match Cramer's V.
    pub cramers_v_post: f64,
}

/// Full balance report across numeric and categorical covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReport {
    /// Numeric covariate summaries.
    pub numeric: Vec<NumericBalance>,
    /// Categorical covariate summaries.
    pub categorical: Vec<CategoricalBalance>,
}

/// Optional transformed numeric covariates included in balance summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NumericBalanceTransform {
    /// Include only observed covariates.
    #[default]
    None,
    /// Include squared numeric terms (for example `age^2`).
    Squares,
    /// Include squared terms and pairwise interactions (for example `age * income`).
    SquaresAndPairwiseInteractions,
}

/// Configuration for balance reporting behavior.
#[derive(bon::Builder, Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReportOptions {
    /// Which transformed numeric terms to include.
    #[builder(default)]
    #[serde(default)]
    pub numeric_transforms: NumericBalanceTransform,
    /// Optional supplemental covariates for balance-only diagnostics.
    #[builder(default)]
    #[serde(default)]
    pub supplemental_covariates: SupplementalBalanceCovariates,
}

impl Default for BalanceReportOptions {
    fn default() -> Self {
        Self {
            numeric_transforms: NumericBalanceTransform::None,
            supplemental_covariates: SupplementalBalanceCovariates::default(),
        }
    }
}

/// Supplemental covariates used only for balance diagnostics.
///
/// This enables `addlvariables`-style workflows where diagnostics include
/// variables not stored in the matching records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SupplementalBalanceCovariates {
    /// Supplemental covariates for anchor/case records keyed by record id.
    #[serde(default)]
    pub cases: HashMap<String, HashMap<String, CovariateValue>>,
    /// Supplemental covariates for candidate/control records keyed by record id.
    #[serde(default)]
    pub controls: HashMap<String, HashMap<String, CovariateValue>>,
}

/// Threshold configuration for post-match balance checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceThresholds {
    /// Maximum allowed absolute post-match SMD.
    pub smd_abs_max: Option<f64>,
    /// Minimum allowed post-match variance ratio.
    pub var_ratio_min: Option<f64>,
    /// Maximum allowed post-match variance ratio.
    pub var_ratio_max: Option<f64>,
    /// Maximum allowed post-match eCDF max distance.
    pub ecdf_max_diff_max: Option<f64>,
    /// Maximum allowed post-match eQQ max distance.
    pub eqq_max_diff_max: Option<f64>,
}

impl Default for BalanceThresholds {
    fn default() -> Self {
        Self {
            smd_abs_max: Some(0.1),
            var_ratio_min: Some(0.8),
            var_ratio_max: Some(1.25),
            ecdf_max_diff_max: Some(0.1),
            eqq_max_diff_max: None,
        }
    }
}

impl BalanceThresholds {
    /// Strict threshold profile for diagnostics-sensitive applications.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            smd_abs_max: Some(0.05),
            var_ratio_min: Some(0.9),
            var_ratio_max: Some(1.11),
            ecdf_max_diff_max: Some(0.05),
            eqq_max_diff_max: None,
        }
    }

    /// Moderate threshold profile aligned with crate defaults.
    #[must_use]
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Lenient threshold profile for exploratory checks.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            smd_abs_max: Some(0.15),
            var_ratio_min: Some(0.67),
            var_ratio_max: Some(1.5),
            ecdf_max_diff_max: Some(0.15),
            eqq_max_diff_max: None,
        }
    }
}

/// Threshold-check result for one numeric covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericBalanceThresholdCheck {
    /// Covariate name.
    pub name: String,
    /// Whether post-match SMD is within threshold (or `None` when disabled).
    pub smd_post_ok: Option<bool>,
    /// Whether post-match variance ratio is within threshold (or `None` when disabled).
    pub var_ratio_post_ok: Option<bool>,
    /// Whether post-match eCDF max distance is within threshold (or `None` when disabled).
    pub ecdf_max_diff_post_ok: Option<bool>,
    /// Whether post-match eQQ max distance is within threshold (or `None` when disabled).
    pub eqq_max_diff_post_ok: Option<bool>,
    /// Aggregate pass/fail over enabled checks.
    pub all_enabled_checks_ok: bool,
}

/// Summary of threshold checks across numeric covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceThresholdSummary {
    /// Per-covariate threshold checks.
    pub numeric: Vec<NumericBalanceThresholdCheck>,
    /// Aggregate pass/fail over all numeric covariates and enabled checks.
    pub all_enabled_checks_ok: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_thresholds_default_is_reasonable() {
        let thresholds = BalanceThresholds::default();
        assert_eq!(thresholds.smd_abs_max, Some(0.1));
        assert_eq!(thresholds.var_ratio_min, Some(0.8));
        assert_eq!(thresholds.var_ratio_max, Some(1.25));
        assert_eq!(thresholds.ecdf_max_diff_max, Some(0.1));
        assert_eq!(thresholds.eqq_max_diff_max, None);
    }

    #[test]
    fn balance_report_options_defaults_to_no_transforms() {
        let options = BalanceReportOptions::default();
        assert_eq!(options.numeric_transforms, NumericBalanceTransform::None);
        assert!(options.supplemental_covariates.cases.is_empty());
        assert!(options.supplemental_covariates.controls.is_empty());
    }

    #[test]
    fn balance_threshold_presets_are_ordered_by_strictness() {
        let strict = BalanceThresholds::strict();
        let moderate = BalanceThresholds::moderate();
        let lenient = BalanceThresholds::lenient();

        assert!(strict.smd_abs_max < moderate.smd_abs_max);
        assert!(moderate.smd_abs_max < lenient.smd_abs_max);
        assert!(strict.ecdf_max_diff_max < moderate.ecdf_max_diff_max);
        assert!(moderate.ecdf_max_diff_max < lenient.ecdf_max_diff_max);
    }
}
