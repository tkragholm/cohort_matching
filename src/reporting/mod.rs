use crate::types::{BalanceReport, MatchOutcome};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Configuration for report output behavior.
#[derive(Debug, Clone, bon::Builder)]
pub struct ReportConfig {
    /// Number of decimal places for numeric outputs (default).
    #[builder(default = 4)]
    pub decimal_places: usize,
    /// CSV delimiter character.
    #[builder(default = ',')]
    pub delimiter: char,
    /// Per-field decimal precision overrides.
    #[builder(default)]
    pub field_precision: HashMap<String, usize>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            decimal_places: 4,
            delimiter: ',',
            field_precision: HashMap::new(),
        }
    }
}

impl ReportConfig {
    /// Get the precision for a specific field, falling back to the default.
    #[must_use]
    pub fn precision_for(&self, field: &str) -> usize {
        self.field_precision
            .get(field)
            .copied()
            .unwrap_or(self.decimal_places)
    }
}

/// Text of a balance statistic: fixed decimals, or `NA` when it is undefined.
fn format_cell(value: Option<f64>, precision: usize) -> String {
    value.map_or_else(|| "NA".to_string(), |value| format!("{value:.precision$}"))
}

/// Trait for types that can be rendered to a report.
pub trait ReportSink {
    /// Write the report content to the provided writer.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if writing to the underlying stream fails.
    fn write_report<W: Write>(&self, writer: &mut W, config: &ReportConfig) -> std::io::Result<()>;
}

impl ReportSink for BalanceReport {
    fn write_report<W: Write>(&self, writer: &mut W, config: &ReportConfig) -> std::io::Result<()> {
        let d = config.delimiter;

        // Write Numeric Balance
        writeln!(
            writer,
            "covariate{d}mean_case_pre{d}mean_control_pre{d}smd_pre{d}var_ratio_pre{d}ecdf_mean_diff_pre{d}ecdf_max_diff_pre{d}eqq_mean_diff_pre{d}eqq_max_diff_pre{d}mean_case_post{d}mean_control_post{d}smd_post{d}var_ratio_post{d}ecdf_mean_diff_post{d}ecdf_max_diff_post{d}eqq_mean_diff_post{d}eqq_max_diff_post"
        )?;

        for row in &self.numeric {
            let p = config.precision_for(&row.name);
            let cells = [
                row.mean_case_pre,
                row.mean_control_pre,
                row.smd_pre,
                row.var_ratio_pre,
                row.ecdf_mean_diff_pre,
                row.ecdf_max_diff_pre,
                row.eqq_mean_diff_pre,
                row.eqq_max_diff_pre,
                row.mean_case_post,
                row.mean_control_post,
                row.smd_post,
                row.var_ratio_post,
                row.ecdf_mean_diff_post,
                row.ecdf_max_diff_post,
                row.eqq_mean_diff_post,
                row.eqq_max_diff_post,
            ]
            .map(|value| format_cell(value, p));
            writeln!(writer, "{}{d}{}", row.name, cells.join(&d.to_string()))?;
        }

        // Write Categorical Balance (often in a separate file, but here we provide a choice or a combined report)
        // Given existing usage, we might want separate sinks or separate methods.
        Ok(())
    }
}

/// Separate sink for categorical balance to follow existing patterns.
pub struct CategoricalBalanceReport<'a>(pub &'a BalanceReport);

impl ReportSink for CategoricalBalanceReport<'_> {
    fn write_report<W: Write>(&self, writer: &mut W, config: &ReportConfig) -> std::io::Result<()> {
        let d = config.delimiter;

        writeln!(
            writer,
            "covariate{d}level{d}p_case_pre{d}p_control_pre{d}smd_pre{d}p_case_post{d}p_control_post{d}smd_post{d}cramers_v_pre{d}cramers_v_post"
        )?;

        for covariate in &self.0.categorical {
            let p = config.precision_for(&covariate.name);
            for level in &covariate.levels {
                let cells = [
                    level.p_case_pre,
                    level.p_control_pre,
                    level.smd_pre,
                    level.p_case_post,
                    level.p_control_post,
                    level.smd_post,
                    covariate.cramers_v_pre,
                    covariate.cramers_v_post,
                ]
                .map(|value| format_cell(value, p));
                writeln!(
                    writer,
                    "{}{d}{}{d}{}",
                    covariate.name,
                    level.level,
                    cells.join(&d.to_string())
                )?;
            }
        }
        Ok(())
    }
}

/// Wrapper for matching summary report.
pub struct MatchSummaryReport<'a> {
    pub outcome: &'a MatchOutcome,
}

impl ReportSink for MatchSummaryReport<'_> {
    fn write_report<W: Write>(&self, writer: &mut W, config: &ReportConfig) -> std::io::Result<()> {
        let p = config.decimal_places;
        let d = config.delimiter;

        writeln!(writer, "metric{d}value")?;
        writeln!(
            writer,
            "total_anchors_evaluated{d}{}",
            self.outcome.diagnostics.total_anchors_evaluated
        )?;
        writeln!(writer, "matched_anchors{d}{}", self.outcome.matched_cases)?;
        writeln!(
            writer,
            "unmatched_anchors{d}{}",
            self.outcome.unmatched_cases
        )?;
        writeln!(writer, "pairs_selected{d}{}", self.outcome.pairs.len())?;
        writeln!(
            writer,
            "anchors_with_no_candidates{d}{}",
            self.outcome.diagnostics.anchors_with_no_candidates
        )?;
        writeln!(
            writer,
            "anchors_below_required_ratio{d}{}",
            self.outcome.diagnostics.anchors_below_required_ratio
        )?;
        writeln!(writer, "used_controls{d}{}", self.outcome.used_controls)?;
        writeln!(
            writer,
            "avg_controls_per_case{d}{:.*}",
            p, self.outcome.avg_controls_per_case
        )?;
        writeln!(
            writer,
            "requested_estimand{d}{:?}",
            self.outcome.diagnostics.requested_estimand
        )?;
        writeln!(
            writer,
            "realized_estimand{d}{:?}",
            self.outcome.diagnostics.realized_estimand
        )?;

        Ok(())
    }
}

/// Wrapper for exclusion counts report.
pub struct ExclusionCountsReport<'a> {
    pub outcome: &'a MatchOutcome,
}

impl ReportSink for ExclusionCountsReport<'_> {
    fn write_report<W: Write>(&self, writer: &mut W, config: &ReportConfig) -> std::io::Result<()> {
        let d = config.delimiter;
        writeln!(writer, "reason{d}count")?;

        if self.outcome.diagnostics.exclusion_counts.is_empty() {
            writeln!(writer, "none{d}0")?;
        } else {
            for (reason, count) in &self.outcome.diagnostics.exclusion_counts {
                writeln!(writer, "{reason}{d}{count}")?;
            }
        }
        Ok(())
    }
}

// Convenience methods for writing to paths
impl BalanceReport {
    /// Write numeric balance report to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the file cannot be created or written to.
    pub fn write_numeric_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        self.write_report(&mut file, &ReportConfig::default())
    }

    /// Write categorical balance report to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the file cannot be created or written to.
    pub fn write_categorical_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        CategoricalBalanceReport(self).write_report(&mut file, &ReportConfig::default())
    }
}

impl MatchOutcome {
    /// Write matching summary to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the file cannot be created or written to.
    pub fn write_summary_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        MatchSummaryReport { outcome: self }.write_report(&mut file, &ReportConfig::default())
    }

    /// Write exclusion counts to a CSV file.
    ///
    /// # Errors
    ///
    /// Returns [`std::io::Error`] if the file cannot be created or written to.
    pub fn write_exclusion_counts_csv(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut file = File::create(path)?;
        ExclusionCountsReport { outcome: self }.write_report(&mut file, &ReportConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MatchDiagnostics, MatchedPair, NumericBalance};

    #[test]
    fn balance_report_numeric_csv_format() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "age".to_string(),
                mean_case_pre: Some(30.5),
                mean_control_pre: Some(31.2),
                smd_pre: Some(0.05),
                var_ratio_pre: Some(1.0),
                ecdf_mean_diff_pre: Some(0.01),
                ecdf_max_diff_pre: Some(0.02),
                eqq_mean_diff_pre: Some(0.0),
                eqq_max_diff_pre: Some(0.0),
                mean_case_post: Some(30.8),
                mean_control_post: Some(30.8),
                smd_post: Some(0.0),
                var_ratio_post: Some(1.0),
                ecdf_mean_diff_post: Some(0.0),
                ecdf_max_diff_post: Some(0.0),
                eqq_mean_diff_post: Some(0.0),
                eqq_max_diff_post: Some(0.0),
            }],
            categorical: vec![],
        };

        let mut buffer = Vec::new();
        report
            .write_report(&mut buffer, &ReportConfig::default())
            .unwrap();
        let output = String::from_utf8(buffer).unwrap();

        assert!(output.contains("covariate,mean_case_pre,mean_control_pre"));
        assert!(output.contains("age,30.5000,31.2000,0.0500"));
    }

    #[test]
    fn balance_report_respects_per_field_precision() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "propensity_score".to_string(),
                mean_case_pre: Some(0.123_456_7),
                mean_control_pre: Some(0.123_456_7),
                smd_pre: Some(0.0),
                ..NumericBalance::default()
            }],
            categorical: vec![],
        };

        let mut field_precision = HashMap::new();
        field_precision.insert("propensity_score".to_string(), 6);

        let config = ReportConfig {
            decimal_places: 2,
            delimiter: ',',
            field_precision,
        };

        let mut buffer = Vec::new();
        report.write_report(&mut buffer, &config).unwrap();
        let output = String::from_utf8(buffer).unwrap();

        assert!(output.contains("propensity_score,0.123457,0.123457,0.000000"));
    }

    #[test]
    fn undefined_statistics_are_written_as_na() {
        let report = BalanceReport {
            numeric: vec![NumericBalance {
                name: "x".to_string(),
                mean_case_pre: Some(1.0),
                ..NumericBalance::default()
            }],
            categorical: vec![crate::types::CategoricalBalance {
                name: "sex".to_string(),
                levels: vec![crate::types::CategoricalLevelBalance {
                    level: "f".to_string(),
                    p_case_pre: Some(0.5),
                    p_control_pre: None,
                    smd_pre: None,
                    p_case_post: None,
                    p_control_post: None,
                    smd_post: None,
                }],
                cramers_v_pre: None,
                cramers_v_post: None,
            }],
        };

        let mut buffer = Vec::new();
        report
            .write_report(&mut buffer, &ReportConfig::default())
            .unwrap();
        let numeric = String::from_utf8(buffer).unwrap();
        assert!(numeric.contains("x,1.0000,NA,NA,NA"));

        let mut buffer = Vec::new();
        CategoricalBalanceReport(&report)
            .write_report(&mut buffer, &ReportConfig::default())
            .unwrap();
        let categorical = String::from_utf8(buffer).unwrap();
        assert!(categorical.contains("sex,f,0.5000,NA,NA,NA,NA,NA,NA,NA"));
    }

    #[test]
    fn match_summary_csv_format() {
        let outcome = MatchOutcome {
            pairs: vec![MatchedPair::new("a1", "c1")],
            unmatched_cases: 0,
            used_controls: 1,
            matched_cases: 1,
            avg_controls_per_case: 1.0,
            diagnostics: MatchDiagnostics {
                total_anchors_evaluated: 1,
                ..MatchDiagnostics::default()
            },
        };

        let mut buffer = Vec::new();
        MatchSummaryReport { outcome: &outcome }
            .write_report(&mut buffer, &ReportConfig::default())
            .unwrap();
        let output = String::from_utf8(buffer).unwrap();

        assert!(output.contains("metric,value"));
        assert!(output.contains("total_anchors_evaluated,1"));
        assert!(output.contains("matched_anchors,1"));
        assert!(output.contains("avg_controls_per_case,1.0000"));
    }
}
