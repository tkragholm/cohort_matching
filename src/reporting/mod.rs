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
            writeln!(
                writer,
                "{}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}",
                row.name,
                p,
                row.mean_case_pre,
                p,
                row.mean_control_pre,
                p,
                row.smd_pre,
                p,
                row.var_ratio_pre,
                p,
                row.ecdf_mean_diff_pre,
                p,
                row.ecdf_max_diff_pre,
                p,
                row.eqq_mean_diff_pre,
                p,
                row.eqq_max_diff_pre,
                p,
                row.mean_case_post,
                p,
                row.mean_control_post,
                p,
                row.smd_post,
                p,
                row.var_ratio_post,
                p,
                row.ecdf_mean_diff_post,
                p,
                row.ecdf_max_diff_post,
                p,
                row.eqq_mean_diff_post,
                p,
                row.eqq_max_diff_post,
            )?;
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
                writeln!(
                    writer,
                    "{}{d}{}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}{d}{:.*}",
                    covariate.name,
                    level.level,
                    p,
                    level.p_case_pre,
                    p,
                    level.p_control_pre,
                    p,
                    level.smd_pre,
                    p,
                    level.p_case_post,
                    p,
                    level.p_control_post,
                    p,
                    level.smd_post,
                    p,
                    covariate.cramers_v_pre,
                    p,
                    covariate.cramers_v_post,
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
                mean_case_pre: 30.5,
                mean_control_pre: 31.2,
                smd_pre: 0.05,
                var_ratio_pre: 1.0,
                ecdf_mean_diff_pre: 0.01,
                ecdf_max_diff_pre: 0.02,
                eqq_mean_diff_pre: 0.0,
                eqq_max_diff_pre: 0.0,
                mean_case_post: 30.8,
                mean_control_post: 30.8,
                smd_post: 0.0,
                var_ratio_post: 1.0,
                ecdf_mean_diff_post: 0.0,
                ecdf_max_diff_post: 0.0,
                eqq_mean_diff_post: 0.0,
                eqq_max_diff_post: 0.0,
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
                mean_case_pre: 0.123_456_7,
                mean_control_pre: 0.123_456_7,
                smd_pre: 0.0,
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
