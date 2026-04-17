use crate::types::{BaseRecord, MatchDiagnostics, MatchOutcome, MatchedPair};
use chrono::NaiveDate;

/// Helper for creating a `NaiveDate` in tests.
///
/// # Panics
/// Panics if the provided date components are invalid.
#[must_use]
pub const fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
}

/// Helper for creating a `BaseRecord` in tests.
#[must_use]
pub fn record(id: &str, birth_date: NaiveDate) -> BaseRecord {
    BaseRecord::new(id, birth_date)
}

/// Helper for creating a `MatchOutcome` in tests.
///
/// # Panics
/// Panics if `pairs.len()` or `matched_cases` cannot be represented as `u32`.
#[must_use]
pub fn test_outcome(
    pairs: Vec<MatchedPair>,
    matched_cases: usize,
    unmatched_cases: usize,
) -> MatchOutcome {
    MatchOutcome {
        avg_controls_per_case: if matched_cases == 0 {
            0.0
        } else {
            f64::from(u32::try_from(pairs.len()).expect("small test vector length"))
                / f64::from(u32::try_from(matched_cases).expect("small matched case count"))
        },
        used_controls: pairs.len(),
        pairs,
        matched_cases,
        unmatched_cases,
        diagnostics: MatchDiagnostics::default(),
    }
}
