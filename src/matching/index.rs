use super::constraints::build_strata_values;
use super::records::MatchingRecord;
use chrono::{Datelike, NaiveDate};
use std::collections::HashMap;

/// Candidate lookup index for strata/birthdate prefiltering.
pub struct CandidateIndex<'a> {
    required_strata: &'a [String],
    global: Vec<(i32, usize)>,
    by_strata: HashMap<Vec<Option<String>>, Vec<(i32, usize)>>,
}

impl<'a> CandidateIndex<'a> {
    #[must_use]
    pub(crate) fn new<R: MatchingRecord>(controls: &'a [R], required_strata: &'a [String]) -> Self {
        let mut index = Self {
            required_strata,
            global: Vec::new(),
            by_strata: HashMap::new(),
        };

        if required_strata.is_empty() {
            index.global = controls
                .iter()
                .enumerate()
                .map(|(idx, control)| (control.birth_date().num_days_from_ce(), idx))
                .collect();
            index.global.sort_unstable_by_key(|entry| entry.0);
            return index;
        }

        for (idx, control) in controls.iter().enumerate() {
            let key = to_strata_key(&build_strata_values(control.strata(), required_strata));
            index
                .by_strata
                .entry(key)
                .or_default()
                .push((control.birth_date().num_days_from_ce(), idx));
        }
        for entries in index.by_strata.values_mut() {
            entries.sort_unstable_by_key(|entry| entry.0);
        }

        index
    }

    #[must_use]
    pub(crate) fn candidate_indices(
        &self,
        birth_date: NaiveDate,
        window_days: i64,
        case_strata_values: Option<&[Option<&str>]>,
    ) -> Vec<usize> {
        let center = birth_date.num_days_from_ce();
        let Some(window) = i32::try_from(window_days).ok() else {
            return Vec::new();
        };
        let lower = center.saturating_sub(window);
        let upper = center.saturating_add(window);

        let entries = if self.required_strata.is_empty() {
            &self.global
        } else {
            let Some(values) = case_strata_values else {
                return Vec::new();
            };
            let key = to_strata_key(values);
            let Some(entries) = self.by_strata.get(&key) else {
                return Vec::new();
            };
            entries
        };

        let start = lower_bound(entries, lower);
        let end = upper_bound(entries, upper);
        entries[start..end].iter().map(|entry| entry.1).collect()
    }
}

fn to_strata_key(values: &[Option<&str>]) -> Vec<Option<String>> {
    values
        .iter()
        .map(|value| value.map(str::to_string))
        .collect()
}

fn lower_bound(entries: &[(i32, usize)], target: i32) -> usize {
    let mut left = 0usize;
    let mut right = entries.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if entries[mid].0 < target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn upper_bound(entries: &[(i32, usize)], target: i32) -> usize {
    let mut left = 0usize;
    let mut right = entries.len();
    while left < right {
        let mid = left + (right - left) / 2;
        if entries[mid].0 <= target {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn control(id: &str, birth_date: NaiveDate, municipality: Option<&str>) -> BaseRecord {
        let mut row = BaseRecord::new(id, birth_date);
        if let Some(value) = municipality {
            row.strata
                .insert("municipality".to_string(), value.to_string());
        }
        row
    }

    #[test]
    fn candidate_indices_include_window_boundaries() {
        let controls = vec![
            control("a", date(2010, 1, 1), None),
            control("b", date(2010, 1, 3), None),
            control("c", date(2010, 1, 5), None),
        ];
        let index = CandidateIndex::new(&controls, &[]);
        let result = index.candidate_indices(date(2010, 1, 3), 2, None);
        assert_eq!(result, vec![0, 1, 2]);
    }

    #[test]
    fn candidate_indices_respect_required_strata() {
        let required = vec!["municipality".to_string()];
        let controls = vec![
            control("a", date(2010, 1, 1), Some("0101")),
            control("b", date(2010, 1, 2), Some("0202")),
            control("c", date(2010, 1, 3), Some("0101")),
        ];
        let index = CandidateIndex::new(&controls, &required);
        let case_values = [Some("0101")];
        let result = index.candidate_indices(date(2010, 1, 2), 10, Some(&case_values));
        assert_eq!(result, vec![0, 2]);
    }

    #[test]
    fn missing_or_unknown_strata_returns_empty_candidates() {
        let required = vec!["municipality".to_string()];
        let controls = vec![control("a", date(2010, 1, 1), Some("0101"))];
        let index = CandidateIndex::new(&controls, &required);
        assert!(
            index
                .candidate_indices(date(2010, 1, 1), 30, None)
                .is_empty()
        );
        let missing_values = [Some("9999")];
        assert!(
            index
                .candidate_indices(date(2010, 1, 1), 30, Some(&missing_values))
                .is_empty()
        );
    }

    #[test]
    fn oversized_window_is_rejected() {
        let controls = vec![control("a", date(2010, 1, 1), None)];
        let index = CandidateIndex::new(&controls, &[]);
        assert!(
            index
                .candidate_indices(date(2010, 1, 1), i64::MAX, None)
                .is_empty()
        );
    }
}
