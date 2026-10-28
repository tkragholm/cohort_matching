use super::records::MatchingRecord;
use chrono::{Datelike, NaiveDate};
use rustc_hash::FxHashMap;

/// Symbol of an absent strata value. Present values are numbered from 1.
const ABSENT: u32 = 0;

/// Strata keys of up to this many required fields are looked up without allocating.
const INLINE_KEYS: usize = 8;

/// Strata values of every control, laid out as one row of `width` values per control.
pub(crate) struct StrataValues<'a> {
    width: usize,
    flat: Vec<Option<&'a str>>,
}

impl<'a> StrataValues<'a> {
    /// Read the `required` strata of every control once.
    pub(crate) fn of_controls<R: MatchingRecord>(controls: &'a [R], required: &[String]) -> Self {
        let width = required.len();
        let mut flat = Vec::with_capacity(controls.len() * width);
        for control in controls {
            let strata = control.strata();
            flat.extend(
                required
                    .iter()
                    .map(|key| strata.get(key).map(String::as_str)),
            );
        }
        Self { width, flat }
    }

    /// The values of control `idx`, in the order of the required keys.
    pub(crate) fn row(&self, idx: usize) -> Option<&[Option<&'a str>]> {
        let start = idx.checked_mul(self.width)?;
        self.flat.get(start..start.checked_add(self.width)?)
    }
}

/// Candidate lookup index for strata/birthdate prefiltering.
///
/// Strata values are interned to small integers when the index is built, so a
/// stratum is identified by a short integer key and a lookup hashes integers
/// rather than strings.
pub struct CandidateIndex<'a> {
    required_strata: &'a [String],
    global: Vec<(i32, usize)>,
    symbols: FxHashMap<&'a str, u32>,
    by_strata: FxHashMap<Vec<u32>, Vec<(i32, usize)>>,
}

impl<'a> CandidateIndex<'a> {
    #[must_use]
    #[cfg(test)]
    pub(crate) fn new<R: MatchingRecord>(controls: &'a [R], required_strata: &'a [String]) -> Self {
        let values = StrataValues::of_controls(controls, required_strata);
        Self::from_values(controls, required_strata, &values)
    }

    #[must_use]
    pub(crate) fn from_values<R: MatchingRecord>(
        controls: &'a [R],
        required_strata: &'a [String],
        values: &StrataValues<'a>,
    ) -> Self {
        let mut index = Self {
            required_strata,
            global: Vec::new(),
            symbols: FxHashMap::default(),
            by_strata: FxHashMap::default(),
        };

        if required_strata.is_empty() {
            index.global = controls
                .iter()
                .enumerate()
                .map(|(idx, control)| (control.birth_date().num_days_from_ce(), idx))
                .collect();
            index
                .global
                .sort_unstable_by_key(|entry| (entry.0, entry.1));
            return index;
        }

        let mut key = vec![ABSENT; required_strata.len()];
        for (idx, control) in controls.iter().enumerate() {
            let row = values.row(idx).unwrap_or(&[]);
            for (slot, value) in key.iter_mut().zip(row) {
                *slot = match value {
                    None => ABSENT,
                    Some(text) => {
                        let next = u32::try_from(index.symbols.len() + 1).unwrap_or(u32::MAX);
                        *index.symbols.entry(text).or_insert(next)
                    }
                };
            }
            let entry = (control.birth_date().num_days_from_ce(), idx);
            if let Some(entries) = index.by_strata.get_mut(key.as_slice()) {
                entries.push(entry);
            } else {
                index.by_strata.insert(key.clone(), vec![entry]);
            }
        }
        for entries in index.by_strata.values_mut() {
            entries.sort_unstable_by_key(|entry| (entry.0, entry.1));
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
            let Some(entries) = self.stratum(values) else {
                return Vec::new();
            };
            entries
        };

        let start = lower_bound(entries, lower);
        let end = upper_bound(entries, upper);
        entries[start..end].iter().map(|entry| entry.1).collect()
    }

    /// The entries of the stratum with these values, if any control belongs to it.
    ///
    /// A value no control carries names no stratum, so it is not looked up.
    fn stratum(&self, values: &[Option<&str>]) -> Option<&Vec<(i32, usize)>> {
        let mut inline = [ABSENT; INLINE_KEYS];
        let mut spill = Vec::new();
        let key: &mut [u32] = if values.len() <= INLINE_KEYS {
            &mut inline[..values.len()]
        } else {
            spill.resize(values.len(), ABSENT);
            &mut spill
        };
        for (slot, value) in key.iter_mut().zip(values) {
            *slot = match value {
                None => ABSENT,
                Some(text) => *self.symbols.get(text)?,
            };
        }
        self.by_strata.get(&*key)
    }
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
    use crate::date;
    use crate::types::BaseRecord;

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
    fn an_absent_value_is_a_stratum_of_its_own() {
        let required = vec!["municipality".to_string()];
        let controls = vec![
            control("a", date(2010, 1, 1), None),
            control("b", date(2010, 1, 2), Some("0101")),
            control("c", date(2010, 1, 3), None),
        ];
        let index = CandidateIndex::new(&controls, &required);
        assert_eq!(
            index.candidate_indices(date(2010, 1, 2), 10, Some(&[None])),
            vec![0, 2]
        );
        assert_eq!(
            index.candidate_indices(date(2010, 1, 2), 10, Some(&[Some("0101")])),
            vec![1]
        );
    }

    #[test]
    fn a_value_equal_to_another_in_text_only_matches_that_stratum() {
        let required = vec!["a".to_string(), "b".to_string()];
        let mut first = BaseRecord::new("x", date(2010, 1, 1));
        first.strata.insert("a".to_string(), "1".to_string());
        first.strata.insert("b".to_string(), "2".to_string());
        let mut second = BaseRecord::new("y", date(2010, 1, 1));
        second.strata.insert("a".to_string(), "2".to_string());
        second.strata.insert("b".to_string(), "1".to_string());
        let controls = vec![first, second];
        let index = CandidateIndex::new(&controls, &required);
        assert_eq!(
            index.candidate_indices(date(2010, 1, 1), 0, Some(&[Some("1"), Some("2")])),
            vec![0]
        );
        assert_eq!(
            index.candidate_indices(date(2010, 1, 1), 0, Some(&[Some("2"), Some("1")])),
            vec![1]
        );
        assert!(
            index
                .candidate_indices(date(2010, 1, 1), 0, Some(&[Some("1"), Some("1")]))
                .is_empty()
        );
    }

    #[test]
    fn more_required_keys_than_the_inline_key_holds() {
        let required: Vec<String> = (0..INLINE_KEYS + 3).map(|i| format!("k{i}")).collect();
        let mut near = BaseRecord::new("near", date(2010, 1, 1));
        let mut far = BaseRecord::new("far", date(2010, 1, 1));
        for (i, key) in required.iter().enumerate() {
            near.strata.insert(key.clone(), format!("v{i}"));
            far.strata.insert(
                key.clone(),
                if i == required.len() - 1 {
                    "other".to_string()
                } else {
                    format!("v{i}")
                },
            );
        }
        let controls = vec![near, far];
        let index = CandidateIndex::new(&controls, &required);
        let values: Vec<String> = (0..required.len()).map(|i| format!("v{i}")).collect();
        let case: Vec<Option<&str>> = values.iter().map(|v| Some(v.as_str())).collect();
        assert_eq!(
            index.candidate_indices(date(2010, 1, 1), 0, Some(&case)),
            vec![0]
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
