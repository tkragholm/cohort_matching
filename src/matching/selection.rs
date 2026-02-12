use super::records::MatchingRecord;
use rand::prelude::*;

/// Strategy for selecting one control from a candidate index set.
pub trait SelectionStrategy<R: MatchingRecord> {
    /// Returns selected candidate position index from `candidate_indices`.
    fn select_position(
        &mut self,
        case: &R,
        controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize>;
}

/// Random control selection using a reproducible RNG seed.
pub struct RandomSelection {
    rng: rand::rngs::StdRng,
}

impl RandomSelection {
    #[must_use]
    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }
}

impl<R: MatchingRecord> SelectionStrategy<R> for RandomSelection {
    fn select_position(
        &mut self,
        _case: &R,
        _controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        if candidate_indices.is_empty() {
            None
        } else {
            Some(self.rng.random_range(0..candidate_indices.len()))
        }
    }
}

/// Select the candidate with smallest birth-date distance to the case.
pub struct NearestBirthDateSelection;

impl<R: MatchingRecord> SelectionStrategy<R> for NearestBirthDateSelection {
    fn select_position(
        &mut self,
        case: &R,
        controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        candidate_indices
            .iter()
            .enumerate()
            .min_by_key(|(_, idx)| {
                (controls[**idx].birth_date() - case.birth_date())
                    .num_days()
                    .unsigned_abs()
            })
            .map(|(pos, _)| pos)
    }
}

/// Deterministic selection: lowest candidate index first.
pub struct DeterministicSelection;

impl<R: MatchingRecord> SelectionStrategy<R> for DeterministicSelection {
    fn select_position(
        &mut self,
        _case: &R,
        _controls: &[R],
        candidate_indices: &[usize],
    ) -> Option<usize> {
        if candidate_indices.is_empty() {
            None
        } else {
            Some(0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn date(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
    }

    fn record(id: &str, birth_date: NaiveDate) -> BaseRecord {
        BaseRecord::new(id, birth_date)
    }

    #[test]
    fn deterministic_selection_returns_first_or_none() {
        let mut strategy = DeterministicSelection;
        let case = record("case", date(2010, 1, 2));
        let controls = vec![record("a", date(2010, 1, 1)), record("b", date(2010, 1, 3))];
        assert_eq!(strategy.select_position(&case, &controls, &[5, 7]), Some(0));
        assert_eq!(strategy.select_position(&case, &controls, &[]), None);
    }

    #[test]
    fn nearest_birth_date_selection_prefers_smallest_distance() {
        let mut strategy = NearestBirthDateSelection;
        let case = record("case", date(2010, 1, 5));
        let controls = vec![
            record("a", date(2010, 1, 1)),
            record("b", date(2010, 1, 6)),
            record("c", date(2010, 1, 10)),
        ];
        assert_eq!(
            strategy.select_position(&case, &controls, &[0, 1, 2]),
            Some(1)
        );
    }

    #[test]
    fn nearest_birth_date_selection_is_stable_on_ties() {
        let mut strategy = NearestBirthDateSelection;
        let case = record("case", date(2010, 1, 5));
        let controls = vec![record("a", date(2010, 1, 4)), record("b", date(2010, 1, 6))];
        assert_eq!(strategy.select_position(&case, &controls, &[1, 0]), Some(0));
    }

    #[test]
    fn random_selection_is_seeded_and_reproducible() {
        let case = record("case", date(2010, 1, 1));
        let controls = vec![record("a", date(2010, 1, 1)); 4];
        let candidates = [0usize, 1, 2, 3];
        let mut left = RandomSelection::seeded(42);
        let mut right = RandomSelection::seeded(42);

        let left_positions = (0..5)
            .map(|_| {
                left.select_position(&case, &controls, &candidates)
                    .expect("position from non-empty candidates")
            })
            .collect::<Vec<_>>();
        let right_positions = (0..5)
            .map(|_| {
                right
                    .select_position(&case, &controls, &candidates)
                    .expect("position from non-empty candidates")
            })
            .collect::<Vec<_>>();
        assert_eq!(left_positions, right_positions);
    }
}
