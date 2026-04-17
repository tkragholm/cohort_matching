use crate::types::MatchRatio;
use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct RatioPolicy {
    primary_ratio: usize,
    ratios: Vec<usize>,
    strict: bool,
}

impl RatioPolicy {
    #[must_use]
    pub fn from_primary(primary_ratio: usize) -> Self {
        Self {
            primary_ratio,
            ratios: vec![primary_ratio],
            strict: false,
        }
    }

    #[must_use]
    pub fn from_fallback(primary_ratio: usize, ratio_fallback: &[MatchRatio]) -> Self {
        if ratio_fallback.is_empty() {
            return Self::from_primary(primary_ratio);
        }

        let ratios = ratio_fallback
            .iter()
            .map(|ratio| ratio.get())
            .sorted_by(|left, right| right.cmp(left))
            .unique()
            .collect_vec();

        if ratios.is_empty() {
            Self::from_primary(primary_ratio)
        } else {
            Self {
                primary_ratio,
                ratios,
                strict: true,
            }
        }
    }

    #[must_use]
    pub fn target_ratio(&self, available_controls: usize) -> Option<usize> {
        let selected = self
            .ratios
            .iter()
            .copied()
            .find(|ratio| available_controls >= *ratio);

        if self.strict {
            selected
        } else {
            selected.or_else(|| Some(available_controls.min(self.primary_ratio)))
        }
    }

    #[must_use]
    pub const fn is_shortfall(&self, available_controls: usize) -> bool {
        if self.strict {
            false
        } else {
            available_controls < self.primary_ratio
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RatioPolicy;
    use crate::types::MatchRatio;

    #[test]
    fn primary_ratio_allows_partial_fallback_to_available() {
        let policy = RatioPolicy::from_primary(3);
        assert_eq!(policy.target_ratio(1), Some(1));
        assert!(policy.is_shortfall(1));
    }

    #[test]
    fn explicit_ratio_fallback_is_strict() {
        let policy = RatioPolicy::from_fallback(
            3,
            &[
                MatchRatio::new(3).expect("non-zero"),
                MatchRatio::new(2).expect("non-zero"),
            ],
        );
        assert_eq!(policy.target_ratio(1), None);
        assert!(!policy.is_shortfall(1));
    }
}
