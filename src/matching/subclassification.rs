use super::records::MatchingRecord;
use crate::types::Estimand;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::BuildHasher;

/// Reference population used to derive subclass cutpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubclassReferenceGroup {
    /// Derive quantiles from anchor scores.
    Anchors,
    /// Derive quantiles from candidate scores.
    Candidates,
    /// Derive quantiles from pooled anchor + candidate scores.
    Pooled,
}

/// Configuration for propensity-score subclassification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclassificationConfig {
    /// Requested number of subclasses (must be at least 1).
    pub subclasses: usize,
    /// Optional reference group used for cutpoints.
    ///
    /// Defaults by estimand:
    /// - ATT -> `Anchors`
    /// - ATC -> `Candidates`
    /// - ATE/ATM -> `Pooled`
    pub reference_group: Option<SubclassReferenceGroup>,
}

impl Default for SubclassificationConfig {
    fn default() -> Self {
        Self {
            subclasses: 6,
            reference_group: None,
        }
    }
}

/// Per-subclass summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubclassSummary {
    /// Subclass index.
    pub subclass: usize,
    /// Number of anchor units in subclass.
    pub anchor_count: usize,
    /// Number of candidate units in subclass.
    pub candidate_count: usize,
    /// Anchor share in subclass (`anchor_count / total`) when total > 0.
    pub subclass_propensity: Option<f64>,
}

/// Result of propensity-score subclassification.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubclassificationOutcome {
    /// Requested estimand for weighting.
    pub estimand: Estimand,
    /// Requested subclass count.
    pub requested_subclasses: usize,
    /// Realized subclass count after non-overlap filtering.
    pub realized_subclasses: usize,
    /// Subclass assignment per retained unit id.
    pub assignments: HashMap<String, usize>,
    /// Subclassification weights per retained unit id.
    pub weights: HashMap<String, f64>,
    /// Unit ids excluded due to missing/non-finite score or non-overlap subclass.
    pub dropped_ids: Vec<String>,
    /// Subclass summaries.
    pub subclasses: Vec<SubclassSummary>,
}

impl SubclassificationOutcome {
    /// Number of retained units.
    #[must_use]
    pub fn retained_units(&self) -> usize {
        self.assignments.len()
    }
}

/// Build propensity-score subclasses and estimand-specific subclass weights.
#[must_use]
pub fn subclassify_by_propensity_score_map<R: MatchingRecord, H: BuildHasher>(
    anchors: &[R],
    candidates: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    estimand: Estimand,
    config: &SubclassificationConfig,
) -> SubclassificationOutcome {
    let requested_subclasses = config.subclasses.max(1);
    let reference_group = resolve_reference_group(config.reference_group, estimand);
    let reference_scores =
        sorted_reference_scores(anchors, candidates, propensity_scores, reference_group);
    if reference_scores.is_empty() {
        return dropped_only_outcome(anchors, candidates, estimand, requested_subclasses);
    }

    let cutpoints = quantile_cutpoints(&reference_scores, requested_subclasses);
    let subclass_count = cutpoints.len() + 1;
    let mut anchor_counts = vec![0usize; subclass_count];
    let mut candidate_counts = vec![0usize; subclass_count];
    let (anchor_assignments, missing_anchor_ids) =
        assign_records(anchors, propensity_scores, &cutpoints, &mut anchor_counts);
    let (candidate_assignments, missing_candidate_ids) = assign_records(
        candidates,
        propensity_scores,
        &cutpoints,
        &mut candidate_counts,
    );

    let valid_subclasses = valid_subclasses(&anchor_counts, &candidate_counts);
    let subclasses = build_subclass_summaries(&anchor_counts, &candidate_counts);
    let realized_subclasses = valid_subclasses
        .iter()
        .filter(|is_valid| **is_valid)
        .count();

    let mut accumulator = SubclassificationAccumulator::default();
    accumulator.dropped_ids.extend(missing_anchor_ids);
    accumulator.dropped_ids.extend(missing_candidate_ids);

    apply_assignments_and_weights(
        &anchor_assignments,
        &valid_subclasses,
        estimand,
        &subclasses,
        true,
        &mut accumulator,
    );
    apply_assignments_and_weights(
        &candidate_assignments,
        &valid_subclasses,
        estimand,
        &subclasses,
        false,
        &mut accumulator,
    );

    SubclassificationOutcome {
        estimand,
        requested_subclasses,
        realized_subclasses,
        assignments: accumulator.assignments,
        weights: accumulator.weights,
        dropped_ids: accumulator.dropped_ids,
        subclasses,
    }
}

fn resolve_reference_group(
    reference_group: Option<SubclassReferenceGroup>,
    estimand: Estimand,
) -> SubclassReferenceGroup {
    reference_group.unwrap_or(match estimand {
        Estimand::Att => SubclassReferenceGroup::Anchors,
        Estimand::Atc => SubclassReferenceGroup::Candidates,
        Estimand::Ate | Estimand::Atm => SubclassReferenceGroup::Pooled,
    })
}

fn sorted_reference_scores<R: MatchingRecord, H: BuildHasher>(
    anchors: &[R],
    candidates: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    reference_group: SubclassReferenceGroup,
) -> Vec<f64> {
    let mut scores = match reference_group {
        SubclassReferenceGroup::Anchors => collect_finite_scores(anchors, propensity_scores),
        SubclassReferenceGroup::Candidates => collect_finite_scores(candidates, propensity_scores),
        SubclassReferenceGroup::Pooled => {
            let mut pooled = collect_finite_scores(anchors, propensity_scores);
            pooled.extend(collect_finite_scores(candidates, propensity_scores));
            pooled
        }
    };
    scores.sort_unstable_by(f64::total_cmp);
    scores
}

fn collect_finite_scores<R: MatchingRecord, H: BuildHasher>(
    rows: &[R],
    propensity_scores: &HashMap<String, f64, H>,
) -> Vec<f64> {
    rows.iter()
        .filter_map(|row| propensity_scores.get(row.id()).copied())
        .filter(|score| score.is_finite())
        .collect()
}

fn quantile_cutpoints(sorted_scores: &[f64], subclasses: usize) -> Vec<f64> {
    if subclasses <= 1 || sorted_scores.len() <= 1 {
        return Vec::new();
    }

    let mut cutpoints = Vec::with_capacity(subclasses.saturating_sub(1));
    for bucket in 1..subclasses {
        let idx = (bucket * (sorted_scores.len() - 1)) / subclasses;
        cutpoints.push(sorted_scores[idx]);
    }
    cutpoints
}

fn assign_records<R: MatchingRecord, H: BuildHasher>(
    rows: &[R],
    propensity_scores: &HashMap<String, f64, H>,
    cutpoints: &[f64],
    counts: &mut [usize],
) -> (Vec<(String, usize)>, Vec<String>) {
    let mut assignments = Vec::with_capacity(rows.len());
    let mut missing_or_invalid_ids = Vec::new();

    for row in rows {
        let Some(score) = propensity_scores.get(row.id()).copied() else {
            missing_or_invalid_ids.push(row.id().to_string());
            continue;
        };
        if !score.is_finite() {
            missing_or_invalid_ids.push(row.id().to_string());
            continue;
        }

        let subclass = cutpoints.partition_point(|cutpoint| score > *cutpoint);
        counts[subclass] += 1;
        assignments.push((row.id().to_string(), subclass));
    }

    (assignments, missing_or_invalid_ids)
}

fn valid_subclasses(anchor_counts: &[usize], candidate_counts: &[usize]) -> Vec<bool> {
    anchor_counts
        .iter()
        .zip(candidate_counts)
        .map(|(anchor_count, candidate_count)| *anchor_count > 0 && *candidate_count > 0)
        .collect()
}

fn build_subclass_summaries(
    anchor_counts: &[usize],
    candidate_counts: &[usize],
) -> Vec<SubclassSummary> {
    anchor_counts
        .iter()
        .zip(candidate_counts)
        .enumerate()
        .map(|(subclass, (anchor_count, candidate_count))| {
            let total = *anchor_count + *candidate_count;
            let subclass_propensity = if total == 0 {
                None
            } else {
                Some(to_f64(*anchor_count) / to_f64(total))
            };

            SubclassSummary {
                subclass,
                anchor_count: *anchor_count,
                candidate_count: *candidate_count,
                subclass_propensity,
            }
        })
        .collect()
}

#[derive(Default)]
struct SubclassificationAccumulator {
    assignments: HashMap<String, usize>,
    weights: HashMap<String, f64>,
    dropped_ids: Vec<String>,
}

fn apply_assignments_and_weights(
    assignments_by_unit: &[(String, usize)],
    valid_subclasses: &[bool],
    estimand: Estimand,
    subclasses: &[SubclassSummary],
    is_anchor_unit: bool,
    accumulator: &mut SubclassificationAccumulator,
) {
    for (id, subclass) in assignments_by_unit {
        if !valid_subclasses.get(*subclass).copied().unwrap_or(false) {
            accumulator.dropped_ids.push(id.clone());
            continue;
        }

        let propensity = subclasses
            .get(*subclass)
            .and_then(|summary| summary.subclass_propensity);
        let Some(propensity) = propensity else {
            accumulator.dropped_ids.push(id.clone());
            continue;
        };

        accumulator.assignments.insert(id.clone(), *subclass);
        accumulator.weights.insert(
            id.clone(),
            subclass_weight_for_estimand(estimand, propensity, is_anchor_unit),
        );
    }
}

fn subclass_weight_for_estimand(estimand: Estimand, propensity: f64, is_anchor_unit: bool) -> f64 {
    match estimand {
        Estimand::Att => {
            if is_anchor_unit {
                1.0
            } else {
                propensity / (1.0 - propensity)
            }
        }
        Estimand::Atc => {
            if is_anchor_unit {
                (1.0 - propensity) / propensity
            } else {
                1.0
            }
        }
        Estimand::Ate | Estimand::Atm => {
            if is_anchor_unit {
                1.0 / propensity
            } else {
                1.0 / (1.0 - propensity)
            }
        }
    }
}

fn dropped_only_outcome<R: MatchingRecord>(
    anchors: &[R],
    candidates: &[R],
    estimand: Estimand,
    requested_subclasses: usize,
) -> SubclassificationOutcome {
    let mut dropped_ids = anchors
        .iter()
        .map(|row| row.id().to_string())
        .collect::<Vec<_>>();
    dropped_ids.extend(candidates.iter().map(|row| row.id().to_string()));

    SubclassificationOutcome {
        estimand,
        requested_subclasses,
        ..SubclassificationOutcome {
            dropped_ids,
            ..SubclassificationOutcome::default()
        }
    }
}

use super::to_f64;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    fn record(id: &str) -> BaseRecord {
        BaseRecord::new(
            id,
            NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid test date"),
        )
    }

    fn assert_close(value: f64, expected: f64) {
        assert!((value - expected).abs() < 1e-12);
    }

    #[test]
    fn pooled_subclassification_preserves_overlap_strata() {
        let anchors = vec![record("a1"), record("a2")];
        let candidates = vec![record("c1"), record("c2")];
        let scores = HashMap::from([
            (String::from("a1"), 0.10),
            (String::from("a2"), 0.30),
            (String::from("c1"), 0.20),
            (String::from("c2"), 0.40),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
        };

        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Att,
            &config,
        );

        assert_eq!(outcome.realized_subclasses, 2);
        assert_eq!(outcome.retained_units(), 4);
        assert!(outcome.dropped_ids.is_empty());
        assert_close(*outcome.weights.get("a1").expect("anchor weight"), 1.0);
        assert_close(*outcome.weights.get("c1").expect("candidate weight"), 1.0);
    }

    #[test]
    fn non_overlap_subclasses_are_dropped() {
        let anchors = vec![record("a1"), record("a2")];
        let candidates = vec![record("c1"), record("c2")];
        let scores = HashMap::from([
            (String::from("a1"), 0.10),
            (String::from("a2"), 0.20),
            (String::from("c1"), 0.80),
            (String::from("c2"), 0.90),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
        };

        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Ate,
            &config,
        );

        assert_eq!(outcome.realized_subclasses, 0);
        assert!(outcome.assignments.is_empty());
        assert_eq!(outcome.dropped_ids.len(), 4);
    }

    #[test]
    fn ate_weights_follow_subclass_propensities() {
        let anchors = vec![record("a1"), record("a2"), record("a3")];
        let candidates = vec![record("c1"), record("c2")];
        let scores = HashMap::from([
            (String::from("a1"), 0.10),
            (String::from("a2"), 0.11),
            (String::from("a3"), 0.90),
            (String::from("c1"), 0.12),
            (String::from("c2"), 0.92),
        ]);
        let config = SubclassificationConfig {
            subclasses: 2,
            reference_group: Some(SubclassReferenceGroup::Pooled),
        };

        let outcome = subclassify_by_propensity_score_map(
            &anchors,
            &candidates,
            &scores,
            Estimand::Ate,
            &config,
        );

        assert_close(*outcome.weights.get("a1").expect("a1 weight"), 1.5);
        assert_close(*outcome.weights.get("a2").expect("a2 weight"), 1.5);
        assert_close(*outcome.weights.get("c1").expect("c1 weight"), 3.0);
        assert_close(*outcome.weights.get("a3").expect("a3 weight"), 2.0);
        assert_close(*outcome.weights.get("c2").expect("c2 weight"), 2.0);
    }
}
