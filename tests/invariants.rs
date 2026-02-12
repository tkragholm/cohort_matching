use chrono::{Months, NaiveDate};
use cohort_matching::{
    BaseRecord, DeterministicSelection, MatchingCriteria, MatchingCriteriaBuilder, MatchingRecord,
    RoleTransitionOptions, RoleTransitionRecord, match_anchors_to_candidates,
    match_with_role_transition_with_strategy,
};
use std::collections::{HashMap, HashSet};

const fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).expect("valid fixed test date")
}

fn base_record(id: &str, day_offset: i64) -> BaseRecord {
    let base = date(2010, 1, 1);
    BaseRecord::new(id, base + chrono::TimeDelta::days(day_offset))
}

#[test]
fn property_no_self_match_and_no_reuse_without_replacement() {
    let criteria = MatchingCriteria {
        birth_date_window_days: 365,
        match_ratio: 2,
        allow_replacement: false,
        ..MatchingCriteria::default()
    };

    for seed in 0_u64..100 {
        let anchors = (0..20)
            .map(|idx| base_record(&format!("id_{idx}"), i64::from(idx)))
            .collect::<Vec<_>>();
        let candidates = (0..40)
            .map(|idx| base_record(&format!("id_{idx}"), i64::from(idx % 20)))
            .collect::<Vec<_>>();

        let outcome = match_anchors_to_candidates(&anchors, &candidates, &criteria, seed);

        for pair in &outcome.pairs {
            assert_ne!(pair.anchor_id(), pair.comparator_id());
        }

        let unique_controls = outcome
            .pairs
            .iter()
            .map(cohort_matching::MatchedPair::comparator_id)
            .collect::<HashSet<_>>();
        assert_eq!(unique_controls.len(), outcome.pairs.len());
        assert_eq!(outcome.used_comparators(), outcome.pairs.len());
    }
}

#[test]
fn property_role_transition_respects_age_threshold() {
    let criteria = MatchingCriteria {
        birth_date_window_days: 730,
        allow_replacement: false,
        ..MatchingCriteria::default()
    };
    let options = RoleTransitionOptions {
        transition_age_limit_years: 6,
        ratio_fallback: vec![1],
    };

    for seed in 0_i64..80 {
        let cohort = (0..25)
            .map(|idx| {
                let row = base_record(
                    &format!("p_{idx}"),
                    i64::try_from(idx).expect("small non-negative index"),
                );

                let birth = row.birth_date;
                let transition_date = match (idx + usize::try_from(seed % 3).expect("small")) % 4 {
                    0 => Some(birth + chrono::TimeDelta::days(365 * 4)),
                    1 => Some(birth + chrono::TimeDelta::days(365 * 7)),
                    2 => None,
                    _ => Some(birth + chrono::TimeDelta::days(365 * 2)),
                };

                RoleTransitionRecord::from_record(row, transition_date)
            })
            .collect::<Vec<_>>();

        let eligible_ids = cohort
            .iter()
            .filter_map(|row| {
                let event = row.transition_date?;
                let age_limit = row
                    .record
                    .birth_date
                    .checked_add_months(Months::new(6 * 12))
                    .expect("valid age-limit date");
                if event < age_limit {
                    Some(row.record.id.as_str())
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();

        let outcome = match_with_role_transition_with_strategy(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
        );

        for pair in &outcome.pairs {
            assert!(eligible_ids.contains(pair.anchor_id()));
            assert_ne!(pair.anchor_id(), pair.comparator_id());
        }
    }
}

#[test]
fn property_anchor_candidate_api_supports_multiple_record_shapes() {
    #[derive(Clone)]
    struct CompactRecord {
        id: String,
        when: NaiveDate,
        strata: HashMap<String, String>,
    }

    impl MatchingRecord for CompactRecord {
        fn id(&self) -> &str {
            &self.id
        }

        fn birth_date(&self) -> NaiveDate {
            self.when
        }

        fn strata(&self) -> &HashMap<String, String> {
            &self.strata
        }

        fn unique_key(&self) -> Option<&str> {
            None
        }
    }

    let criteria = MatchingCriteria {
        birth_date_window_days: 365,
        match_ratio: 1,
        allow_replacement: false,
        ..MatchingCriteria::default()
    };

    let anchors_a = vec![base_record("a0", 0), base_record("a1", 1)];
    let candidates_a = vec![base_record("c0", 0), base_record("c1", 2)];
    let outcome_a = match_anchors_to_candidates(&anchors_a, &candidates_a, &criteria, 1);
    assert_eq!(outcome_a.matched_cases, 2);

    let anchors_b = vec![CompactRecord {
        id: "a0".to_string(),
        when: date(2010, 1, 1),
        strata: HashMap::new(),
    }];
    let candidates_b = vec![CompactRecord {
        id: "c0".to_string(),
        when: date(2010, 1, 2),
        strata: HashMap::new(),
    }];
    let outcome_b = match_anchors_to_candidates(&anchors_b, &candidates_b, &criteria, 1);
    assert_eq!(outcome_b.matched_cases, 1);
    assert_eq!(outcome_b.pairs.len(), 1);
}

#[test]
fn criteria_builder_validates() {
    assert!(
        MatchingCriteriaBuilder::new()
            .match_ratio(0)
            .build()
            .is_err()
    );

    let validated = MatchingCriteriaBuilder::new()
        .birth_date_window_days(30)
        .match_ratio(1)
        .build()
        .expect("valid criteria from builder");
    assert_eq!(validated.as_ref().match_ratio, 1);
}
