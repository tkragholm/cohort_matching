use chrono::NaiveDate;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct ReconstructedEpisode {
    pub person_id: String,
    pub index_date: NaiveDate,
    pub exposed: bool,
    pub sex: Option<i32>,
    pub municipality: Option<String>,
    pub region: Option<String>,
    pub family_id: Option<String>,
    pub birth_date: Option<NaiveDate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructedMatchingTier {
    Municipality,
    RegionFallback,
}

#[derive(Debug, Clone)]
pub struct ReconstructedMatchedPair {
    pub case_id: String,
    pub control_id: String,
    pub index_date: NaiveDate,
    pub tier: ReconstructedMatchingTier,
    pub birth_date_distance_days: Option<i64>,
}

#[derive(Debug, Clone, Copy)]
pub struct ReconstructedMatchingOptions {
    pub max_ratio: usize,
    pub match_caliper_days: i64,
}

#[derive(Debug, Default)]
pub struct ReconstructedMatchingOutput {
    pub eligible_case_ids: HashSet<String>,
    pub matched_case_ids: HashSet<String>,
    pub matched_control_ids: HashSet<String>,
    pub municipality_case_ids: HashSet<String>,
    pub municipality_control_ids: HashSet<String>,
    pub region_case_ids: HashSet<String>,
    pub region_control_ids: HashSet<String>,
    pub matched_pairs: Vec<ReconstructedMatchedPair>,
}

#[derive(Clone, Copy)]
struct CaseState<'a> {
    row: &'a ReconstructedEpisode,
    assigned: usize,
}

type CaseGroupIndex = HashMap<(NaiveDate, i32), HashMap<String, Vec<usize>>>;

#[must_use]
pub fn reconstruct_case_control_pairs(
    rows: &[ReconstructedEpisode],
    options: ReconstructedMatchingOptions,
) -> ReconstructedMatchingOutput {
    let max_ratio = options.max_ratio.max(1);
    let caliper_days = options.match_caliper_days.max(0);

    let mut case_states = dedup_case_states(rows);
    if case_states.is_empty() {
        return ReconstructedMatchingOutput::default();
    }

    let (municipality_case_index, region_case_index) = build_case_indices(&case_states);

    let mut output = ReconstructedMatchingOutput {
        eligible_case_ids: case_states
            .iter()
            .map(|case| case.row.person_id.clone())
            .collect(),
        ..ReconstructedMatchingOutput::default()
    };

    for control_row in rows.iter().filter(|row| !row.exposed) {
        let selected = select_case_for_control(
            control_row,
            &case_states,
            &municipality_case_index,
            &region_case_index,
            caliper_days,
            max_ratio,
        );

        if let Some((case_idx, tier)) = selected {
            case_states[case_idx].assigned = case_states[case_idx].assigned.saturating_add(1);
            let case_row = case_states[case_idx].row;
            record_reconstructed_match(&mut output, case_row, control_row, tier);
        }
    }

    output
}

fn dedup_case_states(rows: &[ReconstructedEpisode]) -> Vec<CaseState<'_>> {
    let mut case_states = Vec::<CaseState<'_>>::new();
    let mut seen_case_episodes = HashSet::<(&str, NaiveDate)>::new();
    for row in rows.iter().filter(|row| row.exposed) {
        let key = (row.person_id.as_str(), row.index_date);
        if seen_case_episodes.insert(key) {
            case_states.push(CaseState { row, assigned: 0 });
        }
    }
    case_states
}

fn build_case_indices(case_states: &[CaseState<'_>]) -> (CaseGroupIndex, CaseGroupIndex) {
    let mut municipality_case_index = CaseGroupIndex::new();
    let mut region_case_index = CaseGroupIndex::new();
    for (case_idx, case) in case_states.iter().enumerate() {
        let sex_key = case.row.sex.unwrap_or(-1);
        if let Some(municipality) = normalized_group_value(case.row.municipality.as_deref()) {
            municipality_case_index
                .entry((case.row.index_date, sex_key))
                .or_default()
                .entry(municipality.to_string())
                .or_default()
                .push(case_idx);
        }
        if let Some(region) = normalized_group_value(case.row.region.as_deref()) {
            region_case_index
                .entry((case.row.index_date, sex_key))
                .or_default()
                .entry(region.to_string())
                .or_default()
                .push(case_idx);
        }
    }
    (municipality_case_index, region_case_index)
}

fn select_case_for_control(
    control_row: &ReconstructedEpisode,
    case_states: &[CaseState<'_>],
    municipality_case_index: &CaseGroupIndex,
    region_case_index: &CaseGroupIndex,
    caliper_days: i64,
    max_ratio: usize,
) -> Option<(usize, ReconstructedMatchingTier)> {
    let sex_key = control_row.sex.unwrap_or(-1);
    normalized_group_value(control_row.municipality.as_deref())
        .and_then(|municipality| {
            municipality_case_index
                .get(&(control_row.index_date, sex_key))
                .and_then(|index| index.get(municipality))
                .and_then(|candidate_indices| {
                    select_best_case(
                        case_states,
                        candidate_indices,
                        control_row,
                        caliper_days,
                        max_ratio,
                    )
                })
                .map(|case_idx| (case_idx, ReconstructedMatchingTier::Municipality))
        })
        .or_else(|| {
            normalized_group_value(control_row.region.as_deref()).and_then(|region| {
                region_case_index
                    .get(&(control_row.index_date, sex_key))
                    .and_then(|index| index.get(region))
                    .and_then(|candidate_indices| {
                        select_best_case(
                            case_states,
                            candidate_indices,
                            control_row,
                            caliper_days,
                            max_ratio,
                        )
                    })
                    .map(|case_idx| (case_idx, ReconstructedMatchingTier::RegionFallback))
            })
        })
}

fn record_reconstructed_match(
    output: &mut ReconstructedMatchingOutput,
    case_row: &ReconstructedEpisode,
    control_row: &ReconstructedEpisode,
    tier: ReconstructedMatchingTier,
) {
    output.matched_case_ids.insert(case_row.person_id.clone());
    output
        .matched_control_ids
        .insert(control_row.person_id.clone());

    match tier {
        ReconstructedMatchingTier::Municipality => {
            output
                .municipality_case_ids
                .insert(case_row.person_id.clone());
            output
                .municipality_control_ids
                .insert(control_row.person_id.clone());
        }
        ReconstructedMatchingTier::RegionFallback => {
            output.region_case_ids.insert(case_row.person_id.clone());
            output
                .region_control_ids
                .insert(control_row.person_id.clone());
        }
    }

    output.matched_pairs.push(ReconstructedMatchedPair {
        case_id: case_row.person_id.clone(),
        control_id: control_row.person_id.clone(),
        index_date: case_row.index_date,
        tier,
        birth_date_distance_days: birth_date_distance_days(case_row, control_row),
    });
}

fn select_best_case(
    case_states: &[CaseState<'_>],
    candidate_indices: &[usize],
    control_row: &ReconstructedEpisode,
    caliper_days: i64,
    max_ratio: usize,
) -> Option<usize> {
    let mut selected: Option<(usize, usize, i64, &str)> = None;
    for case_idx in candidate_indices {
        let Some(case_state) = case_states.get(*case_idx) else {
            continue;
        };
        if case_state.assigned >= max_ratio {
            continue;
        }
        let case_row = case_state.row;
        if !case_control_eligible(case_row, control_row, caliper_days) {
            continue;
        }
        let distance_days = birth_date_distance_days(case_row, control_row).unwrap_or(i64::MAX);
        let candidate = (
            *case_idx,
            case_state.assigned,
            distance_days,
            case_row.person_id.as_str(),
        );
        match selected {
            None => selected = Some(candidate),
            Some(current) => {
                if candidate.1 < current.1
                    || (candidate.1 == current.1 && candidate.2 < current.2)
                    || (candidate.1 == current.1
                        && candidate.2 == current.2
                        && candidate.3 < current.3)
                {
                    selected = Some(candidate);
                }
            }
        }
    }
    selected.map(|choice| choice.0)
}

fn case_control_eligible(
    case_row: &ReconstructedEpisode,
    control_row: &ReconstructedEpisode,
    caliper_days: i64,
) -> bool {
    if case_row.person_id == control_row.person_id {
        return false;
    }
    if let (Some(case_family), Some(control_family)) = (
        normalized_group_value(case_row.family_id.as_deref()),
        normalized_group_value(control_row.family_id.as_deref()),
    ) && case_family == control_family
    {
        return false;
    }
    if caliper_days > 0
        && let Some(distance_days) = birth_date_distance_days(case_row, control_row)
        && distance_days > caliper_days
    {
        return false;
    }
    true
}

fn birth_date_distance_days(
    case_row: &ReconstructedEpisode,
    control_row: &ReconstructedEpisode,
) -> Option<i64> {
    Some(
        case_row
            .birth_date?
            .signed_duration_since(control_row.birth_date?)
            .num_days()
            .abs(),
    )
}

fn normalized_group_value(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::{
        ReconstructedEpisode, ReconstructedMatchingOptions, ReconstructedMatchingTier,
        reconstruct_case_control_pairs,
    };
    use chrono::NaiveDate;

    fn d(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).expect("hardcoded valid date")
    }

    #[test]
    fn reconstructs_pairs_with_municipality_then_region_fallback() {
        let rows = vec![
            ReconstructedEpisode {
                person_id: "case-a".to_string(),
                index_date: d(2021, 1, 1),
                exposed: true,
                sex: Some(1),
                municipality: Some("0101".to_string()),
                region: Some("H".to_string()),
                family_id: Some("fam-a".to_string()),
                birth_date: Some(d(2010, 1, 1)),
            },
            ReconstructedEpisode {
                person_id: "ctrl-a".to_string(),
                index_date: d(2021, 1, 1),
                exposed: false,
                sex: Some(1),
                municipality: Some("0101".to_string()),
                region: Some("H".to_string()),
                family_id: Some("fam-z".to_string()),
                birth_date: Some(d(2010, 1, 2)),
            },
            ReconstructedEpisode {
                person_id: "ctrl-b".to_string(),
                index_date: d(2021, 1, 1),
                exposed: false,
                sex: Some(1),
                municipality: Some("9999".to_string()),
                region: Some("H".to_string()),
                family_id: Some("fam-y".to_string()),
                birth_date: Some(d(2010, 1, 3)),
            },
        ];
        let output = reconstruct_case_control_pairs(
            &rows,
            ReconstructedMatchingOptions {
                max_ratio: 2,
                match_caliper_days: 31,
            },
        );
        assert_eq!(output.matched_pairs.len(), 2);
        assert!(
            output
                .matched_pairs
                .iter()
                .any(|pair| pair.tier == ReconstructedMatchingTier::Municipality)
        );
        assert!(
            output
                .matched_pairs
                .iter()
                .any(|pair| pair.tier == ReconstructedMatchingTier::RegionFallback)
        );
    }
}
