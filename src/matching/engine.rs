use super::constraints::{
    CaliperConstraint, Constraint, ConstraintContext, ConstraintGroup, ExactMatchConstraint,
    NoSelfMatchConstraint, ReplacementConstraint, UniqueKeyConstraint, UsedControlsVec,
    build_strata_values, unique_value,
};
use super::distance::{
    DateDistance, DistanceChannel, DistanceConfig, DistanceMetric, IdMapMahalanobisDistance,
    IdMapPropensityScoreDistance,
};
use super::index::CandidateIndex;
use super::ratio::RatioPolicy;
use super::records::MatchingRecord;
use super::selection::SelectionStrategy;
use crate::types::{
    CommonSupport, CommonSupportFailureReason, ConstraintReason, ControlIdx, DistanceCaliper,
    DistanceCaliperReason, Estimand, EstimandDriftReason, ExclusionReason, InvalidCriteriaReason,
    MatchDiagnostics, MatchOutcome, MatchRatio, MatchedPair, MatchingCriteria, UniqueValueId,
};
use rapidhash::RapidHashMap;
use std::collections::HashMap;
use std::collections::HashSet;

fn initial_diagnostics(
    total_anchors: usize,
    requested_estimand: Estimand,
    common_support_policy: Option<CommonSupport>,
) -> MatchDiagnostics {
    MatchDiagnostics {
        total_anchors_evaluated: total_anchors,
        requested_estimand,
        realized_estimand: requested_estimand,
        common_support_policy,
        ..MatchDiagnostics::default()
    }
}

fn push_drift_reason(diagnostics: &mut MatchDiagnostics, reason: EstimandDriftReason) {
    if diagnostics.estimand_drift_reasons.contains(&reason) {
        return;
    }
    diagnostics.estimand_drift_reasons.push(reason);
}

pub fn finalize_estimand_diagnostics(
    diagnostics: &mut MatchDiagnostics,
    requested_estimand: Estimand,
    eligible_anchors: usize,
    distance_caliper_reason: Option<&str>,
) {
    diagnostics.requested_estimand = requested_estimand;

    if diagnostics.common_support_trimmed_anchors > 0
        || diagnostics.common_support_trimmed_candidates > 0
    {
        push_drift_reason(diagnostics, EstimandDriftReason::CommonSupportTrimming);
    }
    if eligible_anchors > diagnostics.matched_anchors {
        push_drift_reason(diagnostics, EstimandDriftReason::UnmatchedAnchors);
    }
    if diagnostics.anchors_below_required_ratio > 0 {
        push_drift_reason(diagnostics, EstimandDriftReason::RatioShortfall);
    }
    if let Some(reason) = distance_caliper_reason {
        let caliper_exclusions = diagnostics
            .exclusion_counts
            .get(&ExclusionReason::DistanceCaliper(
                DistanceCaliperReason::from_reason_str(reason),
            ))
            .copied()
            .unwrap_or(0);
        if caliper_exclusions > 0 {
            push_drift_reason(diagnostics, EstimandDriftReason::DistanceCaliperExclusion);
            if diagnostics.common_support_trimmed_anchors > 0
                || diagnostics.common_support_trimmed_candidates > 0
            {
                push_drift_reason(
                    diagnostics,
                    EstimandDriftReason::CommonSupportCaliperInteraction,
                );
            }
        }
    }

    diagnostics.realized_estimand = if diagnostics.estimand_drift_reasons.is_empty() {
        requested_estimand
    } else {
        Estimand::Atm
    };
}

/// Reusable matching engine with precomputed lookups and pluggable selection.
pub struct MatchEngine<'a, R: MatchingRecord, S: SelectionStrategy<R>> {
    pub(crate) controls: &'a [R],
    pub(crate) criteria: &'a MatchingCriteria,
    pub(crate) precomputed: MatchingPrecomputed<'a>,
    pub(crate) candidate_index: CandidateIndex<'a>,
    pub(crate) selector: S,
}

/// Unified request for standard anchor-to-candidate matching.
#[derive(Clone, Copy)]
pub struct StandardMatchRequest<'a, S, G: ?Sized> {
    pub criteria: &'a MatchingCriteria,
    pub strategy: S,
    pub constraints: &'a G,
    pub ratio_fallback: &'a [MatchRatio],
    pub distance_config: Option<&'a DistanceConfig>,
}

impl<'a, S, G: ?Sized> StandardMatchRequest<'a, S, G> {
    #[must_use]
    pub const fn new(criteria: &'a MatchingCriteria, strategy: S, constraints: &'a G) -> Self {
        Self {
            criteria,
            strategy,
            constraints,
            ratio_fallback: &[],
            distance_config: None,
        }
    }

    #[must_use]
    pub const fn with_ratio_fallback(mut self, ratio_fallback: &'a [MatchRatio]) -> Self {
        self.ratio_fallback = ratio_fallback;
        self
    }

    crate::impl_with_distance_config!();
}

use rayon::prelude::*;
use rustc_hash::FxHashSet;

/// Internal candidate-pool request used by role-transition and engine helpers.
#[derive(Clone, Copy)]
pub struct CandidatePoolRequest<'a, R: MatchingRecord, D: DistanceMetric<R>> {
    pub case: &'a R,
    pub used_controls: &'a UsedControlsVec,
    pub used_unique: &'a FxHashSet<UniqueValueId>,
    pub unique_interner: &'a RapidHashMap<String, UniqueValueId>,
    pub distance_channel: DistanceChannel<'a, R, D>,
    pub use_birth_date_index: bool,
}

#[derive(Clone, Copy)]
pub struct CandidatePoolConfig<'a, D> {
    pub used_controls: &'a UsedControlsVec,
    pub used_unique: &'a FxHashSet<UniqueValueId>,
    pub unique_interner: &'a RapidHashMap<String, UniqueValueId>,
    pub distance_metric: &'a D,
    pub distance_caliper: Option<DistanceCaliper>,
    pub use_birth_date_index: bool,
    pub caliper_reason: &'static str,
}

#[derive(Clone, Copy)]
pub struct DistanceRunConfig<'a, D> {
    pub distance_metric: &'a D,
    pub distance_caliper: Option<DistanceCaliper>,
    pub use_birth_date_index: bool,
    pub caliper_reason: &'static str,
}

#[derive(Clone, Copy, Default)]
pub struct TrimmedIndexConfig<'a> {
    pub anchors: Option<&'a HashSet<usize>>,
    pub candidates: Option<&'a HashSet<usize>>,
}

fn intersect_candidate_indices(candidate_indices: &mut Vec<usize>, indexed_indices: Vec<usize>) {
    let pre_filter_count = candidate_indices.len();
    if indexed_indices.len() < pre_filter_count {
        let set: HashSet<usize> = candidate_indices.drain(..).collect();
        candidate_indices.extend(indexed_indices.into_iter().filter(|idx| set.contains(idx)));
    } else {
        let set: HashSet<usize> = indexed_indices.into_iter().collect();
        candidate_indices.retain(|idx| set.contains(idx));
    }
}

fn first_blocking_constraint_reason<R: MatchingRecord, D: DistanceMetric<R> + Sync>(
    case: &R,
    control: &R,
    builtin_constraints: &[&dyn Constraint<R>],
    caliper_constraint: Option<&CaliperConstraint<'_, R, D>>,
    extra_constraints: &(impl ConstraintGroup<R> + ?Sized),
    context: &ConstraintContext<'_>,
) -> Option<&'static str> {
    for constraint in builtin_constraints.iter().copied().chain(
        caliper_constraint
            .as_ref()
            .map(|constraint| *constraint as &dyn Constraint<R>),
    ) {
        if !constraint.allows(case, control, context) {
            return Some(constraint.reason());
        }
    }
    extra_constraints.first_blocking_reason(case, control, context)
}

impl<'a, R: MatchingRecord + Sync, S: SelectionStrategy<R> + Clone + Send + Sync>
    MatchEngine<'a, R, S>
{
    /// Construct a reusable engine for a fixed control pool and criteria.
    #[must_use]
    pub fn new(controls: &'a [R], criteria: &'a MatchingCriteria, selector: S) -> Self {
        let control_strata_values = precompute_strata_values(controls, &criteria.required_strata);
        let precomputed = MatchingPrecomputed {
            criteria,
            control_strata_values,
        };

        Self {
            controls,
            criteria,
            precomputed,
            candidate_index: CandidateIndex::new(controls, &criteria.required_strata),
            selector,
        }
    }

    #[must_use]
    pub const fn controls(&self) -> &'a [R] {
        self.controls
    }

    pub(crate) fn candidate_pool_for_request_and_constraint_group<
        F,
        D: DistanceMetric<R> + Sync,
        G: ConstraintGroup<R> + ?Sized,
    >(
        &self,
        request: &CandidatePoolRequest<'_, R, D>,
        extra_constraints: &G,
        additional_filter: F,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<usize>
    where
        F: FnMut(usize, &R) -> bool,
    {
        self.candidate_pool_with_distance_and_constraint_group(
            request.case,
            &CandidatePoolConfig {
                used_controls: request.used_controls,
                used_unique: request.used_unique,
                unique_interner: request.unique_interner,
                distance_metric: request.distance_channel.metric(),
                distance_caliper: request.distance_channel.typed_caliper(),
                use_birth_date_index: request.use_birth_date_index,
                caliper_reason: request.distance_channel.reason(),
            },
            extra_constraints,
            additional_filter,
            diagnostics,
        )
    }

    pub(crate) fn candidate_pool_with_distance_and_constraint_group<
        F,
        D: DistanceMetric<R> + Sync,
        G: ConstraintGroup<R> + ?Sized,
    >(
        &self,
        case: &R,
        pool_config: &CandidatePoolConfig<'_, D>,
        extra_constraints: &G,
        mut additional_filter: F,
        diagnostics: &mut MatchDiagnostics,
    ) -> Vec<usize>
    where
        F: FnMut(usize, &R) -> bool,
    {
        let case_strata_values = (!self.criteria.required_strata.is_empty())
            .then(|| build_strata_values(case.strata(), &self.criteria.required_strata));

        let mut candidate_indices = if pool_config.use_birth_date_index {
            let window_days = self
                .criteria
                .typed_birth_date_window()
                .map_or(0_i64, |days| i64::from(days.get()));
            self.candidate_index.candidate_indices(
                case.birth_date(),
                window_days,
                case_strata_values.as_deref(),
            )
        } else {
            (0..self.controls.len()).collect()
        };

        if let Some(max_distance) = pool_config.distance_caliper
            && let Some(indexed_indices) = pool_config
                .distance_metric
                .candidate_indices(case, max_distance.get())
        {
            let pre_filter_count = candidate_indices.len();
            intersect_candidate_indices(&mut candidate_indices, indexed_indices);
            let excluded_by_index = pre_filter_count.saturating_sub(candidate_indices.len());
            if excluded_by_index > 0 {
                let reason = pool_config.caliper_reason;
                *diagnostics
                    .exclusion_counts
                    .entry(ExclusionReason::DistanceCaliper(
                        DistanceCaliperReason::from_reason_str(reason),
                    ))
                    .or_insert(0) += excluded_by_index;
            }
        }
        let mut eligible = Vec::new();

        let replacement_constraint = ReplacementConstraint;
        let no_self_match_constraint = NoSelfMatchConstraint;
        let strata_constraint = ExactMatchConstraint;
        let unique_constraint = UniqueKeyConstraint;
        let builtin_constraints: [&dyn Constraint<R>; 4] = [
            &replacement_constraint,
            &no_self_match_constraint,
            &strata_constraint,
            &unique_constraint,
        ];
        let caliper_constraint = pool_config.distance_caliper.map(|max_distance| {
            CaliperConstraint::new(
                pool_config.distance_metric,
                max_distance.get(),
                pool_config.caliper_reason,
            )
        });

        for idx in candidate_indices {
            let control = &self.controls[idx];

            if !additional_filter(idx, control) {
                bump_count(diagnostics, ExclusionReason::AdditionalFilter);
                continue;
            }

            let control_strata_values = self
                .precomputed
                .control_strata_values
                .as_deref()
                .and_then(|values| values.get(idx).map(Vec::as_slice));
            let context = ConstraintContext {
                criteria: self.precomputed.criteria,
                used_controls: pool_config.used_controls,
                used_unique: pool_config.used_unique,
                unique_interner: pool_config.unique_interner,
                control_idx: ControlIdx::new(idx),
                case_strata_values: case_strata_values.as_deref(),
                control_strata_values,
            };

            if let Some(reason) = first_blocking_constraint_reason(
                case,
                control,
                &builtin_constraints,
                caliper_constraint.as_ref(),
                extra_constraints,
                &context,
            ) {
                bump_count(
                    diagnostics,
                    ExclusionReason::Constraint(ConstraintReason::from_reason_str(reason)),
                );
                continue;
            }

            eligible.push(idx);
        }

        eligible
    }

    pub fn select_control(
        &mut self,
        case: &R,
        candidate_indices: &mut Vec<usize>,
    ) -> Option<ControlIdx> {
        let selected_pos = self
            .selector
            .select_position(case, self.controls, candidate_indices)?;
        Some(ControlIdx::new(candidate_indices.swap_remove(selected_pos)))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "parallel matching workflow keeps shared state local"
    )]
    pub(crate) fn run_anchor_matching_with_distance_internal<
        D: DistanceMetric<R> + Sync,
        G: ConstraintGroup<R> + ?Sized,
    >(
        &self,
        anchors: &[R],
        distance_config: &DistanceRunConfig<'_, D>,
        extra_constraints: &G,
        ratio_policy: &RatioPolicy,
        trimmed_indices: TrimmedIndexConfig<'_>,
        mut diagnostics: MatchDiagnostics,
    ) -> MatchOutcome {
        struct GroupResult {
            pairs: Vec<MatchedPair>,
            matched_cases: usize,
            eligible_anchors: usize,
            used_controls_count: usize,
            diagnostics: MatchDiagnostics,
        }

        if self.criteria.allow_replacement && self.criteria.unique_by_key.is_none() {
            return self.run_anchor_matching_parallel(
                anchors,
                distance_config,
                extra_constraints,
                ratio_policy,
                trimmed_indices,
                diagnostics,
            );
        }

        // Group anchors by strata key so each group can run independently in parallel.
        // Controls in different strata groups are never interchangeable (the CandidateIndex
        // guarantees strata-exact matching), so `used_controls` is strata-local.
        // `used_unique` (family_id) is also tracked per strata group; cross-strata family_id
        // conflicts are rare in practice since strata differ by parity/birth_type.
        let n_controls = self.controls.len();
        let strata_groups =
            group_anchors_by_strata(anchors, &self.criteria.required_strata, &trimmed_indices);

        let group_results: Vec<GroupResult> = strata_groups
            .into_par_iter()
            .map(|anchor_indices| {
                let mut used_controls = UsedControlsVec::with_capacity(n_controls);
                let mut used_unique = FxHashSet::<UniqueValueId>::default();
                let mut unique_interner = RapidHashMap::<String, UniqueValueId>::default();
                let mut pairs = Vec::new();
                let mut matched_cases = 0usize;
                let mut eligible_anchors = 0usize;
                let mut local_diagnostics = MatchDiagnostics::default();
                let mut local_selector = self.selector.clone();

                for anchor_idx in anchor_indices {
                    let case = &anchors[anchor_idx];
                    eligible_anchors += 1;

                    let mut candidates = self.candidate_pool_with_distance_and_constraint_group(
                        case,
                        &CandidatePoolConfig {
                            used_controls: &used_controls,
                            used_unique: &used_unique,
                            unique_interner: &unique_interner,
                            distance_metric: distance_config.distance_metric,
                            distance_caliper: distance_config.distance_caliper,
                            use_birth_date_index: distance_config.use_birth_date_index,
                            caliper_reason: distance_config.caliper_reason,
                        },
                        extra_constraints,
                        |idx, _control| {
                            !trimmed_indices
                                .candidates
                                .is_some_and(|trimmed| trimmed.contains(&idx))
                        },
                        &mut local_diagnostics,
                    );
                    if candidates.is_empty() {
                        local_diagnostics.anchors_with_no_candidates += 1;
                        continue;
                    }

                    let Some(target_ratio) = ratio_policy.target_ratio(candidates.len()) else {
                        local_diagnostics.anchors_below_required_ratio += 1;
                        continue;
                    };
                    if ratio_policy.is_shortfall(candidates.len()) {
                        local_diagnostics.anchors_below_required_ratio += 1;
                    }

                    let mut local_matches = 0usize;
                    for _ in 0..target_ratio {
                        let Some(pos) =
                            local_selector.select_position(case, self.controls, &candidates)
                        else {
                            break;
                        };
                        let raw_idx = candidates.swap_remove(pos);
                        let selected = ControlIdx::new(raw_idx);
                        let control = &self.controls[raw_idx];
                        pairs.push(MatchedPair::new(case.id(), control.id()));
                        local_matches += 1;

                        if !self.criteria.allow_replacement {
                            used_controls.insert(selected);
                        }
                        if let Some(value) = unique_value(control, self.criteria) {
                            let next_id = unique_interner.len();
                            let id = *unique_interner
                                .entry(value.to_string())
                                .or_insert_with(|| UniqueValueId::new(next_id));
                            used_unique.insert(id);
                        }
                    }
                    if local_matches > 0 {
                        matched_cases += 1;
                    }
                }

                GroupResult {
                    pairs,
                    matched_cases,
                    eligible_anchors,
                    used_controls_count: used_controls.len(),
                    diagnostics: local_diagnostics,
                }
            })
            .collect();

        let mut pairs = Vec::new();
        let mut matched_cases = 0usize;
        let mut eligible_anchors = 0usize;
        let mut used_controls_count = 0usize;
        for gr in group_results {
            pairs.extend(gr.pairs);
            matched_cases += gr.matched_cases;
            eligible_anchors += gr.eligible_anchors;
            used_controls_count += gr.used_controls_count;
            diagnostics.merge(gr.diagnostics);
        }

        diagnostics.matched_anchors = matched_cases;
        diagnostics.pairs_selected = pairs.len();
        finalize_estimand_diagnostics(
            &mut diagnostics,
            self.criteria.estimand,
            eligible_anchors,
            distance_config
                .distance_caliper
                .is_some()
                .then_some(distance_config.caliper_reason),
        );
        build_outcome(
            pairs,
            matched_cases,
            eligible_anchors,
            used_controls_count,
            diagnostics,
        )
    }

    fn run_anchor_matching_parallel<D: DistanceMetric<R> + Sync, G: ConstraintGroup<R> + ?Sized>(
        &self,
        anchors: &[R],
        distance_config: &DistanceRunConfig<'_, D>,
        extra_constraints: &G,
        ratio_policy: &RatioPolicy,
        trimmed_indices: TrimmedIndexConfig<'_>,
        mut diagnostics: MatchDiagnostics,
    ) -> MatchOutcome
    where
        S: Clone + Send + Sync,
        R: Sync,
    {
        let empty_controls = UsedControlsVec::with_capacity(0);
        let empty_unique = FxHashSet::<UniqueValueId>::default();
        let empty_interner = RapidHashMap::<String, UniqueValueId>::default();

        let results: Vec<(Vec<MatchedPair>, MatchDiagnostics)> = anchors
            .par_iter()
            .enumerate()
            .filter_map(|(anchor_idx, case)| {
                if trimmed_indices
                    .anchors
                    .is_some_and(|trimmed| trimmed.contains(&anchor_idx))
                {
                    return None;
                }

                let mut local_diagnostics = MatchDiagnostics::default();
                let mut local_selector = self.selector.clone();

                let mut candidates = self.candidate_pool_with_distance_and_constraint_group(
                    case,
                    &CandidatePoolConfig {
                        used_controls: &empty_controls,
                        used_unique: &empty_unique,
                        unique_interner: &empty_interner,
                        distance_metric: distance_config.distance_metric,
                        distance_caliper: distance_config.distance_caliper,
                        use_birth_date_index: distance_config.use_birth_date_index,
                        caliper_reason: distance_config.caliper_reason,
                    },
                    extra_constraints,
                    |idx, _control| {
                        !trimmed_indices
                            .candidates
                            .is_some_and(|trimmed| trimmed.contains(&idx))
                    },
                    &mut local_diagnostics,
                );

                if candidates.is_empty() {
                    local_diagnostics.anchors_with_no_candidates += 1;
                    return Some((Vec::new(), local_diagnostics));
                }

                let Some(target_ratio) = ratio_policy.target_ratio(candidates.len()) else {
                    local_diagnostics.anchors_below_required_ratio += 1;
                    return Some((Vec::new(), local_diagnostics));
                };
                if ratio_policy.is_shortfall(candidates.len()) {
                    local_diagnostics.anchors_below_required_ratio += 1;
                }

                let mut local_pairs = Vec::new();
                for _ in 0..target_ratio {
                    let Some(selected_idx) =
                        local_selector.select_position(case, self.controls, &candidates)
                    else {
                        break;
                    };
                    let control_idx = candidates[selected_idx];
                    let control = &self.controls[control_idx];
                    local_pairs.push(MatchedPair::new(case.id(), control.id()));
                    candidates.remove(selected_idx);
                }

                if !local_pairs.is_empty() {
                    local_diagnostics.matched_anchors = 1;
                }
                local_diagnostics.pairs_selected = local_pairs.len();
                local_diagnostics.total_anchors_evaluated = 1;

                Some((local_pairs, local_diagnostics))
            })
            .collect();

        let mut all_pairs = Vec::new();
        let mut eligible_anchors = 0usize;
        for (mut pairs, diag) in results {
            all_pairs.append(&mut pairs);
            eligible_anchors += diag.total_anchors_evaluated;
            diagnostics.merge(diag);
        }

        finalize_estimand_diagnostics(
            &mut diagnostics,
            self.criteria.estimand,
            eligible_anchors,
            distance_config
                .distance_caliper
                .is_some()
                .then_some(distance_config.caliper_reason),
        );

        build_outcome(
            all_pairs,
            diagnostics.matched_anchors,
            eligible_anchors,
            0, // with replacement, unique candidates is 0 by convention or not tracked
            diagnostics,
        )
    }
}

/// Run anchor/candidate matching with an explicit control selection strategy.
#[must_use]
pub fn match_standard<
    R: MatchingRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
>(
    anchors: &[R],
    candidates: &[R],
    request: StandardMatchRequest<'_, S, G>,
) -> MatchOutcome {
    let validated = match request.criteria.validate() {
        Ok(validated) => validated,
        Err(err) => {
            return invalid_criteria_outcome(anchors.len(), InvalidCriteriaReason::from(err));
        }
    };
    let criteria = validated.as_ref();
    let engine = MatchEngine::new(candidates, criteria, request.strategy);
    let ratio_policy = RatioPolicy::from_fallback(validated.match_ratio, request.ratio_fallback);
    let diagnostics =
        initial_diagnostics(anchors.len(), criteria.estimand, criteria.common_support);

    match request.distance_config {
        None => {
            let date_distance = DateDistance;
            engine.run_anchor_matching_with_distance_internal(
                anchors,
                &DistanceRunConfig {
                    distance_metric: &date_distance,
                    distance_caliper: criteria.typed_birth_date_caliper(),
                    use_birth_date_index: true,
                    caliper_reason: "date_caliper",
                },
                request.constraints,
                &ratio_policy,
                TrimmedIndexConfig::default(),
                diagnostics,
            )
        }
        Some(distance_config) => match distance_config {
            DistanceConfig::Date { caliper, reason } => {
                let metric = DateDistance;
                engine.run_anchor_matching_with_distance_internal(
                    anchors,
                    &DistanceRunConfig {
                        distance_metric: &metric,
                        distance_caliper: *caliper,
                        use_birth_date_index: false,
                        caliper_reason: reason,
                    },
                    request.constraints,
                    &ratio_policy,
                    TrimmedIndexConfig::default(),
                    diagnostics,
                )
            }
            DistanceConfig::PropensityScoreMap {
                scores,
                caliper,
                reason,
            } => {
                let metric = IdMapPropensityScoreDistance::new(scores);
                engine.run_anchor_matching_with_distance_internal(
                    anchors,
                    &DistanceRunConfig {
                        distance_metric: &metric,
                        distance_caliper: *caliper,
                        use_birth_date_index: false,
                        caliper_reason: reason,
                    },
                    request.constraints,
                    &ratio_policy,
                    TrimmedIndexConfig::default(),
                    diagnostics,
                )
            }
            DistanceConfig::MahalanobisMap {
                vectors,
                inverse_covariance,
                dimension,
                caliper,
                reason,
            } => {
                let metric = IdMapMahalanobisDistance::new(vectors, inverse_covariance, *dimension);
                engine.run_anchor_matching_with_distance_internal(
                    anchors,
                    &DistanceRunConfig {
                        distance_metric: &metric,
                        distance_caliper: *caliper,
                        use_birth_date_index: false,
                        caliper_reason: reason,
                    },
                    request.constraints,
                    &ratio_policy,
                    TrimmedIndexConfig::default(),
                    diagnostics,
                )
            }
        },
    }
}

/// Partition `anchors` by strata key, returning one `Vec<usize>` of anchor indices per group.
/// Anchors trimmed by `trimmed_indices` are excluded.
fn group_anchors_by_strata<R: MatchingRecord>(
    anchors: &[R],
    required_strata: &[String],
    trimmed_indices: &TrimmedIndexConfig<'_>,
) -> Vec<Vec<usize>> {
    let mut groups: HashMap<Vec<Option<String>>, Vec<usize>> = HashMap::new();
    for (anchor_idx, case) in anchors.iter().enumerate() {
        if trimmed_indices
            .anchors
            .is_some_and(|trimmed| trimmed.contains(&anchor_idx))
        {
            continue;
        }
        let key: Vec<Option<String>> = required_strata
            .iter()
            .map(|k| case.strata().get(k).cloned())
            .collect();
        groups.entry(key).or_default().push(anchor_idx);
    }
    groups.into_values().collect()
}

pub struct MatchingPrecomputed<'a> {
    pub criteria: &'a MatchingCriteria,
    pub control_strata_values: Option<Vec<Vec<Option<&'a str>>>>,
}

fn precompute_strata_values<'a, R: MatchingRecord>(
    controls: &'a [R],
    strata_keys: &[String],
) -> Option<Vec<Vec<Option<&'a str>>>> {
    if strata_keys.is_empty() {
        return None;
    }
    Some(
        controls
            .iter()
            .map(|c| build_strata_values(c.strata(), strata_keys))
            .collect(),
    )
}

fn bump_count(diagnostics: &mut MatchDiagnostics, reason: ExclusionReason) {
    *diagnostics.exclusion_counts.entry(reason).or_insert(0) += 1;
}

pub fn build_outcome(
    pairs: Vec<MatchedPair>,
    matched_cases: usize,
    anchor_count: usize,
    used_controls: usize,
    diagnostics: MatchDiagnostics,
) -> MatchOutcome {
    let avg_controls_per_case = if matched_cases == 0 {
        0.0
    } else {
        usize_to_f64(pairs.len()) / usize_to_f64(matched_cases)
    };

    MatchOutcome {
        pairs,
        unmatched_cases: anchor_count.saturating_sub(matched_cases),
        used_controls,
        matched_cases,
        avg_controls_per_case,
        diagnostics,
    }
}

pub fn to_f64(n: usize) -> f64 {
    usize_to_f64(n)
}

fn usize_to_f64(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

pub fn invalid_criteria_outcome(
    anchor_count: usize,
    reason: InvalidCriteriaReason,
) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics
        .exclusion_counts
        .insert(ExclusionReason::InvalidCriteria(reason), 1);

    MatchOutcome {
        pairs: Vec::new(),
        unmatched_cases: anchor_count,
        used_controls: 0,
        matched_cases: 0,
        avg_controls_per_case: 0.0,
        diagnostics,
    }
}

pub fn invalid_common_support_outcome(
    anchor_count: usize,
    reason: CommonSupportFailureReason,
) -> MatchOutcome {
    let mut diagnostics = MatchDiagnostics {
        total_anchors_evaluated: anchor_count,
        ..MatchDiagnostics::default()
    };
    diagnostics
        .exclusion_counts
        .insert(ExclusionReason::CommonSupportFailure(reason), 1);

    MatchOutcome {
        pairs: Vec::new(),
        unmatched_cases: anchor_count,
        used_controls: 0,
        matched_cases: 0,
        avg_controls_per_case: 0.0,
        diagnostics,
    }
}
