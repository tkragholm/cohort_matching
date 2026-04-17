use crate::matching::{
    CandidatePoolRequest, ConstraintGroup, DateDistance, DistanceChannel, DistanceConfig,
    DistanceMetric, IdMapMahalanobisDistance, IdMapPropensityScoreDistance, MatchEngine,
    RoleIndexedRecord, SelectionStrategy, UsedControlsVec, build_outcome,
    finalize_estimand_diagnostics, invalid_common_support_outcome, invalid_criteria_outcome,
    ratio::RatioPolicy, unique_value,
};
use crate::types::{
    CommonSupport, CommonSupportFailureReason, InvalidCriteriaReason, MatchDiagnostics,
    MatchOutcome, MatchRatio, MatchedPair, MatchingCriteria, RoleTransitionOptions, UniqueValueId,
};
use chrono::{Months, NaiveDate};
use itertools::Itertools;
use rapidhash::RapidHashMap;
use std::collections::HashSet;

/// Policy hook for role-transition risk-set construction.
///
/// This allows downstream orchestrators to inject study-specific entry/exit logic
/// (e.g. alive/resident checks) while keeping matching mechanics generic.
pub trait RiskSetPolicy<R: RoleIndexedRecord> {
    /// Whether a record should be treated as an eligible case.
    ///
    /// Default behavior requires an event date before the configured age limit.
    fn is_case_eligible(&self, row: &R, age_limit_years: u8) -> bool {
        is_case_before_age_limit(row, age_limit_years)
    }

    /// Whether a record is at risk (eligible as a candidate) at the case event date.
    ///
    /// Default behavior allows candidates with no event date or with an event date
    /// strictly after the case event date.
    fn is_control_at_risk(&self, _case: &R, control: &R, case_event_date: NaiveDate) -> bool {
        default_control_is_at_risk(control, case_event_date)
    }
}

/// Default role-transition risk-set policy used by existing APIs.
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultRiskSetPolicy;

impl<R: RoleIndexedRecord> RiskSetPolicy<R> for DefaultRiskSetPolicy {}

/// Unified request for role-transition matching.
#[derive(Clone, Copy)]
pub struct TransitionMatchRequest<'a, S, G: ?Sized, P> {
    pub criteria: &'a MatchingCriteria,
    pub options: &'a RoleTransitionOptions,
    pub strategy: S,
    pub constraints: &'a G,
    pub distance_config: Option<&'a DistanceConfig>,
    pub risk_set_policy: &'a P,
}

impl<'a, S, G: ?Sized> TransitionMatchRequest<'a, S, G, DefaultRiskSetPolicy> {
    #[must_use]
    pub const fn new(
        criteria: &'a MatchingCriteria,
        options: &'a RoleTransitionOptions,
        strategy: S,
        constraints: &'a G,
    ) -> Self {
        Self {
            criteria,
            options,
            strategy,
            constraints,
            distance_config: None,
            risk_set_policy: &DefaultRiskSetPolicy,
        }
    }
}

impl<'a, S, G: ?Sized, P> TransitionMatchRequest<'a, S, G, P> {
    crate::impl_with_distance_config!();

    #[must_use]
    pub fn with_risk_set_policy<P2>(
        self,
        risk_set_policy: &'a P2,
    ) -> TransitionMatchRequest<'a, S, G, P2> {
        TransitionMatchRequest {
            criteria: self.criteria,
            options: self.options,
            strategy: self.strategy,
            constraints: self.constraints,
            distance_config: self.distance_config,
            risk_set_policy,
        }
    }
}

/// Risk-set matching with role transitions using neutral terminology.
#[must_use]
pub fn match_transition<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R>,
>(
    cohort: &[R],
    request: TransitionMatchRequest<'_, S, G, P>,
) -> MatchOutcome {
    if let Some(distance_config) = request.distance_config {
        return match_with_role_transition_with_strategy_and_constraint_group_and_distance_config_and_policy(
            cohort,
            request.criteria,
            request.options,
            request.strategy,
            request.constraints,
            distance_config,
            request.risk_set_policy,
        );
    }

    match_with_role_transition_with_strategy_and_constraint_group_and_policy(
        cohort,
        request.criteria,
        request.options,
        request.strategy,
        request.constraints,
        request.risk_set_policy,
    )
}

fn match_with_role_transition_with_strategy_and_constraint_group_and_policy<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    P: RiskSetPolicy<R>,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    options: &RoleTransitionOptions,
    strategy: S,
    extra_constraints: &G,
    risk_set_policy: &P,
) -> MatchOutcome {
    match_with_role_indexing_and_policy_with_constraint_group(
        cohort,
        criteria,
        options.transition_age_limit_years.get(),
        &options.ratio_fallback,
        strategy,
        extra_constraints,
        risk_set_policy,
    )
}

fn match_with_role_transition_with_strategy_and_constraint_group_and_distance_config_and_policy<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    P: RiskSetPolicy<R>,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    options: &RoleTransitionOptions,
    strategy: S,
    extra_constraints: &G,
    distance_config: &DistanceConfig,
    risk_set_policy: &P,
) -> MatchOutcome {
    match_with_role_transition_with_strategy_and_constraints_and_distance_config_internal(
        cohort,
        criteria,
        options,
        strategy,
        extra_constraints,
        distance_config,
        risk_set_policy,
    )
}

fn match_with_role_transition_with_strategy_and_constraints_and_distance_config_internal<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    P: RiskSetPolicy<R>,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    options: &RoleTransitionOptions,
    strategy: S,
    extra_constraints: &G,
    distance_config: &DistanceConfig,
    risk_set_policy: &P,
) -> MatchOutcome {
    match distance_config {
        DistanceConfig::Date { caliper, reason } => {
            if criteria.common_support.is_some() {
                return invalid_common_support_outcome(
                    cohort.len(),
                    CommonSupportFailureReason::RequiresPropensityScoreMap,
                );
            }
            let metric = DateDistance;
            match_with_role_indexing_with_channel_internal(
                cohort,
                strategy,
                &RoleIndexingConfig {
                    criteria,
                    age_limit_years: options.transition_age_limit_years.get(),
                    ratio_fallback: &options.ratio_fallback,
                    extra_constraints,
                    distance_channel: DistanceChannel::new(&metric, *caliper).with_reason(reason),
                    use_birth_date_index: false,
                    common_support_scores: None,
                    risk_set_policy,
                },
            )
        }
        DistanceConfig::PropensityScoreMap {
            scores,
            caliper,
            reason,
        } => {
            let metric = IdMapPropensityScoreDistance::new(scores);
            match_with_role_indexing_with_channel_internal(
                cohort,
                strategy,
                &RoleIndexingConfig {
                    criteria,
                    age_limit_years: options.transition_age_limit_years.get(),
                    ratio_fallback: &options.ratio_fallback,
                    extra_constraints,
                    distance_channel: DistanceChannel::new(&metric, *caliper).with_reason(reason),
                    use_birth_date_index: false,
                    common_support_scores: Some(scores),
                    risk_set_policy,
                },
            )
        }
        DistanceConfig::MahalanobisMap {
            vectors,
            inverse_covariance,
            dimension,
            caliper,
            reason,
        } => {
            if criteria.common_support.is_some() {
                return invalid_common_support_outcome(
                    cohort.len(),
                    CommonSupportFailureReason::RequiresPropensityScoreMap,
                );
            }
            let metric = IdMapMahalanobisDistance::new(vectors, inverse_covariance, *dimension);
            match_with_role_indexing_with_channel_internal(
                cohort,
                strategy,
                &RoleIndexingConfig {
                    criteria,
                    age_limit_years: options.transition_age_limit_years.get(),
                    ratio_fallback: &options.ratio_fallback,
                    extra_constraints,
                    distance_channel: DistanceChannel::new(&metric, *caliper).with_reason(reason),
                    use_birth_date_index: false,
                    common_support_scores: None,
                    risk_set_policy,
                },
            )
        }
    }
}

#[derive(Clone, Copy)]
struct RoleIndexingConfig<
    'a,
    R: RoleIndexedRecord,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
> {
    criteria: &'a MatchingCriteria,
    age_limit_years: u8,
    ratio_fallback: &'a [MatchRatio],
    extra_constraints: &'a G,
    distance_channel: DistanceChannel<'a, R, D>,
    use_birth_date_index: bool,
    common_support_scores: Option<&'a RapidHashMap<String, f64>>,
    risk_set_policy: &'a P,
}

#[derive(Default)]
struct TransitionCommonSupportFilters {
    trimmed_case_indices: HashSet<usize>,
    trimmed_candidate_indices: HashSet<usize>,
    overlap: Option<(f64, f64)>,
    case_score_bounds: Option<(f64, f64)>,
    candidate_score_bounds: Option<(f64, f64)>,
    policy: Option<CommonSupport>,
}

fn score_bounds_for_indices<R: RoleIndexedRecord>(
    cohort: &[R],
    indices: &[usize],
    scores: &RapidHashMap<String, f64>,
) -> Option<(f64, f64)> {
    let mut min_score = f64::INFINITY;
    let mut max_score = f64::NEG_INFINITY;
    let mut found = false;

    for idx in indices {
        let Some(row) = cohort.get(*idx) else {
            continue;
        };
        let Some(score) = scores.get(row.id()).copied() else {
            continue;
        };
        if !score.is_finite() {
            continue;
        }

        min_score = min_score.min(score);
        max_score = max_score.max(score);
        found = true;
    }

    found.then_some((min_score, max_score))
}

fn support_interval(left: (f64, f64), right: (f64, f64)) -> Option<(f64, f64)> {
    let lower = left.0.max(right.0);
    let upper = left.1.min(right.1);
    (lower <= upper).then_some((lower, upper))
}

fn score_inside_interval<R: RoleIndexedRecord>(
    row: &R,
    scores: &RapidHashMap<String, f64>,
    interval: (f64, f64),
) -> bool {
    scores
        .get(row.id())
        .copied()
        .is_some_and(|score| score.is_finite() && score >= interval.0 && score <= interval.1)
}

fn transition_common_support_filters<R: RoleIndexedRecord>(
    cohort: &[R],
    case_indices: &[usize],
    policy: CommonSupport,
    scores: &RapidHashMap<String, f64>,
) -> Option<TransitionCommonSupportFilters> {
    let candidate_indices = (0..cohort.len()).collect_vec();
    let case_bounds = score_bounds_for_indices(cohort, case_indices, scores)?;
    let candidate_bounds = score_bounds_for_indices(cohort, &candidate_indices, scores)?;
    let overlap = support_interval(case_bounds, candidate_bounds)?;

    let mut filters = TransitionCommonSupportFilters {
        overlap: Some(overlap),
        case_score_bounds: Some(case_bounds),
        candidate_score_bounds: Some(candidate_bounds),
        policy: Some(policy),
        ..TransitionCommonSupportFilters::default()
    };

    if matches!(policy, CommonSupport::Treated | CommonSupport::Both) {
        filters.trimmed_case_indices = case_indices
            .iter()
            .filter_map(|idx| {
                let row = cohort.get(*idx)?;
                (!score_inside_interval(row, scores, overlap)).then_some(*idx)
            })
            .collect();
    }

    if matches!(policy, CommonSupport::Control | CommonSupport::Both) {
        filters.trimmed_candidate_indices = candidate_indices
            .into_iter()
            .filter(|idx| {
                cohort
                    .get(*idx)
                    .is_some_and(|row| !score_inside_interval(row, scores, overlap))
            })
            .collect();
    }

    Some(filters)
}

fn resolve_transition_common_support<R: RoleIndexedRecord>(
    cohort: &[R],
    case_indices: &[usize],
    criteria: &MatchingCriteria,
    scores: Option<&RapidHashMap<String, f64>>,
) -> Result<TransitionCommonSupportFilters, CommonSupportFailureReason> {
    if criteria.common_support.is_some() && scores.is_none() {
        return Err(CommonSupportFailureReason::RequiresPropensityScoreMap);
    }

    if let (Some(policy), Some(scores)) = (criteria.common_support, scores) {
        return transition_common_support_filters(cohort, case_indices, policy, scores)
            .ok_or(CommonSupportFailureReason::NoOverlap);
    }

    Ok(TransitionCommonSupportFilters::default())
}

fn transition_initial_diagnostics(
    criteria: &MatchingCriteria,
    case_count: usize,
    filters: &TransitionCommonSupportFilters,
) -> MatchDiagnostics {
    MatchDiagnostics {
        total_anchors_evaluated: case_count,
        requested_estimand: criteria.estimand,
        realized_estimand: criteria.estimand,
        common_support_trimmed_anchors: filters.trimmed_case_indices.len(),
        common_support_trimmed_candidates: filters.trimmed_candidate_indices.len(),
        common_support_overlap: filters.overlap,
        common_support_policy: filters.policy.or(criteria.common_support),
        common_support_anchor_score_bounds: filters.case_score_bounds,
        common_support_candidate_score_bounds: filters.candidate_score_bounds,
        ..MatchDiagnostics::default()
    }
}

fn match_with_role_indexing_and_policy_with_constraint_group<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    P: RiskSetPolicy<R>,
    G: ConstraintGroup<R> + ?Sized,
>(
    cohort: &[R],
    criteria: &MatchingCriteria,
    age_limit_years: u8,
    ratio_fallback: &[MatchRatio],
    strategy: S,
    extra_constraints: &G,
    risk_set_policy: &P,
) -> MatchOutcome {
    let date_distance = DateDistance;
    match_with_role_indexing_with_channel_internal(
        cohort,
        strategy,
        &RoleIndexingConfig {
            criteria,
            age_limit_years,
            ratio_fallback,
            extra_constraints,
            distance_channel: DistanceChannel::new(
                &date_distance,
                criteria.typed_birth_date_caliper(),
            ),
            use_birth_date_index: true,
            common_support_scores: None,
            risk_set_policy,
        },
    )
}

fn match_with_role_indexing_with_channel_internal<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
>(
    cohort: &[R],
    strategy: S,
    config: &RoleIndexingConfig<'_, R, D, G, P>,
) -> MatchOutcome {
    let validated = match config.criteria.validate() {
        Ok(validated) => validated,
        Err(err) => {
            return invalid_criteria_outcome(cohort.len(), InvalidCriteriaReason::from(err));
        }
    };

    let ratio_policy = RatioPolicy::from_fallback(validated.match_ratio, config.ratio_fallback);
    let case_indices =
        eligible_case_indices(cohort, config.age_limit_years, config.risk_set_policy);
    let common_support_filters = match resolve_transition_common_support(
        cohort,
        &case_indices,
        validated.as_ref(),
        config.common_support_scores,
    ) {
        Ok(filters) => filters,
        Err(reason) => {
            return invalid_common_support_outcome(case_indices.len(), reason);
        }
    };

    let mut engine = MatchEngine::new(cohort, validated.as_ref(), strategy);

    let mut run_state = TransitionRunState::new(cohort.len());
    run_state.diagnostics = transition_initial_diagnostics(
        validated.as_ref(),
        case_indices.len(),
        &common_support_filters,
    );

    run_transition_case_loop(
        &TransitionLoopConfig {
            cohort,
            case_indices: &case_indices,
            config,
            validated: validated.as_ref(),
            ratio_policy: &ratio_policy,
            common_support_filters: &common_support_filters,
        },
        &mut engine,
        &mut run_state,
    );

    run_state.diagnostics.matched_anchors = run_state.matched_cases;
    run_state.diagnostics.pairs_selected = run_state.pairs.len();
    finalize_estimand_diagnostics(
        &mut run_state.diagnostics,
        validated.estimand,
        run_state.eligible_cases,
        config
            .distance_channel
            .caliper()
            .is_some()
            .then_some(config.distance_channel.reason()),
    );
    build_outcome(
        run_state.pairs,
        run_state.matched_cases,
        run_state.eligible_cases,
        run_state.used_controls.len(),
        run_state.diagnostics,
    )
}

use rustc_hash::FxHashSet;

struct TransitionRunState {
    pairs: Vec<MatchedPair>,
    used_controls: UsedControlsVec,
    used_unique: FxHashSet<UniqueValueId>,
    unique_interner: RapidHashMap<String, UniqueValueId>,
    matched_cases: usize,
    eligible_cases: usize,
    diagnostics: MatchDiagnostics,
}

impl TransitionRunState {
    fn new(n_controls: usize) -> Self {
        Self {
            pairs: Vec::new(),
            used_controls: UsedControlsVec::with_capacity(n_controls),
            used_unique: FxHashSet::default(),
            unique_interner: RapidHashMap::default(),
            matched_cases: 0,
            eligible_cases: 0,
            diagnostics: MatchDiagnostics::default(),
        }
    }
}

struct TransitionLoopConfig<
    'a,
    R: RoleIndexedRecord,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
> {
    cohort: &'a [R],
    case_indices: &'a [usize],
    config: &'a RoleIndexingConfig<'a, R, D, G, P>,
    validated: &'a MatchingCriteria,
    ratio_policy: &'a RatioPolicy,
    common_support_filters: &'a TransitionCommonSupportFilters,
}

fn run_transition_case_loop<
    R: RoleIndexedRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    D: DistanceMetric<R> + Sync,
    G: ConstraintGroup<R> + ?Sized,
    P: RiskSetPolicy<R> + ?Sized,
>(
    loop_config: &TransitionLoopConfig<'_, R, D, G, P>,
    engine: &mut MatchEngine<'_, R, S>,
    state: &mut TransitionRunState,
) {
    for case_idx in loop_config.case_indices {
        if loop_config
            .common_support_filters
            .trimmed_case_indices
            .contains(case_idx)
        {
            continue;
        }
        state.eligible_cases += 1;

        let case = &loop_config.cohort[*case_idx];
        let Some(case_event_date) = case.event_date() else {
            continue;
        };

        let mut candidates = engine.candidate_pool_for_request_and_constraint_group(
            &CandidatePoolRequest {
                case,
                used_controls: &state.used_controls,
                used_unique: &state.used_unique,
                unique_interner: &state.unique_interner,
                distance_channel: loop_config.config.distance_channel,
                use_birth_date_index: loop_config.config.use_birth_date_index,
            },
            loop_config.config.extra_constraints,
            |idx, control| {
                loop_config.config.risk_set_policy.is_control_at_risk(
                    case,
                    control,
                    case_event_date,
                ) && !loop_config
                    .common_support_filters
                    .trimmed_candidate_indices
                    .contains(&idx)
            },
            &mut state.diagnostics,
        );

        if candidates.is_empty() {
            state.diagnostics.anchors_with_no_candidates += 1;
            continue;
        }

        let Some(target_ratio) = loop_config.ratio_policy.target_ratio(candidates.len()) else {
            state.diagnostics.anchors_below_required_ratio += 1;
            continue;
        };
        if loop_config.ratio_policy.is_shortfall(candidates.len()) {
            state.diagnostics.anchors_below_required_ratio += 1;
        }

        let mut local_matches = 0usize;
        for _ in 0..target_ratio {
            let Some(selected_idx) = engine.select_control(case, &mut candidates) else {
                break;
            };
            let control = &engine.controls()[selected_idx.get()];
            state.pairs.push(MatchedPair::new(case.id(), control.id()));
            local_matches += 1;

            if !loop_config.validated.allow_replacement {
                state.used_controls.insert(selected_idx);
            }
            if let Some(value) = unique_value(control, loop_config.validated) {
                let next_id = state.unique_interner.len();
                let id = *state
                    .unique_interner
                    .entry(value.to_string())
                    .or_insert_with(|| UniqueValueId::new(next_id));
                state.used_unique.insert(id);
            }
        }

        if local_matches > 0 {
            state.matched_cases += 1;
        }
    }
}

fn eligible_case_indices<R: RoleIndexedRecord, P: RiskSetPolicy<R> + ?Sized>(
    cohort: &[R],
    age_limit_years: u8,
    risk_set_policy: &P,
) -> Vec<usize> {
    cohort
        .iter()
        .enumerate()
        .filter_map(|(idx, row)| {
            if risk_set_policy.is_case_eligible(row, age_limit_years) {
                Some(idx)
            } else {
                None
            }
        })
        .sorted_by(|left_idx, right_idx| {
            let left = &cohort[*left_idx];
            let right = &cohort[*right_idx];
            left.event_date()
                .cmp(&right.event_date())
                .then_with(|| left.id().cmp(right.id()))
        })
        .collect_vec()
}

fn is_case_before_age_limit<R: RoleIndexedRecord>(row: &R, age_limit_years: u8) -> bool {
    let Some(event_date) = row.event_date() else {
        return false;
    };
    let age_limit_months = u32::from(age_limit_years) * 12;
    let Some(age_limit_date) = row
        .birth_date()
        .checked_add_months(Months::new(age_limit_months))
    else {
        return false;
    };

    event_date < age_limit_date
}

fn default_control_is_at_risk<R: RoleIndexedRecord>(
    control: &R,
    case_event_date: NaiveDate,
) -> bool {
    control
        .event_date()
        .is_none_or(|event_date| event_date > case_event_date)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::matching::{DeterministicSelection, DistanceConfig, RandomSelection};
    use crate::types::{AgeLimitYears, BaseRecord, Estimand, MatchRatio, RoleTransitionRecord};
    use chrono::NaiveDate;
    use std::collections::HashMap;

    fn ratio(value: usize) -> MatchRatio {
        MatchRatio::new(value).expect("non-zero ratio")
    }

    fn age_limit(value: u8) -> AgeLimitYears {
        AgeLimitYears::new(value).expect("positive age limit")
    }

    fn run_transition<R, S, G, P>(
        cohort: &[R],
        criteria: &MatchingCriteria,
        options: &RoleTransitionOptions,
        strategy: S,
        constraints: &G,
        distance_config: Option<&DistanceConfig>,
        risk_set_policy: &P,
    ) -> MatchOutcome
    where
        R: RoleIndexedRecord,
        S: SelectionStrategy<R> + Clone + Send + Sync,
        G: ConstraintGroup<R> + ?Sized,
        P: RiskSetPolicy<R>,
    {
        match_transition(
            cohort,
            TransitionMatchRequest {
                criteria,
                options,
                strategy,
                constraints,
                distance_config,
                risk_set_policy,
            },
        )
    }

    #[derive(Clone)]
    struct ResidencyRecord {
        base: BaseRecord,
        resident_until: Option<NaiveDate>,
    }

    impl crate::MatchingRecord for ResidencyRecord {
        crate::delegate_matching_record!(base);
    }

    impl ResidencyRecord {
        fn new(id: &str, birth_date: NaiveDate, resident_until: Option<NaiveDate>) -> Self {
            Self {
                base: BaseRecord::new(id, birth_date),
                resident_until,
            }
        }
    }

    struct ResidencyPolicy;

    impl RiskSetPolicy<RoleTransitionRecord<ResidencyRecord>> for ResidencyPolicy {
        fn is_case_eligible(
            &self,
            row: &RoleTransitionRecord<ResidencyRecord>,
            age_limit_years: u8,
        ) -> bool {
            is_case_before_age_limit(row, age_limit_years)
                && row.record.resident_until.is_none_or(|until| {
                    row.event_date()
                        .is_some_and(|event_date| until >= event_date)
                })
        }

        fn is_control_at_risk(
            &self,
            case: &RoleTransitionRecord<ResidencyRecord>,
            control: &RoleTransitionRecord<ResidencyRecord>,
            case_event_date: NaiveDate,
        ) -> bool {
            default_control_is_at_risk(control, case_event_date)
                && control
                    .record
                    .resident_until
                    .is_none_or(|until| until >= case_event_date)
                && case.record.base.id != control.record.base.id
        }
    }

    #[test]
    fn transition_matching_is_generic_over_record_shape() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }

    #[test]
    fn transition_works_with_base_record_default_type() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            RandomSelection::seeded(9),
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
    }

    #[test]
    fn transition_policy_filters_non_resident_candidates() {
        let criteria = MatchingCriteria::default();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("case", date(2010, 1, 1), None),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new(
                    "candidate_not_resident",
                    date(2010, 1, 2),
                    Some(date(2013, 12, 31)),
                ),
                None,
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("candidate_resident", date(2010, 1, 3), None),
                None,
            ),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &ResidencyPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.pairs[0].comparator_id(), "candidate_resident");
    }

    #[test]
    fn transition_policy_filters_non_resident_cases() {
        let criteria = MatchingCriteria::builder().allow_replacement(true).build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                ResidencyRecord::new(
                    "ineligible_case",
                    date(2010, 1, 1),
                    Some(date(2013, 12, 31)),
                ),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("eligible_case", date(2010, 1, 2), None),
                Some(date(2014, 2, 1)),
            ),
            RoleTransitionRecord::from_record(
                ResidencyRecord::new("candidate", date(2010, 1, 3), None),
                None,
            ),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &ResidencyPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.pairs[0].anchor_id(), "eligible_case");
    }

    #[test]
    fn transition_supports_reusable_distance_channels() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(0)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 7, 1)), None),
        ];

        let scores: RapidHashMap<String, f64> = [
            ("case".to_string(), 0.20_f64),
            ("candidate".to_string(), 0.24_f64),
        ]
        .into_iter()
        .collect();
        let config = DistanceConfig::propensity_score_map(
            scores,
            Some(crate::types::DistanceCaliper::new(0.05).expect("valid positive caliper")),
        )
        .with_reason("ps_transition");
        let outcome_with_config = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            Some(&config),
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome_with_config.matched_cases, 1);
        assert_eq!(outcome_with_config.pairs.len(), 1);
    }

    #[test]
    fn transition_default_ratio_behavior_allows_partial_matching() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(10)
            .match_ratio(2)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: Vec::new(),
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.pairs.len(), 1);
        assert_eq!(outcome.diagnostics.anchors_below_required_ratio, 1);
    }

    #[test]
    fn transition_explicit_ratio_fallback_is_strict() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(10)
            .match_ratio(2)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(2)],
        };

        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 0);
        assert!(outcome.pairs.is_empty());
        assert_eq!(outcome.diagnostics.anchors_below_required_ratio, 1);
    }

    #[test]
    fn transition_common_support_requires_propensity_score_map() {
        let criteria = MatchingCriteria::builder()
            .common_support(CommonSupport::Both)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(BaseRecord::new("candidate", date(2010, 1, 2)), None),
        ];

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            RandomSelection::seeded(12),
            &(),
            None,
            &DefaultRiskSetPolicy,
        );
        assert!(outcome.pairs.is_empty());
        assert_eq!(
            outcome
                .diagnostics
                .exclusion_counts
                .get(&crate::types::ExclusionReason::CommonSupportFailure(
                    crate::types::CommonSupportFailureReason::RequiresPropensityScoreMap,
                ))
                .copied(),
            Some(1)
        );
    }

    #[test]
    fn transition_propensity_common_support_updates_estimand_diagnostics() {
        let criteria = MatchingCriteria::builder()
            .birth_date_window_days(0)
            .common_support(CommonSupport::Control)
            .build();
        let options = RoleTransitionOptions {
            transition_age_limit_years: age_limit(6),
            ratio_fallback: vec![ratio(1)],
        };
        let cohort = vec![
            RoleTransitionRecord::from_record(
                BaseRecord::new("case", date(2010, 1, 1)),
                Some(date(2014, 1, 1)),
            ),
            RoleTransitionRecord::from_record(
                BaseRecord::new("candidate_overlap", date(2010, 1, 2)),
                None,
            ),
            RoleTransitionRecord::from_record(
                BaseRecord::new("candidate_trimmed", date(2010, 1, 3)),
                None,
            ),
        ];
        let scores: RapidHashMap<String, f64> = [
            ("case".to_string(), 0.35_f64),
            ("candidate_overlap".to_string(), 0.35_f64),
            ("candidate_trimmed".to_string(), 0.90_f64),
        ]
        .into_iter()
        .collect();
        let config = DistanceConfig::propensity_score_map(
            scores,
            Some(crate::types::DistanceCaliper::new(1.0).expect("valid positive caliper")),
        );

        let outcome = run_transition(
            &cohort,
            &criteria,
            &options,
            DeterministicSelection,
            &(),
            Some(&config),
            &DefaultRiskSetPolicy,
        );
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.diagnostics.common_support_trimmed_candidates, 1);
        assert_eq!(
            outcome.diagnostics.common_support_policy,
            Some(CommonSupport::Control)
        );
        assert_eq!(
            outcome.diagnostics.common_support_anchor_score_bounds,
            Some((0.35, 0.35))
        );
        assert_eq!(
            outcome.diagnostics.common_support_candidate_score_bounds,
            Some((0.35, 0.90))
        );
        assert_eq!(outcome.diagnostics.requested_estimand, Estimand::Att);
        assert_eq!(outcome.diagnostics.realized_estimand, Estimand::Atm);
    }
}
