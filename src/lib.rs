//! # `cohort_matching`
//!
//! Caliper-based cohort matching utilities for observational cohort studies.
//! The crate exposes a neutral core API (`anchor/candidate`, `role transition`) and
//! a compatibility facade under [`compat`] for case/control naming.

mod balance;
pub mod compat;
mod matching;
mod role_transition;
mod types;

pub use balance::{balance_diagnostics, balance_report};
pub use matching::{
    Constraint, ConstraintContext, DeterministicSelection, EngineRunConfig, ExactMatchConstraint,
    MatchEngine, MatchingRecord, NearestBirthDateSelection, RandomSelection, RoleIndexedRecord,
    SelectionStrategy, StrataExactConstraint, match_anchors_to_candidates,
    match_anchors_to_candidates_with_constraints, match_anchors_to_candidates_with_strategy,
    match_anchors_to_candidates_with_strategy_and_constraints,
};
pub use role_transition::{
    match_with_role_transition, match_with_role_transition_with_strategy,
    match_with_role_transition_with_strategy_and_constraints,
};
pub use types::{
    AnchorRecord, BalanceDiagnostics, BalanceReport, BaseRecord, CandidateRecord,
    CategoricalBalance, CategoricalLevelBalance, CovariateValue, CriteriaValidationError,
    MatchDiagnostics, MatchOutcome, MatchedPair, MatchingCriteria, MatchingCriteriaBuilder,
    NumericBalance, RoleTransitionOptions, RoleTransitionRecord, ValidatedMatchingCriteria,
};

/// Common imports for consumers of this crate.
pub mod prelude {
    pub use crate::{
        AnchorRecord, BalanceDiagnostics, BalanceReport, BaseRecord, CandidateRecord,
        CategoricalBalance, CategoricalLevelBalance, Constraint, ConstraintContext, CovariateValue,
        CriteriaValidationError, EngineRunConfig, ExactMatchConstraint, MatchDiagnostics,
        MatchOutcome, MatchedPair, MatchingCriteria, MatchingCriteriaBuilder, MatchingRecord,
        NumericBalance, RoleIndexedRecord, RoleTransitionOptions, RoleTransitionRecord,
        StrataExactConstraint, ValidatedMatchingCriteria, balance_diagnostics, balance_report,
        match_anchors_to_candidates, match_anchors_to_candidates_with_constraints,
        match_anchors_to_candidates_with_strategy,
        match_anchors_to_candidates_with_strategy_and_constraints, match_with_role_transition,
        match_with_role_transition_with_strategy,
        match_with_role_transition_with_strategy_and_constraints,
    };
}
