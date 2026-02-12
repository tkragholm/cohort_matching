mod constraints;
mod engine;
mod index;
mod records;
mod selection;

pub use constraints::{Constraint, ConstraintContext, ExactMatchConstraint, StrataExactConstraint};
pub use engine::{
    EngineRunConfig, MatchEngine, match_anchors_to_candidates,
    match_anchors_to_candidates_with_constraints, match_anchors_to_candidates_with_strategy,
    match_anchors_to_candidates_with_strategy_and_constraints,
};
pub use records::{MatchingRecord, RoleIndexedRecord};
pub use selection::{
    DeterministicSelection, NearestBirthDateSelection, RandomSelection, SelectionStrategy,
};

pub use constraints::unique_value;
pub use engine::{build_outcome, invalid_criteria_outcome, to_f64};
