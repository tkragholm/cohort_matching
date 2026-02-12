# cohort_matching

Caliper-based cohort matching for observational studies.

The crate supports two workflows:

- Core neutral matching (`anchor/candidate`, `role transition`) for arbitrary record shapes.
- Compatibility wrappers for case/control and diagnosis role-switching (`cohort_matching::compat`).
- Constraint and selection hooks for domain-specific matching rules.

Backward-compatible case/control naming lives under `cohort_matching::compat`.

## Installation

```toml
[dependencies]
cohort_matching = "0.1"
```

## Basic anchor/candidate matching (core)

```rust
use chrono::NaiveDate;
use cohort_matching::{match_anchors_to_candidates, BaseRecord, MatchingCriteria};

let anchor = BaseRecord::new("anchor_1", NaiveDate::from_ymd_opt(2010, 1, 1).unwrap());
let candidate = BaseRecord::new("candidate_1", NaiveDate::from_ymd_opt(2010, 1, 2).unwrap());

let outcome = match_anchors_to_candidates(&[anchor], &[candidate], &MatchingCriteria::default(), 42);
assert_eq!(outcome.matched_cases, 1);
```

## Role-transition matching (core)

```rust
use chrono::NaiveDate;
use cohort_matching::{
    match_with_role_transition, BaseRecord, MatchingCriteria, RoleTransitionOptions,
    RoleTransitionRecord,
};

let cohort = vec![
    RoleTransitionRecord::from_record(
        BaseRecord::new("a", NaiveDate::from_ymd_opt(2010, 1, 1).unwrap()),
        Some(NaiveDate::from_ymd_opt(2014, 1, 1).unwrap()),
    ),
    RoleTransitionRecord::from_record(
        BaseRecord::new("b", NaiveDate::from_ymd_opt(2010, 1, 2).unwrap()),
        Some(NaiveDate::from_ymd_opt(2015, 1, 1).unwrap()),
    ),
    RoleTransitionRecord::from_record(
        BaseRecord::new("c", NaiveDate::from_ymd_opt(2010, 1, 3).unwrap()),
        None,
    ),
];

let options = RoleTransitionOptions {
    transition_age_limit_years: 6,
    ratio_fallback: vec![1],
};

let outcome = match_with_role_transition(&cohort, &MatchingCriteria::default(), &options, 42);
assert_eq!(outcome.matched_cases, 2);
```

## Role-switching matching (compat)

```rust
use chrono::NaiveDate;
use cohort_matching::{compat, MatchingCriteria};

let cohort = vec![
    compat::RoleSwitchingRecord::from_participant(
        compat::ParticipantRecord::new("a", NaiveDate::from_ymd_opt(2010, 1, 1).unwrap()),
        Some(NaiveDate::from_ymd_opt(2014, 1, 1).unwrap()),
    ),
    compat::RoleSwitchingRecord::from_participant(
        compat::ParticipantRecord::new("b", NaiveDate::from_ymd_opt(2010, 1, 2).unwrap()),
        None,
    ),
];

let outcome = compat::match_with_role_switching(
    &cohort,
    &MatchingCriteria::default(),
    &compat::RoleSwitchingOptions::default(),
    7,
);
assert!(outcome.matched_cases <= 1);
```

## Custom constraints

```rust
use cohort_matching::{
    match_anchors_to_candidates_with_constraints, BaseRecord, Constraint, ConstraintContext,
    MatchingCriteria,
};

struct NeverAllow;

impl Constraint<BaseRecord> for NeverAllow {
    fn reason(&self) -> &'static str {
        "never_allow"
    }

    fn allows(
        &self,
        _case: &BaseRecord,
        _control: &BaseRecord,
        _ctx: &ConstraintContext<'_>,
    ) -> bool {
        false
    }
}

let anchor = BaseRecord::new("a", chrono::NaiveDate::from_ymd_opt(2010, 1, 1).unwrap());
let candidate = BaseRecord::new("c", chrono::NaiveDate::from_ymd_opt(2010, 1, 2).unwrap());

let outcome = match_anchors_to_candidates_with_constraints(
    &[anchor],
    &[candidate],
    &MatchingCriteria::default(),
    42,
    &[&NeverAllow],
);
assert!(outcome.pairs.is_empty());
```

## License

Licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
