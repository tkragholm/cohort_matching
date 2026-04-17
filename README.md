# cohort_matching

Caliper-based cohort matching for observational studies.

`cohort_matching` provides a robust engine for creating matched cohorts in observational research. It supports both standard anchor-to-candidate matching and longitudinal role-transition (risk-set) matching.

## Key Features

- **Unified MatchJob API**: Statically-typed builder for configuring and executing matches.
- **Declarative Balance Records**: builder-based covariate definition with canonical missing-value policies.
- **Standard Constraint Helpers**: Built-in builder helpers like `.with_gender_match()`, `.with_alive_check()`, and `.with_resident_check(...)`.
- **Research-Ready Exports**: Integrated CSV reporting for balance diagnostics and matching summaries.
- **Flexible Distance Metrics**: Support for exact match, birth date calipers, Propensity Scores, and Mahalanobis distance.

## Installation

```toml
[dependencies]
cohort_matching = { version = "0.2.0", features = ["reporting"] }
```

## Quick Start: Standard Matching

```rust
use cohort_matching::prelude::*;

let anchors = vec![/* ... */];
let candidates = vec![/* ... */];

let outcome = MatchJob::new_standard(&anchors, &candidates, 42)
    .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
    .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
    .with_gender_match()
    .with_exact_matches(["sex", "region"])
    .run();

assert_eq!(outcome.matched_cases, 1);
```

## Quick Start: Role-Transition (Risk-Set) Matching

```rust
use cohort_matching::prelude::*;

let cohort = vec![/* ... */];

let outcome = MatchJob::new_transition(&cohort, 42)
    .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
    .with_age_limit(AgeLimitYears::new(6).expect("positive age limit"))
    .with_alive_check()
    .with_resident_check()
    .run();
```

## Declarative Balance Records

```rust
use cohort_matching::prelude::*;

let record = BalanceRecord::builder("person_1", birth_date)
    .numeric("age", Some(32.5))
    .categorical("municipality", Some("Copenhagen".into()))
    .build();
```

## Reporting and Exports

Enable the `reporting` feature to access built-in CSV exporters.

```rust
// Write matching diagnostics
let job = MatchJob::new_standard(&cases, &controls, 42);
let outcome = job.run();
outcome.write_summary_csv("matching_summary.csv")?;

// Write balance reports
let report = balance_report(&cases, &controls, &outcome);
report.write_numeric_csv("balance_numeric.csv")?;
```

## Documentation

For more detailed usage, see the [GUIDE.md](GUIDE.md).

## Advanced Usage (Core API)

The crate also exposes lower-level functions if you need absolute control over the matching process:

- `match_anchors_to_candidates`: Basic anchor/candidate matching.
- `match_with_role_transition`: Basic risk-set matching.
- `subclassify_by_propensity_score_map`: Propensity score subclassification.

## License

Licensed under either:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
