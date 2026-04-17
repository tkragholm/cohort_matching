# Guide: Research-Ready Matching in `cohort_matching`

This guide covers the refined API for `cohort_matching`, designed to reduce boilerplate and standardize diagnostics for observational cohort studies.

## 1. Defining Balance Records

The `BalanceRecordBuilder` provides a declarative way to define covariates with built-in policies for missing and non-finite values.

```rust
use cohort_matching::prelude::*;

let record = BalanceRecord::builder("person_1", birth_date)
    .numeric("age", Some(32.5))
    .numeric("income", Some(f64::NAN)) // Automatically mapped to Missing
    .categorical("municipality", Some("Copenhagen".into()))
    .categorical("sex", None)          // Automatically mapped to Missing
    .build();
```

## 2. Configuring and Running a Match

The `MatchJob` is the unified entrypoint for both standard and role-transition matching. It uses a statically-typed pattern to ensure that only valid configuration options are available for the chosen matching mode.

### Standard Matching
Match a fixed group of anchors to a fixed group of candidates.

```rust
let outcome = MatchJob::new_standard(&anchors, &candidates, seed)
    .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
    .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
    .with_gender_match()
    .with_exact_match("region")
    .run();
```

### Role-Transition Matching (Risk-Set Sampling)
Match individuals within a longitudinal cohort where they can transition from candidate to case.

```rust
let outcome = MatchJob::new_transition(&cohort, seed)
    .with_ratio(MatchRatio::new(1).expect("non-zero ratio"))
    .with_age_limit(AgeLimitYears::new(6).expect("positive age limit"))
    .with_alive_check() // Automatically uses the record's death_date() method
    .with_constraint(MustBeResident::at_index_date(|r, date| r.is_resident_at(date)))
    .run();
```

### One-shot matching and balance
Both matching modes support `run_with_balance` to get both the outcome and a balance report in one call:

```rust
let (outcome, report) = job.run_with_balance(&cases, &controls);
```

### Advanced Configuration
`MatchJob` supports advanced matching techniques like Propensity Score or Mahalanobis matching:

```rust
let outcome = MatchJob::new_standard(&anchors, &candidates, seed)
    .with_ratio_fallback(vec![4, 3, 2, 1]) // Greedy descending ratio
    .with_distance_config(propensity_config)
    .run();
```

## 3. Standard Constraints

The `constraints` module provides common matching logic that tracks rejection reasons automatically.

- `GenderMatch`: Ensures exact match on a categorical strata key (defaults to "sex").
- `DateWindow`: Ensures a date field (e.g., birth date) is within a specified window.
- `Caliper`: A generic caliper for any numeric field (e.g., `.with_constraint(Caliper::on(|r| r.score, 0.1))`).
- `MustBeAlive`: Checks if a control is alive at the case's index date.
- `MustBeResident`: Checks if a control is resident at the case's index date.

## 4. Reporting and Exports

Enable the `reporting` feature to access built-in CSV exporters.

### Per-field precision
You can specify decimal precision per field in the `ReportConfig`:

```rust
let mut precision = HashMap::new();
precision.insert("propensity_score".to_string(), 6);

let config = ReportConfig::builder()
    .decimal_places(2) // Global default
    .field_precision(precision)
    .build();

report.write_report(&mut file, &config)?;
```

### Convenience exporters
```rust
// Write matching diagnostics
outcome.write_summary_csv("matching_summary.csv")?;
outcome.write_exclusion_counts_csv("exclusion_counts.csv")?;

// Write balance reports
report.write_numeric_csv("balance_numeric.csv")?;
report.write_categorical_csv("balance_categorical.csv")?;
```

The CSV output follows a standardized format with configurable numeric precision.

## 5. Summary of Policies

- **Missing Values**: `None` or non-finite numbers are treated as `Missing`. In balance reporting, missing values are tracked separately.
- **Precision**: Floats are formatted to 4 decimal places in reports by default, overrideable per field.
- **Diagnostics**: Every constraint provides a string-based reason (e.g., `"gender_mismatch"`) which is aggregated in the `MatchOutcome` diagnostics.
