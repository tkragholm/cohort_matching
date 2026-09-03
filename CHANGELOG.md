# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-09-03

### Added
- **A constraint group whose length is decided at run time.** `MatchJob` composes
  constraints into a tuple, so `.with_constraint(a).with_constraint(b)` is
  `(((), A), B)` and the number of constraints is fixed at compile time. A caller
  whose matching rules come from configuration does not have that number until it
  reads the file. `Vec<C>` is now a `ConstraintGroup`, `Chained<G1, G2>` composes
  two groups so one can be appended to a statically built chain, and
  `MatchJob::with_constraints(vec)` / `with_constraint_group(group)` append them.
  An empty list adds nothing and refuses nothing.
- **`Box<C>` is a `Constraint`,** so `Vec<Box<dyn Constraint<R>>>` is a group: the
  nameable form, and the only one that mixes constraint kinds in a single runtime
  list. Two compile-fail cases asserted the opposite on the stance that the group
  is statically dispatched. The crate has always accepted `[&dyn Constraint<R>]`,
  so dynamic dispatch was never actually excluded; what `Box` adds is OWNERSHIP,
  without which a caller cannot build its constraints in a helper and return them.
  Those two cases are replaced by an integration test asserting they are accepted,
  at the same low-level entry points that used to reject them.
- `caliper_on_field` is re-exported from `prelude::constraints`.

### Changed
- **`caliper_on_field` takes `&str` rather than `impl Into<String>`.** Under
  edition 2024 an `impl Trait` return captures every lifetime in scope, so a
  caller passing a non-`'static` `&str` got back a caliper borrowing it, which
  could not be returned from the helper that built it -- and building rules
  somewhere and using them elsewhere is the case this function exists for. The
  return type now says `use<R>`, which requires that the field parameter not be an
  anonymous type parameter. Pass `&field` where you passed an owned `String`;
  string literals and `&str` are unaffected.

## [0.3.1] - 2026-09-03

### Fixed
- **`numeric()` is forwarded by the wrapper record types.** `RoleTransitionRecord`
  and `BalanceRecord` forward every other accessor to the record they wrap and did
  not forward this one, so both fell through to the trait default of `None` -- and
  a caliper reads `None` as a refusal. Every named caliper therefore refused every
  pair, on a field the inner record actually carried. It compiled because the
  method is defaulted, and it does not look like a bug from the outside: it looks
  like a caliper set far too tight. `RoleTransitionRecord` is the record type
  risk-set matching takes, so on 0.3.0 named calipers were unusable in exactly the
  incidence-density design they were added for. Two tests pin the forwarding, one
  per wrapper, and the trait doc now says a wrapper must forward it.

## [0.3.0] - 2026-09-03

### Added
- **Calipers you can address by name, so a caller can configure N of them.**
  `MatchingRecord::numeric(name)` is the numeric counterpart to `strata()`: it
  returns `Option<f64>` and is defaulted to `None`, so no existing implementor
  changes. `BaseRecord` carries `numerics: HashMap<String, f64>` behind
  `#[serde(default)]`, with a `with_numeric()` builder -- a cohort serialised
  before this field deserialises with an empty map, and every caliper over it
  then REFUSES the pair rather than admitting it, which is the safe direction
  for a constraint. `caliper_on_field(field, window)` turns a `(field, window)`
  pair read from configuration into a constraint, which `Caliper::on` could not:
  it takes a selector, so every field needed a closure written at compile time,
  and a list of pairs could not become a list of constraints.

  The case that found it: a study matching children on birth date and their
  parents on birth year measured 94.4% of matched comparators outside the
  parental window its own methods section claims. The one date caliper the
  matcher exposes was spent on the child, so the parental rule was measured
  after matching and never imposed during it.

## [0.2.0] - 2026-08-26

### Fixed
- **Strata group order is deterministic across processes.** `group_anchors_by_strata`
  returned `HashMap::into_values()`, and `std`'s `RandomState` is seeded per process,
  so the strata groups — and the order of the pairs collected from them — differed on
  every run. The matched set was the same size with the same membership, in a
  different row order. That is enough to change results for any caller that resamples
  in row order: measured on a downstream epidemiological pipeline, two identical runs
  reported a different point estimate on 38 of 40 analysis slices, because its
  clustered bootstrap draws in the order rows arrive. Groups now come back in
  first-appearance order, which makes the output order a function of the input order
  and therefore something the caller can control.

### Changed
- `usize_to_f64` no longer saturates. It was
  `f64::from(u32::try_from(n).unwrap_or(u32::MAX))`, so every cohort above
  4,294,967,295 counted as exactly that many; a count that silently stops
  rising is the wrong failure mode for a diagnostic. `as f64` is exact to 2^53
  and agrees with the old form everywhere below saturation.
- `split_covariate_keys_by_kind` builds the kind map in one sweep instead of
  asking `covariate_kind` per key, which rescanned the case list and then the
  control list for every key. First-wins in the same order, so the answer is
  unchanged, including the fall-through to Categorical for a key absent
  everywhere.
- `refresh_predictions` replaces four copies of the same nine-line IRLS
  prediction refresh, so a clip or a dot product has one place to drift.
- `to_dimension` uses `isqrt` instead of searching upward for an integer square
  root.

- Added `itertools` to simplify collection, sorting, and deduplication logic in matching and role-transition helpers.
- Refactored balance statistics internals into a documented `stats` module with explicit formula references for SMD and Cramer's V.
- Corrected Cramer's V scaling to use `min(r - 1, c - 1)` for 2xK tables.
- Refactored role-switching and role-transition APIs to share a single generic role-indexing engine without intermediate conversion/cloning.
- Added neutral convenience accessors on `MatchOutcome` and `BalanceDiagnostics` to reduce case/control-specific coupling.
- Split matching internals into `matching/mod.rs`, `matching/engine.rs`, `matching/constraints.rs`, and `matching/records.rs` for clearer separation of responsibilities.
- Added reusable `MatchEngine` with explicit `EngineRunConfig` and precomputed state.
- Added pluggable selection strategies via `SelectionStrategy` (`RandomSelection`, `NearestBirthDateSelection`, `DeterministicSelection`).
- Added `MatchingCriteriaBuilder`, criteria validation, and `ValidatedMatchingCriteria`.
- Added candidate indexing (`matching/index.rs`) for strata/birthdate prefiltering.
- Extended `MatchOutcome` with structured `MatchDiagnostics` including exclusion counters.
- Added `compat` module to isolate case/control naming wrappers from the neutral core API.
- Added property-style invariant tests (`tests/invariants.rs`) for no-self-match, no-reuse-without-replacement, and age-threshold role-transition behavior.
- Added pluggable constraint hooks (`Constraint`, `ConstraintContext`) with public APIs for anchor/candidate and role-transition matching.
- Split shared types into `types/core.rs` and `types/domain.rs` to separate generic primitives from compatibility/domain records.
- Generalized core matching APIs to be generic over any `MatchingRecord` (not only `ParticipantRecord`-based aliases).
- Moved project-specific constraints (`GenderConstraint`, `ParentDateConstraint`) into `compat` wrappers; core built-ins remain exact-match/date-caliper/uniqueness.
- Moved role-switching APIs to `compat`; kept role-transition APIs in core.
- Added compatibility options (`ParticipantConstraintOptions`) to preserve study-specific matching behavior without coupling core criteria.
- Expanded invariant tests to cover multiple synthetic record shapes and generic transition behavior.

## [0.1.0] - 2026-02-12

### Added
- Split the library into logical modules (`types`, `matching`, `role_switching`, `balance`).
- Added risk-set matching with role switching via `match_with_role_switching`.
- Added `RoleSwitchingRecord` and `RoleSwitchingOptions` for protocol-style matching.
- Added tests for role-switching behavior and case-level unmatched-count semantics.
- Added crates.io packaging metadata and publish assets (`README`, licenses, changelog).
