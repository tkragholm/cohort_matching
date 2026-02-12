# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
