//! The elastic-net logistic fit checked against R's `glmnet`.
//!
//! The reference is `tests/glmnet_reference.json`, written by
//! `tests/glmnet_reference.R` with `standardize = FALSE` so that the penalty
//! sits on the raw covariate scale, as in the Rust fit. Both minimise the mean
//! log-loss plus `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)` with an
//! unpenalised intercept.
//!
//! The fit is IRLS with cyclic coordinate descent, as in `glmnet`. The tests
//! run every reference design, including the correlated ones and the one with a
//! covariate near 40, at the default configuration.

use chrono::NaiveDate;
use cohort_matching::prelude::*;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    designs: Vec<Design>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct Design {
    name: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct RFit {
    design: String,
    alpha: f64,
    lambda: f64,
    coefficients: Vec<f64>,
}

fn reference() -> Reference {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/glmnet_reference.json"
    ))
    .expect("fixture is readable");
    serde_json::from_str(&text).expect("fixture parses")
}

fn records(design: &Design) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let birth_date = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (idx, (row, label)) in design.x.iter().zip(&design.y).enumerate() {
        let mut record = BalanceRecord::new(format!("r{idx}"), birth_date);
        for (col, value) in row.iter().enumerate() {
            record
                .covariates
                .insert(format!("x{}", col + 1), CovariateValue::Numeric(*value));
        }
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates)
}

struct Fitted {
    coefficients: Vec<f64>,
    converged: bool,
    iterations: usize,
}

fn fit_with(design: &Design, config: ElasticNetLogisticConfig) -> Fitted {
    let (anchors, candidates) = records(design);
    let prepared = prepare_propensity_distance_config(
        &anchors,
        &candidates,
        &PropensityScoreConfig::builder()
            .estimator(PropensityEstimator::ElasticNetLogit(config))
            .build(),
    )
    .expect("elastic net fit succeeds");
    let names = &prepared.estimate.column_names;
    // Reorder to glmnet's layout: intercept, then x1, x2, ...
    let coefficients = (0..=design.x[0].len())
        .map(|idx| {
            let name = if idx == 0 {
                "intercept".to_string()
            } else {
                format!("x{idx}")
            };
            let position = names
                .iter()
                .position(|n| *n == name)
                .expect("column exists");
            prepared.estimate.coefficients[position]
        })
        .collect();
    Fitted {
        coefficients,
        converged: prepared.estimate.converged,
        iterations: prepared.estimate.iterations,
    }
}

fn design_of<'a>(reference: &'a Reference, fit: &RFit) -> &'a Design {
    reference
        .designs
        .iter()
        .find(|d| d.name == fit.design)
        .expect("design exists")
}

/// At the default configuration every fit converges and agrees with glmnet's
/// coefficients, on the correlated designs and the one with a covariate near 40
/// as well. glmnet's own error at `thresh = 1e-14` is about 1e-7.
#[test]
fn default_fit_converges_and_matches_glmnet() {
    let reference = reference();
    for design in &reference.designs {
        let mut worst = 0.0_f64;
        let mut max_iterations = 0;
        for fit in reference.fits.iter().filter(|f| f.design == design.name) {
            let config = ElasticNetLogisticConfig::builder()
                .alpha(fit.alpha)
                .lambda(fit.lambda)
                .build();
            let fitted = fit_with(design, config);
            assert!(
                fitted.converged,
                "{} alpha {} lambda {}",
                fit.design, fit.alpha, fit.lambda
            );
            let dev = fitted
                .coefficients
                .iter()
                .zip(&fit.coefficients)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f64, f64::max);
            max_iterations = max_iterations.max(fitted.iterations);
            worst = worst.max(dev);
        }
        println!(
            "{}: worst deviation {worst:e}, up to {max_iterations} IRLS steps",
            design.name
        );
        assert!(worst < 1e-6, "{}: worst deviation {worst}", design.name);
    }
}

/// A fit that is stopped after one iteration reports that it did not converge.
#[test]
fn truncated_fit_reports_not_converged() {
    let reference = reference();
    let fit = &reference.fits[0];
    let config = ElasticNetLogisticConfig::builder()
        .alpha(fit.alpha)
        .lambda(fit.lambda)
        .max_iter(1)
        .build();
    assert!(!fit_with(design_of(&reference, fit), config).converged);
}
