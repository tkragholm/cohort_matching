//! The elastic-net logistic fit checked against R's `glmnet`.
//!
//! The reference is `tests/glmnet_reference.json`, written by
//! `tests/glmnet_reference.R` with `standardize = FALSE` so that the penalty
//! sits on the raw covariate scale, as in the Rust fit. Both minimise the mean
//! log-loss plus `lambda * (alpha * |b|_1 + (1 - alpha) / 2 * |b|_2^2)` with an
//! unpenalised intercept.
//!
//! The fit is a proximal gradient with one fixed step set by the largest row
//! norm of the design, so it converges slowly when a covariate has a large
//! scale. The tests therefore compare coefficients on designs with covariates
//! near unit scale and a tight stopping rule, and only the objective value on a
//! design with a covariate near 40.

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
    }
}

fn design_of<'a>(reference: &'a Reference, fit: &RFit) -> &'a Design {
    reference
        .designs
        .iter()
        .find(|d| d.name == fit.design)
        .expect("design exists")
}

fn config_for(fit: &RFit, max_iter: usize, tolerance: f64) -> ElasticNetLogisticConfig {
    ElasticNetLogisticConfig::builder()
        .alpha(fit.alpha)
        .lambda(fit.lambda)
        .max_iter(max_iter)
        .tolerance(tolerance)
        .build()
}

/// Mean log-loss plus the elastic-net penalty, the objective glmnet minimises
/// with `standardize = FALSE`.
fn objective(design: &Design, beta: &[f64], alpha: f64, lambda: f64) -> f64 {
    let n = design.x.len() as f64;
    let loss: f64 = design
        .x
        .iter()
        .zip(&design.y)
        .map(|(row, y)| {
            let eta = beta[0] + row.iter().zip(&beta[1..]).map(|(x, b)| x * b).sum::<f64>();
            (1.0 + eta.exp()).ln() - y * eta
        })
        .sum();
    let l1: f64 = beta[1..].iter().map(|b| b.abs()).sum();
    let l2: f64 = beta[1..].iter().map(|b| b * b).sum();
    loss / n + lambda * (alpha * l1 + (1.0 - alpha) / 2.0 * l2)
}

/// With a tight stopping rule the coefficients agree with glmnet's to about
/// 1e-7 on unit-scale designs, which is glmnet's own convergence error at
/// `thresh = 1e-14` on a nearly flat objective.
#[test]
fn coefficients_match_glmnet_on_unit_scale_designs() {
    let reference = reference();
    let mut worst = 0.0_f64;
    for fit in reference.fits.iter().filter(|f| f.design != "raw_scale_4") {
        let fitted = fit_with(design_of(&reference, fit), config_for(fit, 100_000, 1e-13));
        assert!(
            fitted.converged,
            "{} alpha {} lambda {}",
            fit.design, fit.alpha, fit.lambda
        );
        for (actual, expected) in fitted.coefficients.iter().zip(&fit.coefficients) {
            worst = worst.max((actual - expected).abs());
        }
    }
    assert!(worst < 1e-6, "worst deviation {worst}");
}

/// The default settings stop at 500 iterations on the correlated designs, and
/// the objective is then close to glmnet's but not equal. On the design with a
/// covariate near 40 it is further away. The fit never goes below glmnet's
/// optimum, on any design.
#[test]
fn objective_never_beats_the_glmnet_optimum() {
    let reference = reference();
    for fit in &reference.fits {
        let design = design_of(&reference, fit);
        let fitted = fit_with(design, config_for(fit, 500, 1e-7));
        let rust = objective(design, &fitted.coefficients, fit.alpha, fit.lambda);
        let glmnet = objective(design, &fit.coefficients, fit.alpha, fit.lambda);
        assert!(
            rust >= glmnet - 1e-12,
            "{} alpha {} lambda {}: {rust} against {glmnet}",
            fit.design,
            fit.alpha,
            fit.lambda
        );
        assert!(
            fit.design == "raw_scale_4" || rust - glmnet < 0.05,
            "{} alpha {} lambda {}: {rust} against {glmnet}",
            fit.design,
            fit.alpha,
            fit.lambda
        );
    }
}
