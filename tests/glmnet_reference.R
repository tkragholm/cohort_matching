# Reference fits for the elastic-net logistic estimator, from R's glmnet.
#
# Run from the crate root:
#   PATH=/opt/homebrew/opt/r/bin:$PATH Rscript tests/glmnet_reference.R
#
# glmnet minimises, per observation,
#   -loglik / n + lambda * (alpha * ||b||_1 + (1 - alpha) / 2 * ||b||_2^2)
# with the intercept unpenalised. standardize = FALSE keeps the penalty on the
# raw covariate scale, as the Rust fit has it. thresh and maxit are set so that
# glmnet's own convergence error is far below the comparison tolerance.

suppressMessages({
  library(glmnet)
  library(jsonlite)
})

set.seed(20261119)

make_design <- function(n, p, rho, beta, intercept, binary_last = FALSE,
                        raw_scale = FALSE) {
  z <- matrix(rnorm(n * p), n, p)
  x <- z
  if (rho > 0) {
    shared <- rnorm(n)
    x <- sqrt(1 - rho) * z + sqrt(rho) * shared
  }
  if (raw_scale) {
    x[, 1] <- x[, 1] * 3 + 40
  }
  if (binary_last) {
    x[, p] <- as.numeric(x[, p] > 0.3)
  }
  eta <- intercept + as.vector(scale(x, center = TRUE, scale = FALSE) %*% beta)
  y <- rbinom(n, 1, plogis(eta))
  list(x = x, y = y)
}

designs <- list(
  list(name = "uncorrelated_4", n = 200, p = 4, rho = 0,
       beta = c(0.3, 0.8, -0.6, 0), intercept = -0.3, binary_last = FALSE),
  list(name = "correlated_6_sparse", n = 150, p = 6, rho = 0.6,
       beta = c(0, 1.0, 0, 0, -0.7, 0), intercept = 0.2, binary_last = FALSE),
  list(name = "small_imbalanced_with_binary", n = 60, p = 4, rho = 0.2,
       beta = c(0.2, 0.7, -0.5, 0.9), intercept = -1.5, binary_last = TRUE),
  # First covariate at mean 40, sd 3: the fixed step of the Rust fit is set by
  # the largest row norm, so this design converges very slowly.
  list(name = "raw_scale_4", n = 200, p = 4, rho = 0,
       beta = c(0.05, 0.8, -0.6, 0), intercept = -0.3, binary_last = FALSE,
       raw_scale = TRUE)
)

settings <- expand.grid(alpha = c(0, 0.5, 1), lambda = c(0.2, 0.05, 0.01))

design_out <- list()
fits <- list()
for (d in designs) {
  data <- make_design(d$n, d$p, d$rho, d$beta, d$intercept, d$binary_last,
                     isTRUE(d$raw_scale))
  colnames(data$x) <- paste0("x", seq_len(d$p))
  design_out[[length(design_out) + 1]] <- list(
    name = d$name, x = unname(split(data$x, row(data$x))), y = data$y)
  for (i in seq_len(nrow(settings))) {
    a <- settings$alpha[i]
    l <- settings$lambda[i]
    fit <- glmnet(data$x, data$y, family = "binomial", alpha = a, lambda = l,
                  standardize = FALSE, intercept = TRUE,
                  control = list(thresh = 1e-14, maxit = 1e7))
    cf <- as.vector(coef(fit))
    fits[[length(fits) + 1]] <- list(
      design = d$name, alpha = a, lambda = l,
      coefficients = cf  # intercept first, then x1, x2, ...
    )
  }
}

out <- list(glmnet_version = as.character(packageVersion("glmnet")),
            r_version = R.version.string, designs = design_out, fits = fits)
write_json(out, "tests/glmnet_reference.json", digits = NA, auto_unbox = TRUE)
