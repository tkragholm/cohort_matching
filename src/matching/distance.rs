use super::records::MatchingRecord;
use crate::types::DistanceCaliper;
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, get_global_parallelism};
use ordered_float::OrderedFloat;
use rapidhash::RapidHashMap;
use std::collections::BTreeMap;
use std::marker::PhantomData;

/// Distance metric used to compare an anchor and a candidate.
///
/// Returning `None` indicates the pair is ineligible for this metric.
pub trait DistanceMetric<R: MatchingRecord> {
    /// Stable metric channel name used in diagnostics.
    fn channel(&self) -> &'static str;

    /// Compute pair distance. Smaller values indicate closer matches.
    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64>;

    /// Optional candidate pre-filtering by caliper using an internal index.
    /// Returns `None` if this metric does not support indexed queries.
    #[must_use]
    fn candidate_indices(&self, _anchor: &R, _caliper: f64) -> Option<Vec<usize>> {
        None
    }
}

/// Reusable named distance channel with optional caliper.
pub struct DistanceChannel<'a, R: MatchingRecord, D: DistanceMetric<R>> {
    metric: &'a D,
    caliper: Option<DistanceCaliper>,
    reason: &'static str,
    marker: PhantomData<R>,
}

impl<R: MatchingRecord, D: DistanceMetric<R>> Copy for DistanceChannel<'_, R, D> {}

impl<R: MatchingRecord, D: DistanceMetric<R>> Clone for DistanceChannel<'_, R, D> {
    fn clone(&self) -> Self {
        *self
    }
}

/// Named and reusable distance setup independent of concrete record type.
#[derive(Debug, Clone)]
pub enum DistanceConfig {
    Date {
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
    PropensityScoreMap {
        scores: RapidHashMap<String, f64>,
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
    MahalanobisMap {
        vectors: RapidHashMap<String, Vec<f64>>,
        inverse_covariance: Vec<f64>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
        reason: &'static str,
    },
}

impl DistanceConfig {
    #[must_use]
    pub const fn date(caliper: Option<DistanceCaliper>) -> Self {
        Self::Date {
            caliper,
            reason: "date_caliper",
        }
    }

    #[must_use]
    pub const fn propensity_score_map(
        scores: RapidHashMap<String, f64>,
        caliper: Option<DistanceCaliper>,
    ) -> Self {
        Self::PropensityScoreMap {
            scores,
            caliper,
            reason: "propensity_score_caliper",
        }
    }

    /// Build Mahalanobis config from id-indexed vectors and inverse covariance.
    ///
    /// # Errors
    ///
    /// Returns [`MahalanobisError`] when `dimension` is zero or matrix shape is invalid.
    pub fn mahalanobis_map(
        vectors: RapidHashMap<String, Vec<f64>>,
        inverse_covariance: Vec<f64>,
        dimension: usize,
        caliper: Option<DistanceCaliper>,
    ) -> Result<Self, MahalanobisError> {
        if dimension == 0 {
            return Err(MahalanobisError::ZeroDimension);
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return Err(MahalanobisError::InvalidInverseCovarianceLength);
        }

        Ok(Self::MahalanobisMap {
            vectors,
            inverse_covariance,
            dimension,
            caliper,
            reason: "mahalanobis_caliper",
        })
    }

    #[must_use]
    pub const fn with_reason(mut self, reason: &'static str) -> Self {
        match &mut self {
            Self::Date { reason: slot, .. }
            | Self::PropensityScoreMap { reason: slot, .. }
            | Self::MahalanobisMap { reason: slot, .. } => *slot = reason,
        }
        self
    }

    #[must_use]
    pub const fn caliper(&self) -> Option<f64> {
        match self {
            Self::Date { caliper, .. }
            | Self::PropensityScoreMap { caliper, .. }
            | Self::MahalanobisMap { caliper, .. } => match caliper {
                Some(caliper) => Some(caliper.get()),
                None => None,
            },
        }
    }

    #[must_use]
    pub const fn typed_caliper(&self) -> Option<DistanceCaliper> {
        match self {
            Self::Date { caliper, .. }
            | Self::PropensityScoreMap { caliper, .. }
            | Self::MahalanobisMap { caliper, .. } => *caliper,
        }
    }

    #[must_use]
    pub const fn reason(&self) -> &'static str {
        match self {
            Self::Date { reason, .. }
            | Self::PropensityScoreMap { reason, .. }
            | Self::MahalanobisMap { reason, .. } => reason,
        }
    }
}

impl DistanceConfig {
    /// The metric this configuration describes, borrowing its maps.
    pub(crate) fn metric(&self) -> ConfiguredMetric<'_> {
        match self {
            Self::Date { .. } => ConfiguredMetric::Date(DateDistance),
            Self::PropensityScoreMap { scores, .. } => {
                ConfiguredMetric::PropensityScore(IdMapPropensityScoreDistance::new(scores))
            }
            Self::MahalanobisMap {
                vectors,
                inverse_covariance,
                dimension,
                ..
            } => ConfiguredMetric::Mahalanobis(IdMapMahalanobisDistance::new(
                vectors,
                inverse_covariance,
                *dimension,
            )),
        }
    }

    /// The scores common-support trimming needs, which only a propensity
    /// configuration has.
    pub(crate) const fn propensity_scores(&self) -> Option<&RapidHashMap<String, f64>> {
        match self {
            Self::PropensityScoreMap { scores, .. } => Some(scores),
            Self::Date { .. } | Self::MahalanobisMap { .. } => None,
        }
    }
}

/// One of the metrics a [`DistanceConfig`] can name, so the matchers take a
/// single metric type whichever variant the caller configured.
pub(crate) enum ConfiguredMetric<'a> {
    Date(DateDistance),
    PropensityScore(IdMapPropensityScoreDistance<'a>),
    Mahalanobis(IdMapMahalanobisDistance<'a>),
}

impl<R: MatchingRecord> DistanceMetric<R> for ConfiguredMetric<'_> {
    fn channel(&self) -> &'static str {
        match self {
            Self::Date(metric) => DistanceMetric::<R>::channel(metric),
            Self::PropensityScore(metric) => DistanceMetric::<R>::channel(metric),
            Self::Mahalanobis(metric) => DistanceMetric::<R>::channel(metric),
        }
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        match self {
            Self::Date(metric) => metric.distance(anchor, candidate),
            Self::PropensityScore(metric) => metric.distance(anchor, candidate),
            Self::Mahalanobis(metric) => metric.distance(anchor, candidate),
        }
    }

    fn candidate_indices(&self, anchor: &R, caliper: f64) -> Option<Vec<usize>> {
        match self {
            Self::Date(metric) => metric.candidate_indices(anchor, caliper),
            Self::PropensityScore(metric) => metric.candidate_indices(anchor, caliper),
            Self::Mahalanobis(metric) => metric.candidate_indices(anchor, caliper),
        }
    }
}

/// Propensity score distance keyed by [`MatchingRecord::id`].
#[derive(Debug, Clone)]
pub struct IdMapPropensityScoreDistance<'a> {
    scores: &'a RapidHashMap<String, f64>,
    index: Option<BTreeMap<OrderedFloat<f64>, Vec<usize>>>,
}

impl<'a> IdMapPropensityScoreDistance<'a> {
    #[must_use]
    pub const fn new(scores: &'a RapidHashMap<String, f64>) -> Self {
        Self {
            scores,
            index: None,
        }
    }

    /// Build a sorted propensity index for faster caliper queries.
    #[must_use]
    pub fn with_index<R: MatchingRecord>(mut self, controls: &[R]) -> Self {
        let mut index: BTreeMap<OrderedFloat<f64>, Vec<usize>> = BTreeMap::new();
        for (idx, record) in controls.iter().enumerate() {
            if let Some(&score) = self.scores.get(record.id())
                && score.is_finite()
            {
                index.entry(OrderedFloat(score)).or_default().push(idx);
            }
        }
        self.index = Some(index);
        self
    }

    /// Query indices within a propensity score caliper.
    #[must_use]
    pub fn query_caliper(&self, center: f64, caliper: f64) -> Option<Vec<usize>> {
        let index = self.index.as_ref()?;
        let lower = OrderedFloat(center - caliper);
        let upper = OrderedFloat(center + caliper);
        Some(
            index
                .range(lower..=upper)
                .flat_map(|(_, ids)| ids.iter().copied())
                .collect(),
        )
    }

    #[must_use]
    pub const fn scores(&self) -> &'a RapidHashMap<String, f64> {
        self.scores
    }
}

impl<R: MatchingRecord> DistanceMetric<R> for IdMapPropensityScoreDistance<'_> {
    fn channel(&self) -> &'static str {
        "propensity_score_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = *self.scores.get(anchor.id())?;
        let right = *self.scores.get(candidate.id())?;
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        Some((left - right).abs())
    }

    fn candidate_indices(&self, anchor: &R, caliper: f64) -> Option<Vec<usize>> {
        let center = *self.scores.get(anchor.id())?;
        if !center.is_finite() {
            return Some(Vec::new());
        }
        self.query_caliper(center, caliper)
    }
}

/// Mahalanobis distance keyed by [`MatchingRecord::id`].
#[derive(Debug, Clone)]
pub struct IdMapMahalanobisDistance<'a> {
    vectors: &'a RapidHashMap<String, Vec<f64>>,
    inverse_covariance_valid: bool,
    inverse_covariance_matrix: Mat<f64>,
}

impl<'a> IdMapMahalanobisDistance<'a> {
    #[must_use]
    pub fn new(
        vectors: &'a RapidHashMap<String, Vec<f64>>,
        inverse_covariance: &'a [f64],
        dimension: usize,
    ) -> Self {
        let inverse_covariance_valid =
            inverse_covariance.len() == dimension.saturating_mul(dimension);
        let inverse_covariance_matrix = if inverse_covariance_valid {
            Mat::from_fn(dimension, dimension, |row, col| {
                inverse_covariance[row * dimension + col]
            })
        } else {
            Mat::zeros(dimension, dimension)
        };

        Self {
            vectors,
            inverse_covariance_valid,
            inverse_covariance_matrix,
        }
    }
}

/// Dimensions up to this size keep the two scratch vectors on the stack.
const STACK_DIMENSION: usize = 32;

/// `delta' * M * delta` through faer's matmul, called exactly as `&Mat * &Mat`
/// calls it: same kernel, same operand layout, `Accum::Replace`, global
/// parallelism, but into scratch buffers rather than three new matrices. faer
/// picks its kernel from shape, stride and the CPU's SIMD level, so the result
/// is bit-identical to the operator form. It must stay so: any other summation
/// order moves the last bits, and with them pairs at the caliper boundary.
fn mahalanobis_distance_faer(
    left: &[f64],
    right: &[f64],
    inverse_covariance_matrix: &Mat<f64>,
) -> Option<f64> {
    let dimension = inverse_covariance_matrix.nrows();
    if left.len() != dimension || right.len() != dimension {
        return None;
    }
    if inverse_covariance_matrix.ncols() != dimension {
        return None;
    }

    let mut stack = [0.0_f64; 2 * STACK_DIMENSION];
    let mut heap = Vec::new();
    let scratch = if dimension <= STACK_DIMENSION {
        &mut stack[..2 * dimension]
    } else {
        heap.resize(2 * dimension, 0.0);
        heap.as_mut_slice()
    };
    let (delta, transformed) = scratch.split_at_mut(dimension);
    for ((slot, left), right) in delta.iter_mut().zip(left).zip(right) {
        *slot = left - right;
    }
    let delta = MatRef::from_column_major_slice(delta, dimension, 1);

    matmul(
        MatMut::from_column_major_slice_mut(transformed, dimension, 1),
        Accum::Replace,
        inverse_covariance_matrix.as_ref(),
        delta,
        1.0,
        get_global_parallelism(),
    );
    let mut distance_sq = [0.0_f64];
    matmul(
        MatMut::from_column_major_slice_mut(&mut distance_sq, 1, 1),
        Accum::Replace,
        delta.transpose(),
        MatRef::from_column_major_slice(transformed, dimension, 1),
        1.0,
        get_global_parallelism(),
    );
    let distance_sq = distance_sq[0];

    if !distance_sq.is_finite() || distance_sq < -1e-12 {
        return None;
    }
    Some(distance_sq.max(0.0).sqrt())
}

impl<R: MatchingRecord> DistanceMetric<R> for IdMapMahalanobisDistance<'_> {
    fn channel(&self) -> &'static str {
        "mahalanobis_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = self.vectors.get(anchor.id())?;
        let right = self.vectors.get(candidate.id())?;
        if !self.inverse_covariance_valid {
            return None;
        }
        mahalanobis_distance_faer(left, right, &self.inverse_covariance_matrix)
    }
}

impl<'a, R: MatchingRecord, D: DistanceMetric<R>> DistanceChannel<'a, R, D> {
    /// Create a channel using the metric's default diagnostics channel name.
    #[must_use]
    pub fn new(metric: &'a D, caliper: Option<DistanceCaliper>) -> Self {
        Self {
            metric,
            caliper,
            reason: metric.channel(),
            marker: PhantomData,
        }
    }

    crate::impl_with_reason!();

    #[must_use]
    pub const fn metric(&self) -> &'a D {
        self.metric
    }

    #[must_use]
    pub const fn caliper(&self) -> Option<f64> {
        match self.caliper {
            Some(caliper) => Some(caliper.get()),
            None => None,
        }
    }

    #[must_use]
    pub const fn typed_caliper(&self) -> Option<DistanceCaliper> {
        self.caliper
    }

    #[must_use]
    pub const fn reason(&self) -> &'static str {
        self.reason
    }
}

/// Absolute date distance in days based on [`MatchingRecord::birth_date`].
#[derive(Debug, Clone, Copy, Default)]
pub struct DateDistance;

impl<R: MatchingRecord> DistanceMetric<R> for DateDistance {
    fn channel(&self) -> &'static str {
        "date_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let diff_days = (candidate.birth_date() - anchor.birth_date()).num_days();
        let abs_days = diff_days.unsigned_abs();
        let capped = u32::try_from(abs_days).unwrap_or(u32::MAX);
        Some(f64::from(capped))
    }
}

/// Distance on scalar propensity scores.
///
/// The supplied accessor should return a finite score for each record.
#[derive(Debug, Clone, Copy)]
pub struct PropensityScoreDistance<F> {
    accessor: F,
}

impl<F> PropensityScoreDistance<F> {
    #[must_use]
    pub const fn new(accessor: F) -> Self {
        Self { accessor }
    }
}

impl<R: MatchingRecord, F> DistanceMetric<R> for PropensityScoreDistance<F>
where
    F: Fn(&R) -> Option<f64>,
{
    fn channel(&self) -> &'static str {
        "propensity_score_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = (self.accessor)(anchor)?;
        let right = (self.accessor)(candidate)?;
        if !left.is_finite() || !right.is_finite() {
            return None;
        }
        Some((left - right).abs())
    }
}

/// Errors constructing [`MahalanobisDistance`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MahalanobisError {
    ZeroDimension,
    InvalidInverseCovarianceLength,
}

/// Mahalanobis distance using a precomputed inverse covariance matrix.
///
/// The inverse covariance matrix is supplied in row-major format.
#[derive(Debug, Clone)]
pub struct MahalanobisDistance<F> {
    accessor: F,
    inverse_covariance_matrix: Mat<f64>,
}

impl<F> MahalanobisDistance<F> {
    /// Construct a Mahalanobis metric from an inverse covariance matrix.
    ///
    /// # Errors
    ///
    /// Returns [`MahalanobisError`] when `dimension` is zero or the matrix length
    /// does not equal `dimension * dimension`.
    pub fn new(
        accessor: F,
        inverse_covariance: Vec<f64>,
        dimension: usize,
    ) -> Result<Self, MahalanobisError> {
        if dimension == 0 {
            return Err(MahalanobisError::ZeroDimension);
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return Err(MahalanobisError::InvalidInverseCovarianceLength);
        }
        let inverse_covariance = inverse_covariance.into_boxed_slice();
        let inverse_covariance_matrix = Mat::from_fn(dimension, dimension, |row, col| {
            inverse_covariance[row * dimension + col]
        });

        Ok(Self {
            accessor,
            inverse_covariance_matrix,
        })
    }
}

impl<R: MatchingRecord, F> DistanceMetric<R> for MahalanobisDistance<F>
where
    F: Fn(&R) -> Option<Vec<f64>>,
{
    fn channel(&self) -> &'static str {
        "mahalanobis_caliper"
    }

    fn distance(&self, anchor: &R, candidate: &R) -> Option<f64> {
        let left = (self.accessor)(anchor)?;
        let right = (self.accessor)(candidate)?;
        mahalanobis_distance_faer(&left, &right, &self.inverse_covariance_matrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::types::BaseRecord;
    use chrono::NaiveDate;

    #[derive(Clone)]
    struct DemoRecord {
        base: BaseRecord,
        numeric: Vec<f64>,
    }

    impl DemoRecord {
        fn new(id: &str, birth_date: NaiveDate, numeric: Vec<f64>) -> Self {
            Self {
                base: BaseRecord::new(id, birth_date),
                numeric,
            }
        }
    }

    impl crate::MatchingRecord for DemoRecord {
        crate::delegate_matching_record!(base);
    }

    #[test]
    fn date_distance_returns_absolute_day_difference() {
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("c", date(2010, 1, 6));
        let metric = DateDistance;
        let distance = metric
            .distance(&anchor, &candidate)
            .expect("date distance should exist");
        assert!((distance - 5.0).abs() < 1e-12);
    }

    #[test]
    fn propensity_score_distance_handles_missing_values() {
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("c", date(2010, 1, 2));
        let scores: RapidHashMap<String, f64> =
            std::iter::once(("a".to_string(), 0.25_f64)).collect();
        let metric =
            PropensityScoreDistance::new(|row: &BaseRecord| scores.get(row.id.as_str()).copied());
        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    #[test]
    fn distance_channel_defaults_to_metric_channel_and_allows_override() {
        let metric = DateDistance;
        let default_channel: DistanceChannel<'_, BaseRecord, _> = DistanceChannel::new(
            &metric,
            Some(DistanceCaliper::new(3.0).expect("valid positive caliper")),
        );
        assert_eq!(default_channel.reason(), "date_caliper");
        assert_eq!(default_channel.caliper(), Some(3.0));

        let custom = default_channel.with_reason("custom_date_band");
        assert_eq!(custom.reason(), "custom_date_band");
    }

    #[test]
    fn distance_config_constructors_and_reason_override() {
        let date = DistanceConfig::date(Some(
            DistanceCaliper::new(10.0).expect("valid positive date caliper"),
        ))
        .with_reason("date_band");
        assert_eq!(date.reason(), "date_band");
        assert_eq!(date.caliper(), Some(10.0));

        let ps = DistanceConfig::propensity_score_map(
            std::iter::once(("a".to_string(), 0.2_f64)).collect(),
            Some(DistanceCaliper::new(0.05).expect("valid positive ps caliper")),
        );
        assert_eq!(ps.reason(), "propensity_score_caliper");

        let maha = DistanceConfig::mahalanobis_map(
            RapidHashMap::default(),
            vec![1.0],
            1,
            Some(DistanceCaliper::new(3.0).expect("valid positive mahalanobis caliper")),
        )
        .expect("valid mahalanobis config");
        assert_eq!(maha.reason(), "mahalanobis_caliper");
        assert_eq!(maha.caliper(), Some(3.0));
    }

    #[test]
    fn mahalanobis_distance_matches_identity_matrix_expectation() {
        let anchor = DemoRecord::new("a", date(2010, 1, 1), vec![1.0, 2.0]);
        let candidate = DemoRecord::new("c", date(2010, 1, 2), vec![4.0, 6.0]);
        let identity = vec![1.0, 0.0, 0.0, 1.0];
        let metric =
            MahalanobisDistance::new(|row: &DemoRecord| Some(row.numeric.clone()), identity, 2)
                .expect("valid identity matrix");

        let distance = metric
            .distance(&anchor, &candidate)
            .expect("distance should be computable");
        assert!((distance - 5.0).abs() < 1e-12);
    }

    #[test]
    fn mahalanobis_distance_rejects_invalid_matrix_length() {
        let result =
            MahalanobisDistance::new(|_row: &DemoRecord| None::<Vec<f64>>, vec![1.0, 0.0, 0.0], 2);
        assert!(matches!(
            result,
            Err(MahalanobisError::InvalidInverseCovarianceLength)
        ));
    }

    #[test]
    fn id_map_mahalanobis_distance_handles_invalid_matrix_length() {
        let vectors: RapidHashMap<String, Vec<f64>> = [
            ("a".to_string(), vec![1.0, 2.0]),
            ("b".to_string(), vec![3.0, 4.0]),
        ]
        .into_iter()
        .collect();
        let metric = IdMapMahalanobisDistance::new(&vectors, &[1.0, 0.0, 0.0], 2);
        let anchor = BaseRecord::new("a", date(2010, 1, 1));
        let candidate = BaseRecord::new("b", date(2010, 1, 2));

        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    #[test]
    fn mahalanobis_distance_rejects_non_finite_result() {
        let anchor = DemoRecord::new("a", date(2010, 1, 1), vec![1.0, 2.0]);
        let candidate = DemoRecord::new("c", date(2010, 1, 2), vec![4.0, 6.0]);
        let invalid_inverse = vec![f64::NAN, 0.0, 0.0, 1.0];
        let metric = MahalanobisDistance::new(
            |row: &DemoRecord| Some(row.numeric.clone()),
            invalid_inverse,
            2,
        )
        .expect("matrix shape is valid");

        assert!(metric.distance(&anchor, &candidate).is_none());
    }

    #[test]
    fn mahalanobis_faer_matches_scalar_reference() {
        let left = vec![1.0, 3.0, -2.0];
        let right = vec![0.5, 1.0, 2.0];
        let inverse_covariance = vec![2.0, 0.1, 0.0, 0.1, 1.5, 0.2, 0.0, 0.2, 1.0];

        let scalar = mahalanobis_distance_scalar_reference(&left, &right, &inverse_covariance, 3)
            .expect("scalar distance should be computable");
        let inverse_covariance_matrix =
            Mat::from_fn(3, 3, |row, col| inverse_covariance[row * 3 + col]);
        let faer = mahalanobis_distance_faer(&left, &right, &inverse_covariance_matrix)
            .expect("faer distance should be computable");

        assert!((scalar - faer).abs() < 1e-12);
    }

    /// The same quadratic form through faer's operators, which allocate.
    fn mahalanobis_distance_operator_form(
        left: &[f64],
        right: &[f64],
        inverse_covariance_matrix: &Mat<f64>,
    ) -> Option<f64> {
        let dimension = inverse_covariance_matrix.nrows();
        let delta = Mat::from_fn(dimension, 1, |row, _| left[row] - right[row]);
        let transformed = inverse_covariance_matrix * &delta;
        let distance_sq = (delta.transpose() * transformed)[(0, 0)];
        if !distance_sq.is_finite() || distance_sq < -1e-12 {
            return None;
        }
        Some(distance_sq.max(0.0).sqrt())
    }

    #[test]
    fn mahalanobis_matches_operator_form_bit_for_bit() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            let exponent = i32::try_from(state % 7).expect("small") - 3;
            (2.0f64.mul_add(unit, -1.0)) * 10f64.powi(exponent)
        };
        // Both sides of STACK_DIMENSION, so the heap path is covered too.
        for dimension in (0..=12).chain([31, 32, 33, 40]) {
            for _ in 0..200 {
                let inverse = Mat::from_fn(dimension, dimension, |_, _| next());
                let left = (0..dimension).map(|_| next()).collect::<Vec<_>>();
                let right = (0..dimension).map(|_| next()).collect::<Vec<_>>();
                let expected = mahalanobis_distance_operator_form(&left, &right, &inverse);
                let actual = mahalanobis_distance_faer(&left, &right, &inverse);
                assert_eq!(
                    expected.map(f64::to_bits),
                    actual.map(f64::to_bits),
                    "dimension {dimension}"
                );
            }
        }
    }

    fn mahalanobis_distance_scalar_reference(
        left: &[f64],
        right: &[f64],
        inverse_covariance: &[f64],
        dimension: usize,
    ) -> Option<f64> {
        if left.len() != dimension || right.len() != dimension {
            return None;
        }
        if inverse_covariance.len() != dimension.saturating_mul(dimension) {
            return None;
        }

        let mut distance_sq = 0.0_f64;
        for row in 0..dimension {
            let mut row_dot = 0.0_f64;
            for col in 0..dimension {
                let idx = row * dimension + col;
                let delta_col = left[col] - right[col];
                row_dot = inverse_covariance[idx].mul_add(delta_col, row_dot);
            }
            let delta_row = left[row] - right[row];
            distance_sq = delta_row.mul_add(row_dot, distance_sq);
        }

        if !distance_sq.is_finite() || distance_sq < -1e-12 {
            return None;
        }
        Some(distance_sq.max(0.0).sqrt())
    }
}
