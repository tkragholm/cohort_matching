//! Validated scalar newtypes: ratios, windows, calipers and indices.

use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

/// Strictly positive requested matching ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MatchRatio(NonZeroUsize);

impl MatchRatio {
    /// Build a validated ratio.
    #[must_use]
    pub const fn new(value: usize) -> Option<Self> {
        match NonZeroUsize::new(value) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }

    /// Access the raw ratio value.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// Validated non-negative birth-date window (days).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BirthDateWindowDays(i32);

impl BirthDateWindowDays {
    #[must_use]
    pub const fn new(value: i32) -> Option<Self> {
        if value >= 0 { Some(Self(value)) } else { None }
    }

    #[must_use]
    pub const fn get(self) -> i32 {
        self.0
    }
}

/// Validated age limit (years) for transition-case eligibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgeLimitYears(pub(super) u8);

impl AgeLimitYears {
    #[must_use]
    pub const fn new(value: u8) -> Option<Self> {
        if value > 0 { Some(Self(value)) } else { None }
    }

    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }
}

/// Non-negative finite distance-caliper value.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct DistanceCaliper(f64);

impl DistanceCaliper {
    /// Build a validated caliper value.
    #[must_use]
    pub fn new(value: f64) -> Option<Self> {
        if value.is_finite() && value >= 0.0 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Access the inner caliper distance.
    #[must_use]
    pub const fn get(self) -> f64 {
        self.0
    }
}

/// Typed index into the control slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ControlIdx(usize);

impl ControlIdx {
    /// Wrap a raw control index.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Access the wrapped control index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Typed identifier for deduplicated unique-key values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct UniqueValueId(usize);

impl UniqueValueId {
    /// Wrap a raw unique-value index.
    #[must_use]
    pub const fn new(value: usize) -> Self {
        Self(value)
    }

    /// Access the wrapped unique-value index.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0
    }
}
