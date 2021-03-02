use crate::{Rng, Uncertain};

/// An uncertain value which is distributed as a
/// point mass around a single value. This is useful
/// to express computations involving known values more
/// easily.
#[derive(Clone, Copy)]
pub struct PointMass<T>
where
    T: Clone,
{
    value: T,
}

impl<T> PointMass<T>
where
    T: Clone,
{
    /// Create a new `PointMass` centered on
    /// the given value.
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T> Uncertain for PointMass<T>
where
    T: Clone,
{
    type Value = T;

    fn sample(&self, _rng: &mut Rng, _epoch: usize) -> Self::Value {
        self.value.clone()
    }
}
