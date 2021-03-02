use crate::{Rng, UncertainBase};
use std::marker::PhantomData;

/// Wraps a [`Distribution`](rand::distributions::Distribution) to create uncertain
/// values from probability distributions and ensure they have the
/// correct [`Copy`] and [`Clone`] semantics.
pub struct Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    dist: D,
    _p: PhantomData<T>,
}

impl<T, D> Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    /// Construct a new [`impl Uncertain`] from a
    /// distribution.
    ///
    /// [`impl Uncertain`]: crate::Uncertain
    pub fn from(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}

impl<T, D> UncertainBase for Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    type Value = T;

    fn sample(&self, rng: &mut Rng, _epoch: usize) -> Self::Value {
        self.dist.sample(rng)
    }
}

impl<T, D> From<D> for Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    fn from(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}
