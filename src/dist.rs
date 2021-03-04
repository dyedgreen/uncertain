use crate::{Rng, Uncertain};
use std::marker::PhantomData;

/// Wraps a [`Distribution`](rand::distributions::Distribution) and implements
/// [`Uncertain`](Uncertain).
///
/// Uncertain values need to observe the correct `epoch` semantics if
/// they implement [`Copy`] or [`Clone`], or want to implement [`Uncertain`](Uncertain)
/// for references.
/// Since most uncertain values model distributions but distributions
/// themselves do not require any special `epoch` semantics when they can be `Cloned` or
/// are implemented on references, this wrapper type exists.
///
/// See [`Uncertain::sample`](Uncertain::sample) for more details on the required semantics.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use uncertain::{Uncertain, Distribution};
/// use rand_distr::StandardNormal;
///
/// let x = Distribution::from(StandardNormal);
/// assert!(x.map(|x: f64| x.abs() < 1.0).pr(0.68));
/// ```
pub struct Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    dist: D,
    _p: PhantomData<T>,
}

impl<T, D> Uncertain for Distribution<T, D>
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
