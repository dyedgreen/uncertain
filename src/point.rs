use crate::{Rng, Uncertain};

/// An uncertain value which always yields the same
/// value.
///
/// This type can be useful when fixed values should
/// be conditionally returned e.g. from [`flat_map`](Uncertain::flat_map).
/// If you only need a fixed value as part of an uncertain computation, consider
/// using [`map`](Uncertain::map).
///
/// # Examples
///
/// Basic usage: conditional distribution.
///
/// ```
/// use uncertain::{Uncertain, PointMass, Distribution};
/// use rand_distr::Normal;
///
/// let a = Distribution::from(Normal::new(2.0, 1.0).unwrap());
/// let b = a.flat_map(|a| if a < 1.5 {
///     Distribution::from(Normal::new(0.0, 1.0).unwrap()).into_boxed()
/// } else {
///     PointMass::new(1.0).into_boxed()
/// });
/// assert!(b.map(|b| b > 0.5).pr(0.9));
/// ```
///
/// In most cases you can use [`map`](Uncertain::map) instead:
///
/// ```
/// use uncertain::{Uncertain, PointMass, Distribution};
/// use rand_distr::StandardNormal;
///
/// let x = 1.0;
/// let y = Distribution::<f64, _>::from(StandardNormal).into_ref();
/// let a = PointMass::new(x).add(&y);
/// let b = (&y).map(|v: f64| v + x);
/// assert!(a.join(b, |a, b| a == b).pr(0.999));
/// ```
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Uncertain;
    use rand_pcg::Pcg32;

    #[test]
    fn samples_are_always_the_same() {
        let val = 42.0;
        let point = PointMass::new(val);
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..100 {
            assert_eq!(val, point.sample(&mut rng, epoch));
        }
    }
}
