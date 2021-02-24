use adapters::*;
use rand::Rng;
use rand_pcg::Pcg32;
use std::cell::Cell;
use std::rc::Rc;

mod adapters;
mod dist;
mod sprt;

pub use dist::Distribution;

#[must_use = "uncertain values are lazy and do nothing unless queried"]
pub trait Uncertain {
    type Value;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value;

    fn from_dist<D>(dist: D) -> dist::Distribution<Self::Value, D>
    where
        D: rand::distributions::Distribution<Self::Value>,
    {
        dist::Distribution::new(dist)
    }

    /// Determine if the probability of obtaining `true` form this uncertain
    /// value is at least `probability`.
    ///
    /// This function evaluates a statistical test by sampling the underlying
    /// uncertain value and determining if it is plausible that it has been
    /// generated from a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution)
    /// with a value of p of *at least* `probability`. (I.e. if hypothesis
    /// `H_0: p >= probability` is plausible.)
    ///
    /// # Panics
    ///
    /// Panics if `probability <= 0 || probability >= 1`.
    fn pr(&self, probability: f32) -> bool
    where
        Self::Value: Into<bool>,
    {
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        self.pr_with(&mut rng, probability)
    }

    /// Same as [pr](Uncertain::pr), but generic over the random number
    /// generator used to produce samples.
    fn pr_with<R: Rng>(&self, rng: &mut R, probability: f32) -> bool
    where
        Self::Value: Into<bool>,
    {
        if probability <= 0.0 || probability >= 1.0 {
            panic!("Probability {:?} must be in (0, 1)", probability);
        }
        sprt::sequential_probability_ratio_test(probability, self, rng)
    }

    fn into_boxed(self) -> BoxedUncertain<Self>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
    {
        BoxedUncertain::new(self)
    }

    /// Takes an uncertain value and produces another which
    /// generates values by calling a closure when sampling.
    fn map<O, F>(self, func: F) -> Map<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Value) -> O,
    {
        Map::new(self, func)
    }

    /// Combine two uncertain values using a closure. The closure
    /// `func` receives `self` as the first, and `other` as the
    /// second argument.
    fn join<O, U, F>(self, other: U, func: F) -> Join<Self, U, F>
    where
        Self: Sized,
        U: Uncertain,
        F: Fn(Self::Value, U::Value) -> O,
    {
        Join::new(self, other, func)
    }

    /// Negate the boolean contained in self.
    fn not(self) -> Not<Self>
    where
        Self: Sized,
        Self::Value: Into<bool>,
    {
        Not::new(self)
    }

    /// Combine two uncertain values by computing their
    /// sum.
    fn add<U>(self, other: U) -> Sum<Self, U>
    where
        Self: Sized,
        U: Uncertain,
        Self::Value: std::ops::Add<U::Value>,
    {
        Sum::new(self, other)
    }
}

pub struct BoxedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    ptr: Rc<U>,
    cache: Rc<Cell<Option<(usize, U::Value)>>>,
}

impl<U> BoxedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    fn new(contained: U) -> Self {
        BoxedUncertain {
            ptr: Rc::new(contained),
            cache: Rc::new(Cell::new(None)),
        }
    }
}

impl<U> Clone for BoxedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    fn clone(&self) -> Self {
        BoxedUncertain {
            ptr: self.ptr.clone(),
            cache: self.cache.clone(),
        }
    }
}

impl<U> Uncertain for BoxedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    type Value = U::Value;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let cache = self.cache.take();
        let value = match cache {
            Some((cache_epoch, cache_value)) => {
                if cache_epoch == epoch {
                    cache_value
                } else {
                    self.ptr.sample(rng, epoch)
                }
            }
            None => self.ptr.sample(rng, epoch),
        };
        self.cache.set(Some((epoch, value.clone())));
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Bernoulli, Normal};

    #[test]
    fn clone_shares_values() {
        let x = Distribution::new(Normal::new(10.0, 1.0).unwrap());
        let x = x.into_boxed();
        let y = x.clone();
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..1000 {
            assert_eq!(x.sample(&mut rng, epoch), y.sample(&mut rng, epoch));
        }
    }

    #[test]
    fn basic_positive_pr() {
        let cases: Vec<f32> = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.89];
        for p in cases {
            let p_true = p + 0.1;
            let x = Distribution::new(Bernoulli::new(p_true.into()).unwrap());
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for p in cases {
            let p_true_much_higher = p + 0.49;
            let x = Distribution::new(Bernoulli::new(p_true_much_higher.into()).unwrap());
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3];
        for p in cases {
            let p_tru_way_higher = p + 0.6;
            let x = Distribution::new(Bernoulli::new(p_tru_way_higher.into()).unwrap());
            assert!(x.pr(p));
        }
    }

    #[test]
    fn basic_negative_pr() {
        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for p in cases {
            let p_too_high = p + 0.1;
            let x = Distribution::new(Bernoulli::new(p.into()).unwrap());
            assert!(!x.pr(p_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for p in cases {
            let p_way_too_high = p + 0.2;
            let x = Distribution::new(Bernoulli::new(p.into()).unwrap());
            assert!(!x.pr(p_way_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for p in cases {
            let p_very_way_too_high = p + 0.49;
            let x = Distribution::new(Bernoulli::new(p.into()).unwrap());
            assert!(!x.pr(p_very_way_too_high));
        }
    }

    #[test]
    fn basic_gaussian_pr() {
        let x = Distribution::new(Normal::new(5.0, 3.0).unwrap());
        let more_than_mean = x.map(|num| num > 5.0);

        assert!(more_than_mean.pr(0.1));
        assert!(more_than_mean.pr(0.2));
        assert!(more_than_mean.pr(0.3));
        assert!(more_than_mean.pr(0.4));

        assert!(!more_than_mean.pr(0.6));
        assert!(!more_than_mean.pr(0.7));
        assert!(!more_than_mean.pr(0.8));
        assert!(!more_than_mean.pr(0.9));
    }

    #[test]
    fn very_certain() {
        let x = Distribution::new(Bernoulli::new(0.1).unwrap());
        assert!(x.pr(1e-5))
    }

    #[test]
    fn not() {
        let x = Distribution::new(Bernoulli::new(0.7).unwrap());
        assert!(x.pr(0.2));
        assert!(x.pr(0.6));
        let not_x = x.not();
        assert!(not_x.pr(0.2));
        assert!(!not_x.pr(0.6));
    }
}
