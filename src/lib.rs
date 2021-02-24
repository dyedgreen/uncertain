use rand::{distributions::Distribution, Rng};
use rand_pcg::Pcg32;
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

mod sprt;

pub trait Uncertain {
    type Value;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value;

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
        Map {
            uncertain: self,
            func,
        }
    }

    /// Combine two uncertain values using a closure.
    fn join<O, U, F>(self, other: U, func: F) -> Join<Self, U, F>
    where
        Self: Sized,
        U: Uncertain,
        F: Fn(Self::Value, U::Value) -> O,
    {
        Join {
            a: self,
            b: other,
            func,
        }
    }

    /// Equivalent to `self.map(|v| !v.into::<bool>())`.
    fn not(self) -> Not<Self>
    where
        Self: Sized,
        Self::Value: Into<bool>,
    {
        Not { uncertain: self }
    }

    /// Combine two uncertain values by computing their
    /// sum.
    fn add<U>(self, other: U) -> Sum<Self, U>
    where
        Self: Sized,
        U: Uncertain,
        Self::Value: std::ops::Add<U::Value>,
    {
        Sum { a: self, b: other }
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

pub struct Map<U, F> {
    uncertain: U,
    func: F,
}

impl<T, U, F> Uncertain for Map<U, F>
where
    U: Uncertain,
    F: Fn(U::Value) -> T,
{
    type Value = T;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let v = self.uncertain.sample(rng, epoch);
        (self.func)(v)
    }
}

pub struct Join<A, B, F> {
    a: A,
    b: B,
    func: F,
}

impl<O, A, B, F> Uncertain for Join<A, B, F>
where
    A: Uncertain,
    B: Uncertain,
    F: Fn(A::Value, B::Value) -> O,
{
    type Value = O;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        (self.func)(a, b)
    }
}

pub struct Sum<A, B> {
    a: A,
    b: B,
}

impl<A, B> Uncertain for Sum<A, B>
where
    A: Uncertain,
    B: Uncertain,
    A::Value: std::ops::Add<B::Value>,
{
    type Value = <A::Value as std::ops::Add<B::Value>>::Output;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        a + b
    }
}

pub struct Not<U>
where
    U: Uncertain,
    U::Value: Into<bool>,
{
    uncertain: U,
}

impl<U> Uncertain for Not<U>
where
    U: Uncertain,
    U::Value: Into<bool>,
{
    type Value = bool;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        !self.uncertain.sample(rng, epoch).into()
    }
}

pub struct UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    dist: D,
    _p: PhantomData<T>,
}

impl<T, D> Uncertain for UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    type Value = T;

    fn sample<R: Rng>(&self, rng: &mut R, _epoch: usize) -> Self::Value {
        self.dist.sample(rng)
    }
}

impl<T, D> From<D> for UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    fn from(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Bernoulli, Normal};

    #[test]
    fn clone_shares_values() {
        let x: UncertainDistribution<f32, _> = Normal::new(10.0, 1.0).unwrap().into();
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
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p_true.into()).unwrap().into();
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for p in cases {
            let p_true_much_higher = p + 0.49;
            let x: UncertainDistribution<bool, _> =
                Bernoulli::new(p_true_much_higher.into()).unwrap().into();
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3];
        for p in cases {
            let p_tru_way_higher = p + 0.6;
            let x: UncertainDistribution<bool, _> =
                Bernoulli::new(p_tru_way_higher.into()).unwrap().into();
            assert!(x.pr(p));
        }
    }

    #[test]
    fn basic_negative_pr() {
        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for p in cases {
            let p_too_high = p + 0.1;
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p.into()).unwrap().into();
            assert!(!x.pr(p_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for p in cases {
            let p_way_too_high = p + 0.2;
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p.into()).unwrap().into();
            assert!(!x.pr(p_way_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for p in cases {
            let p_very_way_too_high = p + 0.49;
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p.into()).unwrap().into();
            assert!(!x.pr(p_very_way_too_high));
        }
    }

    #[test]
    fn basic_gaussian_pr() {
        let x: UncertainDistribution<f64, _> = Normal::new(5.0, 3.0).unwrap().into();
        let more_than_mean = x.map(|num| num > 5.0);

        assert!(more_than_mean.pr(0.2));
        assert!(more_than_mean.pr(0.3));
        assert!(more_than_mean.pr(0.4));
        // assert!(more_than_mean.pr(0.5));

        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        let mut positive = 0;
        for epoch in 0..20 {
            if more_than_mean.sample(&mut rng, epoch) {
                positive += 1;
            }
        }
        print!("{:?}/20", positive);

        // assert!(!more_than_mean.pr(0.5001));
        assert!(!more_than_mean.pr(0.6));
        assert!(!more_than_mean.pr(0.7));
        assert!(!more_than_mean.pr(0.8));
        assert!(!more_than_mean.pr(0.9));
    }

    #[test]
    fn not() {
        let x: UncertainDistribution<bool, _> = Bernoulli::new(0.7).unwrap().into();
        assert!(x.pr(0.2));
        assert!(x.pr(0.6));
        let not_x = x.not();
        assert!(not_x.pr(0.2));
        assert!(!not_x.pr(0.6));
    }
}
