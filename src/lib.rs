use rand::{distributions::Distribution, Rng};
use rand_pcg::Pcg32;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

const ALPHA: f32 = 1e-8; // prob of rejecting a true null hypothesis
const BETA: f32 = 1e-8; // prob of accepting a false null hypothesis

const LOWER_LIMIT: f32 = BETA / (1.0 - ALPHA);
const UPPER_LIMIT: f32 = (1.0 - BETA) / ALPHA;

const STEP_SIZE: usize = 10; // TODO: Tune, 10 is from paper
const MAX_STEPS: usize = 1000; // TODO: What is good(?)

fn accept_likelyhood(prob: f32, val: bool) -> f32 {
    let p = 0.5 * (1.0 + prob);
    if val {
        p
    } else {
        1.0 - p
    }
}

fn reject_likelyhood(prob: f32, val: bool) -> f32 {
    let p = 0.5 * prob;
    if val {
        p
    } else {
        1.0 - p
    }
}

fn log_likelyhood_ratio(prob: f32, val: bool) -> f32 {
    let ratio = accept_likelyhood(prob, val) / reject_likelyhood(prob, val);
    ratio.ln()
}

pub trait Uncertain {
    type Value;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value;

    /// Determine if the probability of obtaining `true` form this uncertain
    /// value is at least `probability`.
    ///
    /// This function evaluates a statistical test by sampling the underlying
    /// uncertain value and determining if it is plausible that it has been
    /// generated from a [Bernoulli distribution](wiki) with a value of p of
    /// *at least* `probability`. (I.e. if hypothesis `H_0: p >= probability`
    /// is plausible.)
    ///
    /// # Panics
    ///
    /// Panics if `probability <= 0 || probability >= 1`.
    ///
    /// [wiki]: https://en.wikipedia.org/wiki/Bernoulli_distribution
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

        let value = self.sample(rng, 0).into();
        let mut ratio_sum = log_likelyhood_ratio(probability, value);

        let mut steps: usize = 1;

        let lower = LOWER_LIMIT.ln();
        let upper = UPPER_LIMIT.ln();

        for step in 0..MAX_STEPS {
            for n in 0..STEP_SIZE {
                let epoch = step * STEP_SIZE + n + 1;
                let value = self.sample(rng, epoch).into();
                ratio_sum += log_likelyhood_ratio(probability, value);
                steps += 1;
            }
            if ratio_sum <= lower || ratio_sum >= upper {
                break;
            }
        }

        println!(
            "p = {}, a = {}, b = {}, S = {}, n = {}",
            probability, lower, upper, ratio_sum, steps
        );

        if ratio_sum <= lower {
            true
        } else {
            // either we accepted H_1, or we ran
            // out of samples and can't accept H_0.
            false
        }
    }

    fn into_boxed(self) -> BoxedUncertain<Self>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
    {
        BoxedUncertain {
            ptr: Rc::new(self),
            cache: Rc::new(RefCell::new(None)),
        }
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
{
    ptr: Rc<U>,
    cache: Rc<RefCell<Option<(usize, U::Value)>>>,
}

impl<U> Clone for BoxedUncertain<U>
where
    U: Uncertain,
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
        let mut cache = self.cache.borrow_mut();
        if let Some((last_epoch, last_value)) = &*cache {
            if *last_epoch == epoch {
                return last_value.clone();
            }
        }
        let value = self.ptr.sample(rng, epoch);
        *cache = Some((epoch, value.clone()));
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
        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        for p in cases {
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p.into()).unwrap().into();
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
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
    }

    #[test]
    fn basic_negative_pr() {
        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for p in cases {
            let p_too_high = p + 0.1;
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p.into()).unwrap().into();
            assert!(!x.pr(p_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        for p in cases {
            let p_slightly_too_high = p + 0.001;
            let x: UncertainDistribution<bool, _> = Bernoulli::new(p.into()).unwrap().into();
            assert!(!x.pr(p_slightly_too_high));
        }
    }

    #[test]
    fn basic_gaussian_pr() {
        let x: UncertainDistribution<f64, _> = Normal::new(5.0, 3.0).unwrap().into();
        let more_than_mean = x.map(|num| num > 5.0);

        assert!(more_than_mean.pr(0.2));
        assert!(more_than_mean.pr(0.3));
        assert!(more_than_mean.pr(0.4));
        assert!(more_than_mean.pr(0.5));

        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        let mut positive = 0;
        for epoch in 0..20 {
            if more_than_mean.sample(&mut rng, epoch) {
                positive += 1;
            }
        }
        print!("{:?}/20", positive);

        assert!(!more_than_mean.pr(0.5001));
        assert!(!more_than_mean.pr(0.6));
        assert!(!more_than_mean.pr(0.7));
        assert!(!more_than_mean.pr(0.8));
        assert!(!more_than_mean.pr(0.9));
    }

    // #[test]
    // fn add() {
    //     let a: UncertainDistribution<f32, _> = Normal::new(0.0, 1.0).unwrap().into();
    //     let b: UncertainDistribution<f32, _> = Normal::new(5.0, 1.0).unwrap().into();
    //     let c = a.add(b);

    //     let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
    //     for epoch in 0..10 {
    //         println!("{:?}", c.sample(&mut rng, epoch));
    //     }
    //     assert!(false);
    // }

    // #[test]
    // fn reused_add() {
    //     let x: UncertainDistribution<f32, _> = Normal::new(10.0, 1.0).unwrap().into();
    //     let y: UncertainDistribution<f32, _> = Normal::new(5.0, 1.0).unwrap().into();
    //     let x = x.into_boxed();
    //     let a = y.add(x.clone()); //x.clone().add(y);
    //     let b = a.add(x);

    //     let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
    //     for epoch in 0..10 {
    //         println!("{:?}", b.sample(&mut rng, epoch));
    //     }
    //     assert!(false);
    // }
}
