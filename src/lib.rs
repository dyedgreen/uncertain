#![warn(missing_docs)]

//! Computation with uncertain values.
//!
//! When working with values which are not exactly determined, such as sensor data, it
//! can be difficult to handle uncertainties correctly.
//!
//! The [`Uncertain`] trait makes such computations as natural as regular computations:
//!
//! ```
//! use uncertain::{Uncertain, Distribution};
//! use rand_distr::Normal;
//!
//! // Some inputs about which we are not sure
//! let x = Distribution::from(Normal::new(5.0, 2.0).unwrap());
//! let y = Distribution::from(Normal::new(7.0, 3.0).unwrap());
//!
//! // Do some computations
//! let distance = x.sub(y).map(|diff: f64| diff.abs());
//!
//! // Ask a question about the result
//! let is_it_far = distance.map(|dist| dist > 2.0);
//!
//! // Check how certain the answer is
//! assert_eq!(is_it_far.pr(0.9), false);
//! assert_eq!(is_it_far.pr(0.5), true);
//! ```
//!
//! # References
//!
//! The [`Uncertain`] trait exported from the library is an implementation of
//! the paper [`Uncertain<T>`][paper].
//!
//! [paper]: https://www.cs.utexas.edu/users/mckinley/papers/uncertainty-asplos-2014.pdf

use adapters::*;
use rand::Rng;
use rand_pcg::Pcg32;

mod adapters;
mod boxed;
mod dist;
mod sprt;

pub use boxed::BoxedUncertain;
pub use dist::Distribution;

/// An interface for using uncertain values in computations.
#[must_use = "uncertain values are lazy and do nothing unless queried"]
pub trait Uncertain {
    /// The type of the contained value.
    type Value;

    /// Generate a random sample from the distribution underlying this
    /// uncertain value. This is similar to [`rand::distributions::Distribution::sample`],
    /// with one important difference:
    ///
    /// If the type which implements [`Uncertain`] is either [`Copy`] or [`Clone`],
    /// then it must guarantee that it will return the same value if queried with
    /// the same epoch (but different rng state) consecutively for multiple times. This
    /// is used to ensure that a single uncertain value is only sampled once, for every
    /// iteration of the statistical test.
    ///
    /// This is important, if a value is reused within a computation. E.g.
    /// `x ~ P; x + x` is different from `x ~ P; x' ~ P; x + x'`.
    ///
    /// If your type is either [`Copy`] or [`Clone`], it is recommended to implement
    /// [`rand::distributions::Distribution`] instead of this trait since any such type
    /// automatically implements [`Into<Distribution>`] in a correct way.
    ///
    /// [`Into<Distribution>`]: Distribution
    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value;

    /// Determine if the probability of obtaining `true` form this uncertain
    /// value is at least `probability`.
    ///
    /// This function evaluates a statistical test by sampling the underlying
    /// uncertain value and determining if it is plausible that it has been
    /// generated from a [Bernoulli distribution][bernoulli]
    /// with a value of p of *at least* `probability`. (I.e. if hypothesis
    /// `H_0: p >= probability` is plausible.)
    ///
    /// The underlying implementation uses the [sequential probability ratio test][sprt],
    /// which takes the least number of samples necessary to establish or reject
    /// a hypothesis. In practice this means that usually only `O(10)` samples
    /// are required.
    ///
    /// [bernoulli]: https://en.wikipedia.org/wiki/Bernoulli_distribution
    /// [sprt]: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
    ///
    /// # Panics
    ///
    /// Panics if `probability <= 0 || probability >= 1`.
    ///
    /// # Examples
    ///
    /// Basic usage: test if some event is more likely than not.
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Bernoulli;
    ///
    /// let x = Distribution::from(Bernoulli::new(0.8).unwrap());
    /// assert_eq!(x.pr(0.5), true);
    ///
    /// let y = Distribution::from(Bernoulli::new(0.3).unwrap());
    /// assert_eq!(y.pr(0.5), false);
    /// ```
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

    /// Box this uncertain value, so it can be reused in a calculation. Usually,
    /// an uncertain value can not be cloned. To ensure that an uncertain value
    /// can be cloned safely, it has to cache it's sampled value such that if
    /// it is queried for the same `epoch` twice, it returns the same value.
    ///
    /// [`BoxedUncertain`] wraps the uncertain value contained in  `self`, and
    /// ensures it behaves correctly if sampled repeatedly.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(5.0, 2.0).unwrap()).into_boxed();
    /// let y = Distribution::from(Normal::new(10.0, 5.0).unwrap());
    /// let a = x.clone().add(y);
    /// let b = a.add(x);
    ///
    /// let bigger_than_twelve = b.map(|v| v > 12.0);
    /// assert!(bigger_than_twelve.pr(0.5));
    /// ```
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

    /// Negate the boolean contained in self. This is a shorthand
    /// for `x.map(|b| !b)`.
    ///
    /// # Examples
    ///
    /// Inverting a Bernoulli distribution:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Bernoulli;
    ///
    /// let x = Distribution::from(Bernoulli::new(0.1).unwrap());
    /// assert!(x.not().pr(0.9));
    /// ```
    fn not(self) -> Not<Self>
    where
        Self: Sized,
        Self::Value: Into<bool>,
    {
        Not::new(self)
    }

    /// Combines two boolean values. This should be preferred over
    /// `x.join(y, |x, y| x && y)`, since it uses short-circuit logic
    /// to avoid sampling `y` if `x` is already false.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Bernoulli;
    ///
    /// let x = Distribution::from(Bernoulli::new(0.5).unwrap());
    /// let y = Distribution::from(Bernoulli::new(0.5).unwrap());
    /// let both = x.and(y);
    /// assert_eq!(both.pr(0.5), false);
    /// assert_eq!(both.not().pr(0.5), true);
    /// ```
    fn and<U>(self, other: U) -> And<Self, U>
    where
        Self: Sized,
        Self::Value: Into<bool>,
        U: Uncertain,
        U::Value: Into<bool>,
    {
        And::new(self, other)
    }

    /// Combines two boolean values. This should be preferred over
    /// `x.join(y, |x, y| x || y)`, since it uses short-circuit logic
    /// to avoid sampling `y` if `x` is already true.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Bernoulli;
    ///
    /// let x = Distribution::from(Bernoulli::new(0.3).unwrap());
    /// let y = Distribution::from(Bernoulli::new(0.3).unwrap());
    /// let either = x.or(y);
    /// assert_eq!(either.pr(0.5), true);
    /// assert_eq!(either.not().pr(0.5), false);
    /// ```
    fn or<U>(self, other: U) -> Or<Self, U>
    where
        Self: Sized,
        Self::Value: Into<bool>,
        U: Uncertain,
        U::Value: Into<bool>,
    {
        Or::new(self, other)
    }

    /// Add two uncertain values. This is a shorthand
    /// for `x.join(y, |x, y| x + y)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(1.0, 1.0).unwrap());
    /// let y = Distribution::from(Normal::new(4.0, 1.0).unwrap());
    /// assert!(x.add(y).map(|sum| sum >= 5.0).pr(0.5));
    /// ```
    fn add<U>(self, other: U) -> Sum<Self, U>
    where
        Self: Sized,
        U: Uncertain,
        Self::Value: std::ops::Add<U::Value>,
    {
        Sum::new(self, other)
    }

    /// Subtract two uncertain values. This is a shorthand
    /// for `x.join(y, |x, y| x - y)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(7.0, 1.0).unwrap());
    /// let y = Distribution::from(Normal::new(2.0, 1.0).unwrap());
    /// assert!(x.sub(y).map(|diff| diff >= 5.0).pr(0.5));
    /// ```
    fn sub<U>(self, other: U) -> Difference<Self, U>
    where
        Self: Sized,
        U: Uncertain,
        Self::Value: std::ops::Sub<U::Value>,
    {
        Difference::new(self, other)
    }

    /// Multiply two uncertain values. This is a shorthand
    /// for `x.join(y, |x, y| x * y)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(4.0, 1.0).unwrap());
    /// let y = Distribution::from(Normal::new(2.0, 1.0).unwrap());
    /// assert!(x.mul(y).map(|prod| prod >= 4.0).pr(0.5));
    /// ```
    fn mul<U>(self, other: U) -> Product<Self, U>
    where
        Self: Sized,
        U: Uncertain,
        Self::Value: std::ops::Mul<U::Value>,
    {
        Product::new(self, other)
    }

    /// Divide two uncertain values. This is a shorthand
    /// for `x.join(y, |x, y| x / y)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(100.0, 1.0).unwrap());
    /// let y = Distribution::from(Normal::new(2.0, 1.0).unwrap());
    /// assert!(x.div(y).map(|prod| prod <= 50.0).pr(0.5));
    /// ```
    fn div<U>(self, other: U) -> Ratio<Self, U>
    where
        Self: Sized,
        U: Uncertain,
        Self::Value: std::ops::Div<U::Value>,
    {
        Ratio::new(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_distr::{Bernoulli, Normal};

    #[test]
    fn basic_positive_pr() {
        let cases: Vec<f32> = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.89];
        for p in cases {
            let p_true = p + 0.1;
            let x = Distribution::from(Bernoulli::new(p_true.into()).unwrap());
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for p in cases {
            let p_true_much_higher = p + 0.49;
            let x = Distribution::from(Bernoulli::new(p_true_much_higher.into()).unwrap());
            assert!(x.pr(p));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3];
        for p in cases {
            let p_tru_way_higher = p + 0.6;
            let x = Distribution::from(Bernoulli::new(p_tru_way_higher.into()).unwrap());
            assert!(x.pr(p));
        }
    }

    #[test]
    fn basic_negative_pr() {
        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for p in cases {
            let p_too_high = p + 0.1;
            let x = Distribution::from(Bernoulli::new(p.into()).unwrap());
            assert!(!x.pr(p_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        for p in cases {
            let p_way_too_high = p + 0.2;
            let x = Distribution::from(Bernoulli::new(p.into()).unwrap());
            assert!(!x.pr(p_way_too_high));
        }

        let cases: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        for p in cases {
            let p_very_way_too_high = p + 0.49;
            let x = Distribution::from(Bernoulli::new(p.into()).unwrap());
            assert!(!x.pr(p_very_way_too_high));
        }
    }

    #[test]
    fn basic_gaussian_pr() {
        let x = Distribution::from(Normal::new(5.0, 3.0).unwrap());
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
        let x = Distribution::from(Bernoulli::new(0.1).unwrap());
        assert!(x.pr(1e-5))
    }

    #[test]
    fn not() {
        let x = Distribution::from(Bernoulli::new(0.7).unwrap());
        assert!(x.pr(0.2));
        assert!(x.pr(0.6));
        let not_x = x.not();
        assert!(not_x.pr(0.2));
        assert!(!not_x.pr(0.6));
    }
}
