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
//! This works by sampling a Bayesian network which is implicitly created by describing the computation
//! on the uncertain type. The [`Uncertain`] trait only permits tests for simple boolean hypotheses. This
//! is by design: using Wald's [sequential probability ratio test][sprt], evaluation typically
//! takes less than `100` samples.
//!
//! # References
//!
//! The [`Uncertain`] trait exported from the library is an implementation of
//! the paper [`Uncertain<T>`][paper].
//!
//! [paper]: https://www.cs.utexas.edu/users/mckinley/papers/uncertainty-asplos-2014.pdf
//! [sprt]: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test

use adapters::*;
use num_traits::{identities, Float};
use rand_pcg::Pcg32;
use reference::RefUncertain;

mod adapters;
mod boxed;
mod dist;
mod expectation;
mod point;
mod reference;
mod sprt;

pub use boxed::BoxedUncertain;
pub use dist::Distribution;
pub use point::PointMass;

pub use expectation::ConvergenceError;

pub(crate) type Rng = Pcg32;

/// An interface for using uncertain values in computations.
#[must_use = "uncertain values are lazy and do nothing unless queried"]
pub trait Uncertain {
    /// The type of the contained value.
    type Value;

    /// Generate a random sample from the distribution of this
    /// uncertain value. This is similar to [`Distribution::sample`],
    /// with one important difference:
    ///
    /// If the type which implements `Uncertain` is either [`Copy`] or [`Clone`], or if
    /// its references implement `Uncertain`, then it must guarantee that it will return
    /// the same value if queried consecutively with the same epoch (but different rng state).
    ///
    /// This is important when a value is reused within a computation. Consider the following
    /// example:
    /// ```text
    /// x ~ Normal(0, 1)
    /// y ~ Normal(0, 1)
    /// a = x + y
    /// b = a + x
    ///
    /// Correct computation graph:      Incorrect computation graph:
    /// x --+---------+                 x (2nd sample) --+
    ///     |         |                                  |
    ///     |        (+) -> b           x --+           (+) -> b
    ///     |         |                     |            |
    ///    (+) -> a --+                    (+) -> a -----+
    ///     |                               |
    /// y --+                           y --+
    /// ```
    ///
    /// If your type is either [`Copy`] or [`Clone`], it is recommended to implement
    /// [`Distribution`] instead of this trait since any such type
    /// automatically implements [`Into<Distribution>`] in a correct way.
    ///
    /// [`Distribution`]: rand::distributions::Distribution
    /// [`Distribution::sample`]: rand::distributions::Distribution::sample
    /// [`Into<Distribution>`]: Distribution
    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value;

    /// Determine if the probability of obtaining `true` form this uncertain
    /// value is at least `probability`.
    ///
    /// This function evaluates a statistical test by sampling the underlying
    /// uncertain value and determining if it is plausible that it has been
    /// generated from a [Bernoulli distribution][bernoulli]
    /// with a value of p of at least `probability`. (I.e. if hypothesis
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
        if probability <= 0.0 || probability >= 1.0 {
            panic!("Probability {:?} must be in (0, 1)", probability);
        }

        sprt::compute(self, probability)
    }

    /// Calculate the expectation of this uncertain value to the desired
    /// precision. This can be useful e.g. when displaying values in a user
    /// interface.
    ///
    /// If the expected value does not converge to within the desired precision,
    /// a [`ConvergenceError`](ConvergenceError) is returned which can be used
    /// to obtain the non converged expectation and estimated error.
    ///
    /// Note that this value should typically not be used in further computation or in
    /// comparisons. It is usually more expensive to calculate than [`pr`](Uncertain::pr) and
    /// calculations or comparisons using the resulting value can be miss-leading.
    ///
    /// # Panics
    ///
    /// Panics if `precision <= 0`.
    ///
    /// # Example
    ///
    /// If we have a [bimodal][multi-modal] distribution, the expected value can lead to confusing
    /// results:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Bernoulli;
    ///
    /// let choice = Distribution::from(Bernoulli::new(0.6).unwrap());
    /// let value = choice.map(|c| if c { 1.0 } else { -1.0 }).into_ref();
    ///
    /// let bigger_eq_zero = (&value).map(|v| v >= 0.0);
    /// let bigger_eq_half = (&value).map(|v| v >= 0.5);
    ///
    /// assert_eq!(bigger_eq_zero.pr(0.5), true);
    /// assert_eq!(bigger_eq_half.pr(0.5), true); // this is true
    ///
    /// let expected_value = value.expect(0.1).unwrap();
    /// assert_eq!(expected_value >= 0.0, true);
    /// assert_eq!(expected_value >= 0.5, false); // but this is not :o
    /// ```
    ///
    /// # Details
    ///
    /// To take as few samples as possible, this method utilizes an online sampling
    /// strategy to compute estimates of the mean and variance of the distribution
    /// modeled by the uncertain value.
    ///
    /// To determine if the mean has converged to the desired precision, the
    /// variance of the estimate (i.e. `var(E(x))`) is computed, assuming
    /// the samples are [identically and independently distributed][iid].
    ///
    /// The function returns if the [two sigma confidence interval][two-sigma] (i.e. `2 * sqrt(var(E(x)))`)
    /// is smaller than the desired precision and the returned estimate lies within
    /// plus/ minus precision of the true value with approximately `95%` probability.
    ///
    /// [iid]: https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables
    /// [multi-modal]: https://en.wikipedia.org/wiki/Multimodal_distribution
    /// [two-sigma]: https://en.wikipedia.org/wiki/68–95–99.7_rule
    fn expect(&self, precision: Self::Value) -> Result<Self::Value, ConvergenceError<Self::Value>>
    where
        Self::Value: Float,
    {
        if precision <= identities::zero() {
            panic!("Precision must be larger than 0");
        }

        expectation::compute(self, precision)
    }

    /// Box this uncertain value, such that it's type becomes opaque. This is
    /// necessary when you want to mix different sources for uncertain values
    /// e.g. to return different distributions inside [`flat_map`](Self::flat_map).
    ///
    /// This boxes the underlying uncertain value using a trait object and should
    /// only be used if necessary.
    ///
    /// # Examples
    ///
    /// Basic example:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution, PointMass};
    /// use rand_distr::{Bernoulli, StandardNormal};
    ///
    /// let choice = Distribution::from(Bernoulli::new(0.5).unwrap());
    /// let value = choice.flat_map(|fixed| if fixed {
    ///     PointMass::new(5.0).into_boxed()
    /// } else {
    ///     Distribution::from(StandardNormal).into_boxed()
    /// });
    /// assert!(value.map(|v| v > 0.25).pr(0.5));
    /// ```
    fn into_boxed(self) -> BoxedUncertain<Self::Value>
    where
        Self: 'static + Sized + Send,
    {
        BoxedUncertain::new(self)
    }

    /// Bundle this uncertain value with a cache, so it can be reused in a calculation.
    ///
    /// Uncertain values should normally not implement `Copy` or `Clone`, since the same value
    /// is only allowed to be sampled once for every epoch (see [`sample`](Self::sample)).
    /// This wrapper allows a value to be reused by caching the sample result for every epoch and
    /// implementing [`Uncertain`] for references.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(5.0, 2.0).unwrap()).into_ref();
    /// let y = Distribution::from(Normal::new(10.0, 5.0).unwrap());
    /// let a = y.add(&x);
    /// let b = a.add(&x);
    ///
    /// let bigger_than_twelve = b.map(|v| v > 12.0);
    /// assert!(bigger_than_twelve.pr(0.5));
    /// ```
    fn into_ref(self) -> RefUncertain<Self>
    where
        Self: Sized,
        Self::Value: Clone,
    {
        RefUncertain::new(self)
    }

    /// Takes an uncertain value and produces another which
    /// generates values by calling a closure.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::Normal;
    ///
    /// let x = Distribution::from(Normal::new(0.0, 1.0).unwrap());
    /// let y = x.map(|x| 5.0 + x);
    /// let bigger_eq_four = y.map(|v| v >= 4.0);
    /// assert!(bigger_eq_four.pr(0.5));
    /// ```
    fn map<O, F>(self, func: F) -> Map<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Value) -> O,
    {
        Map::new(self, func)
    }

    /// Takes an uncertain value and produces another which
    /// generates values by calling a closure to generate
    /// fresh uncertain types that can depend on the value
    /// contained in self.
    ///
    /// This is useful for cases where the distribution of
    /// an uncertain value depends on another.
    ///
    /// # Example
    ///
    /// Basic example: model two poker chip factories. The first of
    /// which produces chips with `N ~ Binomial(20, 0.3)` and
    /// the second of which produces chips with `M ~ Binomial(50, 0.5)`.
    ///
    /// ```
    /// use uncertain::{Uncertain, Distribution};
    /// use rand_distr::{Binomial, Bernoulli};
    ///
    /// let is_first_factory = Distribution::from(Bernoulli::new(0.5).unwrap());
    /// let number_of_chips = is_first_factory
    ///     .flat_map(|is_first| if is_first {
    ///         Distribution::from(Binomial::new(20, 0.3).unwrap())
    ///     } else {
    ///         Distribution::from(Binomial::new(50, 0.5).unwrap())
    ///     });
    /// assert!(number_of_chips.map(|n| n < 25).pr(0.5));
    /// ```
    fn flat_map<O, F>(self, func: F) -> FlatMap<Self, F>
    where
        Self: Sized,
        O: Uncertain,
        F: Fn(Self::Value) -> O,
    {
        FlatMap::new(self, func)
    }

    /// Combine two uncertain values using a closure. The closure
    /// `func` receives `self` as the first, and `other` as the
    /// second argument.
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
    /// let are_equal = x.join(y, |x, y| x == y);
    /// assert!(are_equal.pr(0.5));
    /// ```
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
