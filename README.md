# `Uncertain<T>`

[![crates.io](https://img.shields.io/crates/v/uncertain.svg)](https://crates.io/crates/uncertain)
[![Released API docs](https://docs.rs/uncertain/badge.svg)](https://docs.rs/uncertain)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Fast and correct computations with uncertain values.

When working with values which are not exactly determined, such as sensor data, it
can be difficult to handle uncertainties correctly.

The `Uncertain` trait makes such computations as natural as regular computations:

```rust
use uncertain::{Uncertain, Distribution};
use rand_distr::Normal;

// Some inputs about which we are not sure
let x = Distribution::from(Normal::new(5.0, 2.0).unwrap());
let y = Distribution::from(Normal::new(7.0, 3.0).unwrap());

// Do some computations
let distance = x.sub(y).map(|diff: f64| diff.abs());

// Ask a question about the result
let is_it_far = distance.map(|dist| dist > 2.0);

// Check how certain the answer is
assert_eq!(is_it_far.pr(0.9), false);
assert_eq!(is_it_far.pr(0.5), true);
```

This works by sampling a Bayesian network which is implicitly created by describing the computation
on the uncertain type. The `Uncertain` trait only permits tests for simple boolean hypotheses. This
is by design: using Wald's [sequential probability ratio test](sprt), evaluation typically
takes less than `100` samples.

## Stability

While this crate is released as version `0.x`, breaking API changes should be expected.

## References

The `Uncertain` trait exported from the library is an implementation of
the paper [`Uncertain<T>`][paper].

[paper]: https://www.cs.utexas.edu/users/mckinley/papers/uncertainty-asplos-2014.pdf
[sprt]: https://en.wikipedia.org/wiki/Sequential_probability_ratio_test
