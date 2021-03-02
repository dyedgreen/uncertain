use crate::{Rng, Uncertain, UncertainBase};
use std::boxed::Box;

/// An opaque uncertain value. This is useful when you need to conditionally
/// return different uncertain values. See [`into_boxed`](Uncertain::into_boxed).
pub struct BoxedUncertain<T> {
    ptr: Box<dyn Uncertain<Value = T> + Send>,
}

impl<T> BoxedUncertain<T> {
    pub(crate) fn new<U>(contained: U) -> Self
    where
        U: 'static + Uncertain<Value = T> + Send,
    {
        Self {
            ptr: Box::new(contained),
        }
    }
}

impl<T> UncertainBase for BoxedUncertain<T> {
    type Value = T;

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
        self.ptr.sample(rng, epoch)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, Uncertain};
    use rand_distr::{Bernoulli, Exp1, StandardNormal};

    #[test]
    fn boxed_uncertain_allows_mixed_sources() {
        let choice = Distribution::from(Bernoulli::new(0.5).unwrap());
        let value = choice.flat_map(|choice| {
            if choice {
                Distribution::from(Exp1).into_boxed()
            } else {
                Distribution::from(StandardNormal).into_boxed()
            }
        });

        assert!(value.map(|v: f32| v < 2.0).pr(0.9));
    }

    #[test]
    fn boxed_works_with_into_ref() {
        let x = Distribution::from(Bernoulli::new(0.5).unwrap())
            .into_ref()
            .into_boxed();
        assert!(x.pr(0.5));
    }
}
