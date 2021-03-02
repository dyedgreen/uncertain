use crate::{Rng, UncertainBase};
use std::cell::Cell;

pub struct RefUncertain<U>
where
    U: UncertainBase,
    U::Value: Clone,
{
    uncertain: U,
    cache: Cell<Option<(usize, U::Value)>>,
}

impl<U> RefUncertain<U>
where
    U: UncertainBase,
    U::Value: Clone,
{
    pub(crate) fn new(contained: U) -> Self {
        Self {
            uncertain: contained,
            cache: Cell::new(None),
        }
    }
}

impl<U> UncertainBase for &RefUncertain<U>
where
    U: UncertainBase,
    U::Value: Clone,
{
    type Value = U::Value;

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
        let value = match self.cache.take() {
            Some((cache_epoch, cache_value)) if cache_epoch == epoch => cache_value,
            _ => self.uncertain.sample(rng, epoch),
        };
        self.cache.set(Some((epoch, value.clone())));
        value
    }
}

impl<U> UncertainBase for RefUncertain<U>
where
    U: UncertainBase,
    U::Value: Clone,
{
    type Value = U::Value;

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
        <&Self as UncertainBase>::sample(&self, rng, epoch)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, Uncertain, UncertainBase};
    use rand_distr::Normal;
    use rand_pcg::Pcg32;

    #[test]
    fn ref_uncertain_shares_values() {
        let x = Distribution::from(Normal::new(10.0, 1.0).unwrap());
        let x = x.into_ref();
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..1000 {
            assert_eq!(x.sample(&mut rng, epoch), x.sample(&mut rng, epoch));
        }
    }
}
