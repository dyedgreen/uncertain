use crate::Uncertain;
use rand::Rng;
use std::cell::Cell;

/// Cached uncertainty. Its reference also implements Uncertain.
/// See [`Uncertain::into_cached`].
pub struct CachedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    ptr: U,
    cache: Cell<Option<(usize, U::Value)>>,
}

impl<U> CachedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    pub(crate) fn new(contained: U) -> Self {
        CachedUncertain {
            ptr: contained,
            cache: Cell::new(None),
        }
    }
}

impl<U> Uncertain for &CachedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    type Value = U::Value;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let value = match self.cache.take() {
            Some((cache_epoch, cache_value)) if cache_epoch == epoch => cache_value,
            _ => self.ptr.sample(rng, epoch),
        };
        self.cache.set(Some((epoch, value.clone())));
        value
    }
}

impl<U> Uncertain for CachedUncertain<U>
where
    U: Uncertain,
    U::Value: Clone,
{
    type Value = U::Value;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        <&Self as Uncertain>::sample(&self, rng, epoch)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, Uncertain};
    use rand_distr::Normal;
    use rand_pcg::Pcg32;

    #[test]
    fn cached_uncertain_shares_values() {
        let x = Distribution::from(Normal::new(10.0, 1.0).unwrap());
        let x = x.into_cached();
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..1000 {
            assert_eq!((&x).sample(&mut rng, epoch), (&x).sample(&mut rng, epoch));
        }
    }
}
