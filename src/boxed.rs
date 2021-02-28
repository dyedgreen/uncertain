use crate::Uncertain;
use rand::Rng;
use std::cell::Cell;
use std::rc::Rc;

/// Boxed uncertain value. An uncertain value which can
/// be cloned. See [`Uncertain::into_boxed`].
#[deprecated(since = "0.2.1", note = "Please use `CachedUncertain` instead")]
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
    pub(crate) fn new(contained: U) -> Self {
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

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R, epoch: usize) -> Self::Value {
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
    use crate::{Distribution, Uncertain};
    use rand_distr::Normal;
    use rand_pcg::Pcg32;

    #[test]
    fn cloned_boxed_shares_values() {
        let x = Distribution::from(Normal::new(10.0, 1.0).unwrap());
        let x = x.into_boxed();
        let y = x.clone();
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..1000 {
            assert_eq!(x.sample(&mut rng, epoch), y.sample(&mut rng, epoch));
        }
    }
}
