use crate::Uncertain;
use rand::Rng;
use std::cell::Cell;
use std::rc::Rc;

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
