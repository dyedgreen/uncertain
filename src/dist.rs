use crate::Uncertain;
use rand::Rng;
use std::marker::PhantomData;

pub struct Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    dist: D,
    _p: PhantomData<T>,
}

impl<T, D> Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    pub fn new(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}

impl<T, D> Uncertain for Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    type Value = T;

    fn sample<R: Rng>(&self, rng: &mut R, _epoch: usize) -> Self::Value {
        self.dist.sample(rng)
    }
}

impl<T, D> From<D> for Distribution<T, D>
where
    D: rand::distributions::Distribution<T>,
{
    fn from(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}
