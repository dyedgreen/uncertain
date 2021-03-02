use crate::{Rng, Uncertain};

pub struct Map<U, F> {
    uncertain: U,
    func: F,
}

impl<T, U, F> Map<U, F>
where
    U: Uncertain,
    F: Fn(U::Value) -> T,
{
    pub fn new(uncertain: U, func: F) -> Self {
        Self { uncertain, func }
    }
}

impl<T, U, F> Uncertain for Map<U, F>
where
    U: Uncertain,
    F: Fn(U::Value) -> T,
{
    type Value = T;

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
        let v = self.uncertain.sample(rng, epoch);
        (self.func)(v)
    }
}
