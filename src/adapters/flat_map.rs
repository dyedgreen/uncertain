use crate::{Rng, Uncertain};

pub struct FlatMap<U, F> {
    uncertain: U,
    func: F,
}

impl<O, U, F> FlatMap<U, F>
where
    U: Uncertain,
    O: Uncertain,
    F: Fn(U::Value) -> O,
{
    pub fn new(uncertain: U, func: F) -> Self {
        Self { uncertain, func }
    }
}

impl<O, U, F> Uncertain for FlatMap<U, F>
where
    U: Uncertain,
    O: Uncertain,
    F: Fn(U::Value) -> O,
{
    type Value = O::Value;

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
        let v = self.uncertain.sample(rng, epoch);
        (self.func)(v).sample(rng, epoch)
    }
}
