use crate::{Rng, Uncertain, UncertainBase};

pub struct Join<A, B, F> {
    a: A,
    b: B,
    func: F,
}

impl<O, A, B, F> Join<A, B, F>
where
    A: Uncertain,
    B: Uncertain,
    F: Fn(A::Value, B::Value) -> O,
{
    pub fn new(a: A, b: B, func: F) -> Self {
        Self { a, b, func }
    }
}

impl<O, A, B, F> UncertainBase for Join<A, B, F>
where
    A: Uncertain,
    B: Uncertain,
    F: Fn(A::Value, B::Value) -> O,
{
    type Value = O;

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        (self.func)(a, b)
    }
}
