use crate::Uncertain;
use rand::Rng;

pub struct Not<U>
where
    U: Uncertain,
    U::Value: Into<bool>,
{
    uncertain: U,
}

impl<U> Not<U>
where
    U: Uncertain,
    U::Value: Into<bool>,
{
    pub fn new(uncertain: U) -> Self {
        Self { uncertain }
    }
}

impl<U> Uncertain for Not<U>
where
    U: Uncertain,
    U::Value: Into<bool>,
{
    type Value = bool;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        !self.uncertain.sample(rng, epoch).into()
    }
}

pub struct Sum<A, B>
where
    A: Uncertain,
    B: Uncertain,
    A::Value: std::ops::Add<B::Value>,
{
    a: A,
    b: B,
}

impl<A, B> Sum<A, B>
where
    A: Uncertain,
    B: Uncertain,
    A::Value: std::ops::Add<B::Value>,
{
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A, B> Uncertain for Sum<A, B>
where
    A: Uncertain,
    B: Uncertain,
    A::Value: std::ops::Add<B::Value>,
{
    type Value = <A::Value as std::ops::Add<B::Value>>::Output;

    fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        a + b
    }
}
