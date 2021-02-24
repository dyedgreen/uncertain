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

macro_rules! logic_op {
    ($name: ident, $op: tt) => {
        pub struct $name<A, B>
        where
            A: Uncertain,
            B: Uncertain,
            A::Value: Into<bool>,
            B::Value: Into<bool>,
        {
            a: A,
            b: B,
        }

        impl<A, B> $name<A, B>
        where
            A: Uncertain,
            B: Uncertain,
            A::Value: Into<bool>,
            B::Value: Into<bool>,
        {
            pub fn new(a: A, b: B) -> Self {
                Self { a, b }
            }
        }

        impl<A, B> Uncertain for $name<A, B>
        where
            A: Uncertain,
            B: Uncertain,
            A::Value: Into<bool>,
            B::Value: Into<bool>,
        {
            type Value = bool;

            fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
                self.a.sample(rng, epoch).into() $op self.b.sample(rng, epoch).into()
            }
        }
    };
}

macro_rules! binary_op {
    ($name:ident, $op:tt, $trait:tt) => {
        pub struct $name<A, B>
        where
            A: Uncertain,
            B: Uncertain,
            A::Value: std::ops::$trait<B::Value>,
        {
            a: A,
            b: B,
        }

        impl<A, B> $name<A, B>
        where
            A: Uncertain,
            B: Uncertain,
            A::Value: std::ops::$trait<B::Value>,
        {
            pub fn new(a: A, b: B) -> Self {
                Self { a, b }
            }
        }

        impl<A, B> Uncertain for $name<A, B>
        where
            A: Uncertain,
            B: Uncertain,
            A::Value: std::ops::$trait<B::Value>,
        {
            type Value = <A::Value as std::ops::$trait<B::Value>>::Output;

            fn sample<R: Rng>(&self, rng: &mut R, epoch: usize) -> Self::Value {
                self.a.sample(rng, epoch) $op self.b.sample(rng, epoch)
            }
        }
    };
}

logic_op!(And, &&);
logic_op!(Or, ||);

binary_op!(Sum, +, Add);
binary_op!(Difference, -, Sub);
binary_op!(Product, *, Mul);
binary_op!(Ratio, /, Div);
