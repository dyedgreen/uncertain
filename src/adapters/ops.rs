use crate::{Rng, Uncertain};

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

    fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
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

            fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
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

            fn sample(&self, rng: &mut Rng, epoch: usize) -> Self::Value {
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

#[cfg(test)]
mod tests {
    use crate::{PointMass, Uncertain};

    #[test]
    fn op_not() {
        let a = PointMass::new(false);
        assert!(a.not().pr(0.99999));
    }

    #[test]
    fn op_and() {
        let a = PointMass::new(true);
        let b = PointMass::new(true);
        assert!(a.and(b).pr(0.99999));

        let a = PointMass::new(true);
        let b = PointMass::new(false);
        assert_eq!(a.and(b).pr(0.00001), false);
    }

    #[test]
    fn op_or() {
        let a = PointMass::new(false);
        let b = PointMass::new(true);
        assert!(a.or(b).pr(0.99999));

        let a = PointMass::new(false);
        let b = PointMass::new(false);
        assert_eq!(a.or(b).pr(0.00001), false);
    }

    #[test]
    fn op_add() {
        let a = PointMass::new(5);
        let b = PointMass::new(9);
        assert!(a.add(b).map(|sum| sum == 5 + 9).pr(0.99999));
    }

    #[test]
    fn op_sub() {
        let a = PointMass::new(5);
        let b = PointMass::new(9);
        assert!(a.sub(b).map(|sum| sum == 5 - 9).pr(0.99999));
    }

    #[test]
    fn op_mul() {
        let a = PointMass::new(5);
        let b = PointMass::new(9);
        assert!(a.mul(b).map(|sum| sum == 5 * 9).pr(0.99999));
    }

    #[test]
    fn op_div() {
        let a = PointMass::new(5.0);
        let b = PointMass::new(9.0);
        assert!(a.div(b).map(|sum| sum == 5.0 / 9.0).pr(0.99999));
    }
}
