use rand::{distributions::Distribution, Rng, SeedableRng};
use rand_distr::Normal;
use rand_pcg::{Pcg32, Pcg64};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

pub trait Uncertain<R: Rng + ?Sized = Pcg32> {
    type Value;

    fn sample(&self, rng: &mut R, epoch: usize) -> Self::Value;

    fn into_boxed(self) -> BoxedUncertain<Self, R>
    where
        Self: 'static + Sized,
        Self::Value: Clone,
    {
        BoxedUncertain {
            ptr: Rc::new(self),
            cache: Rc::new(RefCell::new(None)),
        }
    }

    /// Takes an uncertain value and produces another which
    /// generates values by calling a closure when sampling.
    fn map<O, F>(self, func: F) -> Map<Self, F>
    where
        Self: Sized,
        F: Fn(Self::Value) -> O,
    {
        Map {
            uncertain: self,
            func,
        }
    }

    /// Combine two uncertain values using a closure.
    fn join<U, F>(self, other: U, func: F) -> Join<Self, U, F>
    where
        Self: Sized,
        U: Uncertain<R>,
        F: Fn(Self::Value, U::Value),
    {
        Join {
            a: self,
            b: other,
            func,
        }
    }

    /// Combine two uncertain values by computing their
    /// sum.
    fn add<U>(self, other: U) -> Sum<Self, U>
    where
        Self: Sized,
        U: Uncertain<R>,
        Self::Value: std::ops::Add<U::Value>,
    {
        Sum { a: self, b: other }
    }
}

pub struct BoxedUncertain<U, R>
where
    R: Rng + ?Sized,
    U: Uncertain<R>,
{
    ptr: Rc<U>,
    cache: Rc<RefCell<Option<(usize, U::Value)>>>,
}

impl<U, R> Clone for BoxedUncertain<U, R>
where
    R: Rng + ?Sized,
    U: Uncertain<R>,
{
    fn clone(&self) -> Self {
        BoxedUncertain {
            ptr: self.ptr.clone(),
            cache: self.cache.clone(),
        }
    }
}

impl<U, R> Uncertain<R> for BoxedUncertain<U, R>
where
    R: Rng + ?Sized,
    U: Uncertain<R>,
    U::Value: Clone,
{
    type Value = U::Value;

    fn sample(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let mut cache = self.cache.borrow_mut();
        if let Some((last_epoch, last_value)) = &*cache {
            if *last_epoch == epoch {
                return last_value.clone();
            }
        }
        let value = self.ptr.sample(rng, epoch);
        *cache = Some((epoch, value.clone()));
        value
    }
}

pub struct Map<U, F> {
    uncertain: U,
    func: F,
}

impl<R, U, F> Uncertain<R> for Map<U, F>
where
    R: Rng + ?Sized,
    U: Uncertain<R>,
    F: Fn(U::Value),
{
    type Value = F::Output;

    fn sample(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let v = self.uncertain.sample(rng, epoch);
        (self.func)(v)
    }
}

pub struct Join<A, B, F> {
    a: A,
    b: B,
    func: F,
}

impl<A, B, F, R> Uncertain<R> for Join<A, B, F>
where
    R: Rng + ?Sized,
    A: Uncertain<R>,
    B: Uncertain<R>,
    F: Fn(A::Value, B::Value),
{
    type Value = F::Output;

    fn sample(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        (self.func)(a, b)
    }
}

pub struct Sum<A, B> {
    a: A,
    b: B,
}

impl<A, B, R> Uncertain<R> for Sum<A, B>
where
    R: Rng + ?Sized,
    A: Uncertain<R>,
    B: Uncertain<R>,
    A::Value: std::ops::Add<B::Value>,
{
    type Value = <A::Value as std::ops::Add<B::Value>>::Output;

    fn sample(&self, rng: &mut R, epoch: usize) -> Self::Value {
        let a = self.a.sample(rng, epoch);
        let b = self.b.sample(rng, epoch);
        a + b
    }
}

pub struct UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    dist: D,
    _p: PhantomData<T>,
}

impl<T, D, R> Uncertain<R> for UncertainDistribution<T, D>
where
    R: Rng + ?Sized,
    D: Distribution<T>,
{
    type Value = T;

    fn sample(&self, rng: &mut R, _epoch: usize) -> Self::Value {
        self.dist.sample(rng)
    }
}

impl<T, D> From<D> for UncertainDistribution<T, D>
where
    D: Distribution<T>,
{
    fn from(dist: D) -> Self {
        Self {
            dist,
            _p: PhantomData {},
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clone_shares_values() {
        let x: UncertainDistribution<f32, _> = Normal::new(10.0, 1.0).unwrap().into();
        let x = x.into_boxed();
        let y = x.clone();
        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..1000 {
            assert_eq!(x.sample(&mut rng, epoch), y.sample(&mut rng, epoch));
        }
    }

    #[test]
    fn add() {
        let a: UncertainDistribution<f32, _> = Normal::new(0.0, 1.0).unwrap().into();
        let b: UncertainDistribution<f32, _> = Normal::new(5.0, 1.0).unwrap().into();
        let c = <UncertainDistribution<f32, _> as Uncertain<Pcg32>>::add(a, b);

        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..10 {
            println!("{:?}", c.sample(&mut rng, epoch));
        }
        assert!(false);
    }

    #[test]
    fn reused_add() {
        let x: UncertainDistribution<f32, _> = Normal::new(10.0, 1.0).unwrap().into();
        let y: UncertainDistribution<f32, _> = Normal::new(5.0, 1.0).unwrap().into();
        let x = x.into_boxed();
        let a = x.clone().add(y);
        let b = a.add(x);

        let mut rng = Pcg32::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7);
        for epoch in 0..10 {
            println!("{:?}", b.sample(&mut rng, epoch));
        }
        assert!(false);
    }
}
